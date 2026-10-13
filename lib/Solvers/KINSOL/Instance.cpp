#ifdef SUNDIALS_ENABLE

#include "marco/Runtime/Solvers/KINSOL/Instance.h"
#include "kinsol/kinsol.h"
#include "marco/Runtime/Simulation/Options.h"
#include "marco/Runtime/Solvers/KINSOL/Options.h"
#include "marco/Runtime/Solvers/KINSOL/Profiler.h"
#include "marco/Runtime/Solvers/KINSOL/SparseNonlinearSystem.h"
#include "marco/Runtime/Support/MemoryManagement.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <set>
#include <thread>

using namespace marco::runtime::sundials;
using namespace marco::runtime::sundials::kinsol;

//===---------------------------------------------------------------------===//
// Solver
//===---------------------------------------------------------------------===//

namespace marco::runtime::sundials::kinsol {
template <typename Callback>
static double invokeScalarBlockCallback(Callback &&callback) {
#ifdef THREADS_ENABLE
  // 中文：保持已有单 worker 直接执行方程体的异常边界。
  // Match the existing one-worker direct equation body's exception boundary.
  try {
    return callback();
  } catch (...) {
    std::terminate();
  }
#else
  return callback();
#endif
}

static const char *getParallelIterationKindName(
    EquationsParallelIterationKind kind) {
  switch (kind) {
  case EquationsParallelIterationKind::Residuals:
    return "residual";
  case EquationsParallelIterationKind::Jacobian:
    return "Jacobian";
  }

  return "unknown";
}

static void printParallelIterationStats(
    EquationsParallelIterationKind kind,
    const std::vector<::marco::runtime::profiling::ParallelThreadWorkStats>
        &threadWork,
    const std::vector<std::thread::id> &threadIds) {
  std::cerr << "[KINSOL] " << getParallelIterationKindName(kind)
            << " parallel iteration work" << std::endl;

  for (size_t i = 0, e = threadWork.size(); i < e; ++i) {
    std::cerr << "  worker " << i << " (thread " << threadIds[i]
              << "): chunks=" << threadWork[i].chunks
              << ", scalar equations=" << threadWork[i].scalarEquations
              << std::endl;
  }
}

KINSOLInstance::KINSOLInstance() {
  // Initially there is are no variables or equations in the instance.
  variableOffsets.push_back(0);
  equationOffsets.push_back(0);

  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[KINSOL] Instance created" << std::endl;
  }
}

struct KINSOLInstance::C2KernelState {
  MarcoC2KernelV1 descriptor{};
  std::vector<double> values, residuals, matrix;
  std::vector<uint64_t> kinds, offsets{0}, rows, columns;
  uint64_t solved{0}, declined{0}, healthAccepted{0}, healthDeclined{0};
};

// 中文：绑定实际 scalar F/J、getter/setter、配对和 V1 布局；同名函数或数值零都不能替代身份一致。
// English: Bind actual scalar F/J, accessors, pairing and V1 layout; equal names or observed
// zeros cannot replace identity checks.
bool KINSOLInstance::bindC2KernelV1(const MarcoC2KernelV1 &d) {
  auto invalid = [&](const char *reason, uint64_t row = 0,
                     uint64_t column = 0) {
    std::cerr << "[KINSOL] c2_kernel_v1_invalid_binding reason=" << reason
              << " domain=" << d.domain << " epoch=" << d.epoch
              << " component=" << d.component << " row=" << row
              << " column=" << column << '\n';
    return false;
  };
  if (initialized || c2Kernel || d.version != 1 ||
      d.descriptorBytes != sizeof(MarcoC2KernelV1) || !d.rows || d.rows>16 ||
      !d.residuals || !d.jacobians || !d.getters || !d.setters || !d.matchedColumns ||
      !d.residual || !d.jacobian || !d.install || !d.read || !d.health || !d.solve ||
      d.scratchBytes>65536)
    return invalid("descriptor_or_lifecycle");
  const auto n=d.rows;
  if (variableGetters.size()!=n || variableSetters.size()!=n ||
      equationRanges.size()!=n || timedResidualFunctions.size()!=n ||
      timedResidualPostprocessor)
    return invalid("shape_or_postprocessor");
  for (uint64_t row=0; row<n; ++row) {
    if (getVariableFlatSize(row)!=1 || getEquationFlatSize(row)!=1 ||
        reinterpret_cast<const void *>(variableGetters[row])!=d.getters[row] ||
        reinterpret_cast<const void *>(variableSetters[row])!=d.setters[row] ||
        reinterpret_cast<const void *>(timedResidualFunctions[row])!=d.residuals[row])
      return invalid("scalar_callback", row);
    for (uint64_t column=0; column<n; ++column) {
      const void *callback = row<timedJacobianFunctions.size() &&
          column<timedJacobianFunctions[row].size()
              ? reinterpret_cast<const void *>(timedJacobianFunctions[row][column].first)
              : nullptr;
      if (callback!=d.jacobians[row*n+column])
        return invalid("jacobian_callback", row, column);
    }
  }
  // 中文：缺少可选多项式/分区证明不是输入错误。
  // Missing optional polynomial/partition proof is not an input error.
  if (!scalarBlockPlan || !scalarBlockBudget) return true;
  if (scalarBlockRows.size()!=n) return invalid("plan_size");
  for (uint64_t row=0;row<n;++row)
    if (scalarBlockRows[row].matchedColumn!=d.matchedColumns[row])
      return invalid("selected_pairing", row);
  // 中文：显式隔离可支持仿射算术以外的表达式；本后端只有在既有多项式证明其线性时才使用线性求解。
  // Explicit isolation can support more than affine arithmetic. This backend
  // only uses its linear solve when the existing polynomial proves that fact.
  for (const auto &block:scalarBlockPlan->getExecutionBlocks()) {
    if (block.kind!=ScalarBlockPlan::Kind::Explicit) continue;
    uint64_t mask=0;
    for(auto c:block.columns) mask|=uint64_t(1)<<c;
    for(auto row:block.rows)
      for(const auto &term:scalarBlockRows[row].terms)
        if ((term.squared&mask) || __builtin_popcountll(term.linear&mask)>1)
          return true;
  }
  uint64_t bytes=(n*n+10*n+4)*sizeof(uint64_t)+sizeof(C2KernelState);
  if (scalarBlockBudget->work<n*n+4*n || scalarBlockBudget->storage<bytes)
    return true;
  scalarBlockBudget->work-=n*n+4*n;
  scalarBlockBudget->storage-=bytes;
  auto state=std::make_unique<C2KernelState>();
  state->descriptor=d;
  state->values.resize(n);state->residuals.resize(n);state->matrix.resize(n*n);
  for(const auto &block:scalarBlockPlan->getExecutionBlocks()) {
    state->kinds.push_back(static_cast<uint64_t>(block.kind));
    state->rows.insert(state->rows.end(),block.rows.begin(),block.rows.end());
    state->columns.insert(state->columns.end(),block.columns.begin(),block.columns.end());
    state->offsets.push_back(state->rows.size());
  }
  c2Kernel=std::move(state);
  solveEntry = &KINSOLInstance::solveBody<true>;
  return true;
}

// 中文：生成健康计算只能在当前求值作用域内提供数值证据；无法认证时保留原完整检查。
// English: Generated health evaluation supplies numerical evidence only in the current scope;
// unavailable certification retains the original full check.
bool KINSOLInstance::certifyC2KernelPoint(realtype time,
    const std::vector<uint64_t> &rows, const std::vector<uint64_t> &columns,
    SparseNonlinearSystem::ConditionEstimate &chart,
    SparseNonlinearSystem::ConditionEstimate &closure) {
  if (!c2Kernel || !initialized || !nonlinearSystem) return false;
  auto &state=*c2Kernel;
  currentTime=time;
  state.descriptor.read(state.values.data());
  const SuccessfulPoint *point=successfulPoint;
  successfulPoint=nullptr;
  // 中文：成功点残差只能在当前时间及局部值逐位一致时消费一次；Jacobian 始终在实际当前点重算，避免把旧 Newton 点当作健康证据。
  // English: Consume a successful-point residual once only when time and local values match
  //          bitwise. Always recompute the current-point Jacobian instead of treating an old Newton
  //          iterate as health evidence.
  bool same=point && std::memcmp(&point->time,&time,sizeof(realtype))==0 &&
      point->variables.size()==state.values.size() &&
      std::memcmp(point->variables.data(),state.values.data(),
                  state.values.size()*sizeof(double))==0;
  if (same) state.residuals=point->residuals;
  else invokeScalarBlockCallback([&] {
    state.descriptor.residual(time,~uint64_t(0),state.residuals.data());return 0.;
  });
  invokeScalarBlockCallback([&] {
    state.descriptor.jacobian(time,~uint64_t(0),~uint64_t(0),state.matrix.data());return 0.;
  });
  if (marco::runtime::simulation::getOptions().profiling) {
    auto &observation=nonlinearSystem->lastResidualObservation;
    observation={true,true,0,0};
    for(size_t row=0;row<state.residuals.size();++row) {
      auto value=state.residuals[row];
      observation.finite &= std::isfinite(value);
      observation.maximumNorm=std::max(observation.maximumNorm,std::abs(value));
      observation.scaledMaximumNorm=std::max(observation.scaledMaximumNorm,
          std::abs(value)*nonlinearSystem->getResidualScale()[row]);
    }
  }
  // 中文：先排除非有限 F/J，再用生成核的完整闭包健康界与独立 chart condition 共同认证；快速界未命中不等于奇异性证明。
  // English: Reject non-finite F/J before combining the generated closure-health bound with an
  //          independent chart-condition check. A missed fast bound is not a singularity
  //          certificate.
  for(double value:state.residuals) if(!std::isfinite(value)) return false;
  for(double value:state.matrix) if(!std::isfinite(value)) return false;
  double health[3]{};
  if (state.descriptor.health(state.matrix.data(),
      nonlinearSystem->getVariableScale().data(),
      nonlinearSystem->getResidualScale().data(),health)!=1) {
    ++state.healthDeclined;return false;
  }
  closure={};
  closure.rank=state.values.size();
  closure.oneNormCondition=health[0];closure.reciprocalCondition=1./health[0];
  closure.minimumPivot=health[1];closure.maximumPivot=health[2];
  if (!nonlinearSystem->estimateFlatCondition(state.matrix.data(),rows,columns,chart) ||
      chart.rankDeficient || !std::isfinite(chart.oneNormCondition) ||
      chart.oneNormCondition>=500) return false;
  ++state.healthAccepted;
  return true;
}


KINSOLInstance::~KINSOLInstance() {
  if (c2Kernel && marco::runtime::simulation::getOptions().profiling)
    std::cerr << "[c2-kernel-v1] rows=" << c2Kernel->descriptor.rows
              << " solved=" << c2Kernel->solved << " declined=" << c2Kernel->declined
              << " healthy=" << c2Kernel->healthAccepted
              << " health_fallback=" << c2Kernel->healthDeclined << '\n';
  if (marco::runtime::simulation::getOptions().profiling && scalarBlockPlan)
    std::cerr << "[kinsol-scalar-blocks] solver_id=" << getStatisticsId()
              << " blocks=" << scalarBlockPlan->getExecutionBlocks().size()
              << " accepted=" << scalarBlockAccepted
              << " original_solver=" << scalarBlockDeclined << '\n';
  if (marco::runtime::simulation::getOptions().profiling)
    std::cerr << "[kinsol-call-statistics] solver_id=" << getStatisticsId()
              << " scope=lifetime_cumulative scalar_residual_calls="
              << scalarResidualCalls.load() << " scalar_jacobian_calls="
              << scalarJacobianCalls.load() << '\n';
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[KINSOL] Instance destroyed" << std::endl;
  }
}

bool KINSOLInstance::requestImmutableAccessLayout(uint64_t workLimit,
                                                 uint64_t storageLimit) {
  if (initialized || workLimit > 1000000 || storageLimit > 16000000)
    return false;
  immutableAccessLayoutRequested = true;
  layoutWorkLimit = workLimit;
  layoutStorageLimit = storageLimit;
  return true;
}

// 中文：注册几何改变时一并失效生成核、成功点回执、标量分块和访问布局，并恢复普通 solve 入口；不能保留旧绑定的部分优化状态。
// English: Registration changes invalidate the generated kernel, successful-point receipt, scalar
//          blocks, and access layout together, restoring ordinary solve dispatch. No partial
//          optimization may retain stale bindings.
void KINSOLInstance::invalidateAccessLayout() {
  solveEntry = &KINSOLInstance::solveBody<false>;
  c2Kernel.reset();
  successfulPoint = nullptr;
  scalarBlockSolver.reset();
  scalarBlockPlan.reset();
  scalarBlockRows.clear();
  scalarExplicitValues.clear();
  scalarBlockBudget = nullptr;
  scalarBlockPrepared = false;
  immutableAccessLayoutRequested = false;
  accessLayout.reset();
}

bool KINSOLInstance::requestScalarBlockPlan(
    std::vector<ScalarBlockPlan::Row> rows, ScalarBlockPlan::Budget &budget) {
  return requestScalarBlockPlan(std::move(rows), {}, budget);
}

bool KINSOLInstance::requestScalarBlockPlan(
    std::vector<ScalarBlockPlan::Row> rows,
    std::vector<TimedResidualFunction> explicitValues,
    ScalarBlockPlan::Budget &budget) {
  if (initialized) return false;
  if (!explicitValues.empty() && explicitValues.size() != rows.size())
    return false;
  for (size_t row = 0; row < rows.size(); ++row)
    if (rows[row].explicitValueCertified !=
        (!explicitValues.empty() && explicitValues[row] != nullptr))
      return false;
  auto [status, plan] = ScalarBlockPlan::build(rows, budget);
  if (status == ScalarBlockPlan::Status::InvalidInput) return false;
  if (status != ScalarBlockPlan::Status::Ready) return true;
  scalarBlockRows = std::move(rows);
  scalarExplicitValues = std::move(explicitValues);
  scalarBlockPlan = std::move(plan);
  scalarBlockBudget = &budget;
  return true;
}

Variable KINSOLInstance::addVariable(uint64_t rank, const uint64_t *dimensions,
                                     VariableGetter getterFunction,
                                     VariableSetter setterFunction,
                                     const char *name) {
  invalidateAccessLayout();
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[KINSOL] Adding algebraic variable";

    if (name != nullptr) {
      std::cerr << " \"" << name << "\"";
    }

    std::cerr << std::endl;
  }

  // Add variable offset and dimensions.
  assert(variableOffsets.size() == variablesDimensions.size() + 1);

  VariableDimensions varDimension(rank);
  uint64_t flatSize = 1;

  for (uint64_t i = 0; i < rank; ++i) {
    flatSize *= dimensions[i];
    varDimension[i] = dimensions[i];
  }

  variablesDimensions.push_back(std::move(varDimension));

  size_t offset = variableOffsets.back();
  variableOffsets.push_back(offset + flatSize);

  // Store the getter and setter functions.
  variableGetters.push_back(getterFunction);
  variableSetters.push_back(setterFunction);
  // 中文：nominal getter 与 variable ID 平行占位，允许调用方稍后在 initialize
  // 前补充显式尺度，而不改变 descriptor 编号。
  // English: Reserve a nominal-getter slot parallel to the variable ID so a
  // caller may attach explicit scaling before initialize without renumbering
  // descriptors.
  variableNominalGetters.push_back(nullptr);

  // Return the index of the variable.
  Variable id = getNumOfArrayVariables() - 1;

  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "  - ID: " << id << std::endl;
    std::cerr << "  - Rank: " << rank << std::endl;
    std::cerr << "  - Dimensions: [";

    for (uint64_t i = 0; i < rank; ++i) {
      if (i != 0) {
        std::cerr << ",";
      }

      std::cerr << dimensions[i];
    }

    std::cerr << "]" << std::endl;
    std::cerr << "  - Getter function address: "
              << reinterpret_cast<void *>(getterFunction) << std::endl;
    std::cerr << "  - Setter function address: "
              << reinterpret_cast<void *>(setterFunction) << std::endl;
  }

  return id;
}

bool KINSOLInstance::setVariableNominalGetter(
    Variable variable, VariableGetter nominalGetter) {
  // 中文：nominal getter 属于变量描述的一部分，初始化后禁止更改，避免已建立的
  // KINSOL scaling 与 MARCO 元数据分叉。
  // English: A nominal getter is part of the variable descriptor and cannot
  // change after initialization, preventing KINSOL scaling from diverging
  // from MARCO metadata.
  if (initialized || nominalGetter == nullptr ||
      variable >= variableNominalGetters.size()) {
    return false;
  }
  variableNominalGetters[variable] = nominalGetter;
  return true;
}

Equation KINSOLInstance::addEquation(const int64_t *ranges,
                                     uint64_t equationRank,
                                     const char *stringRepresentation) {
  invalidateAccessLayout();
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[KINSOL] Adding equation";

    if (stringRepresentation != nullptr) {
      std::cerr << " \"" << stringRepresentation << "\"";
    }

    std::cerr << std::endl;
  }

  // Add the start and end dimensions of the current equation.
  MultidimensionalRange eqRanges = {};
  uint64_t flatSize = 1;

  for (size_t i = 0, e = equationRank * 2; i < e; i += 2) {
    int64_t begin = ranges[i];
    int64_t end = ranges[i + 1];
    eqRanges.push_back({begin, end});
    flatSize *= end - begin;
  }

  equationRanges.push_back(eqRanges);
  size_t offset = equationOffsets.back();
  equationOffsets.push_back(offset + flatSize);

  // Return the index of the equation.
  Equation id = getNumOfVectorizedEquations() - 1;

  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "  - ID: " << id << std::endl;
    std::cerr << "  - Rank: " << equationRank << std::endl;
    std::cerr << "  - Ranges: [";

    for (uint64_t i = 0; i < equationRank; ++i) {
      if (i != 0) {
        std::cerr << ",";
      }

      std::cerr << "[" << ranges[i * 2] << "," << (ranges[i * 2 + 1] - 1)
                << "]";
    }

    std::cerr << "]" << std::endl;
  }

  return id;
}

void KINSOLInstance::addVariableAccess(Equation equation, Variable variable,
                                       AccessFunction accessFunction) {
  invalidateAccessLayout();
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[KINSOL] Adding access information" << std::endl;
    std::cerr << "  - Equation: " << equation << std::endl;
    std::cerr << "  - Variable: " << variable << std::endl;
    std::cerr << "  - Access function address: "
              << reinterpret_cast<void *>(accessFunction) << std::endl;
  }

  assert(equation < getNumOfVectorizedEquations());
  assert(variable < getNumOfArrayVariables());

  precomputedAccesses = true;

  if (variableAccesses.size() <= (size_t)equation) {
    variableAccesses.resize(equation + 1);
  }

  auto &varAccessList = variableAccesses[equation];
  varAccessList.emplace_back(variable, accessFunction);
}

void KINSOLInstance::setResidualFunction(Equation equation,
                                         ResidualFunction residualFunction) {
  invalidateAccessLayout();
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[KINSOL] Setting residual function for equation " << equation
              << ". Address: " << reinterpret_cast<void *>(residualFunction)
              << std::endl;
  }

  if (residualFunctions.size() <= equation) {
    residualFunctions.resize(equation + 1, nullptr);
  }

  residualFunctions[equation] = residualFunction;
}

void KINSOLInstance::setTimedResidualFunction(
    Equation equation, TimedResidualFunction residualFunction) {
  invalidateAccessLayout();
  // 中文：timed 表与普通表按相同 equation ID 对齐；实际求值优先 timed callback，
  // 因而 component closure 可读取当前 IDA time 而不改变既有 ABI。
  // English: The timed table aligns with ordinary equation IDs. Evaluation
  // prefers the timed callback, letting component closures observe current IDA
  // time without changing the established ABI.
  if (timedResidualFunctions.size() <= equation) {
    timedResidualFunctions.resize(equation + 1, nullptr);
  }
  timedResidualFunctions[equation] = residualFunction;
}

void KINSOLInstance::setTimedResidualPostprocessor(
    TimedResidualPostprocessor callback) {
  assert(!initialized && "KINSOL residual postprocessor must be set before "
                         "initialize");
  invalidateAccessLayout();
  timedResidualPostprocessor = std::move(callback);
}

void KINSOLInstance::addJacobianFunction(Equation equation, Variable variable,
                                         JacobianFunction jacobianFunction,
                                         uint64_t numOfSeeds,
                                         uint64_t *seedSizes) {
  invalidateAccessLayout();
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[KINSOL] Setting jacobian function for equation " << equation
              << " and variable " << variable << "\n"
              << "  - Address: " << reinterpret_cast<void *>(jacobianFunction)
              << "\n"
              << "  - Number of seeds : " << numOfSeeds;

    if (numOfSeeds != 0) {
      std::cerr << "\n" << "  - Seed sizes: ";

      for (uint64_t i = 0; i < numOfSeeds; ++i) {
        if (i != 0) {
          std::cerr << ", ";
        }

        std::cerr << seedSizes[i];
      }
    }

    std::cerr << std::endl;
  }

  if (jacobianFunctions.size() <= equation) {
    jacobianFunctions.resize(equation + 1, {});
  }

  if (jacobianFunctions[equation].size() <= variable) {
    jacobianFunctions[equation].resize(
        variable + 1, std::make_pair(nullptr, std::vector<uint64_t>{}));
  }

  jacobianFunctions[equation][variable].first = jacobianFunction;
  jacobianFunctions[equation][variable].second.resize(numOfSeeds);

  for (uint64_t i = 0; i < numOfSeeds; ++i) {
    assert(seedSizes[i] != 0);
    jacobianFunctions[equation][variable].second[i] = seedSizes[i];
  }
}

void KINSOLInstance::addTimedJacobianFunction(
    Equation equation, Variable variable,
    TimedJacobianFunction jacobianFunction, uint64_t numOfSeeds,
    uint64_t *seedSizes) {
  invalidateAccessLayout();
  // 中文：seed shape 与 callback 一起保存，后续 chunk 构造按真实 callback 地址
  // 分配 AD buffer；timed-only 配置不依赖空的普通 callback。
  // English: Seed shapes are stored with the callback. Chunk construction
  // allocates AD buffers by the actual callback address, so timed-only
  // configurations never depend on a null ordinary callback.
  if (timedJacobianFunctions.size() <= equation) {
    timedJacobianFunctions.resize(equation + 1, {});
  }
  if (timedJacobianFunctions[equation].size() <= variable) {
    timedJacobianFunctions[equation].resize(
        variable + 1,
        std::make_pair(nullptr, std::vector<uint64_t>{}));
  }
  auto &descriptor = timedJacobianFunctions[equation][variable];
  descriptor.first = jacobianFunction;
  descriptor.second.assign(seedSizes, seedSizes + numOfSeeds);
}

bool KINSOLInstance::initialize() {
  assert(!initialized && "KINSOL instance has already been initialized");

  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[KINSOL] Performing initialization" << std::endl;
  }

  memoryPoolId = MemoryPoolManager::getInstance().create();

  // Compute the number of scalar variables.
  scalarVariablesNumber = 0;

  for (Variable var = 0, e = getNumOfArrayVariables(); var < e; ++var) {
    scalarVariablesNumber += getVariableFlatSize(var);
  }

  // Compute the number of scalar equations.
  scalarEquationsNumber = 0;

  for (Equation eq = 0, e = getNumOfVectorizedEquations(); eq < e; ++eq) {
    scalarEquationsNumber += getEquationFlatSize(eq);
  }

  assert(getNumOfScalarVariables() == getNumOfScalarEquations() &&
         "Unbalanced system");

  if (scalarEquationsNumber == 0) {
    // KINSOL has nothing to solve.
    initialized = true;
    return true;
  }

  // 中文：局部 component 可使用普通或带时间参数的 callback；初始化在 release
  // 构建中也执行完整性检查，缺失函数不会依赖 assert 后继续运行。
  // English: A local component may use ordinary or time-aware callbacks.
  // Initialization validates completeness in release builds as well, rather
  // than relying on assertions that may disappear.
  assert(residualFunctions.size() == getNumOfVectorizedEquations() ||
         timedResidualFunctions.size() == getNumOfVectorizedEquations());
  for (Equation equation = 0, e = getNumOfVectorizedEquations(); equation < e;
       ++equation) {
    bool hasResidual = equation < residualFunctions.size() &&
                       residualFunctions[equation] != nullptr;
    bool hasTimedResidual = equation < timedResidualFunctions.size() &&
                            timedResidualFunctions[equation] != nullptr;
    if (!hasResidual && !hasTimedResidual) {
      std::cerr << "KINSOL local component is missing a residual callback"
                << std::endl;
      return false;
    }
  }

  // 中文：没有预计算 access 时，每个 equation-variable 对必须提供普通或 timed
  // Jacobian callback。
  // English: Without precomputed accesses, every equation-variable pair needs
  // either the time-independent or the time-aware Jacobian callback.
  if (!precomputedAccesses) {
    for (Equation equation = 0, e = getNumOfVectorizedEquations(); equation < e;
         ++equation) {
      for (Variable variable = 0, v = variableGetters.size(); variable < v;
           ++variable) {
        bool hasJacobian = equation < jacobianFunctions.size() &&
                           variable < jacobianFunctions[equation].size() &&
                           jacobianFunctions[equation][variable].first !=
                               nullptr;
        bool hasTimedJacobian =
            equation < timedJacobianFunctions.size() &&
            variable < timedJacobianFunctions[equation].size() &&
            timedJacobianFunctions[equation][variable].first != nullptr;
        if (!hasJacobian && !hasTimedJacobian) {
          std::cerr << "KINSOL is missing a Jacobian callback for equation "
                    << equation << " and variable " << variable << std::endl;
          return false;
        }
      }
    }
  }

  // Check that all the getters and setters have been set.
  assert(
      std::none_of(variableGetters.begin(), variableGetters.end(),
                   [](VariableGetter getter) { return getter == nullptr; }) &&
      "Not all the variable getters have been set");

  assert(
      std::none_of(variableSetters.begin(), variableSetters.end(),
                   [](VariableSetter setter) { return setter == nullptr; }) &&
      "Not all the variable setters have been set");

  // Reserve the space for data of the jacobian matrix.
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[KINSOL] Reserving space for the data of the Jacobian matrix"
              << std::endl;
  }

  jacobianMatrixData.resize(scalarEquationsNumber);

  // 中文：计划只拥有不可变几何及固定回调/seed 绑定，不保留数值。
  //       完整构造后才发布；耗尽时丢弃计划，不改变矩阵或必需初始化工作。
  // This plan owns immutable geometry and fixed callback/seed bindings, never
  // numerical values. Publish only after complete construction; exhaustion
  // drops it without changing the matrix or required initialization work.
  std::unique_ptr<ImmutableAccessLayout> layout;
  if (immutableAccessLayoutRequested && layoutWorkLimit != 0 &&
      layoutStorageLimit >= sizeof(ImmutableAccessLayout)) {
    layout = std::make_unique<ImmutableAccessLayout>();
  }
  uint64_t remainingWork = layoutWorkLimit;
  uint64_t remainingStorage = layoutStorageLimit;
  auto charge = [&](uint64_t count, uint64_t bytesPerItem) {
    if (!layout)
      return false;
    if (count > remainingWork ||
        (bytesPerItem != 0 && count > remainingStorage / bytesPerItem)) {
      layout.reset();
      return false;
    }
    remainingWork -= count;
    remainingStorage -= count * bytesPerItem;
    return true;
  };
  if (charge(1, sizeof(ImmutableAccessLayout)) &&
      charge(scalarEquationsNumber, sizeof(std::vector<JacobianColumn>)))
    layout->columns.resize(scalarEquationsNumber);

  uint64_t numOfVectorizedEquations = getNumOfVectorizedEquations();

  for (Equation eq = 0; eq < numOfVectorizedEquations; ++eq) {
    std::vector<int64_t> equationIndices;
    getEquationBeginIndices(eq, equationIndices);

    do {
      uint64_t equationArrayOffset = equationOffsets[eq];

      uint64_t equationScalarOffset =
          getEquationFlatIndex(equationIndices, equationRanges[eq]);

      uint64_t scalarEquationIndex = equationArrayOffset + equationScalarOffset;

      // Compute the column indexes that may be non-zeros.
      std::vector<JacobianColumn> jacobianColumns =
          computeJacobianColumns(eq, equationIndices.data());

      jacobianMatrixData[scalarEquationIndex].resize(jacobianColumns.size());

      if (marco::runtime::simulation::getOptions().debug) {
        std::cerr << "  - Equation " << eq << std::endl;
        std::cerr << "    Equation indices: ";
        printIndices(equationIndices);
        std::cerr << std::endl;

        std::cerr << "    Scalar equation index: " << scalarEquationIndex
                  << std::endl;

        std::cerr << "    Number of possibly non-zero columns: "
                  << jacobianColumns.size() << std::endl;
      }
      if (charge(jacobianColumns.capacity(), sizeof(JacobianColumn))) {
        for (const auto &column : jacobianColumns) {
          if (!charge(column.second.capacity(), sizeof(uint64_t)))
            break;
        }
        if (layout)
          layout->columns[scalarEquationIndex] = std::move(jacobianColumns);
      }
    } while (advanceEquationIndices(equationIndices, equationRanges[eq]));
  }

  // Compute the total amount of non-zero values in the Jacobian Matrix.
  computeNNZ();

  // Compute the equation chunks for each thread.
  computeThreadChunks();

  if (charge(scalarVariablesNumber,
             sizeof(ImmutableAccessLayout::VariableEntry))) {
    layout->variables.reserve(scalarVariablesNumber);
    for (Variable variable = 0; layout && variable < getNumOfArrayVariables();
         ++variable) {
      std::vector<uint64_t> indices;
      getVariableBeginIndices(variable, indices);
      if (getVariableFlatSize(variable) == 0)
        continue;
      do {
        if (!charge(indices.size(), sizeof(uint64_t)))
          break;
        uint64_t offset = variableOffsets[variable] +
                          getVariableFlatIndex(variablesDimensions[variable],
                                               indices.data());
        layout->variables.push_back({variable, indices, offset});
      } while (advanceVariableIndices(indices, variablesDimensions[variable]));
    }
  }
  if (charge(scalarEquationsNumber, sizeof(ImmutableAccessLayout::RowEntry))) {
    layout->rows.reserve(scalarEquationsNumber);
    for (size_t chunk = 0; layout && chunk < threadEquationsChunks.size();
         ++chunk) {
      const auto &descriptor = threadEquationsChunks[chunk];
      auto indices = std::get<1>(descriptor);
      do {
        if (!charge(indices.size(), sizeof(int64_t)))
          break;
        auto equation = std::get<0>(descriptor);
        auto offset = equationOffsets[equation] +
                      getEquationFlatIndex(indices, equationRanges[equation]);
        layout->rows.push_back({chunk, indices, offset});
      } while (advanceEquationIndicesUntil(
          indices, equationRanges[std::get<0>(descriptor)],
          std::get<2>(descriptor)));
    }
  }
  if (layout && (layout->variables.size() != scalarVariablesNumber ||
                 layout->rows.size() != scalarEquationsNumber))
    layout.reset();
  if (layout && threadPool.getNumOfThreads() == 1 &&
      charge(scalarEquationsNumber,
             sizeof(std::vector<ImmutableAccessLayout::JacobianEntry>))) {
    layout->evaluations.resize(scalarEquationsNumber);
    for (size_t row = 0; layout && row < layout->rows.size(); ++row) {
      const auto &entry = layout->rows[row];
      const auto &chunk = threadEquationsChunks[entry.chunk];
      Equation equation = std::get<0>(chunk);
      const auto &seedsMap = std::get<3>(chunk);
      const auto &columns = layout->columns[entry.offset];
      if (!charge(columns.size(), sizeof(ImmutableAccessLayout::JacobianEntry)))
        break;
      auto &evaluations = layout->evaluations[entry.offset];
      evaluations.reserve(columns.size());
      for (const auto &column : columns) {
        Variable variable = column.first;
        if (!charge(column.second.size() + seedsMap.size() + 1, 0))
          break;
        auto callback = equation < jacobianFunctions.size() &&
                                variable < jacobianFunctions[equation].size()
                            ? jacobianFunctions[equation][variable].first
                            : nullptr;
        auto timed = equation < timedJacobianFunctions.size() &&
                             variable < timedJacobianFunctions[equation].size()
                         ? timedJacobianFunctions[equation][variable].first
                         : nullptr;
        // 中文：与普通路径使用相同 seed 键及 timed-callback 优先级；缓冲直到 chunk 销毁均稳定，新注册在修改描述符前使绑定失效。
        // Use the same seed key and timed-callback precedence as the ordinary
        // path. These buffers are stable until chunk destruction; registration
        // invalidates the binding before changing any descriptor.
        uintptr_t key = callback ? reinterpret_cast<uintptr_t>(callback)
                                 : reinterpret_cast<uintptr_t>(timed);
        auto seeds = seedsMap.find(key);
        auto offset = variableOffsets[variable] +
                      getVariableFlatIndex(variablesDimensions[variable],
                                           column.second);
        evaluations.push_back({static_cast<sunindextype>(offset), callback,
                               timed, seeds == seedsMap.end()
                                          ? nullptr : seeds->second.data()});
      }
    }
  }
  accessLayout = std::move(layout);

  if (scalarBlockPlan) {
    if (scalarBlockRows.size() != scalarEquationsNumber ||
        scalarVariablesNumber != variablesDimensions.size() ||
        scalarEquationsNumber != equationRanges.size()) return false;
    if (!accessLayout || accessLayout->evaluations.empty() ||
        timedResidualPostprocessor) {
      // 中文：生成核派生于当前精确执行计划；多 worker 或不支持布局必须让计划和生成核同时失效。
      // Generated kernels are derived from this exact execution plan. A
      // multi-worker or unsupported layout must invalidate both together.
      c2Kernel.reset();
      solveEntry = &KINSOLInstance::solveBody<false>;
      scalarBlockPlan.reset();
      scalarBlockRows.clear();
    } else {
      for (size_t row = 0; row < scalarBlockRows.size(); ++row) {
        uint64_t reads = 0;
        for (const auto &entry : accessLayout->evaluations[row]) {
          if (entry.column < 0 || entry.column >= 64) return false;
          reads |= uint64_t(1) << entry.column;
        }
        if (reads != scalarBlockRows[row].reads) return false;
      }
    }
  }

  realtype effectiveMaximumNewtonStep =
      maximumNewtonStep.value_or(getOptions().maxNewtonStep);
  if (effectiveMaximumNewtonStep <= 0) {
    effectiveMaximumNewtonStep = std::max<realtype>(
        100, 10 * std::sqrt(static_cast<realtype>(scalarVariablesNumber)));
  }
  // 中文：KINSOLInstance 与 constraint-local solve 共享同一个稀疏非线性内核，
  // 因而 scaling、recoverable failure 与 KLU 行为保持一致。
  // English: KINSOLInstance and constraint-local solves share one sparse
  // nonlinear core, keeping scaling, recoverable failure, and KLU semantics
  // consistent.
  SparseNonlinearSystem::Configuration configuration{
      scalarVariablesNumber,
      nonZeroValuesNumber,
      functionNormTolerance.value_or(getOptions().fnormtol),
      scaledStepTolerance.value_or(getOptions().scsteptol),
      effectiveMaximumNewtonStep,
      lineSearchEnabled,
      quietErrors};
  nonlinearSystem = std::make_unique<SparseNonlinearSystem>(
      configuration,
      [this](N_Vector variables, N_Vector residuals) {
        return residualFunction(variables, residuals, this);
      },
      [this](N_Vector variables, N_Vector residuals, SUNMatrix jacobian) {
        return jacobianMatrix(variables, residuals, jacobian, this, nullptr,
                              nullptr);
      });
  if (!nonlinearSystem->initialize()) {
    return false;
  }

  initialized = true;

  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[KINSOL] Initialization completed" << std::endl;
  }

  return true;
}

bool KINSOLInstance::solve() {
  return solve(currentTime);
}

KINSOLInstance::SolveStatus KINSOLInstance::solveWithStatus(realtype time) {
  return solveImpl(time, false);
}

KINSOLInstance::SolveStatus KINSOLInstance::solveAndCheck(
    realtype time, const std::function<void()> &inspection) {
  auto status = solveImpl(time, false);
  if (status != SolveStatus::Success)
    return status;
  if (!lastSolveWasBlocked || !scalarBlockSolver ||
      timedResidualPostprocessor) {
    inspection();
    return status;
  }
  // 中文：完整最终 F 在所有块及根分支检查后、完整已安装解处求值，不能使用 Newton 前一迭代的残差。
  //       owner 检查期间时间、参数、外层输入和 epoch 不变，且唯一允许消费前还逐位核对局部坐标。
  // The complete final F was evaluated after every block and root-branch
  // check, at the full installed solution, not at Newton's previous iterate.
  // Time, parameters, outer inputs and the active epoch are unchanged within
  // this owner-only inspection. Local coordinates are also checked bitwise
  // immediately before the single allowed consumption.
  SuccessfulPoint point{time, scalarBlockValues,
                        scalarBlockSolver->finalResidual};
  successfulPoint = &point;
  struct Release {
    KINSOLInstance &instance;
    ~Release() { instance.successfulPoint = nullptr; }
  } release{*this};
  inspection();
  return status;
}

KINSOLInstance::SolveStatus KINSOLInstance::solveImpl(realtype time,
                                                     bool rollbackOnFailure) {
  if (!initialized && !initialize())
    return SolveStatus::FatalFailure;
  return (this->*solveEntry)(time, rollbackOnFailure);
}

template <bool Generated>
// 中文：generated 与基础后端在初始化时选定；数值 miss 还原 warm start，失败 trial 的健康检查仍由 IDA 事务负责。
// English: Initialization selects generated or base execution; numerical misses restore the
// warm start, while IDA owns failed-trial health checks.
KINSOLInstance::SolveStatus KINSOLInstance::solveBody(realtype time,
                                                    bool rollbackOnFailure) {
  successfulPoint = nullptr;
  // 中文：IDA 在外层事务恢复局部状态前检查失败 trial；只有独立调用在这个边界自行回滚。
  // IDA inspects the failed trial before its enclosing transaction restores
  // local state. Only standalone calls own the rollback at this boundary.
  currentTime = time;
  if (!initialized) {
    if (!initialize()) {
      return SolveStatus::FatalFailure;
    }
  }

  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[KINSOL] Computing solution" << std::endl;
  }

  if (getNumOfScalarEquations() == 0) {
    return SolveStatus::Success;
  }

  N_Vector variables = nonlinearSystem->getVariablesVector();
  copyVariablesFromMARCO(variables);
  lastSolveWasBlocked = false;
  scalarBlockIterations = 0;
  if (scalarBlockPlan && prepareScalarBlockSolver()) {
    scalarBlockSolver->setLineSearchEnabled(lineSearchEnabled);
    const double *input = N_VGetArrayPointer(variables);
    scalarBlockValues.assign(input, input + scalarVariablesNumber);
    auto result = ScalarBlockSolver::Result::RetainOriginal;
    bool generated = false;
    if constexpr (Generated) {
      generated = c2Kernel->descriptor.solve(time,
        scalarBlockValues.data(), nonlinearSystem->getVariableScale().data(),
        nonlinearSystem->getResidualScale().data(),
        functionNormTolerance.value_or(getOptions().fnormtol),
        c2Kernel->kinds.size(), c2Kernel->kinds.data(), c2Kernel->offsets.data(),
        c2Kernel->rows.data(), c2Kernel->columns.data(),
        c2Kernel->residuals.data(), &scalarBlockIterations) == 1;
    }
    if (generated) {
      ++c2Kernel->solved;
      scalarBlockSolver->finalResidual = c2Kernel->residuals;
      result = ScalarBlockSolver::Result::Solved;
    } else {
      if constexpr (Generated) ++c2Kernel->declined;
      scalarBlockValues.assign(input, input + scalarVariablesNumber);
      result = scalarBlockSolver->solve(time, scalarBlockValues);
      scalarBlockIterations = scalarBlockSolver->getLastNonlinearIterations();
    }
    std::copy(scalarBlockValues.begin(), scalarBlockValues.end(),
              N_VGetArrayPointer(variables));
    // 中文：分支或舍入证明不成立时，从原 warm start 进入完全相同的原求解；纯回调不能产生其它输出。
    // A declined branch/roundoff proof re-enters exactly the original solve
    // from the original warm start. Pure callbacks cannot have other outputs.
    copyVariablesIntoMARCO(variables);
    if (result == ScalarBlockSolver::Result::Solved) {
      ++scalarBlockAccepted;
      lastSolveWasBlocked = true;
      return SolveStatus::Success;
    }
    ++scalarBlockDeclined;
  }
  std::vector<realtype> initialValues;
  if (rollbackOnFailure) {
    const realtype *initialData = N_VGetArrayPointer(variables);
    initialValues.assign(initialData, initialData + scalarVariablesNumber);
  }
  nonlinearSystem->setLineSearchEnabled(lineSearchEnabled);
  SparseNonlinearSystem::SolveStatus status = nonlinearSystem->solve(time);
  if (status != SparseNonlinearSystem::SolveStatus::Success) {
    // 中文：residual/Jacobian 回调会通过普通模型 setter 安装 trial；仅省去成功后的最终复制并不会撤销这些写入。
    //       失败时必须精确恢复注册未知量。
    // Residual/Jacobian callbacks install trial values through the ordinary
    // model setters. Skipping the final success copy alone does not roll
    // those writes back. Restore exactly the registered unknowns on failure.
    if (rollbackOnFailure) {
      std::copy(initialValues.begin(), initialValues.end(),
                N_VGetArrayPointer(variables));
      copyVariablesIntoMARCO(variables);
    }
    return status == SparseNonlinearSystem::SolveStatus::RecoverableFailure
               ? SolveStatus::RecoverableFailure
               : SolveStatus::FatalFailure;
  }
  copyVariablesIntoMARCO(variables);
  return SolveStatus::Success;
}

bool KINSOLInstance::solve(realtype time) {
  return solveImpl(time, true) == SolveStatus::Success;
}

bool KINSOLInstance::solve(realtype time, bool enableLineSearch) {
  lineSearchEnabled = enableLineSearch;
  return solve(time);
}

void KINSOLInstance::setFunctionNormTolerance(realtype tolerance) {
  assert(!initialized && "KINSOL tolerance must be set before initialize");
  functionNormTolerance = tolerance;
}

void KINSOLInstance::setScaledStepTolerance(realtype tolerance) {
  assert(!initialized && "KINSOL tolerance must be set before initialize");
  scaledStepTolerance = tolerance;
}

void KINSOLInstance::setMaximumNewtonStep(realtype maximumStep) {
  assert(!initialized && "KINSOL maximum step must be set before initialize");
  maximumNewtonStep = maximumStep;
}

void KINSOLInstance::setLineSearchEnabled(bool enabled) {
  assert(!initialized && "KINSOL strategy must be set before initialize");
  lineSearchEnabled = enabled;
}

void KINSOLInstance::setQuietErrors(bool enabled) {
  assert(!initialized && "KINSOL error handling must be set before initialize");
  quietErrors = enabled;
}

bool KINSOLInstance::configureInitializationAnchorScaling(realtype time) {
  // 中文：scaling 锚定在初始化成功点；显式 nominal 优先，缺失时才用固定的
  // max(1, |z0|)，不会随 Newton 试探点漂移。
  // English: Scaling is anchored at the initialized state. Explicit nominals
  // take precedence and missing values use fixed max(1, |z0|), never a scale
  // that changes with Newton trial points.
  if (initializationAnchorScalingConfigured) {
    return true;
  }
  if (!initialized && !initialize()) {
    return false;
  }
  if (getNumOfScalarEquations() == 0) {
    initializationAnchorScalingConfigured = true;
    return true;
  }
  N_Vector variables = nonlinearSystem->getVariablesVector();
  copyVariablesFromMARCO(variables);
  if (!nonlinearSystem->factorize(time)) {
    return false;
  }
  std::vector<double> variableNominals(scalarVariablesNumber, 1);
  explicitNominalScalars = 0;
  anchorNominalScalars = 0;
  for (Variable variable = 0; variable < getNumOfArrayVariables();
       ++variable) {
    const VariableDimensions &dimensions = variablesDimensions[variable];
    std::vector<uint64_t> indices;
    getVariableBeginIndices(variable, indices);
    uint64_t flat = 0;
    do {
      double nominal = 0;
      if (variableNominalGetters[variable] != nullptr) {
        nominal = variableNominalGetters[variable](indices.data());
        ++explicitNominalScalars;
      } else {
        nominal = std::max(
            1.0, std::abs(variableGetters[variable](indices.data())));
        ++anchorNominalScalars;
      }
      if (!(nominal > 0) || !std::isfinite(nominal)) {
        return false;
      }
      variableNominals[variableOffsets[variable] + flat++] = nominal;
    } while (advanceVariableIndices(indices, dimensions));
  }
  if (explicitNominalScalars + anchorNominalScalars !=
      scalarVariablesNumber) {
    return false;
  }
  if (!nonlinearSystem->setScalingFromVariableNominals(variableNominals)) {
    return false;
  }
  initializationAnchorScalingConfigured = true;
  return true;
}

bool KINSOLInstance::factorizeCurrentJacobian(realtype time) {
  return factorizeCurrentJacobianWithStatus(time) ==
         SparseNonlinearSystem::FactorizationStatus::Success;
}

SparseNonlinearSystem::FactorizationStatus
KINSOLInstance::factorizeCurrentJacobianWithStatus(realtype time) {
  // 中文：始终刷新当前点的 Jacobian 与分解；只有紧接着成功点检查中的完整 F 可代替重复残差求值，不能复用 Newton Jacobian。
  //       后续多右端项仍共享这一次新分解。
  // Always refresh the current-point Jacobian and factorization. Only the
  // complete F from the immediately enclosing successful-point inspection
  // can replace a duplicate F evaluation; a Newton Jacobian is never reused.
  // Subsequent RHS solves share this fresh factorization as before.
  currentTime = time;
  if (!initialized && !initialize()) {
    return SparseNonlinearSystem::FactorizationStatus::SolverFailure;
  }
  if (getNumOfScalarEquations() == 0) {
    return SparseNonlinearSystem::FactorizationStatus::Success;
  }
  copyVariablesFromMARCO(nonlinearSystem->getVariablesVector());
  const SuccessfulPoint *point = successfulPoint;
  // 中文：一份回执只允许一次检查。
  successfulPoint = nullptr; // A receipt can authorize exactly one check.
  if (point && std::memcmp(&point->time, &time, sizeof(realtype)) == 0 &&
      point->variables.size() == scalarVariablesNumber &&
      std::memcmp(point->variables.data(),
                  N_VGetArrayPointer(nonlinearSystem->getVariablesVector()),
                  scalarVariablesNumber * sizeof(realtype)) == 0)
    return nonlinearSystem->factorizeWithResidual(time, point->residuals);
  return nonlinearSystem->factorizeWithStatus(time);
}

bool KINSOLInstance::solveCurrentJacobian(
    const std::vector<double> &rhs, std::vector<double> &solution) {
  return nonlinearSystem != nullptr &&
         nonlinearSystem->solveFactorized(rhs, solution);
}

bool KINSOLInstance::getCurrentJacobianEntries(
    const std::vector<uint64_t> &rows,
    const std::vector<uint64_t> &columns,
    std::vector<double> &values) const {
  return nonlinearSystem != nullptr &&
         nonlinearSystem->extractEntries(rows, columns, values);
}

bool KINSOLInstance::estimateScaledJacobianCondition(
    const std::vector<uint64_t> &rows,
    const std::vector<uint64_t> &columns,
    SparseNonlinearSystem::ConditionEstimate &estimate) const {
  return nonlinearSystem != nullptr &&
         nonlinearSystem->estimateScaledSubmatrixCondition(rows, columns,
                                                            estimate);
}

double KINSOLInstance::getScalarVariableNominal(uint64_t scalarIndex) const {
  return nonlinearSystem == nullptr
             ? 0
             : nonlinearSystem->getVariableNominal(scalarIndex);
}

uint64_t KINSOLInstance::getLastNonlinearIterations() const {
  return scalarBlockIterations +
         (lastSolveWasBlocked || nonlinearSystem == nullptr
              ? 0 : nonlinearSystem->getLastNonlinearIterations());
}

bool KINSOLInstance::prepareScalarBlockSolver() {
  if (scalarBlockPrepared) return scalarBlockSolver != nullptr;
  scalarBlockPrepared = true;
  if (!scalarBlockPlan || !scalarBlockBudget || !accessLayout ||
      accessLayout->evaluations.empty() || timedResidualPostprocessor)
    return false;
  auto install = [this](const std::vector<double> &values) {
    std::copy(values.begin(), values.end(),
              N_VGetArrayPointer(nonlinearSystem->getVariablesVector()));
    copyVariablesIntoMARCO(nonlinearSystem->getVariablesVector());
  };
  auto rowEntry = [this](uint32_t row) -> const ImmutableAccessLayout::RowEntry & {
    // 中文：布局行保持 chunk 顺序；即使不同于注册顺序，原标量 offset 仍是权威。
    // The layout's rows retain chunk order; original scalar offsets remain
    // authoritative even if that order differs from registration order.
    auto it = std::find_if(accessLayout->rows.begin(), accessLayout->rows.end(),
                          [&](const auto &entry) { return entry.offset == row; });
    assert(it != accessLayout->rows.end());
    return *it;
  };
  SparseNonlinearSystem::Configuration configuration{
      scalarVariablesNumber, nonZeroValuesNumber,
      functionNormTolerance.value_or(getOptions().fnormtol),
      scaledStepTolerance.value_or(getOptions().scsteptol),
      maximumNewtonStep.value_or(getOptions().maxNewtonStep),
      lineSearchEnabled, quietErrors};
  if (configuration.maximumNewtonStep <= 0)
    configuration.maximumNewtonStep = std::max<realtype>(
        100, 10 * std::sqrt(static_cast<realtype>(scalarVariablesNumber)));
  scalarBlockSolver = ScalarBlockSolver::create(
      *scalarBlockPlan, configuration,
      [this, rowEntry](uint32_t row, const std::vector<double> &) {
        const auto &entry = rowEntry(row);
        Equation equation = std::get<0>(threadEquationsChunks[entry.chunk]);
        if (marco::runtime::simulation::getOptions().profiling)
          ++scalarResidualCalls;
        auto timed = equation < timedResidualFunctions.size()
                         ? timedResidualFunctions[equation] : nullptr;
        return invokeScalarBlockCallback([&] {
          return timed ? timed(currentTime, entry.indices.data())
                       : residualFunctions[equation](entry.indices.data());
        });
      },
      [this, rowEntry](uint32_t row, uint32_t column,
                       const std::vector<double> &) {
        const auto &entry = rowEntry(row);
        const auto &evaluations = accessLayout->evaluations[row];
        for (size_t i = 0; i < evaluations.size(); ++i) {
          const auto &binding = evaluations[i];
          if (binding.column != column) continue;
          if (marco::runtime::simulation::getOptions().profiling)
            ++scalarJacobianCalls;
          const auto &indices = accessLayout->columns[row][i].second;
          return invokeScalarBlockCallback([&] {
            return binding.timedCallback
                ? binding.timedCallback(currentTime, entry.indices.data(),
                      indices.data(), 0.0, memoryPoolId, binding.seeds)
                : binding.callback(entry.indices.data(), indices.data(),
                                    memoryPoolId, binding.seeds);
          });
        }
        // 中文：该位置不属于完整认证访问关系。
        return 0.0; // Absent from the complete certified access relation.
      }, *scalarBlockBudget, install);
  if (scalarBlockSolver && !scalarExplicitValues.empty())
    scalarBlockSolver->setExplicitValues(
        [this, rowEntry](uint32_t row, const std::vector<double> &) {
          return invokeScalarBlockCallback([&] {
            return scalarExplicitValues[row](currentTime,
                                              rowEntry(row).indices.data());
          });
        });
  if (scalarBlockSolver && !scalarBlockSolver->setScaling(
          nonlinearSystem->getVariableScale(), nonlinearSystem->getResidualScale()))
    scalarBlockSolver.reset();
  return scalarBlockSolver != nullptr;
}

void KINSOLInstance::printInitializationScaling() const {
  if (!nonlinearSystem)
    return;
  const auto &variables = nonlinearSystem->getVariableScale();
  const auto &residuals = nonlinearSystem->getResidualScale();
  size_t count = std::min<size_t>(256, variables.size());
  std::cerr << std::setprecision(17)
            << "[sparse-solver-scaling] id=" << getStatisticsId()
            << " entries=" << variables.size() << " printed=" << count
            << " variable_scale=";
  for (size_t i = 0; i < count; ++i)
    std::cerr << (i ? "," : "") << variables[i];
  std::cerr << " residual_scale=";
  for (size_t i = 0; i < count; ++i)
    std::cerr << (i ? "," : "") << residuals[i];
  std::cerr << '\n';
}

int KINSOLInstance::residualFunction(N_Vector variables, N_Vector residuals,
                                     void *userData) {
  KINSOL_PROFILER_RESIDUALS_CALL_COUNTER_INCREMENT

  realtype *rval = N_VGetArrayPointer(residuals);
  auto *instance = static_cast<KINSOLInstance *>(userData);

  // Copy the values of the variables and derivatives provided by KINSOL into
  // the variables owned by MARCO, so that the residual functions operate on
  // the current iteration values.
  instance->copyVariablesIntoMARCO(variables);

  // For every vectorized equation, set the residual values of the variables
  // it writes into.
  KINSOL_PROFILER_RESIDUALS_START

  instance->equationsParallelIteration(
      EquationsParallelIterationKind::Residuals,
      [&](Equation eq, const std::vector<int64_t> &equationIndices,
          const JacobianSeedsMap &jacobianSeedsMap, uint64_t offset) {
        assert(equationIndices.size() == instance->getEquationRank(eq));

        // 中文：timed callback 专供 IDA component closure；普通 KINSOL 路径仍
        // 使用无时间签名，两者共享相同 residual buffer 和有限值检查。
        // English: Timed callbacks serve IDA component closure while ordinary
        // KINSOL retains the time-independent signature. Both share the same
        // residual buffer and finite-value validation.
        auto residualFn = eq < instance->residualFunctions.size()
                              ? instance->residualFunctions[eq]
                              : nullptr;
        auto timedResidualFn = eq < instance->timedResidualFunctions.size()
                                   ? instance->timedResidualFunctions[eq]
                                   : nullptr;
        auto *eqIndicesPtr = equationIndices.data();

        if (marco::runtime::simulation::getOptions().profiling)
          ++instance->scalarResidualCalls;
        auto residualFunctionResult =
            timedResidualFn != nullptr
                ? timedResidualFn(instance->currentTime, eqIndicesPtr)
                : residualFn(eqIndicesPtr);
        *(rval + offset) = residualFunctionResult;
      });

  if (instance->timedResidualPostprocessor &&
      !instance->timedResidualPostprocessor(instance->currentTime, rval,
                                            instance->scalarEquationsNumber)) {
    return KIN_SYSFUNC_FAIL;
  }
  for (uint64_t i = 0; i < instance->scalarEquationsNumber; ++i) {
    if (!std::isfinite(rval[i])) {
      return KIN_SYSFUNC_FAIL;
    }
  }

  KINSOL_PROFILER_RESIDUALS_STOP

  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[KINSOL] Residuals function called" << std::endl;
    std::cerr << "Variables:" << std::endl;
    instance->printVariablesVector(variables);
    std::cerr << "Residuals vector:" << std::endl;
    instance->printResidualsVector(residuals);
  }

  return KIN_SUCCESS;
}

int KINSOLInstance::jacobianMatrix(N_Vector variables, N_Vector residuals,
                                   SUNMatrix jacobianMatrix, void *userData,
                                   N_Vector tempv1, N_Vector tempv2) {
  KINSOL_PROFILER_PARTIAL_DERIVATIVES_CALL_COUNTER_INCREMENT

  realtype *jacobian = SUNSparseMatrix_Data(jacobianMatrix);
  auto *instance = static_cast<KINSOLInstance *>(userData);

  // Copy the values of the variables and derivatives provided by KINSOL into
  // the variables owned by MARCO, so that the jacobian functions operate on
  // the current iteration values.
  instance->copyVariablesIntoMARCO(variables);

  KINSOL_PROFILER_PARTIAL_DERIVATIVES_START

  instance->equationsParallelIteration(
      EquationsParallelIterationKind::Jacobian,
      [&](Equation eq, const std::vector<int64_t> &equationIndices,
          const JacobianSeedsMap &jacobianSeedsMap,
          uint64_t scalarEquationIndex) {
        assert(scalarEquationIndex < instance->getNumOfScalarEquations());

        if (instance->accessLayout &&
            !instance->accessLayout->evaluations.empty()) {
          const auto &bindings =
              instance->accessLayout->evaluations[scalarEquationIndex];
          const auto &columns =
              instance->accessLayout->columns[scalarEquationIndex];
          for (size_t i = 0; i < bindings.size(); ++i) {
            const auto &binding = bindings[i];
            assert((binding.callback || binding.timedCallback) &&
                   "Missing Jacobian callback");
            if (marco::runtime::simulation::getOptions().profiling)
              ++instance->scalarJacobianCalls;
            double value = binding.timedCallback
                ? binding.timedCallback(instance->currentTime,
                       equationIndices.data(), columns[i].second.data(), 0.0,
                       instance->memoryPoolId, binding.seeds)
                : binding.callback(equationIndices.data(),
                       columns[i].second.data(), instance->memoryPoolId,
                       binding.seeds);
            instance->jacobianMatrixData[scalarEquationIndex][i] =
                {binding.column, value};
          }
          return;
        }

        // Compute the column indexes that may be non-zeros.
        std::vector<JacobianColumn> transientColumns;
        if (!instance->accessLayout)
          transientColumns =
              instance->computeJacobianColumns(eq, equationIndices.data());
        const auto &jacobianColumns = instance->accessLayout
            ? instance->accessLayout->columns[scalarEquationIndex]
            : transientColumns;

        // For every scalar variable with respect to which the equation must be
        // partially differentiated.
        for (size_t i = 0, e = jacobianColumns.size(); i < e; ++i) {
          const JacobianColumn &column = jacobianColumns[i];
          Variable variable = column.first;
          const auto &variableIndices = column.second;

          uint64_t variableArrayOffset = instance->variableOffsets[variable];

          uint64_t variableScalarOffset = getVariableFlatIndex(
              instance->variablesDimensions[variable], column.second);

          auto jacobianFunction =
              eq < instance->jacobianFunctions.size() &&
                      variable < instance->jacobianFunctions[eq].size()
                  ? instance->jacobianFunctions[eq][variable].first
                  : nullptr;
          auto timedJacobianFunction =
              eq < instance->timedJacobianFunctions.size() &&
                      variable < instance->timedJacobianFunctions[eq].size()
                  ? instance->timedJacobianFunctions[eq][variable].first
                  : nullptr;

          // 中文：timed-only Jacobian 是合法配置，seed cache 以实际 callback
          // 地址区分，不能再断言普通 callback 必须存在。
          // English: A timed-only Jacobian is valid. The seed cache keys the
          // actual callback address and must not require an ordinary callback.
          assert((jacobianFunction != nullptr ||
                  timedJacobianFunction != nullptr) &&
                 "Missing Jacobian callback");
          uintptr_t seedKey = jacobianFunction != nullptr
                                  ? reinterpret_cast<uintptr_t>(
                                        jacobianFunction)
                                  : reinterpret_cast<uintptr_t>(
                                        timedJacobianFunction);
          auto seedsMapIt = jacobianSeedsMap.find(seedKey);
          const uint64_t *seedsPtr = nullptr;

          if (seedsMapIt != jacobianSeedsMap.end()) {
            seedsPtr = seedsMapIt->second.data();
          }

          if (marco::runtime::simulation::getOptions().profiling)
            ++instance->scalarJacobianCalls;
          auto jacobianFunctionResult = timedJacobianFunction != nullptr
              ? timedJacobianFunction(
                    instance->currentTime, equationIndices.data(),
                    variableIndices.data(), 0.0, instance->memoryPoolId,
                    seedsPtr)
              : jacobianFunction(equationIndices.data(),
                                 variableIndices.data(),
                                 instance->memoryPoolId, seedsPtr);

          instance->jacobianMatrixData[scalarEquationIndex][i].second =
              jacobianFunctionResult;

          auto index = static_cast<sunindextype>(variableArrayOffset +
                                                 variableScalarOffset);

          instance->jacobianMatrixData[scalarEquationIndex][i].first = index;
        }
      });

  // Move the partial derivatives into the SUNDIALS sparse matrix.
  sunindextype *rowPtrs = SUNSparseMatrix_IndexPointers(jacobianMatrix);
  sunindextype *columnIndices = SUNSparseMatrix_IndexValues(jacobianMatrix);

  sunindextype offset = 0;
  *rowPtrs++ = offset;

  for (const auto &row : instance->jacobianMatrixData) {
    offset += static_cast<sunindextype>(row.size());
    *rowPtrs++ = offset;

    for (const auto &column : row) {
      // 中文：AD callback 的非有限值是结构/数值函数失败，必须在写入 SUNMatrix 时
      // 立即上报，不能让 KLU 接收污染矩阵后给出误导性收敛错误。
      // English: A non-finite AD callback value is a structural/numerical
      // function failure and is reported before KLU receives a contaminated
      // matrix and emits a misleading convergence error.
      if (!std::isfinite(column.second)) {
        return KIN_SYSFUNC_FAIL;
      }
      *columnIndices++ = column.first;
      *jacobian++ = column.second;
    }
  }

  assert(rowPtrs == SUNSparseMatrix_IndexPointers(jacobianMatrix) +
                        instance->getNumOfScalarEquations() + 1);
  assert(columnIndices == SUNSparseMatrix_IndexValues(jacobianMatrix) +
                              instance->nonZeroValuesNumber);
  assert(jacobian ==
         SUNSparseMatrix_Data(jacobianMatrix) + instance->nonZeroValuesNumber);

  KINSOL_PROFILER_PARTIAL_DERIVATIVES_STOP

  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[KINSOL] Jacobian matrix function called" << std::endl;
    std::cerr << "Variables:" << std::endl;
    instance->printVariablesVector(variables);
    std::cerr << "Residuals vector:" << std::endl;
    instance->printResidualsVector(residuals);
    std::cerr << "Jacobian matrix:" << std::endl;
    instance->printJacobianMatrix(jacobianMatrix);
  }

  return KIN_SUCCESS;
}

uint64_t KINSOLInstance::getNumOfArrayVariables() const {
  return variablesDimensions.size();
}

uint64_t KINSOLInstance::getNumOfScalarVariables() const {
  return scalarVariablesNumber;
}

uint64_t KINSOLInstance::getVariableFlatSize(Variable variable) const {
  uint64_t result = 1;

  for (uint64_t dimension : variablesDimensions[variable]) {
    result *= dimension;
  }

  return result;
}

uint64_t KINSOLInstance::getNumOfVectorizedEquations() const {
  return equationRanges.size();
}

uint64_t KINSOLInstance::getNumOfScalarEquations() const {
  return scalarEquationsNumber;
}

uint64_t KINSOLInstance::getEquationRank(Equation equation) const {
  return equationRanges[equation].size();
}

uint64_t KINSOLInstance::getEquationFlatSize(Equation equation) const {
  assert(equation < getNumOfVectorizedEquations());
  uint64_t result = 1;

  for (const Range &range : equationRanges[equation]) {
    result *= range.end - range.begin;
  }

  return result;
}

uint64_t KINSOLInstance::getVariableRank(Variable variable) const {
  return variablesDimensions[variable].rank();
}

void KINSOLInstance::iterateAccessedArrayVariables(
    Equation equation, std::function<void(Variable)> callback) const {
  if (precomputedAccesses) {
    for (const auto &access : variableAccesses[equation]) {
      callback(access.first);
    }
  } else {
    uint64_t numOfArrayVariables = getNumOfArrayVariables();

    for (Variable variable = 0; variable < numOfArrayVariables; ++variable) {
      callback(variable);
    }
  }
}

/// Determine which of the columns of the current Jacobian row has to be
/// populated, and with respect to which variable the partial derivative has
/// to be performed. The row is determined by the indices of the equation.
std::vector<JacobianColumn>
KINSOLInstance::computeJacobianColumns(Equation eq,
                                       const int64_t *equationIndices) const {
  std::set<JacobianColumn> uniqueColumns;

  if (precomputedAccesses) {
    for (const auto &access : variableAccesses[eq]) {
      Variable variable = access.first;
      AccessFunction accessFunction = access.second;

      uint64_t variableRank = getVariableRank(variable);

      std::vector<uint64_t> variableIndices;
      variableIndices.resize(variableRank, 0);
      accessFunction(equationIndices, variableIndices.data());

      bool outsideBlock = false;
      for (uint64_t i = 0; i < variableRank; ++i) {
        if (variableIndices[i] >= variablesDimensions[variable][i]) {
          outsideBlock = true;
          break;
        }
      }
      // 中文：IDA 将原仿射访问平移到各注册变量块，其像只在部分行与当前块相交。
      //       局部 KINSOL 复用映射时必须保留该范围限制；块外像不能算作当前块的列。
      // IDA passes the original affine access translated into each registered
      // variable block. Its image intersects this block only on some rows.
      // Preserve IDA's range restriction when reusing the map in a local
      // KINSOL closure; an outside image is not a column of this block.
      if (outsideBlock) {
        continue;
      }

      uniqueColumns.insert({variable, variableIndices});
    }
  } else {
    for (size_t variableIndex = 0, e = getNumOfArrayVariables();
         variableIndex < e; ++variableIndex) {
      const auto &dimensions = variablesDimensions[variableIndex];

      for (auto indices = dimensions.indicesBegin(),
                end = dimensions.indicesEnd();
           indices != end; ++indices) {
        JacobianColumn column(variableIndex, {});

        for (size_t dim = 0; dim < dimensions.rank(); ++dim) {
          column.second.push_back((*indices)[dim]);
        }

        uniqueColumns.insert(std::move(column));
      }
    }
  }

  // 中文：set 已按变量和索引元组字典序排序；逐轴“任一轴更小”不是严格弱序，两方向可能同时比较为真。
  // The set already orders (variable, index tuple) lexicographically.
  // Re-sorting with a per-axis "any less" comparison is not a strict weak
  // order: both (0,1)<(1,0) and its reverse would compare true.
  return {uniqueColumns.begin(), uniqueColumns.end()};
}

/// Compute the number of non-zero values in the Jacobian Matrix. Also
/// compute the column indexes of all non-zero values in the Jacobian Matrix.
/// This allows to avoid the recomputation of such indexes during the
/// Jacobian evaluation.
void KINSOLInstance::computeNNZ() {
  nonZeroValuesNumber = 0;
  std::vector<int64_t> equationIndices;

  for (size_t eq = 0; eq < getNumOfVectorizedEquations(); ++eq) {
    // Initialize the multidimensional interval of the vector equation.
    uint64_t equationRank = equationRanges[eq].size();
    equationIndices.resize(equationRank);

    for (size_t i = 0; i < equationRank; ++i) {
      const auto &iterationRange = equationRanges[eq][i];
      int64_t beginIndex = iterationRange.begin;
      equationIndices[i] = beginIndex;
    }

    // For every scalar equation in the vector equation.
    do {
      // Compute the column indexes that may be non-zeros
      nonZeroValuesNumber +=
          computeJacobianColumns(eq, equationIndices.data()).size();

    } while (advanceEquationIndices(equationIndices, equationRanges[eq]));
  }
}

void KINSOLInstance::computeThreadChunks() {
  unsigned int numOfThreads = threadPool.getNumOfThreads();

  int64_t chunksFactor =
      std::max<int64_t>(1, getOptions().equationsChunksFactor);
  int64_t numOfChunks = numOfThreads * chunksFactor;

  uint64_t numOfVectorizedEquations = getNumOfVectorizedEquations();
  uint64_t numOfScalarEquations = getNumOfScalarEquations();

  size_t chunkSize = (numOfScalarEquations + numOfChunks - 1) / numOfChunks;

  // The number of vectorized equations whose indices have been completely
  // assigned.
  uint64_t processedEquations = 0;

  while (processedEquations < numOfVectorizedEquations) {
    Equation equation = processedEquations;
    uint64_t equationFlatSize = getEquationFlatSize(equation);
    uint64_t equationFlatIndex = 0;

    // Divide the ranges into chunks.
    while (equationFlatIndex < equationFlatSize) {
      uint64_t beginFlatIndex = equationFlatIndex;

      uint64_t endFlatIndex = std::min(
          beginFlatIndex + static_cast<uint64_t>(chunkSize), equationFlatSize);

      std::vector<int64_t> beginIndices;
      std::vector<int64_t> endIndices;

      getEquationIndicesFromFlatIndex(beginFlatIndex, beginIndices,
                                      equationRanges[equation]);

      if (endFlatIndex == equationFlatSize) {
        getEquationEndIndices(equation, endIndices);
      } else {
        getEquationIndicesFromFlatIndex(endFlatIndex, endIndices,
                                        equationRanges[equation]);
      }

      JacobianSeedsMap jacobianSeedsMap;

      // 中文：普通和 timed Jacobian 共享 chunk-local AD seed pool，但键必须是
      // 实际被调用的函数地址，避免 timed-only 方程错误复用空键。
      // English: Ordinary and timed Jacobians share the chunk-local AD seed
      // pool, keyed by the callback that will actually run so timed-only
      // equations cannot alias a null key.
      iterateAccessedArrayVariables(equation, [&](Variable variable) {
        uintptr_t jacobianFunction = 0;
        const std::vector<uint64_t> *seedSizes = nullptr;
        if (equation < jacobianFunctions.size() &&
            variable < jacobianFunctions[equation].size() &&
            jacobianFunctions[equation][variable].first != nullptr) {
          jacobianFunction = reinterpret_cast<uintptr_t>(
              jacobianFunctions[equation][variable].first);
          seedSizes = &jacobianFunctions[equation][variable].second;
        } else if (equation < timedJacobianFunctions.size() &&
                   variable < timedJacobianFunctions[equation].size()) {
          jacobianFunction = reinterpret_cast<uintptr_t>(
              timedJacobianFunctions[equation][variable].first);
          seedSizes = &timedJacobianFunctions[equation][variable].second;
        }
        assert(jacobianFunction != 0 && seedSizes != nullptr);

        for (const auto &seedSize : *seedSizes) {
          MemoryPool &memoryPool =
              MemoryPoolManager::getInstance().get(memoryPoolId);
          uint64_t seedId = memoryPool.create(seedSize);
          jacobianSeedsMap[jacobianFunction].push_back(seedId);
        }
      });

      threadEquationsChunks.emplace_back(equation, std::move(beginIndices),
                                         std::move(endIndices),
                                         std::move(jacobianSeedsMap));

      // Move to the next chunk.
      equationFlatIndex = endFlatIndex;
    }

    // Move to the next vectorized equation.
    ++processedEquations;
  }
}

void KINSOLInstance::copyVariablesFromMARCO(N_Vector variables) {
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[KINSOL] Copying variables from MARCO" << std::endl;
  }

  KINSOL_PROFILER_COPY_VARS_FROM_MARCO_START

  realtype *varsPtr = N_VGetArrayPointer(variables);
  if (accessLayout && !marco::runtime::simulation::getOptions().debug) {
    for (const auto &entry : accessLayout->variables)
      varsPtr[entry.offset] = variableGetters[entry.variable](entry.indices.data());
    KINSOL_PROFILER_COPY_VARS_FROM_MARCO_STOP
    return;
  }
  uint64_t numOfArrayVariables = getNumOfArrayVariables();

  for (Variable var = 0; var < numOfArrayVariables; ++var) {
    uint64_t variableArrayOffset = variableOffsets[var];
    const auto &dimensions = variablesDimensions[var];

    std::vector<uint64_t> varIndices;
    getVariableBeginIndices(var, varIndices);

    do {
      uint64_t variableScalarOffset =
          getVariableFlatIndex(dimensions, varIndices.data());

      uint64_t offset = variableArrayOffset + variableScalarOffset;

      // Get the variable.
      auto getterFn = variableGetters[var];
      auto value = static_cast<realtype>(getterFn(varIndices.data()));
      varsPtr[offset] = value;

      if (marco::runtime::simulation::getOptions().debug) {
        std::cerr << "Got var " << var << " ";
        printIndices(varIndices);
        std::cerr << " with value " << std::fixed << std::setprecision(9)
                  << value << std::endl;
      }
    } while (advanceVariableIndices(varIndices, variablesDimensions[var]));
  }

  KINSOL_PROFILER_COPY_VARS_FROM_MARCO_STOP
}

void KINSOLInstance::copyVariablesIntoMARCO(N_Vector variables) {
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[KINSOL] Copying variables into MARCO" << std::endl;
  }

  KINSOL_PROFILER_COPY_VARS_INTO_MARCO_START

  realtype *varsPtr = N_VGetArrayPointer(variables);
  if (accessLayout && !marco::runtime::simulation::getOptions().debug) {
    for (const auto &entry : accessLayout->variables) {
      double value = varsPtr[entry.offset];
      variableSetters[entry.variable](value, entry.indices.data());
      assert(variableGetters[entry.variable](entry.indices.data()) == value &&
             "Variable value not set correctly");
    }
    KINSOL_PROFILER_COPY_VARS_INTO_MARCO_STOP
    return;
  }
  uint64_t numOfArrayVariables = getNumOfArrayVariables();

  for (Variable var = 0; var < numOfArrayVariables; ++var) {
    uint64_t variableArrayOffset = variableOffsets[var];
    const auto &dimensions = variablesDimensions[var];

    std::vector<uint64_t> varIndices;
    getVariableBeginIndices(var, varIndices);

    do {
      uint64_t variableScalarOffset =
          getVariableFlatIndex(dimensions, varIndices.data());

      uint64_t offset = variableArrayOffset + variableScalarOffset;

      // Set the variable.
      auto setterFn = variableSetters[var];
      auto value = static_cast<double>(varsPtr[offset]);

      if (marco::runtime::simulation::getOptions().debug) {
        std::cerr << "Setting var " << var << " ";
        printIndices(varIndices);
        std::cerr << " to " << value << std::endl;
      }

      setterFn(value, varIndices.data());

      assert([&]() -> bool {
        auto getterFn = variableGetters[var];
        return getterFn(varIndices.data()) == value;
      }() && "Variable value not set correctly");
    } while (advanceVariableIndices(varIndices, variablesDimensions[var]));
  }

  KINSOL_PROFILER_COPY_VARS_INTO_MARCO_STOP
}

void KINSOLInstance::equationsParallelIteration(
    EquationsParallelIterationKind kind,
    std::function<void(Equation equation,
                       const std::vector<int64_t> &equationIndices,
                       const JacobianSeedsMap &jacobianSeedsMap,
                       uint64_t scalarEquationIndex)>
        processFn) {
  // Shard the work among multiple threads.
  unsigned int numOfThreads = threadPool.getNumOfThreads();
  std::atomic_size_t chunkIndex = 0;
  const auto &simulationOptions = marco::runtime::simulation::getOptions();
  bool collectStats = simulationOptions.debug || simulationOptions.profiling;
  std::vector<::marco::runtime::profiling::ParallelThreadWorkStats> threadWork(
      collectStats ? numOfThreads : 0);
  std::vector<std::thread::id> threadIds(collectStats ? numOfThreads : 0);

  auto job = [&](unsigned int thread) {
      if (collectStats) {
        threadIds[thread] = std::this_thread::get_id();
      }

      if (numOfThreads == 1 && accessLayout) {
        size_t lastChunk = threadEquationsChunks.size();
        for (const auto &row : accessLayout->rows) {
          const auto &chunk = threadEquationsChunks[row.chunk];
          if (collectStats) {
            if (row.chunk != lastChunk)
              ++threadWork[thread].chunks;
            ++threadWork[thread].scalarEquations;
          }
          lastChunk = row.chunk;
          processFn(std::get<0>(chunk), row.indices, std::get<3>(chunk), row.offset);
        }
        return;
      }

      size_t assignedChunk;

      while ((assignedChunk = chunkIndex++) < threadEquationsChunks.size()) {
        if (collectStats) {
          ++threadWork[thread].chunks;
        }

        const ThreadEquationsChunk &chunk =
            threadEquationsChunks[assignedChunk];

        Equation equation = std::get<0>(chunk);
        std::vector<int64_t> equationIndices = std::get<1>(chunk);

        do {
          assert([&]() -> bool {
            if (equationIndices.size() != equationRanges[equation].size()) {
              return false;
            }

            for (size_t i = 0, rank = equationIndices.size(); i < rank; ++i) {
              if (equationIndices[i] < equationRanges[equation][i].begin ||
                  equationIndices[i] >= equationRanges[equation][i].end) {
                return false;
              }
            }

            return true;
          }() && "Invalid equation indices");

          processFn(equation, equationIndices, std::get<3>(chunk),
                    equationOffsets[equation] + getEquationFlatIndex(
                        equationIndices, equationRanges[equation]));

          if (collectStats) {
            ++threadWork[thread].scalarEquations;
          }
        } while (advanceEquationIndicesUntil(
            equationIndices, equationRanges[equation], std::get<2>(chunk)));
      }
  };

  if (numOfThreads == 1) {
    // 中文：单 worker 使用调用线程及其 TLS，保持相同 chunk 和 seed，只省去逐回调队列交接。
    // A single worker uses the caller's thread (and TLS). Keep exactly the
    // same chunks and seeds, without a queue hand-off for each callback.
#ifdef THREADS_ENABLE
    try {
      job(0);
    } catch (...) {
      // 中文：原 worker 线程的未捕获异常也会终止，保持原语义。
      // An uncaught exception on the former worker thread terminated too.
      std::terminate();
    }
#else
    // 中文：未启用线程时 ThreadPool::async 传播异常。
    // ThreadPool::async propagates exceptions in non-threaded builds.
    job(0);
#endif
  } else {
    for (unsigned int thread = 0; thread < numOfThreads; ++thread) {
      threadPool.async([&, thread]() { job(thread); });
    }
    threadPool.wait();
  }

  if (simulationOptions.debug) {
    printParallelIterationStats(kind, threadWork, threadIds);
  }

  if (kind == EquationsParallelIterationKind::Residuals) {
    KINSOL_PROFILER_RESIDUALS_PARALLEL_WORK_RECORD(threadWork)
  } else {
    KINSOL_PROFILER_PARTIAL_DERIVATIVES_PARALLEL_WORK_RECORD(threadWork)
  }
}

void KINSOLInstance::getVariableBeginIndices(
    Variable variable, std::vector<uint64_t> &indices) const {
  uint64_t variableRank = getVariableRank(variable);
  indices.resize(variableRank);

  for (uint64_t i = 0; i < variableRank; ++i) {
    indices[i] = 0;
  }
}

void KINSOLInstance::getVariableEndIndices(
    Variable variable, std::vector<uint64_t> &indices) const {
  uint64_t variableRank = getVariableRank(variable);
  indices.resize(variableRank);

  for (uint64_t i = 0; i < variableRank; ++i) {
    indices[i] = variablesDimensions[variable][i];
  }
}

void KINSOLInstance::getEquationBeginIndices(
    Equation equation, std::vector<int64_t> &indices) const {
  uint64_t equationRank = getEquationRank(equation);
  indices.resize(equationRank);

  for (uint64_t i = 0; i < equationRank; ++i) {
    indices[i] = equationRanges[equation][i].begin;
  }
}

void KINSOLInstance::getEquationEndIndices(
    Equation equation, std::vector<int64_t> &indices) const {
  uint64_t equationRank = getEquationRank(equation);
  indices.resize(equationRank);

  for (uint64_t i = 0; i < equationRank; ++i) {
    indices[i] = equationRanges[equation][i].end;
  }
}

void KINSOLInstance::printVariablesVector(N_Vector variables) const {
  realtype *data = N_VGetArrayPointer(variables);
  uint64_t numOfArrayVariables = getNumOfArrayVariables();

  for (Variable var = 0; var < numOfArrayVariables; ++var) {
    std::vector<uint64_t> indices;
    getVariableBeginIndices(var, indices);

    do {
      std::cerr << "var " << var << " ";
      printIndices(indices);
      std::cerr << "\t" << std::fixed << std::setprecision(9) << *data
                << std::endl;
      ++data;
    } while (advanceVariableIndices(indices, variablesDimensions[var]));
  }
}

void KINSOLInstance::printResidualsVector(N_Vector residuals) const {
  realtype *data = N_VGetArrayPointer(residuals);
  uint64_t numOfVectorizedEquations = getNumOfVectorizedEquations();

  for (Equation eq = 0; eq < numOfVectorizedEquations; ++eq) {
    std::vector<int64_t> equationIndices;
    getEquationBeginIndices(eq, equationIndices);

    do {
      std::cerr << "eq " << eq << " ";
      printIndices(equationIndices);
      std::cerr << "\t" << std::fixed << std::setprecision(9) << *data << "\n";
      ++data;
    } while (advanceEquationIndices(equationIndices, equationRanges[eq]));
  }
}

void KINSOLInstance::printJacobianMatrix(SUNMatrix jacobianMatrix) const {
  uint64_t numOfArrayVariables = getNumOfArrayVariables();

  // Print the heading row.
  for (Variable var = 0; var < numOfArrayVariables; ++var) {
    std::vector<uint64_t> variableIndices;
    getVariableBeginIndices(var, variableIndices);

    do {
      std::cerr << "\tvar " << var << " ";
      printIndices(variableIndices);
    } while (advanceVariableIndices(variableIndices, variablesDimensions[var]));
  }

  std::cerr << std::endl;

  // Print the matrix.
  uint64_t numOfVectorizedEquations = getNumOfVectorizedEquations();
  uint64_t rowFlatIndex = 0;

  for (Equation eq = 0; eq < numOfVectorizedEquations; ++eq) {
    std::vector<int64_t> equationIndices;
    getEquationBeginIndices(eq, equationIndices);

    do {
      std::cerr << "eq " << eq << " ";
      printIndices(equationIndices);

      uint64_t columnFlatIndex = 0;
      bool previousNegative = false;

      for (Variable var = 0; var < numOfArrayVariables; ++var) {
        std::vector<uint64_t> varIndices;
        getVariableBeginIndices(var, varIndices);

        do {
          auto value = getCellFromSparseMatrix(jacobianMatrix, rowFlatIndex,
                                               columnFlatIndex);

          if (!previousNegative) {
            std::cerr << " ";
          }

          std::cerr << "\t" << std::fixed << std::setprecision(9) << value;
          previousNegative = value < 0;

          columnFlatIndex++;
        } while (advanceVariableIndices(varIndices, variablesDimensions[var]));
      }

      std::cerr << std::endl;
      rowFlatIndex++;
    } while (advanceEquationIndices(equationIndices, equationRanges[eq]));
  }
}
} // namespace marco::runtime::sundials::kinsol

//===---------------------------------------------------------------------===//
// Exported functions
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// kinsolCreate

static void *kinsolCreate_pvoid() {
  auto *instance = new KINSOLInstance();
  return static_cast<void *>(instance);
}

RUNTIME_FUNC_DEF(kinsolCreate, PTR(void))

//===---------------------------------------------------------------------===//
// kinsolSolve

static void kinsolSolve_void(void *instance) {
  // 中文：生成程序中的数值失败必须在 release 构建中传播为非零退出，不能依赖
  // 会被 NDEBUG 移除的 assert。
  // English: Numerical failure in generated programs must propagate as a
  // non-zero release exit and cannot rely on an assertion removed by NDEBUG.
  bool result = static_cast<KINSOLInstance *>(instance)->solve();
  if (!result) {
    std::cerr << "KINSOL solve failed." << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

RUNTIME_FUNC_DEF(kinsolSolve, void, PTR(void))

//===---------------------------------------------------------------------===//
// kinsolFree

static void kinsolFree_void(void *instance) {
  delete static_cast<KINSOLInstance *>(instance);
}

RUNTIME_FUNC_DEF(kinsolFree, void, PTR(void))

//===---------------------------------------------------------------------===//
// kinsolAddVariable

static uint64_t kinsolAddVariable_i64(void *instance, uint64_t rank,
                                      uint64_t *dimensions, void *getter,
                                      void *setter, void *name) {
  return static_cast<KINSOLInstance *>(instance)->addVariable(
      rank, dimensions, reinterpret_cast<VariableGetter>(getter),
      reinterpret_cast<VariableSetter>(setter),
      static_cast<const char *>(name));
}

RUNTIME_FUNC_DEF(kinsolAddVariable, uint64_t, PTR(void), uint64_t,
                 PTR(uint64_t), PTR(void), PTR(void), PTR(void))

//===---------------------------------------------------------------------===//
// kinsolAddVariableAccess

static void kinsolAddVariableAccess_void(void *instance, uint64_t equationIndex,
                                         uint64_t variableIndex,
                                         void *accessFunction) {
  static_cast<KINSOLInstance *>(instance)->addVariableAccess(
      equationIndex, variableIndex,
      reinterpret_cast<AccessFunction>(accessFunction));
}

RUNTIME_FUNC_DEF(kinsolAddVariableAccess, void, PTR(void), uint64_t, uint64_t,
                 PTR(void))

//===---------------------------------------------------------------------===//
// kinsolAddEquation

static uint64_t kinsolAddEquation_i64(void *instance, int64_t *ranges,
                                      uint64_t rank,
                                      void *stringRepresentation) {
  return static_cast<KINSOLInstance *>(instance)->addEquation(
      ranges, rank, static_cast<const char *>(stringRepresentation));
}

RUNTIME_FUNC_DEF(kinsolAddEquation, uint64_t, PTR(void), PTR(int64_t), uint64_t,
                 PTR(void))

//===---------------------------------------------------------------------===//
// kinsolSetResidual

static void kinsolSetResidual_void(void *instance, uint64_t equationIndex,
                                   void *residualFunction) {
  static_cast<KINSOLInstance *>(instance)->setResidualFunction(
      equationIndex, reinterpret_cast<ResidualFunction>(residualFunction));
}

RUNTIME_FUNC_DEF(kinsolSetResidual, void, PTR(void), uint64_t, PTR(void))

//===---------------------------------------------------------------------===//
// kinsolAddJacobian

static void kinsolAddJacobian_void(void *instance, uint64_t equationIndex,
                                   uint64_t variableIndex,
                                   void *jacobianFunction, uint64_t numOfSeeds,
                                   uint64_t *seedSizes) {
  static_cast<KINSOLInstance *>(instance)->addJacobianFunction(
      equationIndex, variableIndex,
      reinterpret_cast<JacobianFunction>(jacobianFunction), numOfSeeds,
      seedSizes);
}

RUNTIME_FUNC_DEF(kinsolAddJacobian, void, PTR(void), uint64_t, uint64_t,
                 PTR(void), uint64_t, PTR(uint64_t))

#endif // SUNDIALS_ENABLE
