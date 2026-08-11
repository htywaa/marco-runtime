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

KINSOLInstance::~KINSOLInstance() {
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[KINSOL] Instance destroyed" << std::endl;
  }
}

Variable KINSOLInstance::addVariable(uint64_t rank, const uint64_t *dimensions,
                                     VariableGetter getterFunction,
                                     VariableSetter setterFunction,
                                     const char *name) {
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
  timedResidualPostprocessor = std::move(callback);
}

void KINSOLInstance::addJacobianFunction(Equation equation, Variable variable,
                                         JacobianFunction jacobianFunction,
                                         uint64_t numOfSeeds,
                                         uint64_t *seedSizes) {
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
    } while (advanceEquationIndices(equationIndices, equationRanges[eq]));
  }

  // Compute the total amount of non-zero values in the Jacobian Matrix.
  computeNNZ();

  // Compute the equation chunks for each thread.
  computeThreadChunks();

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
  return solveWithStatus(currentTime) == SolveStatus::Success;
}

KINSOLInstance::SolveStatus KINSOLInstance::solveWithStatus(realtype time) {
  // 中文：facade 负责 MARCO 数组与 solver vector 的双向同步；失败时绝不把
  // 未收敛的 trial values 写回模型数组。
  // English: The facade synchronizes MARCO arrays with the solver vector and
  // never writes unconverged trial values back to model storage on failure.
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
  nonlinearSystem->setLineSearchEnabled(lineSearchEnabled);
  SparseNonlinearSystem::SolveStatus status = nonlinearSystem->solve(time);
  if (status != SparseNonlinearSystem::SolveStatus::Success) {
    return status == SparseNonlinearSystem::SolveStatus::RecoverableFailure
               ? SolveStatus::RecoverableFailure
               : SolveStatus::FatalFailure;
  }
  copyVariablesIntoMARCO(variables);
  return SolveStatus::Success;
}

bool KINSOLInstance::solve(realtype time) {
  return solveWithStatus(time) == SolveStatus::Success;
}

bool KINSOLInstance::solve(realtype time, bool enableLineSearch) {
  lineSearchEnabled = enableLineSearch;
  return solveWithStatus(time) == SolveStatus::Success;
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
  // 中文：Schur/condition 查询总是在当前 MARCO 数组值上刷新一次 Jacobian，
  // 随后的多个 RHS 共享该分解直到下一次显式 factorize。
  // English: Schur and condition queries refresh the Jacobian from current
  // MARCO arrays once; subsequent right-hand sides share that factorization
  // until the next explicit factorize call.
  currentTime = time;
  if (!initialized && !initialize()) {
    return false;
  }
  if (getNumOfScalarEquations() == 0) {
    return true;
  }
  copyVariablesFromMARCO(nonlinearSystem->getVariablesVector());
  return nonlinearSystem->factorize(time);
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
  return nonlinearSystem == nullptr
             ? 0
             : nonlinearSystem->getLastNonlinearIterations();
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
          const JacobianSeedsMap &jacobianSeedsMap) {
        assert(equationIndices.size() == instance->getEquationRank(eq));

        uint64_t equationArrayOffset = instance->equationOffsets[eq];

        uint64_t equationScalarOffset =
            getEquationFlatIndex(equationIndices, instance->equationRanges[eq]);

        uint64_t offset = equationArrayOffset + equationScalarOffset;

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
          const JacobianSeedsMap &jacobianSeedsMap) {
        uint64_t equationArrayOffset = instance->equationOffsets[eq];

        uint64_t equationScalarOffset =
            getEquationFlatIndex(equationIndices, instance->equationRanges[eq]);

        uint64_t scalarEquationIndex =
            equationArrayOffset + equationScalarOffset;

        assert(scalarEquationIndex < instance->getNumOfScalarEquations());

        // Compute the column indexes that may be non-zeros.
        std::vector<JacobianColumn> jacobianColumns =
            instance->computeJacobianColumns(eq, equationIndices.data());

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

      assert([&]() -> bool {
        for (uint64_t i = 0; i < variableRank; ++i) {
          if (variableIndices[i] >= variablesDimensions[variable][i]) {
            return false;
          }
        }

        return true;
      }() && "Access out of bounds");

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

  std::vector<JacobianColumn> orderedColumns;

  for (const JacobianColumn &column : uniqueColumns) {
    orderedColumns.push_back(column);
  }

  std::sort(orderedColumns.begin(), orderedColumns.end(),
            [](const JacobianColumn &first, const JacobianColumn &second) {
              if (first.first != second.first) {
                return first.first < second.first;
              }

              assert(first.second.size() == second.second.size());

              for (size_t i = 0, e = first.second.size(); i < e; ++i) {
                if (first.second[i] < second.second[i]) {
                  return true;
                }
              }

              return false;
            });

  return orderedColumns;
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
                       const JacobianSeedsMap &jacobianSeedsMap)>
        processFn) {
  // Shard the work among multiple threads.
  unsigned int numOfThreads = threadPool.getNumOfThreads();
  std::atomic_size_t chunkIndex = 0;
  const auto &simulationOptions = marco::runtime::simulation::getOptions();
  bool collectStats = simulationOptions.debug || simulationOptions.profiling;
  std::vector<::marco::runtime::profiling::ParallelThreadWorkStats> threadWork(
      numOfThreads);
  std::vector<std::thread::id> threadIds(numOfThreads);

  for (unsigned int thread = 0; thread < numOfThreads; ++thread) {
    threadPool.async([&, thread]() {
      if (collectStats) {
        threadIds[thread] = std::this_thread::get_id();
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

          processFn(equation, equationIndices, std::get<3>(chunk));

          if (collectStats) {
            ++threadWork[thread].scalarEquations;
          }
        } while (advanceEquationIndicesUntil(
            equationIndices, equationRanges[equation], std::get<2>(chunk)));
      }
    });
  }

  threadPool.wait();

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
