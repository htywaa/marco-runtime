#ifdef SUNDIALS_ENABLE

#include "marco/Runtime/Solvers/IDA/Instance.h"
#include "marco/Runtime/Simulation/Options.h"
#include "marco/Runtime/Solvers/IDA/Options.h"
#include "marco/Runtime/Solvers/IDA/Profiler.h"
#include "marco/Runtime/Solvers/KINSOL/Instance.h"
#include "marco/Runtime/Support/MemoryManagement.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <thread>
#include <tuple>

using namespace ::marco::runtime;
using namespace ::marco::runtime::sundials;
using namespace ::marco::runtime::sundials::ida;

//===---------------------------------------------------------------------===//
// Solver
//===---------------------------------------------------------------------===//

namespace marco::runtime::sundials::ida {
// 中文：运行期 IDA 探针，用于定位 residual/Jacobian/初始化耗时和并行遍历
// 进度；默认关闭，设置 MARCO_RUNTIME_IDA_PROBE=0 也视为关闭。
// English: Runtime IDA probe for residual, Jacobian, initialization timing, and
// parallel traversal progress. It is disabled by default, and
// MARCO_RUNTIME_IDA_PROBE=0 is treated as disabled.
static bool isRuntimeIDAProbeEnabled() {
  static const bool enabled = []() {
    const char *value = std::getenv("MARCO_RUNTIME_IDA_PROBE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();

  return enabled;
}

enum class Stage6GRuntimeProbeMode { Off, Summary, Trace, Invalid };

static Stage6GRuntimeProbeMode getStage6GRuntimeProbeMode() {
  // 中文：summary 只维护峰值证书，trace 额外记录 accepted/output 样本；非法值
  // 保留为独立状态，由初始化阶段给出确定性诊断，而非静默关闭 probe。
  // English: Summary retains peak certificates only, while trace also records
  // accepted/output samples. Invalid values remain explicit so initialization
  // diagnoses them deterministically instead of silently disabling the probe.
  static const Stage6GRuntimeProbeMode mode = []() {
    const char *value = std::getenv("MARCO_STAGE6G_RUNTIME_PROBE");
    if (!value || value[0] == '\0') {
      return Stage6GRuntimeProbeMode::Off;
    }
    if (std::string(value) == "summary") {
      return Stage6GRuntimeProbeMode::Summary;
    }
    if (std::string(value) == "trace") {
      return Stage6GRuntimeProbeMode::Trace;
    }
    return Stage6GRuntimeProbeMode::Invalid;
  }();
  return mode;
}

static bool isStage6GRuntimeProbeEnabled() {
  Stage6GRuntimeProbeMode mode = getStage6GRuntimeProbeMode();
  return mode == Stage6GRuntimeProbeMode::Summary ||
         mode == Stage6GRuntimeProbeMode::Trace;
}

static bool isConstraintBasisSwitchDisabled() {
  static const bool disabled = []() {
    const char *value = std::getenv("MARCO_STAGE6G_DISABLE_BASIS_SWITCH");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return disabled;
}

static uint64_t elapsedMillisecondsSince(
    std::chrono::steady_clock::time_point start) {
  auto elapsed = std::chrono::steady_clock::now() - start;
  return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
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

struct IDAInstance::ConstraintComponentExecutionRuntime {
  // 中文：一个 runtime execution group 可包含多个语义 component，并拥有一个
  // 稀疏 local closure、DAG 依赖、condition block 与 Schur 响应缓存。
  // English: One runtime execution group may contain several semantic
  // components and owns one sparse local closure, DAG dependencies, condition
  // block, and Schur-response cache.
  uint64_t executionGroupId{0};
  std::set<uint64_t> semanticComponentIds;
  std::optional<ConstraintSwitchScope> switchScope;
  std::vector<uint64_t> dependencies;
  uint64_t topologicalOrder{0};
  uint64_t stateSetId{0};
  std::unique_ptr<kinsol::KINSOLInstance> solver;
  std::vector<Variable> globalVariables;
  std::vector<Equation> globalEquations;
  std::vector<Equation> outerEquations;
  std::map<Variable, Variable> localVariableIds;
  std::map<Equation, Equation> localEquationIds;
  std::map<Variable, uint64_t> localVariableOffsets;
  std::map<Equation, uint64_t> localEquationOffsets;
  std::vector<JacobianColumn> localScalarColumns;
  std::vector<std::pair<Equation, std::vector<int64_t>>> localScalarRows;
  std::vector<JacobianColumn> outerCouplingColumns;
  std::map<JacobianColumn, uint64_t> localColumnIndices;
  std::map<JacobianColumn, std::vector<double>> totalResponses;
  std::vector<uint64_t> localRowComponents;
  std::map<JacobianColumn, uint64_t> localColumnComponents;
  std::map<uint64_t, std::vector<JacobianColumn>>
      componentOuterCouplingColumns;
  std::map<JacobianFunction, std::vector<uint64_t>> couplingSeeds;
  std::vector<uint64_t> conditionRows;
  std::vector<uint64_t> conditionColumns;
  uint64_t localScalars{0};
  bool scalingConfigured{false};
  uint64_t localSolveAttempts{0};
  uint64_t localSolveSuccesses{0};
  uint64_t localSolveRecoverableFailures{0};
  uint64_t localSolveFatalFailures{0};
  uint64_t localNonlinearIterations{0};
  double maximumLocalResidual{0};
  double maximumLocalResidualTime{0};
  double maximumCondition{0};
  double maximumConditionTime{0};
  double maximumChartCondition{0};
  double maximumLocalClosureCondition{0};
  double maximumResponseAmplification{0};
  double maximumResponseAmplificationTime{0};
  ConstraintBasisHealthCertificate lastHealthCertificate;
  uint64_t schurFactorizations{0};
  uint64_t schurRightHandSides{0};
  uint64_t schurFillEntries{0};
};

static void printParallelIterationStats(
    EquationsParallelIterationKind kind,
    const std::vector<profiling::ParallelThreadWorkStats> &threadWork,
    const std::vector<std::thread::id> &threadIds) {
  std::cerr << "[IDA] " << getParallelIterationKindName(kind)
            << " parallel iteration work" << std::endl;

  for (size_t i = 0, e = threadWork.size(); i < e; ++i) {
    std::cerr << "  worker " << i << " (thread " << threadIds[i]
              << "): chunks=" << threadWork[i].chunks
              << ", scalar equations=" << threadWork[i].scalarEquations
              << std::endl;
  }
}

ConstraintConditionEstimate
estimateConstraintCondition(const std::vector<double> &matrix, uint64_t size) {
  // 中文：该 dense LU 仅服务小型局部 condition block 的诊断/测试；完整
  // component Jacobian 始终留在 sparse KLU 路径，不构造全局 dense 矩阵。
  // English: This dense LU is only for diagnostics/tests of small local
  // condition blocks. Full component Jacobians stay on the sparse KLU path and
  // are never materialized as global dense matrices.
  ConstraintConditionEstimate result;
  if (size == 0 || matrix.size() != size * size) {
    result.rankDeficient = true;
    result.oneNormCondition = std::numeric_limits<double>::infinity();
    return result;
  }

  double infinityNorm = 0;
  double oneNorm = 0;
  for (uint64_t row = 0; row < size; ++row) {
    double rowSum = 0;
    for (uint64_t column = 0; column < size; ++column) {
      rowSum += std::abs(matrix[row * size + column]);
    }
    infinityNorm = std::max(infinityNorm, rowSum);
  }
  for (uint64_t column = 0; column < size; ++column) {
    double columnSum = 0;
    for (uint64_t row = 0; row < size; ++row) {
      columnSum += std::abs(matrix[row * size + column]);
    }
    oneNorm = std::max(oneNorm, columnSum);
  }

  const double threshold =
      100 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, infinityNorm);
  std::vector<double> lu = matrix;
  std::vector<uint64_t> permutation(size);
  for (uint64_t i = 0; i < size; ++i) {
    permutation[i] = i;
  }

  result.minimumPivot = std::numeric_limits<double>::infinity();
  for (uint64_t column = 0; column < size; ++column) {
    uint64_t pivotRow = column;
    double pivotMagnitude = std::abs(lu[column * size + column]);
    for (uint64_t row = column + 1; row < size; ++row) {
      double candidate = std::abs(lu[row * size + column]);
      if (candidate > pivotMagnitude) {
        pivotMagnitude = candidate;
        pivotRow = row;
      }
    }

    if (pivotMagnitude <= threshold || !std::isfinite(pivotMagnitude)) {
      result.rankDeficient = true;
      break;
    }
    if (pivotRow != column) {
      for (uint64_t j = 0; j < size; ++j) {
        std::swap(lu[column * size + j], lu[pivotRow * size + j]);
      }
      std::swap(permutation[column], permutation[pivotRow]);
    }

    double pivot = std::abs(lu[column * size + column]);
    result.minimumPivot = std::min(result.minimumPivot, pivot);
    result.maximumPivot = std::max(result.maximumPivot, pivot);
    ++result.rank;
    for (uint64_t row = column + 1; row < size; ++row) {
      lu[row * size + column] /= lu[column * size + column];
      for (uint64_t j = column + 1; j < size; ++j) {
        lu[row * size + j] -=
            lu[row * size + column] * lu[column * size + j];
      }
    }
  }

  if (result.rank != size) {
    result.rankDeficient = true;
    result.oneNormCondition = std::numeric_limits<double>::infinity();
    if (!std::isfinite(result.minimumPivot)) {
      result.minimumPivot = 0;
    }
    return result;
  }

  double inverseOneNorm = 0;
  std::vector<double> rhs(size);
  std::vector<double> y(size);
  std::vector<double> x(size);
  for (uint64_t inverseColumn = 0; inverseColumn < size; ++inverseColumn) {
    for (uint64_t row = 0; row < size; ++row) {
      rhs[row] = permutation[row] == inverseColumn ? 1.0 : 0.0;
      y[row] = rhs[row];
      for (uint64_t j = 0; j < row; ++j) {
        y[row] -= lu[row * size + j] * y[j];
      }
    }
    for (uint64_t reverse = size; reverse > 0; --reverse) {
      uint64_t row = reverse - 1;
      x[row] = y[row];
      for (uint64_t j = row + 1; j < size; ++j) {
        x[row] -= lu[row * size + j] * x[j];
      }
      x[row] /= lu[row * size + row];
    }
    double columnSum = 0;
    for (double value : x) {
      columnSum += std::abs(value);
    }
    inverseOneNorm = std::max(inverseOneNorm, columnSum);
  }

  result.oneNormCondition = oneNorm * inverseOneNorm;
  result.reciprocalCondition =
      result.oneNormCondition > 0 ? 1.0 / result.oneNormCondition : 1;
  return result;
}

bool requiresDynamicStateSelection(
    const ConstraintConditionEstimate &estimate, bool failedLocalTrial) {
  constexpr double hardConditionLimit = 1e8;
  constexpr double failedTrialConditionLimit = 1e6;
  return estimate.rankDeficient ||
         !std::isfinite(estimate.oneNormCondition) ||
         estimate.oneNormCondition >= hardConditionLimit ||
         (failedLocalTrial &&
          estimate.oneNormCondition >= failedTrialConditionLimit);
}

bool shouldRequestConstraintStateSetSwitch(
    const ConstraintConditionEstimate &estimate, bool failedLocalTrial) {
  constexpr double switchHysteresisLimit = 1e3;
  return requiresDynamicStateSelection(estimate, failedLocalTrial) ||
         (std::isfinite(estimate.oneNormCondition) &&
          estimate.oneNormCondition >= switchHysteresisLimit);
}

ConstraintBasisHealthAction classifyConstraintBasisHealth(
    const ConstraintBasisHealthCertificate &certificate,
    bool failedLocalTrial) {
  constexpr double responseHardLimit = 1e8;
  constexpr double responseSwitchLimit = 1e4;
  // 中文：切换依据是 chart、完整 local closure 与 Schur response amplification
  // 的联合证书；单独一个未经缩放的坐标条件数不构成充分触发条件。
  // English: Switching uses a joint certificate over chart conditioning, full
  // local closure, and Schur response amplification. One unscaled coordinate
  // condition number is not sufficient evidence by itself.
  bool conditionUnsafe =
      requiresDynamicStateSelection(certificate.chart, failedLocalTrial) ||
      requiresDynamicStateSelection(certificate.localClosure,
                                    failedLocalTrial);
  bool responseUnsafe =
      certificate.responseAvailable &&
      (!std::isfinite(certificate.responseAmplification) ||
       certificate.responseAmplification >= responseHardLimit);
  if (conditionUnsafe || responseUnsafe) {
    return ConstraintBasisHealthAction::Unsafe;
  }
  bool conditionRequestsSwitch =
      shouldRequestConstraintStateSetSwitch(certificate.chart,
                                            failedLocalTrial) ||
      shouldRequestConstraintStateSetSwitch(certificate.localClosure,
                                            failedLocalTrial);
  bool responseRequestsSwitch =
      certificate.responseAvailable &&
      certificate.responseAmplification >= responseSwitchLimit;
  return conditionRequestsSwitch || responseRequestsSwitch
             ? ConstraintBasisHealthAction::RequestSwitch
             : ConstraintBasisHealthAction::Accept;
}

IDARestartRecoveryState createIDARestartRecoveryState(
    double acceptedStep, double outputStep, double configuredInitialStep,
    double configuredMinimumStep, double configuredMaximumStep) {
  // 中文：IDAReInit 丢弃高阶历史，因此以最近 accepted step 为锚点选择较小
  // 初始步长，并在后续 accepted steps 中逐级恢复 step/order 上限。
  // English: IDAReInit discards high-order history. Anchor a smaller initial
  // step to the latest accepted step and restore step/order limits gradually
  // over subsequent accepted steps.
  IDARestartRecoveryState state;
  double referenceStep = std::abs(acceptedStep);
  if (!(referenceStep > 0) || !std::isfinite(referenceStep)) {
    referenceStep = std::abs(configuredInitialStep);
  }
  if (!(referenceStep > 0) || !std::isfinite(referenceStep)) {
    referenceStep = std::abs(outputStep);
  }
  if (!(referenceStep > 0) || !std::isfinite(referenceStep)) {
    return state;
  }
  if (configuredMaximumStep > 0 &&
      std::isfinite(configuredMaximumStep)) {
    referenceStep = std::min(referenceStep, configuredMaximumStep);
  }
  double minimumStep =
      configuredMinimumStep > 0 && std::isfinite(configuredMinimumStep)
          ? configuredMinimumStep
          : 0;
  state.initialStep = std::max(minimumStep, referenceStep * 0.25);
  state.targetStep = std::max(state.initialStep, referenceStep);
  state.currentStepLimit =
      std::min(state.targetStep, state.initialStep * 2);
  state.currentMaximumOrder = 1;
  state.targetMaximumOrder = 5;
  state.active = true;
  return state;
}

bool advanceIDARestartRecoveryState(IDARestartRecoveryState &state) {
  if (!state.active) {
    return true;
  }
  ++state.acceptedSteps;
  state.currentMaximumOrder =
      std::min(state.targetMaximumOrder, state.currentMaximumOrder + 1);
  state.currentStepLimit =
      std::min(state.targetStep, state.currentStepLimit * 2);
  bool complete = state.currentMaximumOrder == state.targetMaximumOrder &&
                  state.currentStepLimit >= state.targetStep;
  if (complete) {
    state.active = false;
  }
  return complete;
}

bool isSafeConstraintStateSetCandidate(bool localSolveSucceeded,
                                       double maximumCondition,
                                       double maximumConstraintResidual,
                                       double maximumOuterStateJump) {
  return localSolveSucceeded && std::isfinite(maximumCondition) &&
         maximumCondition <= 1e4 &&
         std::isfinite(maximumConstraintResidual) &&
         maximumConstraintResidual <= 1e-10 &&
         std::isfinite(maximumOuterStateJump) &&
         maximumOuterStateJump <= 1e-9;
}

bool hasCompleteConstraintResidualCertificate(
    const ConstraintResidualRoleCounts &expected,
    const ConstraintResidualRoleCounts &observed) {
  return observed.position == expected.position &&
         observed.tangent == expected.tangent &&
         observed.highest == expected.highest &&
         observed.support == expected.support;
}

std::optional<double> getMaximumConstraintSwitchScopeCondition(
    const std::vector<ConstraintSwitchHealthSample> &samples,
    ConstraintSwitchScope scope) {
  std::optional<double> result;
  for (const ConstraintSwitchHealthSample &sample : samples) {
    if (sample.scope != scope) {
      continue;
    }
    if (!std::isfinite(sample.condition)) {
      return std::numeric_limits<double>::infinity();
    }
    result = std::max(result.value_or(0), sample.condition);
  }
  return result;
}

std::optional<uint64_t> selectConstraintExecutionEpoch(
    const std::vector<ConstraintExecutionEpochCandidate> &candidates) {
  // 中文：只在 safe candidates 中按 stateSelect 软代价、condition 和稳定 ID
  // 排序；硬约束已由编译期 epoch certificate 保证。
  // English: Rank only safe candidates by stateSelect soft cost, condition,
  // and stable ID. Compile-time epoch certificates already enforce hard
  // constraints.
  const ConstraintExecutionEpochCandidate *best = nullptr;
  auto ordering = [](const ConstraintExecutionEpochCandidate &candidate) {
    return std::tuple(candidate.preferDummyScalars,
                      candidate.defaultDummyScalars,
                      std::numeric_limits<uint64_t>::max() -
                          candidate.avoidDummyScalars,
                      candidate.condition, candidate.epochId);
  };
  for (const ConstraintExecutionEpochCandidate &candidate : candidates) {
    if (!candidate.safe) {
      continue;
    }
    if (best == nullptr || ordering(candidate) < ordering(*best)) {
      best = &candidate;
    }
  }
  return best == nullptr ? std::nullopt
                         : std::optional<uint64_t>(best->epochId);
}

bool shouldSuppressIDAError(int errorCode, bool pendingBasisSwitch,
                            bool expectedConstraintFailure) {
  return (pendingBasisSwitch || expectedConstraintFailure) &&
         (errorCode == IDA_RES_FAIL || errorCode == IDA_REP_RES_ERR);
}

bool composeExecutionGroupDerivative(
    double directDerivative,
    const std::vector<double> &crossDerivatives,
    const std::vector<double> &dependencyResponses, double &result) {
  if (!std::isfinite(directDerivative) ||
      crossDerivatives.size() != dependencyResponses.size()) {
    return false;
  }
  // 中文：DAG sensitivity 按拓扑序合成 direct term 与上游 local response；
  // 非有限输入立即拒绝，不能静默污染 Schur correction。
  // English: Compose the direct term with upstream local responses in DAG
  // order. Reject non-finite inputs immediately rather than silently
  // contaminating the Schur correction.
  result = directDerivative;
  for (uint64_t i = 0; i < crossDerivatives.size(); ++i) {
    if (!std::isfinite(crossDerivatives[i]) ||
        !std::isfinite(dependencyResponses[i])) {
      return false;
    }
    result -= crossDerivatives[i] * dependencyResponses[i];
  }
  return std::isfinite(result);
}

IDAInstance::IDAInstance()
    : startTime(simulation::getOptions().startTime),
      endTime(simulation::getOptions().endTime),
      timeStep(getOptions().timeStep) {
#if SUNDIALS_VERSION_MAJOR >= 7
#ifdef MPI_ENABLE
  comm = MPI_COMM_WORLD;
#else
  comm = SUN_COMM_NULL;
#endif
#endif

  // Initially there is are no variables or equations in the instance.
  variableOffsets.push_back(0);
  equationOffsets.push_back(0);

  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[IDA] Instance created" << std::endl;
  }
}

IDAInstance::~IDAInstance() {
  if (isStage6GRuntimeProbeEnabled() && !constraintComponents.empty()) {
    printConstraintProbeSummary();
  }

  if (getNumOfScalarEquations() != 0) {
    N_VDestroy(variablesVector);
    N_VDestroy(derivativesVector);
    N_VDestroy(idVector);
    N_VDestroy(tolerancesVector);

    IDAFree(&idaMemory);
    SUNLinSolFree(linearSolver);
    SUNMatDestroy(sparseMatrix);
  }

#if SUNDIALS_VERSION_MAJOR >= 6
  if (ctx != nullptr) {
    SUNContext_Free(&ctx);
  }
#endif

  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[IDA] Instance destroyed" << std::endl;
  }
}

void IDAInstance::setStartTime(double time) {
  startTime = time;

  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[IDA] Start time set to " << startTime << std::endl;
  }
}

void IDAInstance::setEndTime(double time) {
  endTime = time;

  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[IDA] End time set to " << endTime << std::endl;
  }
}

void IDAInstance::setTimeStep(double step) {
  assert(step > 0);
  timeStep = step;

  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[IDA] Time step set to " << timeStep << std::endl;
  }
}

void IDAInstance::setSuppressAlgebraicErrorTest(bool suppress) {
  modelSuppressAlgebraicErrorTest = suppress ? SUNTRUE : SUNFALSE;
}

void IDAInstance::registerConstraintComponent(uint64_t componentId) {
  // 中文：注册阶段只收集编译期 descriptor，不分配 IDA/KINSOL 对象；所有 owner
  // 与 square 证书会在 initialize 前统一验证。
  // English: Registration only collects compile-time descriptors and allocates
  // no IDA/KINSOL objects. Ownership and square certificates are validated as a
  // whole before initialization.
  constraintComponents[componentId] = {componentId, 0, 0, 0, 0,
                                       0,           0, 0, 0};
}

void IDAInstance::registerConstraintComponentRoleOwnership(
    uint64_t componentId, const char *role, uint64_t ownedScalars,
    uint64_t residualScalars) {
  if (constraintComponents.find(componentId) == constraintComponents.end()) {
    registerConstraintComponent(componentId);
  }
  ConstraintComponentDescriptor &component =
      constraintComponents[componentId];
  std::string roleName = role ? role : "unknown";
  if (roleName == "position") {
    component.positionScalars = ownedScalars;
    component.residualPositionScalars = residualScalars;
  } else if (roleName == "tangent") {
    component.tangentScalars = ownedScalars;
    component.residualTangentScalars = residualScalars;
  } else if (roleName == "highest") {
    component.highestScalars = ownedScalars;
    component.residualHighestScalars = residualScalars;
  } else if (roleName == "auxiliary_support") {
    component.supportScalars = ownedScalars;
    component.residualSupportScalars = residualScalars;
  }
}

void IDAInstance::registerConstraintEquation(uint64_t componentId,
                                             Equation equation,
                                             const char *role,
                                             uint64_t order,
                                             const char *matchedVariableName) {
  if (constraintComponents.find(componentId) == constraintComponents.end()) {
    registerConstraintComponent(componentId);
  }
  constraintEquations.push_back({componentId, equation,
                                 role ? role : "unknown", order,
                                 matchedVariableName ? matchedVariableName
                                                     : ""});
}

void IDAInstance::registerConstraintVariable(uint64_t componentId,
                                             Variable variable,
                                             const char *role,
                                             const char *name) {
  if (constraintComponents.find(componentId) == constraintComponents.end()) {
    registerConstraintComponent(componentId);
  }
  constraintVariables.push_back(
      {componentId, variable, role ? role : "unknown", name ? name : ""});
}

void IDAInstance::registerSolverExecutionGroup(uint64_t groupId,
                                               const char *kind,
                                               const char *mode,
                                               uint64_t topologicalOrder) {
  SolverExecutionGroupDescriptor descriptor{
      groupId,
      kind ? kind : "unknown",
      mode ? mode : "unknown",
      topologicalOrder,
      0,
      0,
      0,
      0,
      {},
      {}};
  auto [it, inserted] =
      solverExecutionGroups.emplace(groupId, std::move(descriptor));
  if (!inserted) {
    it->second.kind = "invalid_duplicate";
  }
}

void IDAInstance::registerSolverExecutionGroupCertificate(
    uint64_t groupId, uint64_t outerEquationScalars,
    uint64_t outerVariableScalars, uint64_t localEquationScalars,
    uint64_t localVariableScalars) {
  auto group = solverExecutionGroups.find(groupId);
  if (group == solverExecutionGroups.end()) {
    return;
  }
  group->second.outerEquationScalars = outerEquationScalars;
  group->second.outerVariableScalars = outerVariableScalars;
  group->second.localEquationScalars = localEquationScalars;
  group->second.localVariableScalars = localVariableScalars;
}

void IDAInstance::registerSolverExecutionGroupMember(uint64_t groupId,
                                                     uint64_t componentId) {
  auto group = solverExecutionGroups.find(groupId);
  if (group == solverExecutionGroups.end() ||
      !group->second.semanticComponentIds.insert(componentId).second) {
    if (group != solverExecutionGroups.end()) {
      group->second.kind = "invalid_duplicate_member";
    }
    return;
  }
}

void IDAInstance::registerSolverExecutionGroupDependency(
    uint64_t groupId, uint64_t dependencyGroupId) {
  auto group = solverExecutionGroups.find(groupId);
  if (group == solverExecutionGroups.end() ||
      !group->second.dependencies.insert(dependencyGroupId).second ||
      groupId == dependencyGroupId) {
    if (group != solverExecutionGroups.end()) {
      group->second.kind = "invalid_dependency";
    }
    return;
  }
}

void IDAInstance::enableConstraintComponentExecution(
    uint64_t executionGroupId) {
  if (solverExecutionGroups.find(executionGroupId) ==
      solverExecutionGroups.end()) {
    SolverExecutionGroupDescriptor legacy{
        executionGroupId, "constraint", "exact_causalized", 0,
        0, 0, 0, 0, {}, {}};
    legacy.semanticComponentIds.insert(executionGroupId);
    solverExecutionGroups.emplace(executionGroupId, std::move(legacy));
    if (constraintComponents.find(executionGroupId) ==
        constraintComponents.end()) {
      registerConstraintComponent(executionGroupId);
    }
  }
  executableConstraintComponents.insert(executionGroupId);
}

void IDAInstance::registerConstraintVariableExecution(
    uint64_t executionGroupId, Variable variable, const char *owner) {
  enableConstraintComponentExecution(executionGroupId);
  constraintVariableExecutions.push_back(
      {executionGroupId, variable, owner ? owner : "unknown"});
}

void IDAInstance::registerConstraintEquationExecution(
    uint64_t executionGroupId, Equation equation, const char *owner) {
  enableConstraintComponentExecution(executionGroupId);
  constraintEquationExecutions.push_back(
      {executionGroupId, equation, owner ? owner : "unknown"});
}

void IDAInstance::registerConstraintStateSet(uint64_t componentId,
                                             uint64_t stateSetId,
                                             bool active,
                                             const char *canonicalKey) {
  constraintStateSets.push_back(
      {componentId, stateSetId, active,
       canonicalKey ? canonicalKey : ""});
}

void IDAInstance::registerConstraintStateSetVariable(
    uint64_t componentId, uint64_t stateSetId, Variable variable,
    const char *owner, const char *name) {
  constraintStateSetVariables.push_back(
      {componentId, stateSetId, variable, owner ? owner : "unknown",
       name ? name : ""});
}

void IDAInstance::registerConstraintStateSetEquation(
    uint64_t componentId, uint64_t stateSetId, Equation equation,
    const char *owner, const char *role, uint64_t order,
    const char *matchedVariableName) {
  constraintStateSetEquations.push_back(
      {componentId, stateSetId, equation, owner ? owner : "unknown",
       role ? role : "unknown", order,
      matchedVariableName ? matchedVariableName : ""});
}

void IDAInstance::registerConstraintExecutionEpoch(
    uint64_t domainId, uint64_t epochId, bool active,
    const char *canonicalKey, uint64_t preferDummyScalars,
    uint64_t defaultDummyScalars, uint64_t avoidDummyScalars) {
  auto key = std::make_pair(domainId, epochId);
  ConstraintExecutionEpochDescriptor descriptor{
      domainId,
      epochId,
      active,
      canonicalKey ? canonicalKey : "",
      preferDummyScalars,
      defaultDummyScalars,
      avoidDummyScalars,
      {}};
  if (!constraintExecutionEpochs.emplace(key, std::move(descriptor)).second) {
    constraintExecutionEpochs[key].canonicalKey = "invalid_duplicate";
  }
}

void IDAInstance::registerConstraintExecutionEpochTransition(
    uint64_t domainId, uint64_t fromEpochId, uint64_t toEpochId) {
  auto epoch = constraintExecutionEpochs.find({domainId, fromEpochId});
  if (epoch == constraintExecutionEpochs.end() || fromEpochId == toEpochId ||
      !epoch->second.transitions.insert(toEpochId).second) {
    if (epoch != constraintExecutionEpochs.end()) {
      epoch->second.canonicalKey = "invalid_transition";
    }
  }
}

void IDAInstance::registerConstraintExecutionEpochGroup(
    uint64_t domainId, uint64_t epochId, uint64_t executionGroupId,
    const char *kind, const char *mode, uint64_t topologicalOrder,
    uint64_t outerEquationScalars, uint64_t outerVariableScalars,
    uint64_t localEquationScalars, uint64_t localVariableScalars) {
  auto key = std::make_tuple(domainId, epochId, executionGroupId);
  ConstraintExecutionEpochGroupDescriptor descriptor{
      domainId,
      epochId,
      SolverExecutionGroupDescriptor{
          executionGroupId,
          kind ? kind : "unknown",
          mode ? mode : "unknown",
          topologicalOrder,
          outerEquationScalars,
          outerVariableScalars,
          localEquationScalars,
          localVariableScalars,
          {},
          {}}};
  if (!constraintExecutionEpochGroups.emplace(key, std::move(descriptor))
           .second) {
    constraintExecutionEpochGroups[key].group.kind = "invalid_duplicate";
  }
}

void IDAInstance::registerConstraintExecutionEpochGroupCertificate(
    uint64_t domainId, uint64_t epochId, uint64_t executionGroupId,
    uint64_t outerEquationScalars, uint64_t outerVariableScalars,
    uint64_t localEquationScalars, uint64_t localVariableScalars) {
  auto group = constraintExecutionEpochGroups.find(
      {domainId, epochId, executionGroupId});
  if (group == constraintExecutionEpochGroups.end()) {
    return;
  }
  group->second.group.outerEquationScalars = outerEquationScalars;
  group->second.group.outerVariableScalars = outerVariableScalars;
  group->second.group.localEquationScalars = localEquationScalars;
  group->second.group.localVariableScalars = localVariableScalars;
}

void IDAInstance::registerConstraintExecutionEpochGroupMember(
    uint64_t domainId, uint64_t epochId, uint64_t executionGroupId,
    uint64_t componentId) {
  auto group = constraintExecutionEpochGroups.find(
      {domainId, epochId, executionGroupId});
  if (group == constraintExecutionEpochGroups.end() ||
      !group->second.group.semanticComponentIds.insert(componentId).second) {
    if (group != constraintExecutionEpochGroups.end()) {
      group->second.group.kind = "invalid_duplicate_member";
    }
  }
}

void IDAInstance::registerConstraintExecutionEpochGroupDependency(
    uint64_t domainId, uint64_t epochId, uint64_t executionGroupId,
    uint64_t dependencyGroupId) {
  auto group = constraintExecutionEpochGroups.find(
      {domainId, epochId, executionGroupId});
  if (group == constraintExecutionEpochGroups.end() ||
      executionGroupId == dependencyGroupId ||
      !group->second.group.dependencies.insert(dependencyGroupId).second) {
    if (group != constraintExecutionEpochGroups.end()) {
      group->second.group.kind = "invalid_dependency";
    }
  }
}

void IDAInstance::registerConstraintExecutionEpochVariable(
    uint64_t domainId, uint64_t epochId, uint64_t executionGroupId,
    Variable variable, const char *owner, const char *name) {
  constraintExecutionEpochVariables.push_back(
      {domainId, epochId, executionGroupId, variable,
       owner ? owner : "unknown", name ? name : ""});
}

void IDAInstance::registerConstraintExecutionEpochEquation(
    uint64_t domainId, uint64_t epochId, uint64_t executionGroupId,
    Equation equation, const char *owner) {
  constraintExecutionEpochEquations.push_back(
      {domainId, epochId, executionGroupId, equation,
       owner ? owner : "unknown", "unknown", 0, ""});
}

void IDAInstance::registerConstraintExecutionEpochEquationMetadata(
    uint64_t domainId, uint64_t epochId, Equation equation,
    const char *role, uint64_t order, const char *matchedVariableName) {
  auto descriptor = std::find_if(
      constraintExecutionEpochEquations.begin(),
      constraintExecutionEpochEquations.end(),
      [&](const ConstraintExecutionEpochEquationDescriptor &candidate) {
        return candidate.domainId == domainId &&
               candidate.epochId == epochId &&
               candidate.equation == equation;
      });
  if (descriptor == constraintExecutionEpochEquations.end() ||
      descriptor->role != "unknown") {
    return;
  }
  descriptor->role = role ? role : "unknown";
  descriptor->order = order;
  descriptor->matchedVariableName =
      matchedVariableName ? matchedVariableName : "";
}

bool IDAInstance::validateConstraintStateSetRegistrations() {
  // 中文：验证 legacy state sets 或 Stage 7D execution epochs（二者不可混用），
  // 确保每套配置 owner 唯一、维数一致且 transition 引用完整。
  // English: Validate either legacy state sets or Stage 7D execution epochs
  // (never both), ensuring unique owners, equal dimensions, and complete
  // transition references for every configuration.
  constraintStateSetRegistrationsCertified = false;
  activeConstraintStateSets.clear();
  constraintExecutionEpochResidualCertificates.clear();

  // 中文：新 epoch registry 与 legacy 单-component state-set registry 使用
  // 两套互斥验证路径，防止同一 handle 同时被两种切换协议认领。
  // English: The new epoch registry and legacy single-component state-set
  // registry use mutually exclusive validation paths, preventing one handle
  // from being claimed by both switching protocols.
  if (!constraintExecutionEpochs.empty()) {
    if (!constraintStateSets.empty() || !constraintStateSetVariables.empty() ||
        !constraintStateSetEquations.empty()) {
      std::cerr << "stage7d_invalid_execution_epoch_registration "
                   "reason=mixed_legacy_state_set_registry"
                << std::endl;
      return false;
    }

    struct EpochCounts {
      uint64_t outerVariables{0};
      uint64_t localVariables{0};
      uint64_t outerEquations{0};
      uint64_t localEquations{0};
    };
    using EpochKey = std::pair<uint64_t, uint64_t>;
    using GroupKey = std::tuple<uint64_t, uint64_t, uint64_t>;
    std::map<EpochKey, EpochCounts> counts;
    std::map<GroupKey, EpochCounts> groupCounts;
    std::map<uint64_t, uint64_t> outerDimensions;
    std::set<std::tuple<uint64_t, uint64_t, Variable>> variableOwners;
    std::set<std::tuple<uint64_t, uint64_t, Equation>> equationOwners;
    std::set<uint64_t> domains;

    for (const auto &[key, epoch] : constraintExecutionEpochs) {
      if (epoch.domainId != key.first || epoch.epochId != key.second ||
          epoch.canonicalKey == "invalid_duplicate" ||
          epoch.canonicalKey == "invalid_transition") {
        std::cerr << "stage7d_invalid_execution_epoch_registration domain="
                  << key.first << " epoch=" << key.second
                  << " reason=invalid_epoch_descriptor" << std::endl;
        return false;
      }
      counts.emplace(key, EpochCounts{});
      constraintExecutionEpochResidualCertificates.emplace(
          key, ConstraintResidualRoleCounts{});
      domains.insert(epoch.domainId);
      if (epoch.active &&
          !activeConstraintStateSets
               .emplace(epoch.domainId, epoch.epochId)
               .second) {
        std::cerr << "stage7d_invalid_execution_epoch_registration domain="
                  << epoch.domainId << " reason=multiple_active_epochs"
                  << std::endl;
        return false;
      }
      for (uint64_t target : epoch.transitions) {
        if (constraintExecutionEpochs.find({epoch.domainId, target}) ==
            constraintExecutionEpochs.end()) {
          std::cerr << "stage7d_invalid_execution_epoch_registration domain="
                    << epoch.domainId << " epoch=" << epoch.epochId
                    << " reason=transition_target_missing" << std::endl;
          return false;
        }
      }
    }
    if (activeConstraintStateSets.size() != domains.size()) {
      std::cerr << "stage7d_invalid_execution_epoch_registration "
                   "reason=missing_active_epoch"
                << std::endl;
      return false;
    }

    for (const auto &[key, descriptor] : constraintExecutionEpochGroups) {
      const SolverExecutionGroupDescriptor &group = descriptor.group;
      if (constraintExecutionEpochs.find(
              {descriptor.domainId, descriptor.epochId}) ==
              constraintExecutionEpochs.end() ||
          group.groupId != std::get<2>(key) ||
          group.kind != "constraint" ||
          (group.mode != "exact_causalized" &&
           group.mode != "implicit_closure") ||
          group.outerEquationScalars != group.outerVariableScalars ||
          group.localEquationScalars != group.localVariableScalars ||
          group.semanticComponentIds.empty()) {
        std::cerr << "stage7d_invalid_execution_epoch_registration domain="
                  << descriptor.domainId << " epoch=" << descriptor.epochId
                  << " group=" << group.groupId
                  << " reason=invalid_group_certificate" << std::endl;
        return false;
      }
      for (uint64_t dependency : group.dependencies) {
        auto dependencyIt = constraintExecutionEpochGroups.find(
            {descriptor.domainId, descriptor.epochId, dependency});
        if (dependencyIt == constraintExecutionEpochGroups.end() ||
            dependencyIt->second.group.topologicalOrder >=
                group.topologicalOrder) {
          std::cerr
              << "stage7d_invalid_execution_epoch_registration domain="
              << descriptor.domainId << " epoch=" << descriptor.epochId
              << " group=" << group.groupId
              << " reason=invalid_group_dependency" << std::endl;
          return false;
        }
      }
      groupCounts.emplace(key, EpochCounts{});
    }

    for (const ConstraintExecutionEpochVariableDescriptor &descriptor :
         constraintExecutionEpochVariables) {
      EpochKey epochKey{descriptor.domainId, descriptor.epochId};
      GroupKey groupKey{descriptor.domainId, descriptor.epochId,
                        descriptor.executionGroupId};
      if (counts.find(epochKey) == counts.end() ||
          groupCounts.find(groupKey) == groupCounts.end() ||
          descriptor.variable >= getNumOfArrayVariables() ||
          (descriptor.owner != "outer" && descriptor.owner != "local") ||
          !variableOwners
               .emplace(descriptor.domainId, descriptor.epochId,
                        descriptor.variable)
               .second) {
        std::cerr << "stage7d_invalid_execution_epoch_registration domain="
                  << descriptor.domainId << " epoch=" << descriptor.epochId
                  << " reason=invalid_or_duplicate_variable_owner"
                  << std::endl;
        return false;
      }
      uint64_t scalars = getVariableFlatSize(descriptor.variable);
      if (descriptor.owner == "outer") {
        if (getVariableKind(descriptor.variable) != VariableKind::STATE) {
          std::cerr
              << "stage7d_invalid_execution_epoch_registration domain="
              << descriptor.domainId << " epoch=" << descriptor.epochId
              << " reason=outer_variable_without_state_pair" << std::endl;
          return false;
        }
        counts[epochKey].outerVariables += scalars;
        groupCounts[groupKey].outerVariables += scalars;
      } else {
        counts[epochKey].localVariables += scalars;
        groupCounts[groupKey].localVariables += scalars;
      }
    }

    for (const ConstraintExecutionEpochEquationDescriptor &descriptor :
         constraintExecutionEpochEquations) {
      EpochKey epochKey{descriptor.domainId, descriptor.epochId};
      GroupKey groupKey{descriptor.domainId, descriptor.epochId,
                        descriptor.executionGroupId};
      if (counts.find(epochKey) == counts.end() ||
          groupCounts.find(groupKey) == groupCounts.end() ||
          descriptor.equation >= getNumOfVectorizedEquations() ||
          (descriptor.owner != "outer" && descriptor.owner != "local") ||
          !equationOwners
               .emplace(descriptor.domainId, descriptor.epochId,
                        descriptor.equation)
               .second) {
        std::cerr << "stage7d_invalid_execution_epoch_registration domain="
                  << descriptor.domainId << " epoch=" << descriptor.epochId
                  << " reason=invalid_or_duplicate_equation_owner"
                  << std::endl;
        return false;
      }
      if ((descriptor.role != "position" &&
           descriptor.role != "tangent" &&
           descriptor.role != "highest" &&
           descriptor.role != "auxiliary_support") ||
          descriptor.matchedVariableName.empty()) {
        std::cerr << "stage7d_invalid_execution_epoch_registration domain="
                  << descriptor.domainId << " epoch=" << descriptor.epochId
                  << " equation=" << descriptor.equation
                  << " reason=incomplete_equation_metadata" << std::endl;
        return false;
      }
      uint64_t scalars = getEquationFlatSize(descriptor.equation);
      ConstraintResidualRoleCounts &residualCounts =
          constraintExecutionEpochResidualCertificates.at(epochKey);
      if (descriptor.role == "position") {
        residualCounts.position += scalars;
      } else if (descriptor.role == "tangent") {
        residualCounts.tangent += scalars;
      } else if (descriptor.role == "highest") {
        residualCounts.highest += scalars;
      } else {
        assert(descriptor.role == "auxiliary_support");
        residualCounts.support += scalars;
      }
      if (descriptor.owner == "outer") {
        counts[epochKey].outerEquations += scalars;
        groupCounts[groupKey].outerEquations += scalars;
      } else {
        counts[epochKey].localEquations += scalars;
        groupCounts[groupKey].localEquations += scalars;
      }
    }

    for (const auto &[key, groupCount] : groupCounts) {
      const SolverExecutionGroupDescriptor &group =
          constraintExecutionEpochGroups.at(key).group;
      if (groupCount.outerVariables != group.outerVariableScalars ||
          groupCount.outerEquations != group.outerEquationScalars ||
          groupCount.localVariables != group.localVariableScalars ||
          groupCount.localEquations != group.localEquationScalars) {
        std::cerr << "stage7d_invalid_execution_epoch_registration domain="
                  << std::get<0>(key) << " epoch=" << std::get<1>(key)
                  << " group=" << std::get<2>(key)
                  << " reason=group_runtime_certificate_mismatch"
                  << std::endl;
        return false;
      }
    }
    for (const auto &[key, epochCount] : counts) {
      if (epochCount.outerVariables != epochCount.outerEquations ||
          epochCount.localVariables != epochCount.localEquations) {
        std::cerr << "stage7d_invalid_execution_epoch_registration domain="
                  << key.first << " epoch=" << key.second
                  << " reason=epoch_runtime_square_certificate_failed"
                  << std::endl;
        return false;
      }
      auto [it, inserted] =
          outerDimensions.emplace(key.first, epochCount.outerVariables);
      if (!inserted && it->second != epochCount.outerVariables) {
        std::cerr << "stage7d_invalid_execution_epoch_registration domain="
                  << key.first << " epoch=" << key.second
                  << " reason=unequal_outer_dimensions" << std::endl;
        return false;
      }
    }

    constraintStateSetRegistrationsCertified = true;
    if (isStage6GRuntimeProbeEnabled()) {
      std::cerr << "[stage6g-runtime-probe] stage7d_runtime_contract domains="
                << domains.size() << " epochs=" << counts.size()
                << " groups=" << groupCounts.size()
                << " residual_certificates="
                << constraintExecutionEpochResidualCertificates.size()
                << " registration_certified=1" << std::endl;
    }
    return true;
  }

  if (constraintStateSets.empty()) {
    if (!constraintStateSetVariables.empty() ||
        !constraintStateSetEquations.empty()) {
      std::cerr << "stage6gd_invalid_state_set_registration "
                   "reason=descriptor_without_state_set"
                << std::endl;
      return false;
    }
    constraintStateSetRegistrationsCertified = true;
    return true;
  }

  struct StateSetCounts {
    uint64_t outerVariables{0};
    uint64_t localVariables{0};
    uint64_t outerEquations{0};
    uint64_t localEquations{0};
  };

  using StateSetKey = std::pair<uint64_t, uint64_t>;
  std::map<StateSetKey, StateSetCounts> counts;
  std::map<uint64_t, uint64_t> expectedOuterDimensions;
  std::set<std::tuple<uint64_t, uint64_t, Variable>> variableOwners;
  std::set<std::tuple<uint64_t, uint64_t, Equation>> equationOwners;
  std::set<uint64_t> registeredComponents;
  std::map<uint64_t, uint64_t> semanticToExecutionGroup;
  for (const auto &[groupId, group] : solverExecutionGroups) {
    if (group.kind != "constraint") {
      continue;
    }
    for (uint64_t componentId : group.semanticComponentIds) {
      if (!semanticToExecutionGroup.emplace(componentId, groupId).second) {
        std::cerr << "stage6gd_invalid_state_set_registration component="
                  << componentId
                  << " reason=duplicate_semantic_execution_owner"
                  << std::endl;
        return false;
      }
    }
  }

  for (const ConstraintStateSetDescriptor &stateSet : constraintStateSets) {
    StateSetKey key{stateSet.componentId, stateSet.stateSetId};
    if (!counts.emplace(key, StateSetCounts{}).second) {
      std::cerr << "stage6gd_invalid_state_set_registration component="
                << stateSet.componentId << " state_set="
                << stateSet.stateSetId << " reason=duplicate_state_set"
                << std::endl;
      return false;
    }
    auto executionGroup = semanticToExecutionGroup.find(stateSet.componentId);
    if (constraintComponents.find(stateSet.componentId) ==
            constraintComponents.end() ||
        executionGroup == semanticToExecutionGroup.end() ||
        executableConstraintComponents.find(executionGroup->second) ==
            executableConstraintComponents.end()) {
      std::cerr << "stage6gd_invalid_state_set_registration component="
                << stateSet.componentId
                << " reason=unregistered_execution_component" << std::endl;
      return false;
    }
    registeredComponents.insert(stateSet.componentId);
    if (stateSet.active &&
        !activeConstraintStateSets
             .emplace(stateSet.componentId, stateSet.stateSetId)
             .second) {
      std::cerr << "stage6gd_invalid_state_set_registration component="
                << stateSet.componentId
                << " reason=multiple_active_state_sets" << std::endl;
      return false;
    }
  }

  if (activeConstraintStateSets.size() != registeredComponents.size()) {
    std::cerr << "stage6gd_invalid_state_set_registration "
                 "reason=missing_active_state_set"
              << std::endl;
    return false;
  }

  for (const ConstraintStateSetVariableDescriptor &descriptor :
       constraintStateSetVariables) {
    StateSetKey key{descriptor.componentId, descriptor.stateSetId};
    auto stateSet = counts.find(key);
    if (stateSet == counts.end() ||
        descriptor.variable >= getNumOfArrayVariables() ||
        (descriptor.owner != "outer" && descriptor.owner != "local") ||
        !variableOwners
             .emplace(descriptor.componentId, descriptor.stateSetId,
                      descriptor.variable)
             .second) {
      std::cerr << "stage6gd_invalid_state_set_registration component="
                << descriptor.componentId << " state_set="
                << descriptor.stateSetId
                << " reason=invalid_or_duplicate_variable_owner"
                << std::endl;
      return false;
    }
    uint64_t scalars = getVariableFlatSize(descriptor.variable);
    if (descriptor.owner == "outer") {
      if (getVariableKind(descriptor.variable) != VariableKind::STATE) {
        std::cerr << "stage6gd_invalid_state_set_registration component="
                  << descriptor.componentId << " state_set="
                  << descriptor.stateSetId
                  << " reason=outer_variable_without_state_derivative_pair"
                  << std::endl;
        return false;
      }
      stateSet->second.outerVariables += scalars;
    } else {
      stateSet->second.localVariables += scalars;
    }
  }

  for (const ConstraintStateSetEquationDescriptor &descriptor :
       constraintStateSetEquations) {
    StateSetKey key{descriptor.componentId, descriptor.stateSetId};
    auto stateSet = counts.find(key);
    if (stateSet == counts.end() ||
        descriptor.equation >= getNumOfVectorizedEquations() ||
        (descriptor.owner != "outer" && descriptor.owner != "local") ||
        !equationOwners
             .emplace(descriptor.componentId, descriptor.stateSetId,
                      descriptor.equation)
             .second) {
      std::cerr << "stage6gd_invalid_state_set_registration component="
                << descriptor.componentId << " state_set="
                << descriptor.stateSetId
                << " reason=invalid_or_duplicate_equation_owner"
                << std::endl;
      return false;
    }
    uint64_t scalars = getEquationFlatSize(descriptor.equation);
    if (descriptor.owner == "outer") {
      stateSet->second.outerEquations += scalars;
    } else {
      stateSet->second.localEquations += scalars;
    }
  }

  for (const auto &[key, stateSetCounts] : counts) {
    if (stateSetCounts.outerVariables != stateSetCounts.outerEquations ||
        stateSetCounts.localVariables != stateSetCounts.localEquations) {
      std::cerr << "stage6gd_invalid_state_set_registration component="
                << key.first << " state_set=" << key.second
                << " reason=runtime_square_certificate_failed"
                << " outer_variables=" << stateSetCounts.outerVariables
                << " outer_equations=" << stateSetCounts.outerEquations
                << " local_variables=" << stateSetCounts.localVariables
                << " local_equations=" << stateSetCounts.localEquations
                << std::endl;
      return false;
    }
    auto [expected, inserted] = expectedOuterDimensions.emplace(
        key.first, stateSetCounts.outerVariables);
    if (!inserted && expected->second != stateSetCounts.outerVariables) {
      std::cerr << "stage6gd_invalid_state_set_registration component="
                << key.first << " state_set=" << key.second
                << " reason=unequal_outer_dimensions" << std::endl;
      return false;
    }
  }

  constraintStateSetRegistrationsCertified = true;
  if (isStage6GRuntimeProbeEnabled()) {
    std::cerr << "[stage6g-runtime-probe] "
                 "stage6gd_runtime_contract components="
              << registeredComponents.size() << " state_sets=" << counts.size()
              << " active_state_sets=" << activeConstraintStateSets.size()
              << " registration_certified=1" << std::endl;
  }
  return true;
}

bool IDAInstance::isLocalVariable(Variable variable) const {
  return variable < localConstraintVariables.size() &&
         localConstraintVariables[variable];
}

bool IDAInstance::isLocalEquation(Equation equation) const {
  return equation < localConstraintEquations.size() &&
         localConstraintEquations[equation];
}

bool IDAInstance::prepareConstraintComponentExecutions() {
  // 中文：根据当前 active state set/epoch 重建 outer/local descriptor 布局，按
  // 编译期拓扑序创建每个 local KINSOL；这里不重新 Matching 或猜测 ownership。
  // English: Rebuild outer/local descriptor layouts for the active state
  // set/epoch and create each local KINSOL in compile-time topological order.
  // This stage never reruns Matching or infers ownership.
  localConstraintVariables.assign(getNumOfArrayVariables(), false);
  localConstraintEquations.assign(getNumOfVectorizedEquations(), false);
  constraintComponentExecutions.clear();

  for (uint64_t groupId : activeEpochRuntimeExecutionGroupIds) {
    solverExecutionGroups.erase(groupId);
    executableConstraintComponents.erase(groupId);
  }
  activeEpochRuntimeExecutionGroupIds.clear();
  epochRuntimeExecutionGroupIds.clear();

  // 中文：每次 active epoch 改变都用临时 runtime group IDs 重建当前布局；
  // 编译期 domain/epoch/group IDs 本身保持稳定，便于回滚与 probe。
  // English: Each active-epoch change rebuilds the current layout with
  // temporary runtime group IDs. Compile-time domain/epoch/group IDs remain
  // stable for rollback and diagnostics.
  std::vector<ConstraintVariableExecutionDescriptor> epochVariables;
  std::vector<ConstraintEquationExecutionDescriptor> epochEquations;
  if (!constraintExecutionEpochs.empty()) {
    uint64_t nextRuntimeGroupId = 0;
    if (!solverExecutionGroups.empty()) {
      nextRuntimeGroupId = solverExecutionGroups.rbegin()->first + 1;
    }
    for (const auto &[key, descriptor] : constraintExecutionEpochGroups) {
      (void)descriptor;
      epochRuntimeExecutionGroupIds.emplace(key, nextRuntimeGroupId++);
    }
    for (const auto &[domainId, activeEpochId] :
         activeConstraintStateSets) {
      for (const auto &[key, descriptor] : constraintExecutionEpochGroups) {
        if (descriptor.domainId != domainId ||
            descriptor.epochId != activeEpochId) {
          continue;
        }
        uint64_t runtimeGroupId = epochRuntimeExecutionGroupIds.at(key);
        SolverExecutionGroupDescriptor group = descriptor.group;
        group.groupId = runtimeGroupId;
        std::set<uint64_t> remappedDependencies;
        for (uint64_t dependency : group.dependencies) {
          auto remapped = epochRuntimeExecutionGroupIds.find(
              {domainId, activeEpochId, dependency});
          if (remapped == epochRuntimeExecutionGroupIds.end()) {
            std::cerr << "stage7d_invalid_execution_epoch_registration domain="
                      << domainId << " epoch=" << activeEpochId
                      << " reason=active_dependency_mapping_missing"
                      << std::endl;
            return false;
          }
          remappedDependencies.insert(remapped->second);
        }
        group.dependencies = std::move(remappedDependencies);
        solverExecutionGroups.emplace(runtimeGroupId, std::move(group));
        executableConstraintComponents.insert(runtimeGroupId);
        activeEpochRuntimeExecutionGroupIds.insert(runtimeGroupId);
      }
    }
    for (const ConstraintExecutionEpochVariableDescriptor &descriptor :
         constraintExecutionEpochVariables) {
      auto active = activeConstraintStateSets.find(descriptor.domainId);
      auto runtimeGroup = epochRuntimeExecutionGroupIds.find(
          {descriptor.domainId, descriptor.epochId,
           descriptor.executionGroupId});
      localConstraintVariables[descriptor.variable] = true;
      if (active != activeConstraintStateSets.end() &&
          active->second == descriptor.epochId &&
          runtimeGroup != epochRuntimeExecutionGroupIds.end()) {
        epochVariables.push_back({runtimeGroup->second, descriptor.variable,
                                  descriptor.owner});
        localConstraintVariables[descriptor.variable] = false;
      }
    }
    for (const ConstraintExecutionEpochEquationDescriptor &descriptor :
         constraintExecutionEpochEquations) {
      auto active = activeConstraintStateSets.find(descriptor.domainId);
      auto runtimeGroup = epochRuntimeExecutionGroupIds.find(
          {descriptor.domainId, descriptor.epochId,
           descriptor.executionGroupId});
      localConstraintEquations[descriptor.equation] = true;
      if (active != activeConstraintStateSets.end() &&
          active->second == descriptor.epochId &&
          runtimeGroup != epochRuntimeExecutionGroupIds.end()) {
        epochEquations.push_back({runtimeGroup->second, descriptor.equation,
                                  descriptor.owner});
        localConstraintEquations[descriptor.equation] = false;
      }
    }
  }

  if (executableConstraintComponents.empty()) {
    return true;
  }

  std::map<uint64_t, uint64_t> semanticToExecutionGroup;
  for (const auto &[groupId, group] : solverExecutionGroups) {
    if (group.groupId != groupId || group.kind == "invalid_duplicate" ||
        group.kind == "invalid_duplicate_member" ||
        group.kind == "invalid_dependency" ||
        (group.kind != "constraint" && group.kind != "helper" &&
         group.kind != "ida") ||
        (group.mode != "exact_causalized" &&
         group.mode != "implicit_closure") ||
        group.outerEquationScalars != group.outerVariableScalars ||
        group.localEquationScalars != group.localVariableScalars) {
      std::cerr << "stage7c_invalid_execution_group group=" << groupId
                << " reason=invalid_group_certificate" << std::endl;
      return false;
    }
    for (uint64_t dependency : group.dependencies) {
      auto dependencyIt = solverExecutionGroups.find(dependency);
      if (dependencyIt == solverExecutionGroups.end() ||
          dependencyIt->second.topologicalOrder >= group.topologicalOrder) {
        std::cerr << "stage7c_invalid_execution_group group=" << groupId
                  << " dependency=" << dependency
                  << " reason=invalid_topological_dependency" << std::endl;
        return false;
      }
    }
    for (uint64_t componentId : group.semanticComponentIds) {
      auto [it, inserted] =
          semanticToExecutionGroup.emplace(componentId, groupId);
      if (!inserted && it->second != groupId) {
        std::cerr << "stage7c_invalid_execution_group component="
                  << componentId << " reason=duplicate_semantic_owner"
                  << std::endl;
        return false;
      }
    }
  }
  for (uint64_t groupId : executableConstraintComponents) {
    auto group = solverExecutionGroups.find(groupId);
    if (group == solverExecutionGroups.end() ||
        group->second.kind != "constraint" ||
        group->second.semanticComponentIds.empty()) {
      std::cerr << "stage7c_invalid_execution_group group=" << groupId
                << " reason=enabled_group_not_constraint" << std::endl;
      return false;
    }
  }

  // 中文：将普通 execution descriptors、active legacy state set 和 active
  // epoch descriptors 汇成一个无重叠 active owner 视图。
  // English: Merge ordinary execution descriptors, the active legacy state
  // set, and active epoch descriptors into one non-overlapping active-owner
  // view.
  std::vector<ConstraintVariableExecutionDescriptor> activeVariables;
  std::vector<ConstraintEquationExecutionDescriptor> activeEquations;
  activeVariables.insert(activeVariables.end(), epochVariables.begin(),
                         epochVariables.end());
  activeEquations.insert(activeEquations.end(), epochEquations.begin(),
                         epochEquations.end());
  std::map<uint64_t, uint64_t> activeStateSets;
  std::set<uint64_t> stateSetExecutionGroups;
  if (!constraintStateSets.empty()) {
    if (activeConstraintStateSets.empty()) {
      for (const ConstraintStateSetDescriptor &stateSet : constraintStateSets) {
        if (!stateSet.active) {
          continue;
        }
        if (!activeConstraintStateSets
                 .emplace(stateSet.componentId, stateSet.stateSetId)
                 .second) {
          std::cerr << "stage6gc_invalid_state_set_registry component="
                    << stateSet.componentId
                    << " reason=multiple_active_state_sets" << std::endl;
          return false;
        }
      }
    }
    activeStateSets = activeConstraintStateSets;
    for (const auto &[componentId, stateSetId] : activeStateSets) {
      (void)stateSetId;
      auto group = semanticToExecutionGroup.find(componentId);
      if (group == semanticToExecutionGroup.end() ||
          !stateSetExecutionGroups.insert(group->second).second ||
          solverExecutionGroups[group->second].semanticComponentIds.size() !=
              1) {
        std::cerr << "stage7c_state_set_execution_group_deferred component="
                  << componentId
                  << " reason=state_set_requires_independent_execution_group"
                  << std::endl;
        return false;
      }
    }
  }

  for (const ConstraintVariableExecutionDescriptor &descriptor :
       constraintVariableExecutions) {
    if (stateSetExecutionGroups.find(descriptor.executionGroupId) ==
        stateSetExecutionGroups.end()) {
      activeVariables.push_back(descriptor);
    }
  }
  for (const ConstraintEquationExecutionDescriptor &descriptor :
       constraintEquationExecutions) {
    if (stateSetExecutionGroups.find(descriptor.executionGroupId) ==
        stateSetExecutionGroups.end()) {
      activeEquations.push_back(descriptor);
    }
  }

  if (!constraintStateSets.empty()) {
    for (const ConstraintStateSetVariableDescriptor &descriptor :
         constraintStateSetVariables) {
      if (descriptor.variable >= getNumOfArrayVariables() ||
          (descriptor.owner != "outer" && descriptor.owner != "local")) {
        std::cerr << "stage6gc_invalid_state_set_variable component="
                  << descriptor.componentId << " state_set="
                  << descriptor.stateSetId << std::endl;
        return false;
      }
      // Every state-set-specific handle is excluded until its set is active.
      localConstraintVariables[descriptor.variable] = true;
      auto active = activeStateSets.find(descriptor.componentId);
      if (active != activeStateSets.end() &&
          active->second == descriptor.stateSetId) {
        activeVariables.push_back(
            {semanticToExecutionGroup[descriptor.componentId],
             descriptor.variable, descriptor.owner});
        localConstraintVariables[descriptor.variable] = false;
      }
    }
    for (const ConstraintStateSetEquationDescriptor &descriptor :
         constraintStateSetEquations) {
      if (descriptor.equation >= getNumOfVectorizedEquations() ||
          (descriptor.owner != "outer" && descriptor.owner != "local")) {
        std::cerr << "stage6gc_invalid_state_set_equation component="
                  << descriptor.componentId << " state_set="
                  << descriptor.stateSetId << std::endl;
        return false;
      }
      localConstraintEquations[descriptor.equation] = true;
      auto active = activeStateSets.find(descriptor.componentId);
      if (active != activeStateSets.end() &&
          active->second == descriptor.stateSetId) {
        activeEquations.push_back({semanticToExecutionGroup[descriptor.componentId],
                                   descriptor.equation, descriptor.owner});
        localConstraintEquations[descriptor.equation] = false;
      }
    }
  }

  for (const ConstraintVariableExecutionDescriptor &descriptor :
       activeVariables) {
    if (executableConstraintComponents.find(descriptor.executionGroupId) ==
            executableConstraintComponents.end() ||
        descriptor.variable >= getNumOfArrayVariables() ||
        (descriptor.owner != "outer" && descriptor.owner != "local")) {
      std::cerr << "stage6gb_component_requires_c3 execution_group="
                << descriptor.executionGroupId
                << " reason=invalid_runtime_variable_owner" << std::endl;
      return false;
    }
    if (descriptor.owner == "local") {
      if (localConstraintVariables[descriptor.variable]) {
        std::cerr << "stage6gb_component_requires_c3 execution_group="
                  << descriptor.executionGroupId
                  << " reason=duplicate_runtime_variable_owner"
                  << std::endl;
        return false;
      }
      localConstraintVariables[descriptor.variable] = true;
    }
  }

  for (const ConstraintEquationExecutionDescriptor &descriptor :
       activeEquations) {
    if (executableConstraintComponents.find(descriptor.executionGroupId) ==
            executableConstraintComponents.end() ||
        descriptor.equation >= getNumOfVectorizedEquations() ||
        (descriptor.owner != "outer" && descriptor.owner != "local")) {
      std::cerr << "stage6gb_component_requires_c3 execution_group="
                << descriptor.executionGroupId
                << " reason=invalid_runtime_equation_owner" << std::endl;
      return false;
    }
    if (descriptor.owner == "local") {
      if (localConstraintEquations[descriptor.equation]) {
        std::cerr << "stage6gb_component_requires_c3 execution_group="
                  << descriptor.executionGroupId
                  << " reason=duplicate_runtime_equation_owner"
                  << std::endl;
        return false;
      }
      localConstraintEquations[descriptor.equation] = true;
    }
  }

  // 中文：local solvers 必须按编译期 DAG 拓扑序构造和运行；稳定 group ID 是
  // 同序时的决胜键，保证可重复布局。
  // English: Local solvers are constructed and run in compile-time DAG order;
  // stable group ID breaks ties for reproducible layouts.
  std::vector<uint64_t> orderedExecutionGroups(
      executableConstraintComponents.begin(),
      executableConstraintComponents.end());
  std::sort(orderedExecutionGroups.begin(), orderedExecutionGroups.end(),
            [&](uint64_t left, uint64_t right) {
              const auto &leftGroup = solverExecutionGroups.at(left);
              const auto &rightGroup = solverExecutionGroups.at(right);
              return std::tie(leftGroup.topologicalOrder, left) <
                     std::tie(rightGroup.topologicalOrder, right);
            });

  for (uint64_t executionGroupId : orderedExecutionGroups) {
    const SolverExecutionGroupDescriptor &groupDescriptor =
        solverExecutionGroups.at(executionGroupId);
    auto execution = std::make_unique<ConstraintComponentExecutionRuntime>();
    execution->executionGroupId = executionGroupId;
    execution->semanticComponentIds =
        groupDescriptor.semanticComponentIds;
    // 中文：持久图还包含 helper schedule 和 promoted-IDA dependencies；这些
    // 灵敏度已由生成的 residual/Jacobian callbacks 传播，runtime C2 DAG 只
    // 保留 constraint-to-constraint 边。
    // English: The persistent graph also carries helper scheduling and
    // promoted-IDA dependencies. Generated residual/Jacobian callbacks already
    // propagate those sensitivities, so only constraint-to-constraint edges
    // belong to the runtime C2 DAG.
    for (uint64_t dependencyId : groupDescriptor.dependencies) {
      auto dependency = solverExecutionGroups.find(dependencyId);
      assert(dependency != solverExecutionGroups.end() &&
             "registration validation missed an execution dependency");
      if (dependency->second.kind == "constraint") {
        execution->dependencies.push_back(dependencyId);
      }
    }
    execution->topologicalOrder = groupDescriptor.topologicalOrder;
    auto epochGroup = std::find_if(
        epochRuntimeExecutionGroupIds.begin(),
        epochRuntimeExecutionGroupIds.end(),
        [&](const auto &entry) { return entry.second == executionGroupId; });
    if (epochGroup != epochRuntimeExecutionGroupIds.end()) {
      execution->switchScope = ConstraintSwitchScope{
          ConstraintSwitchScopeKind::ExecutionDomain,
          std::get<0>(epochGroup->first)};
      execution->stateSetId = std::get<1>(epochGroup->first);
    } else if (groupDescriptor.semanticComponentIds.size() == 1) {
      uint64_t semanticComponent = *groupDescriptor.semanticComponentIds.begin();
      if (auto active = activeStateSets.find(semanticComponent);
          active != activeStateSets.end()) {
        execution->switchScope = ConstraintSwitchScope{
            ConstraintSwitchScopeKind::SemanticComponent,
            semanticComponent};
        execution->stateSetId = active->second;
      }
    }
    execution->solver = std::make_unique<kinsol::KINSOLInstance>();
    execution->solver->setFunctionNormTolerance(1e-12);
    execution->solver->setScaledStepTolerance(1e-14);
    execution->solver->setLineSearchEnabled(true);
    execution->solver->setQuietErrors(true);

    for (const ConstraintVariableExecutionDescriptor &descriptor :
         activeVariables) {
      if (descriptor.executionGroupId == executionGroupId &&
          descriptor.owner == "local") {
        execution->globalVariables.push_back(descriptor.variable);
      }
    }
    for (const ConstraintEquationExecutionDescriptor &descriptor :
         activeEquations) {
      if (descriptor.executionGroupId == executionGroupId &&
          descriptor.owner == "local") {
        execution->globalEquations.push_back(descriptor.equation);
      } else if (descriptor.executionGroupId == executionGroupId &&
                 descriptor.owner == "outer") {
        execution->outerEquations.push_back(descriptor.equation);
      }
    }
    std::sort(execution->globalVariables.begin(),
              execution->globalVariables.end());
    std::sort(execution->globalEquations.begin(),
              execution->globalEquations.end());
    std::sort(execution->outerEquations.begin(),
              execution->outerEquations.end());
    execution->globalVariables.erase(
        std::unique(execution->globalVariables.begin(),
                    execution->globalVariables.end()),
        execution->globalVariables.end());
    execution->globalEquations.erase(
        std::unique(execution->globalEquations.begin(),
                    execution->globalEquations.end()),
        execution->globalEquations.end());
    execution->outerEquations.erase(
        std::unique(execution->outerEquations.begin(),
                    execution->outerEquations.end()),
        execution->outerEquations.end());

    uint64_t outerVariableScalars = 0;
    for (const ConstraintVariableExecutionDescriptor &descriptor :
         activeVariables) {
      if (descriptor.executionGroupId == executionGroupId &&
          descriptor.owner == "outer") {
        outerVariableScalars += getVariableFlatSize(descriptor.variable);
      }
    }
    uint64_t outerEquationScalars = 0;
    for (Equation equation : execution->outerEquations) {
      outerEquationScalars += getEquationFlatSize(equation);
    }
    if (outerVariableScalars != groupDescriptor.outerVariableScalars ||
        outerEquationScalars != groupDescriptor.outerEquationScalars) {
      std::cerr << "stage7c_invalid_execution_group group="
                << executionGroupId
                << " reason=runtime_outer_certificate_mismatch"
                << " expected_variables="
                << groupDescriptor.outerVariableScalars
                << " actual_variables=" << outerVariableScalars
                << " expected_equations="
                << groupDescriptor.outerEquationScalars
                << " actual_equations=" << outerEquationScalars << std::endl;
      return false;
    }

    // 中文：为 group-local unknowns/equations 建立连续局部坐标，并把原数组
    // getter/setter 与 residual/Jacobian callbacks 直接复用到 KINSOL。
    // English: Assign contiguous local coordinates to group-local unknowns and
    // equations, reusing original-array getters/setters and existing
    // residual/Jacobian callbacks in KINSOL.
    uint64_t localVariableScalars = 0;
    for (Variable globalVariable : execution->globalVariables) {
      execution->localVariableOffsets[globalVariable] = localVariableScalars;
      const VariableDimensions &dimensions =
          variablesDimensions[globalVariable];
      std::vector<uint64_t> shape(dimensions.begin(), dimensions.end());
      uint64_t flatSize = 1;
      for (uint64_t dimension : shape) {
        flatSize *= dimension;
      }
      localVariableScalars += flatSize;
      Variable localVariable = execution->solver->addVariable(
          shape.size(), shape.data(),
          algebraicAndStateVariablesGetters[globalVariable],
          algebraicAndStateVariablesSetters[globalVariable], nullptr);
      if (globalVariable < variableNominalGetters.size() &&
          variableNominalGetters[globalVariable] != nullptr &&
          !execution->solver->setVariableNominalGetter(
              localVariable, variableNominalGetters[globalVariable])) {
        std::cerr << "stage7e_invalid_nominal_registration execution_group="
                  << executionGroupId << " variable=" << globalVariable
                  << std::endl;
        return false;
      }
      execution->localVariableIds[globalVariable] = localVariable;
      for (auto indices = dimensions.indicesBegin(), end = dimensions.indicesEnd();
           indices != end; ++indices) {
        std::vector<uint64_t> scalarIndices(dimensions.rank());
        for (uint64_t dimension = 0; dimension < dimensions.rank();
             ++dimension) {
          scalarIndices[dimension] = (*indices)[dimension];
        }
        execution->localScalarColumns.emplace_back(globalVariable,
                                                    std::move(scalarIndices));
      }
    }

    uint64_t localEquationScalars = 0;
    for (Equation globalEquation : execution->globalEquations) {
      execution->localEquationOffsets[globalEquation] = localEquationScalars;
      const MultidimensionalRange &range = equationRanges[globalEquation];
      std::vector<int64_t> flattenedRanges;
      for (const Range &dimension : range) {
        flattenedRanges.push_back(dimension.begin);
        flattenedRanges.push_back(dimension.end);
      }
      uint64_t flatSize = 1;
      for (const Range &dimension : range) {
        flatSize *= dimension.end - dimension.begin;
      }
      localEquationScalars += flatSize;
      Equation localEquation = execution->solver->addEquation(
          flattenedRanges.data(), range.size(), nullptr);
      execution->localEquationIds[globalEquation] = localEquation;
      std::vector<int64_t> scalarEquationIndices;
      getEquationBeginIndices(globalEquation, scalarEquationIndices);
      do {
        execution->localScalarRows.emplace_back(globalEquation,
                                                scalarEquationIndices);
      } while (advanceEquationIndices(scalarEquationIndices, range));
      if (globalEquation >= residualFunctions.size() ||
          residualFunctions[globalEquation] == nullptr) {
        std::cerr << "stage6gb_component_requires_c3 component="
                  << executionGroupId
                  << " reason=missing_local_residual_callback"
                  << std::endl;
        return false;
      }
      execution->solver->setTimedResidualFunction(
          localEquation, residualFunctions[globalEquation]);

      if (globalEquation >= variableAccesses.size()) {
        std::cerr << "stage6gb_component_requires_c3 component="
                  << executionGroupId << " reason=missing_local_access_map"
                  << std::endl;
        return false;
      }
      for (const auto &[globalVariable, access] :
           variableAccesses[globalEquation]) {
        auto localVariableIt =
            execution->localVariableIds.find(globalVariable);
        if (localVariableIt == execution->localVariableIds.end()) {
          continue;
        }
        Variable localVariable = localVariableIt->second;
        execution->solver->addVariableAccess(localEquation, localVariable,
                                             access);
        if (globalEquation >= jacobianFunctions.size() ||
            globalVariable >= jacobianFunctions[globalEquation].size() ||
            jacobianFunctions[globalEquation][globalVariable].first ==
                nullptr) {
          std::cerr << "stage6gb_component_requires_c3 component="
                    << executionGroupId
                    << " reason=missing_local_jacobian_callback equation="
                    << globalEquation << " variable=" << globalVariable
                    << std::endl;
          return false;
        }
        const JacobianFunctionDescriptor &jacobian =
            jacobianFunctions[globalEquation][globalVariable];
        execution->solver->addTimedJacobianFunction(
            localEquation, localVariable, jacobian.first,
            jacobian.second.size(),
            const_cast<uint64_t *>(jacobian.second.data()));
      }
    }

    if (localVariableScalars != localEquationScalars) {
      std::cerr << "stage6gb_component_requires_c3 group="
                << executionGroupId
                << " reason=runtime_local_partition_not_square variables="
                << localVariableScalars
                << " equations=" << localEquationScalars << std::endl;
      return false;
    }
    if (localVariableScalars != groupDescriptor.localVariableScalars ||
        localEquationScalars != groupDescriptor.localEquationScalars) {
      std::cerr << "stage7c_invalid_execution_group group="
                << executionGroupId
                << " reason=runtime_local_certificate_mismatch"
                << " expected_variables="
                << groupDescriptor.localVariableScalars
                << " actual_variables=" << localVariableScalars
                << " expected_equations="
                << groupDescriptor.localEquationScalars
                << " actual_equations=" << localEquationScalars << std::endl;
      return false;
    }

    execution->localScalars = localVariableScalars;
    execution->solver->setMaximumNewtonStep(std::max<realtype>(
        100, 10 * std::sqrt(static_cast<realtype>(localVariableScalars))));

    for (uint64_t i = 0; i < execution->localScalarColumns.size(); ++i) {
      execution->localColumnIndices.emplace(
          execution->localScalarColumns[i], i);
    }
    std::vector<uint64_t> parents(2 * execution->localScalars);
    for (uint64_t i = 0; i < parents.size(); ++i) {
      parents[i] = i;
    }
    std::function<uint64_t(uint64_t)> findRoot = [&](uint64_t node) {
      if (parents[node] != node) {
        parents[node] = findRoot(parents[node]);
      }
      return parents[node];
    };
    auto unite = [&](uint64_t first, uint64_t second) {
      first = findRoot(first);
      second = findRoot(second);
      if (first != second) {
        parents[second] = first;
      }
    };

    for (uint64_t row = 0; row < execution->localScalarRows.size(); ++row) {
      const auto &[equation, equationIndices] =
          execution->localScalarRows[row];
      bool hasLocalAccess = false;
      for (const JacobianColumn &column : computeAccessedJacobianColumns(
               equation, equationIndices.data(), true)) {
        auto localColumn = execution->localColumnIndices.find(column);
        if (localColumn == execution->localColumnIndices.end()) {
          continue;
        }
        hasLocalAccess = true;
        unite(row, execution->localScalars + localColumn->second);
      }
      if (!hasLocalAccess) {
        std::cerr << "stage6gb_component_requires_c3 component="
                  << executionGroupId
                  << " reason=local_row_without_local_jacobian_access"
                  << std::endl;
        return false;
      }
    }

    std::map<uint64_t, uint64_t> compactComponents;
    auto getCompactComponent = [&](uint64_t node) {
      uint64_t root = findRoot(node);
      auto [it, inserted] = compactComponents.try_emplace(
          root, static_cast<uint64_t>(compactComponents.size()));
      return it->second;
    };
    execution->localRowComponents.resize(execution->localScalars);
    std::map<uint64_t, uint64_t> rowsPerComponent;
    for (uint64_t row = 0; row < execution->localScalars; ++row) {
      execution->localRowComponents[row] = getCompactComponent(row);
      ++rowsPerComponent[execution->localRowComponents[row]];
    }
    std::map<uint64_t, uint64_t> columnsPerComponent;
    for (uint64_t column = 0; column < execution->localScalars; ++column) {
      uint64_t component =
          getCompactComponent(execution->localScalars + column);
      execution->localColumnComponents.emplace(
          execution->localScalarColumns[column], component);
      ++columnsPerComponent[component];
    }
    for (const auto &[localComponent, columns] : columnsPerComponent) {
      auto rows = rowsPerComponent.find(localComponent);
      if (rows == rowsPerComponent.end()) {
        std::cerr << "stage6gb_component_requires_c3 component="
                  << executionGroupId
                  << " reason=local_column_without_local_jacobian_access"
                  << std::endl;
        return false;
      }
      if (rows->second != columns) {
        std::cerr << "stage6gb_component_requires_c3 component="
                  << executionGroupId
                  << " reason=non_square_local_jacobian_component"
                  << " local_component=" << localComponent
                  << " equations=" << rows->second
                  << " variables=" << columns << std::endl;
        return false;
      }
    }

    // 中文：局部二部图连通分量界定 Schur fill；outer coupling 只在同一局部
    // 连通块上传播，避免把整个 component 扩成 dense pattern。
    // English: Local bipartite connected components bound Schur fill. Outer
    // coupling propagates only within the same local component, avoiding a
    // dense whole-component pattern.
    std::set<JacobianColumn> outerCouplingColumns;
    for (uint64_t row = 0; row < execution->localScalarRows.size(); ++row) {
      const auto &[equation, equationIndices] =
          execution->localScalarRows[row];
      uint64_t localComponent = execution->localRowComponents[row];
      std::set<JacobianColumn> componentColumns(
          execution->componentOuterCouplingColumns[localComponent].begin(),
          execution->componentOuterCouplingColumns[localComponent].end());
      for (const JacobianColumn &column : computeAccessedJacobianColumns(
               equation, equationIndices.data(), true)) {
        if (execution->localColumnIndices.find(column) !=
            execution->localColumnIndices.end()) {
          continue;
        }
        componentColumns.insert(column);
        outerCouplingColumns.insert(column);
      }
      execution->componentOuterCouplingColumns[localComponent].assign(
          componentColumns.begin(), componentColumns.end());
    }
    execution->outerCouplingColumns.assign(outerCouplingColumns.begin(),
                                           outerCouplingColumns.end());

    auto allocateCouplingSeeds = [&](Equation equation, Variable variable) {
      if (equation >= jacobianFunctions.size() ||
          variable >= jacobianFunctions[equation].size()) {
        return false;
      }
      const JacobianFunctionDescriptor &descriptor =
          jacobianFunctions[equation][variable];
      if (descriptor.first == nullptr) {
        return false;
      }
      if (execution->couplingSeeds.find(descriptor.first) !=
          execution->couplingSeeds.end()) {
        return true;
      }
      std::vector<uint64_t> seedIds;
      MemoryPool &memoryPool =
          MemoryPoolManager::getInstance().get(memoryPoolId);
      for (uint64_t seedSize : descriptor.second) {
        seedIds.push_back(memoryPool.create(seedSize));
      }
      execution->couplingSeeds.emplace(descriptor.first, std::move(seedIds));
      return true;
    };

    for (Equation equation : execution->globalEquations) {
      for (const JacobianColumn &column : execution->outerCouplingColumns) {
        allocateCouplingSeeds(equation, column.first);
      }
    }
    for (Equation equation : execution->outerEquations) {
      for (Variable variable : execution->globalVariables) {
        allocateCouplingSeeds(equation, variable);
      }
    }

    struct ConditionEquationDescriptor {
      Equation equation;
      std::string role;
      std::string matchedVariableName;
    };
    std::vector<ConditionEquationDescriptor> conditionEquations;
    if (epochGroup != epochRuntimeExecutionGroupIds.end()) {
      uint64_t domainId = std::get<0>(epochGroup->first);
      uint64_t epochId = std::get<1>(epochGroup->first);
      uint64_t sourceGroupId = std::get<2>(epochGroup->first);
      for (const ConstraintExecutionEpochEquationDescriptor &descriptor :
           constraintExecutionEpochEquations) {
        if (descriptor.domainId == domainId &&
            descriptor.epochId == epochId &&
            descriptor.executionGroupId == sourceGroupId &&
            (descriptor.role == "position" ||
             descriptor.role == "tangent")) {
          conditionEquations.push_back(
              {descriptor.equation, descriptor.role,
               descriptor.matchedVariableName});
        }
      }
    } else if (constraintStateSets.empty()) {
      for (const ConstraintEquationDescriptor &descriptor :
           constraintEquations) {
        if (execution->semanticComponentIds.find(descriptor.componentId) !=
                execution->semanticComponentIds.end() &&
            (descriptor.role == "position" ||
             descriptor.role == "tangent")) {
          conditionEquations.push_back(
              {descriptor.equation, descriptor.role,
               descriptor.matchedVariableName});
        }
      }
    } else {
      for (const ConstraintStateSetEquationDescriptor &descriptor :
           constraintStateSetEquations) {
        auto active = activeStateSets.find(descriptor.componentId);
        if (execution->semanticComponentIds.find(descriptor.componentId) !=
                execution->semanticComponentIds.end() &&
            active != activeStateSets.end() &&
            descriptor.stateSetId == active->second &&
            (descriptor.role == "position" ||
             descriptor.role == "tangent")) {
          conditionEquations.push_back(
              {descriptor.equation, descriptor.role,
               descriptor.matchedVariableName});
        }
      }
    }

    for (const ConditionEquationDescriptor &equationDescriptor :
         conditionEquations) {
      if (equationDescriptor.role != "position" &&
          equationDescriptor.role != "tangent") {
        continue;
      }
      auto equationOffsetIt = execution->localEquationOffsets.find(
          equationDescriptor.equation);
      if (equationOffsetIt == execution->localEquationOffsets.end()) {
        std::cerr << "stage6gb_component_requires_c3 group="
                  << executionGroupId
                  << " reason=condition_equation_not_local" << std::endl;
        return false;
      }
      uint64_t equationSize = 1;
      for (const Range &range : equationRanges[equationDescriptor.equation]) {
        equationSize *= range.end - range.begin;
      }

      std::optional<Variable> matchedVariable;
      if (epochGroup != epochRuntimeExecutionGroupIds.end()) {
        uint64_t domainId = std::get<0>(epochGroup->first);
        uint64_t epochId = std::get<1>(epochGroup->first);
        for (const ConstraintExecutionEpochVariableDescriptor
                 &variableDescriptor : constraintExecutionEpochVariables) {
          if (variableDescriptor.domainId == domainId &&
              variableDescriptor.epochId == epochId &&
              variableDescriptor.name ==
                  equationDescriptor.matchedVariableName &&
              execution->localVariableIds.find(variableDescriptor.variable) !=
                  execution->localVariableIds.end()) {
            matchedVariable = variableDescriptor.variable;
            break;
          }
        }
      } else if (constraintStateSets.empty()) {
        for (const ConstraintVariableDescriptor &variableDescriptor :
             constraintVariables) {
          if (execution->semanticComponentIds.find(
                  variableDescriptor.componentId) !=
                  execution->semanticComponentIds.end() &&
              variableDescriptor.name ==
                  equationDescriptor.matchedVariableName &&
              execution->localVariableIds.find(variableDescriptor.variable) !=
                  execution->localVariableIds.end()) {
            matchedVariable = variableDescriptor.variable;
            break;
          }
        }
      } else {
        for (const ConstraintStateSetVariableDescriptor &variableDescriptor :
             constraintStateSetVariables) {
          auto active = activeStateSets.find(variableDescriptor.componentId);
          if (execution->semanticComponentIds.find(
                  variableDescriptor.componentId) !=
                  execution->semanticComponentIds.end() &&
              active != activeStateSets.end() &&
              variableDescriptor.stateSetId == active->second &&
              variableDescriptor.name ==
                  equationDescriptor.matchedVariableName &&
              execution->localVariableIds.find(variableDescriptor.variable) !=
                  execution->localVariableIds.end()) {
            matchedVariable = variableDescriptor.variable;
            break;
          }
        }
      }
      if (!matchedVariable) {
        std::cerr << "stage6gb_component_requires_c3 group="
                  << executionGroupId
                  << " reason=condition_matched_variable_not_local"
                  << std::endl;
        return false;
      }
      uint64_t variableSize = 1;
      for (uint64_t dimension : variablesDimensions[*matchedVariable]) {
        variableSize *= dimension;
      }
      if (equationSize != variableSize) {
        std::cerr << "stage6gb_component_requires_c3 group="
                  << executionGroupId
                  << " reason=condition_range_size_mismatch equations="
                  << equationSize << " variables=" << variableSize
                  << std::endl;
        return false;
      }
      uint64_t equationOffset = equationOffsetIt->second;
      uint64_t variableOffset =
          execution->localVariableOffsets[*matchedVariable];
      for (uint64_t i = 0; i < equationSize; ++i) {
        execution->conditionRows.push_back(equationOffset + i);
        execution->conditionColumns.push_back(variableOffset + i);
      }
    }
    if (execution->conditionRows.empty() ||
        execution->conditionRows.size() != execution->conditionColumns.size()) {
      std::cerr << "stage6gb_component_requires_c3 group="
                << executionGroupId
                << " reason=condition_block_not_square" << std::endl;
      return false;
    }

    if (getStage6GRuntimeProbeMode() != Stage6GRuntimeProbeMode::Off) {
      std::cerr << "[stage6g-runtime-probe] execution_owner=local"
                << " execution_group=" << executionGroupId
                << " local_connected_blocks=" << compactComponents.size()
                << " outer_coupling_columns="
                << execution->outerCouplingColumns.size()
                << " scaling_source=initialization_anchor" << std::endl;
    }

    constraintComponentExecutions.push_back(std::move(execution));
  }

  std::map<uint64_t, ConstraintComponentExecutionRuntime *> executionsById;
  std::map<JacobianColumn,
           std::pair<ConstraintComponentExecutionRuntime *, uint64_t>>
      localColumnOwners;
  for (const auto &execution : constraintComponentExecutions) {
    executionsById.emplace(execution->executionGroupId, execution.get());
    for (const auto &[column, localIndex] : execution->localColumnIndices) {
      if (!localColumnOwners
               .emplace(column, std::make_pair(execution.get(), localIndex))
               .second) {
        std::cerr << "stage7c_invalid_execution_group group="
                  << execution->executionGroupId
                  << " reason=duplicate_cross_group_local_variable_owner"
                  << std::endl;
        return false;
      }
    }
  }

  for (const auto &executionPtr : constraintComponentExecutions) {
    ConstraintComponentExecutionRuntime &execution = *executionPtr;
    std::set<JacobianColumn> actualOuterColumns;
    for (const JacobianColumn &column : execution.outerCouplingColumns) {
      auto localOwner = localColumnOwners.find(column);
      if (localOwner == localColumnOwners.end()) {
        actualOuterColumns.insert(column);
        continue;
      }
      uint64_t dependencyId =
          localOwner->second.first->executionGroupId;
      if (std::find(execution.dependencies.begin(),
                    execution.dependencies.end(),
                    dependencyId) == execution.dependencies.end()) {
        std::cerr << "stage7c_invalid_execution_group group="
                  << execution.executionGroupId
                  << " dependency=" << dependencyId
                  << " reason=unregistered_cross_group_local_access"
                  << std::endl;
        return false;
      }
    }
    for (uint64_t dependencyId : execution.dependencies) {
      auto dependency = executionsById.find(dependencyId);
      if (dependency == executionsById.end()) {
        std::cerr << "stage7c_invalid_execution_group group="
                  << execution.executionGroupId
                  << " dependency=" << dependencyId
                  << " reason=dependency_not_executable_constraint_group"
                  << std::endl;
        return false;
      }
      actualOuterColumns.insert(
          dependency->second->outerCouplingColumns.begin(),
          dependency->second->outerCouplingColumns.end());
    }
    execution.outerCouplingColumns.assign(actualOuterColumns.begin(),
                                          actualOuterColumns.end());

    for (auto &[localComponent, columns] :
         execution.componentOuterCouplingColumns) {
      (void)localComponent;
      std::set<JacobianColumn> propagated;
      for (const JacobianColumn &column : columns) {
        auto localOwner = localColumnOwners.find(column);
        if (localOwner == localColumnOwners.end()) {
          propagated.insert(column);
        } else {
          propagated.insert(
              localOwner->second.first->outerCouplingColumns.begin(),
              localOwner->second.first->outerCouplingColumns.end());
        }
      }
      columns.assign(propagated.begin(), propagated.end());
    }
  }

  std::set<Equation> allOuterEquations;
  for (const auto &execution : constraintComponentExecutions) {
    allOuterEquations.insert(execution->outerEquations.begin(),
                             execution->outerEquations.end());
  }
  auto allocateCouplingSeeds =
      [&](ConstraintComponentExecutionRuntime &execution, Equation equation,
          Variable variable) {
        if (equation >= jacobianFunctions.size() ||
            variable >= jacobianFunctions[equation].size()) {
          return false;
        }
        const JacobianFunctionDescriptor &descriptor =
            jacobianFunctions[equation][variable];
        if (descriptor.first == nullptr) {
          return false;
        }
        if (execution.couplingSeeds.find(descriptor.first) !=
            execution.couplingSeeds.end()) {
          return true;
        }
        std::vector<uint64_t> seedIds;
        MemoryPool &memoryPool =
            MemoryPoolManager::getInstance().get(memoryPoolId);
        for (uint64_t seedSize : descriptor.second) {
          seedIds.push_back(memoryPool.create(seedSize));
        }
        execution.couplingSeeds.emplace(descriptor.first, std::move(seedIds));
        return true;
      };

  for (const auto &executionPtr : constraintComponentExecutions) {
    ConstraintComponentExecutionRuntime &execution = *executionPtr;
    std::set<Equation> ownedOuter(execution.outerEquations.begin(),
                                  execution.outerEquations.end());
    for (Equation equation : allOuterEquations) {
      bool accessesLocal = false;
      std::set<Variable> accessedLocalVariables;
      std::vector<int64_t> equationIndices;
      getEquationBeginIndices(equation, equationIndices);
      do {
        for (const JacobianColumn &column :
             computeAccessedJacobianColumns(equation,
                                            equationIndices.data(), true)) {
          if (execution.localColumnIndices.find(column) !=
              execution.localColumnIndices.end()) {
            accessesLocal = true;
            accessedLocalVariables.insert(column.first);
          }
        }
      } while (advanceEquationIndices(equationIndices,
                                      equationRanges[equation]));
      if (!accessesLocal) {
        continue;
      }
      ownedOuter.insert(equation);
      // 中文：R_z 只需当前 outer residual 实际访问的 local blocks。若要求
      // 每个 local block 都有 callback，会把稀疏 group 变成 dense contract，
      // 并错误拒绝合法的 mixed helper/IDA ownership。
      // English: R_z needs callbacks only for local blocks actually accessed
      // by this outer residual. Requiring every local block would make a sparse
      // group a dense callback contract and reject valid mixed ownership.
      for (Variable variable : accessedLocalVariables) {
        if (!allocateCouplingSeeds(execution, equation, variable)) {
          std::cerr << "stage7c_invalid_execution_group group="
                    << execution.executionGroupId
                    << " reason=missing_cross_group_outer_jacobian_callback"
                    << " equation=" << equation << " variable=" << variable
                    << std::endl;
          return false;
        }
      }
    }
    execution.outerEquations.assign(ownedOuter.begin(), ownedOuter.end());
  }

  // 中文：完成所有 owner 与 dependency 验证后才重建 outer flattening offsets；
  // local handles 的 flat size 为零，不占用 IDA y/y' 或 residual 行。
  // English: Rebuild outer flattening offsets only after all owner/dependency
  // checks pass. Local handles contribute zero size to IDA y/y' and residual
  // rows.
  variableOffsets.clear();
  variableOffsets.push_back(0);
  for (Variable variable = 0; variable < getNumOfArrayVariables(); ++variable) {
    uint64_t flatSize = 1;
    for (uint64_t dimension : variablesDimensions[variable]) {
      flatSize *= dimension;
    }
    variableOffsets.push_back(variableOffsets.back() +
                              (isLocalVariable(variable) ? 0 : flatSize));
  }

  equationOffsets.clear();
  equationOffsets.push_back(0);
  for (Equation equation = 0; equation < getNumOfVectorizedEquations();
       ++equation) {
    uint64_t flatSize = 1;
    for (const Range &range : equationRanges[equation]) {
      flatSize *= range.end - range.begin;
    }
    equationOffsets.push_back(equationOffsets.back() +
                              (isLocalEquation(equation) ? 0 : flatSize));
  }

  return true;
}

bool IDAInstance::evaluateConstraintJacobian(
    ConstraintComponentExecutionRuntime &execution, Equation equation,
    const std::vector<int64_t> &equationIndices,
    const JacobianColumn &column, realtype time, realtype alpha,
    double &result) {
  if (equation >= jacobianFunctions.size() ||
      column.first >= jacobianFunctions[equation].size()) {
    return false;
  }
  const JacobianFunctionDescriptor &descriptor =
      jacobianFunctions[equation][column.first];
  if (descriptor.first == nullptr) {
    return false;
  }
  auto seedsIt = execution.couplingSeeds.find(descriptor.first);
  if (seedsIt == execution.couplingSeeds.end()) {
    return false;
  }
  result = descriptor.first(time, equationIndices.data(), column.second.data(),
                            alpha, memoryPoolId, seedsIt->second.data());
  return std::isfinite(result);
}

int IDAInstance::solveConstraintComponents(realtype time) {
  // 中文：每次 IDA residual/Jacobian 试探先按 group DAG 刷新局部闭包。普通
  // Newton 失败可恢复；condition/rank 触发只登记 pending switch，真正事务在
  // 最近 accepted checkpoint 上执行。
  // English: Every IDA residual/Jacobian trial refreshes local closures in
  // group-DAG order. Ordinary Newton failure may be recoverable; condition/rank
  // triggers only enqueue a switch, whose transaction runs at the latest
  // accepted checkpoint.
  if (pendingBasisSwitch && !evaluatingBasisCandidate) {
    return -1;
  }
  for (const auto &execution : constraintComponentExecutions) {
    std::vector<double> warmStart;
    warmStart.reserve(execution->localScalarColumns.size());
    for (const JacobianColumn &column : execution->localScalarColumns) {
      warmStart.push_back(algebraicAndStateVariablesGetters[column.first](
          column.second.data()));
    }

    if (!execution->scalingConfigured) {
      if (!execution->solver->configureInitializationAnchorScaling(time)) {
        std::cerr << "stage6gb_component_requires_c3 execution_group="
                  << execution->executionGroupId
                  << " reason=initial_scaling_jacobian_unavailable"
                  << " time=" << time << std::endl;
        return -1;
      }
      execution->scalingConfigured = true;
    }

    auto requestBasisSwitch = [&](const char *phase,
                                  const ConstraintConditionEstimate &estimate,
                                  const char *reason) {
      if (execution->switchScope &&
          !isConstraintBasisSwitchDisabled() &&
          !evaluatingBasisCandidate) {
        pendingBasisSwitch = PendingBasisSwitch{
            *execution->switchScope, execution->stateSetId, time,
            estimate.oneNormCondition, phase};
        if (isStage6GRuntimeProbeEnabled()) {
          std::cerr << std::setprecision(17)
                    << "[stage6g-runtime-probe] basis_epoch=" << basisEpoch
                    << " event=switch_requested switch_scope="
                    << (execution->switchScope->kind ==
                                ConstraintSwitchScopeKind::ExecutionDomain
                            ? "execution_domain"
                            : "semantic_component")
                    << " switch_id=" << execution->switchScope->id
                    << " execution_group=" << execution->executionGroupId
                    << " state_set=" << execution->stateSetId
                    << " trial_time=" << time
                    << " condition=" << estimate.oneNormCondition
                    << " phase=" << phase << " reason=" << reason
                    << std::endl;
        }
        return;
      }
      if (!evaluatingBasisCandidate) {
        expectedConstraintFailure = true;
        std::cerr << std::setprecision(17)
                  << "dynamic_state_selection_required execution_group="
                  << execution->executionGroupId << " time=" << time
                  << " rank=" << estimate.rank
                  << " minimum_pivot=" << estimate.minimumPivot
                  << " maximum_pivot=" << estimate.maximumPivot
                  << " condition=" << estimate.oneNormCondition
                  << " phase=" << phase << " reason=" << reason
                  << std::endl;
      }
    };

    // 中文：pre/post/failed-trial 三个时点共享同一缩放证书检查。先 factorize
    // 当前 sparse J_zz，再分别估计 chart 子块和完整 local closure。
    // English: Pre-solve, post-solve, and failed-trial checks share one scaled
    // certificate path. Factorize the current sparse J_zz first, then estimate
    // both the chart sub-block and complete local closure.
    auto checkConditionCertificate = [&](const char *phase,
                                         bool failedLocalTrial) {
      if (!execution->solver->factorizeCurrentJacobian(time)) {
        ConstraintConditionEstimate estimate;
        estimate.rankDeficient = true;
        estimate.oneNormCondition = std::numeric_limits<double>::infinity();
        requestBasisSwitch(phase, estimate,
                           "local_jacobian_factorization_failed");
        return false;
      }
      uint64_t conditionSize = execution->conditionRows.size();
      kinsol::SparseNonlinearSystem::ConditionEstimate sparseEstimate;
      if (!execution->solver->estimateScaledJacobianCondition(
              execution->conditionRows, execution->conditionColumns,
              sparseEstimate)) {
        std::cerr << "stage6gb_component_requires_c3 execution_group="
                  << execution->executionGroupId
                  << " reason=scaled_condition_estimate_failed" << std::endl;
        return false;
      }
      ConstraintConditionEstimate estimate;
      estimate.rank = sparseEstimate.rank;
      estimate.minimumPivot = sparseEstimate.minimumPivot;
      estimate.maximumPivot = sparseEstimate.maximumPivot;
      estimate.reciprocalCondition = sparseEstimate.reciprocalCondition;
      estimate.oneNormCondition = sparseEstimate.oneNormCondition;
      estimate.rankDeficient = sparseEstimate.rankDeficient;
      estimate.scaled = true;
      estimate.sparseEstimator = sparseEstimate.sparseEstimator;
      std::vector<uint64_t> localIndices(execution->localScalars);
      std::iota(localIndices.begin(), localIndices.end(), 0);
      kinsol::SparseNonlinearSystem::ConditionEstimate localSparseEstimate;
      if (!execution->solver->estimateScaledJacobianCondition(
              localIndices, localIndices, localSparseEstimate)) {
        std::cerr << "stage6gb_component_requires_c3 execution_group="
                  << execution->executionGroupId
                  << " reason=scaled_local_closure_condition_estimate_failed"
                  << std::endl;
        return false;
      }
      ConstraintConditionEstimate localEstimate;
      localEstimate.rank = localSparseEstimate.rank;
      localEstimate.minimumPivot = localSparseEstimate.minimumPivot;
      localEstimate.maximumPivot = localSparseEstimate.maximumPivot;
      localEstimate.reciprocalCondition =
          localSparseEstimate.reciprocalCondition;
      localEstimate.oneNormCondition = localSparseEstimate.oneNormCondition;
      localEstimate.rankDeficient = localSparseEstimate.rankDeficient;
      localEstimate.scaled = true;
      localEstimate.sparseEstimator = localSparseEstimate.sparseEstimator;

      execution->lastHealthCertificate.chart = estimate;
      execution->lastHealthCertificate.localClosure = localEstimate;
      execution->maximumChartCondition = std::max(
          execution->maximumChartCondition, estimate.oneNormCondition);
      execution->maximumLocalClosureCondition =
          std::max(execution->maximumLocalClosureCondition,
                   localEstimate.oneNormCondition);
      double maximumCondition =
          std::max(estimate.oneNormCondition, localEstimate.oneNormCondition);
      if (std::isfinite(maximumCondition) &&
          maximumCondition >= execution->maximumCondition) {
        execution->maximumCondition = maximumCondition;
        execution->maximumConditionTime = time;
      }
      if (getStage6GRuntimeProbeMode() == Stage6GRuntimeProbeMode::Trace) {
        std::cerr << std::setprecision(17)
                  << "[stage6g-runtime-probe] sample=condition_" << phase
                  << " execution_group=" << execution->executionGroupId
                  << " time=" << time << " rank=" << estimate.rank
                  << " size=" << conditionSize
                  << " minimum_pivot=" << estimate.minimumPivot
                  << " maximum_pivot=" << estimate.maximumPivot
                  << " condition=" << estimate.oneNormCondition
                  << " reciprocal_condition="
                  << estimate.reciprocalCondition
                  << " scaled=1 estimator="
                  << (estimate.sparseEstimator ? "sparse_klu" : "dense_lu")
                  << " local_rank=" << localEstimate.rank
                  << " local_size=" << execution->localScalars
                  << " local_condition="
                  << localEstimate.oneNormCondition
                  << " local_estimator="
                  << (localEstimate.sparseEstimator ? "sparse_klu"
                                                    : "dense_lu")
                  << " response_amplification="
                  << (execution->lastHealthCertificate.responseAvailable
                          ? execution->lastHealthCertificate
                                .responseAmplification
                          : 0)
                  << " response_available="
                  << (execution->lastHealthCertificate.responseAvailable ? 1
                                                                         : 0)
                  << std::endl;
      }
      ConstraintBasisHealthAction healthAction = classifyConstraintBasisHealth(
          execution->lastHealthCertificate, failedLocalTrial);
      bool proactiveStateSetSwitch =
          healthAction == ConstraintBasisHealthAction::RequestSwitch &&
          execution->switchScope && !evaluatingBasisCandidate;
      if (proactiveStateSetSwitch ||
          healthAction == ConstraintBasisHealthAction::Unsafe) {
        const ConstraintConditionEstimate &worstEstimate =
            localEstimate.oneNormCondition > estimate.oneNormCondition
                ? localEstimate
                : estimate;
        requestBasisSwitch(
            phase, worstEstimate,
            healthAction == ConstraintBasisHealthAction::Unsafe
                ? "unsafe_basis_health_certificate"
                : proactiveStateSetSwitch
                      ? "basis_health_hysteresis"
                      : "local_closure_failed_in_unsafe_basis");
        return false;
      }
      return true;
    };

    // 中文：KINSOL Newton 前检查 accepted warm-start chart；失败后在最终
    // trial iterate 仍可见时再检查一次，随后才恢复旧 local solution。
    // English: Check the accepted warm-start chart before KINSOL Newton. A
    // failed trial is checked again while its final iterate remains visible,
    // before restoring the previous local solution.
    if (!checkConditionCertificate("pre_solve", false)) {
      return -1;
    }

    ++execution->localSolveAttempts;
    kinsol::KINSOLInstance::SolveStatus status =
        execution->solver->solveWithStatus(time);
    execution->localNonlinearIterations +=
        execution->solver->getLastNonlinearIterations();
    if (status != kinsol::KINSOLInstance::SolveStatus::Success) {
      if (status == kinsol::KINSOLInstance::SolveStatus::RecoverableFailure) {
        ++execution->localSolveRecoverableFailures;
      } else {
        ++execution->localSolveFatalFailures;
      }
      if (!checkConditionCertificate("failed_trial", true)) {
        return -1;
      }
      for (uint64_t i = 0; i < execution->localScalarColumns.size(); ++i) {
        const JacobianColumn &column = execution->localScalarColumns[i];
        algebraicAndStateVariablesSetters[column.first](
            warmStart[i], column.second.data());
      }
      if (status == kinsol::KINSOLInstance::SolveStatus::RecoverableFailure) {
        return 1;
      }
      std::cerr << "stage6gb_local_solve_failed execution_group="
                << execution->executionGroupId << " time=" << time
                << " kind=fatal" << std::endl;
      return -1;
    }
    if (!checkConditionCertificate("post_solve", false)) {
      return -1;
    }
    ++execution->localSolveSuccesses;

    if (isStage6GRuntimeProbeEnabled()) {
      double maximumLocalResidual = 0;
      for (const auto &[equation, equationIndices] :
           execution->localScalarRows) {
        double value = residualFunctions[equation](time,
                                                    equationIndices.data());
        if (!std::isfinite(value)) {
          std::cerr << "stage6gb_local_solve_failed execution_group="
                    << execution->executionGroupId << " time=" << time
                    << " kind=non_finite_residual" << std::endl;
          return -1;
        }
        maximumLocalResidual =
            std::max(maximumLocalResidual, std::abs(value));
      }
      if (maximumLocalResidual >= execution->maximumLocalResidual) {
        execution->maximumLocalResidual = maximumLocalResidual;
        execution->maximumLocalResidualTime = time;
      }
      if (getStage6GRuntimeProbeMode() == Stage6GRuntimeProbeMode::Trace) {
        std::cerr << std::setprecision(17)
                  << "[stage6g-runtime-probe] sample=local_closure "
                     "execution_group="
                  << execution->executionGroupId << " time=" << time
                  << " raw_max=" << maximumLocalResidual
                  << " solver=kinsol_scaled_klu" << std::endl;
      }
    }
  }
  return 0;
}
bool IDAInstance::refreshConstraintComponents(realtype time) {
  return solveConstraintComponents(time) == 0;
}

void IDAInstance::saveAcceptedConstraintCheckpoint(realtype time) {
  // 中文：checkpoint 同时保存 outer y/y'、原模型数组和 local warm start，保证
  // trial failure 后可恢复完整求解器可见状态。
  // English: A checkpoint stores outer y/y', original model arrays, and local
  // warm starts so a failed trial can restore the complete solver-visible state.
  if ((constraintStateSets.empty() && constraintExecutionEpochs.empty()) ||
      !variablesVector || !derivativesVector) {
    return;
  }
  acceptedConstraintCheckpoint.valid = true;
  acceptedConstraintCheckpoint.time = time;
  acceptedConstraintCheckpoint.variables.assign(
      N_VGetArrayPointer(variablesVector),
      N_VGetArrayPointer(variablesVector) + scalarVariablesNumber);
  acceptedConstraintCheckpoint.derivatives.assign(
      N_VGetArrayPointer(derivativesVector),
      N_VGetArrayPointer(derivativesVector) + scalarVariablesNumber);
  acceptedConstraintCheckpoint.modelVariables.clear();
  acceptedConstraintCheckpoint.modelVariables.resize(
      getNumOfArrayVariables());
  for (Variable variable = 0; variable < getNumOfArrayVariables(); ++variable) {
    std::vector<uint64_t> indices;
    getVariableBeginIndices(variable, indices);
    do {
      acceptedConstraintCheckpoint.modelVariables[variable].push_back(
          algebraicAndStateVariablesGetters[variable](indices.data()));
    } while (advanceVariableIndices(indices, variablesDimensions[variable]));
  }
  realtype stepSize = 0;
  int order = 1;
  if (idaMemory != nullptr &&
      IDAGetCurrentStep(idaMemory, &stepSize) == IDA_SUCCESS) {
    acceptedConstraintCheckpoint.stepSize = stepSize;
  }
  if (idaMemory != nullptr &&
      IDAGetCurrentOrder(idaMemory, &order) == IDA_SUCCESS) {
    acceptedConstraintCheckpoint.order = order;
  }
}

bool IDAInstance::restoreAcceptedConstraintCheckpoint() {
  if (!acceptedConstraintCheckpoint.valid ||
      acceptedConstraintCheckpoint.variables.size() != scalarVariablesNumber ||
      acceptedConstraintCheckpoint.derivatives.size() !=
          scalarVariablesNumber ||
      acceptedConstraintCheckpoint.modelVariables.size() !=
          getNumOfArrayVariables()) {
    return false;
  }
  std::copy(acceptedConstraintCheckpoint.variables.begin(),
            acceptedConstraintCheckpoint.variables.end(),
            N_VGetArrayPointer(variablesVector));
  std::copy(acceptedConstraintCheckpoint.derivatives.begin(),
            acceptedConstraintCheckpoint.derivatives.end(),
            N_VGetArrayPointer(derivativesVector));
  for (Variable variable = 0; variable < getNumOfArrayVariables(); ++variable) {
    std::vector<uint64_t> indices;
    getVariableBeginIndices(variable, indices);
    uint64_t scalar = 0;
    do {
      if (scalar >=
          acceptedConstraintCheckpoint.modelVariables[variable].size()) {
        return false;
      }
      algebraicAndStateVariablesSetters[variable](
          acceptedConstraintCheckpoint.modelVariables[variable][scalar++],
          indices.data());
    } while (advanceVariableIndices(indices, variablesDimensions[variable]));
  }
  currentTime = acceptedConstraintCheckpoint.time;
  return true;
}

bool IDAInstance::verifyAcceptedConstraintCheckpointRestored(
    ConstraintSwitchScope scope, uint64_t expectedStateSet,
    double &maximumDifference) {
  maximumDifference = 0;
  if (!acceptedConstraintCheckpoint.valid ||
      acceptedConstraintCheckpoint.variables.size() != scalarVariablesNumber ||
      acceptedConstraintCheckpoint.derivatives.size() !=
          scalarVariablesNumber ||
      acceptedConstraintCheckpoint.modelVariables.size() !=
          getNumOfArrayVariables()) {
    return false;
  }

  auto active = activeConstraintStateSets.find(scope.id);
  if (active == activeConstraintStateSets.end() ||
      active->second != expectedStateSet) {
    return false;
  }

  const realtype *variables = N_VGetArrayPointer(variablesVector);
  const realtype *derivatives = N_VGetArrayPointer(derivativesVector);
  for (uint64_t scalar = 0; scalar < scalarVariablesNumber; ++scalar) {
    if (!std::isfinite(variables[scalar]) ||
        !std::isfinite(derivatives[scalar])) {
      return false;
    }
    maximumDifference = std::max(
        maximumDifference,
        std::abs(variables[scalar] -
                 acceptedConstraintCheckpoint.variables[scalar]));
    maximumDifference = std::max(
        maximumDifference,
        std::abs(derivatives[scalar] -
                 acceptedConstraintCheckpoint.derivatives[scalar]));
  }

  for (Variable variable = 0; variable < getNumOfArrayVariables(); ++variable) {
    std::vector<uint64_t> indices;
    getVariableBeginIndices(variable, indices);
    uint64_t scalar = 0;
    do {
      if (scalar >=
          acceptedConstraintCheckpoint.modelVariables[variable].size()) {
        return false;
      }
      double current =
          algebraicAndStateVariablesGetters[variable](indices.data());
      double expected =
          acceptedConstraintCheckpoint.modelVariables[variable][scalar++];
      if (!std::isfinite(current)) {
        return false;
      }
      maximumDifference =
          std::max(maximumDifference, std::abs(current - expected));
    } while (advanceVariableIndices(indices, variablesDimensions[variable]));
    if (scalar != acceptedConstraintCheckpoint.modelVariables[variable].size()) {
      return false;
    }
  }

  uint64_t matchingExecutions = 0;
  for (const auto &execution : constraintComponentExecutions) {
    if (execution->switchScope && *execution->switchScope == scope) {
      ++matchingExecutions;
      if (execution->stateSetId != expectedStateSet) {
        return false;
      }
    }
  }
  uint64_t expectedExecutions = 1;
  if (scope.kind == ConstraintSwitchScopeKind::ExecutionDomain) {
    expectedExecutions = 0;
    for (const auto &[key, descriptor] : constraintExecutionEpochGroups) {
      if (descriptor.domainId == scope.id &&
          descriptor.epochId == expectedStateSet) {
        ++expectedExecutions;
      }
    }
  }
  return matchingExecutions == expectedExecutions &&
         expectedExecutions != 0 && maximumDifference == 0;
}

bool IDAInstance::rollbackConstraintStateSetSwitch(
    ConstraintSwitchScope scope, uint64_t previousStateSet) {
  // 中文：回滚不仅恢复 active ID，还重建 descriptor 布局并逐值证明 checkpoint
  // 差为零；无法证明时 fail-stop，禁止在部分切换状态上继续积分。
  // English: Rollback restores more than the active ID: it rebuilds descriptor
  // layout and proves every checkpoint value has zero difference. Failure is
  // fail-stop; integration never continues from a partially switched state.
  activeConstraintStateSets[scope.id] = previousStateSet;
  bool executionRestored = prepareConstraintComponentExecutions();
  bool checkpointRestored = restoreAcceptedConstraintCheckpoint();
  double maximumDifference = std::numeric_limits<double>::infinity();
  bool rollbackCertified =
      executionRestored && checkpointRestored &&
      verifyAcceptedConstraintCheckpointRestored(
          scope, previousStateSet, maximumDifference);
  if (!rollbackCertified) {
    expectedConstraintFailure = true;
    std::cerr << "stage6gd_switch_rollback_failed switch_scope="
              << (scope.kind == ConstraintSwitchScopeKind::ExecutionDomain
                      ? "execution_domain"
                      : "semantic_component")
              << " switch_id=" << scope.id
              << " checkpoint_time=" << acceptedConstraintCheckpoint.time
              << " active_state_set=" << previousStateSet
              << " execution_restored=" << (executionRestored ? 1 : 0)
              << " checkpoint_restored=" << (checkpointRestored ? 1 : 0)
              << " maximum_difference=" << maximumDifference << std::endl;
  }
  return rollbackCertified;
}

std::string IDAInstance::getConstraintConfigurationSignature() const {
  std::ostringstream stream;
  bool first = true;
  for (const auto &[componentId, stateSetId] : activeConstraintStateSets) {
    if (!first) {
      stream << ";";
    }
    first = false;
    stream << componentId << "=" << stateSetId;
  }
  return stream.str();
}

bool IDAInstance::rebuildIDAAfterBasisSwitch(realtype time) {
  // 中文：状态基切换后依据新 outer layout 重建 ID vector、稀疏矩阵和线性求解器，
  // 再通过公开 IDAReInit 从 accepted time 以一阶历史重启。
  // English: After a basis switch, rebuild the ID vector, sparse matrix, and
  // linear solver from the new outer layout, then restart at accepted time via
  // public IDAReInit with first-order history.
  uint64_t newVariables = 0;
  for (Variable variable = 0; variable < getNumOfArrayVariables(); ++variable) {
    newVariables += getVariableFlatSize(variable);
  }
  uint64_t newEquations = 0;
  for (Equation equation = 0; equation < getNumOfVectorizedEquations();
       ++equation) {
    newEquations += getEquationFlatSize(equation);
  }
  if (newVariables != scalarVariablesNumber ||
      newEquations != scalarEquationsNumber ||
      newVariables != newEquations) {
    std::cerr << "stage6gc_state_set_dimension_mismatch variables="
              << newVariables << " equations=" << newEquations
              << " expected=" << scalarVariablesNumber << std::endl;
    return false;
  }

  N_VConst(0, idVector);
  N_VConst(0, tolerancesVector);
  for (Variable variable = 0; variable < getNumOfArrayVariables(); ++variable) {
    if (isLocalVariable(variable)) {
      continue;
    }
    uint64_t offset = variableOffsets[variable];
    uint64_t flatSize = getVariableFlatSize(variable);
    VariableKind kind = getVariableKind(variable);
    for (uint64_t scalar = 0; scalar < flatSize; ++scalar) {
      N_VGetArrayPointer(idVector)[offset + scalar] =
          kind == VariableKind::STATE ? 1 : 0;
      N_VGetArrayPointer(tolerancesVector)[offset + scalar] =
          kind == VariableKind::STATE
              ? getOptions().absoluteTolerance
              : std::min(getOptions().maxAlgebraicAbsoluteTolerance,
                         getOptions().absoluteTolerance);
    }
  }

  copyVariablesFromMARCO(variablesVector, derivativesVector);
  threadEquationsChunks.clear();
  rebuildJacobianMatrixStorage();
  computeThreadChunks();

  if (linearSolver) {
    SUNLinSolFree(linearSolver);
    linearSolver = nullptr;
  }
  if (sparseMatrix) {
    SUNMatDestroy(sparseMatrix);
    sparseMatrix = nullptr;
  }
  if (IDAReInit(idaMemory, time, variablesVector, derivativesVector) !=
      IDA_SUCCESS) {
    std::cerr << "stage6gc_ida_reinit_failed time=" << time << std::endl;
    return false;
  }

#if SUNDIALS_VERSION_MAJOR >= 6
  sparseMatrix = SUNSparseMatrix(
      static_cast<sunindextype>(scalarEquationsNumber),
      static_cast<sunindextype>(scalarEquationsNumber),
      static_cast<sunindextype>(nonZeroValuesNumber), CSR_MAT, ctx);
  linearSolver = SUNLinSol_KLU(variablesVector, sparseMatrix, ctx);
#else
  sparseMatrix = SUNSparseMatrix(
      static_cast<sunindextype>(scalarEquationsNumber),
      static_cast<sunindextype>(scalarEquationsNumber),
      static_cast<sunindextype>(nonZeroValuesNumber), CSR_MAT);
  linearSolver = SUNLinSol_KLU(variablesVector, sparseMatrix);
#endif
  if (!checkAllocation(static_cast<void *>(sparseMatrix), "SUNSparseMatrix") ||
      !checkAllocation(static_cast<void *>(linearSolver), "SUNLinSol_KLU")) {
    return false;
  }
  if (IDASetErrHandlerFn(idaMemory, IDAInstance::idaErrorHandler, this) !=
      IDA_SUCCESS) {
    return false;
  }
  if (!idaSVTolerances() || !idaSetLinearSolver() || !idaSetUserData() ||
      !idaSetMaxNumSteps() || !idaSetInitialStepSize() ||
      !idaSetMinStepSize() || !idaSetMaxStepSize() ||
      !idaSetMaxErrTestFails() || !idaSetSuppressAlg() || !idaSetId() ||
      !idaSetJacobianFunction() || !idaSetMaxNonlinIters() ||
      !idaSetMaxConvFails() || !idaSetNonlinConvCoef()) {
    return false;
  }
  restartRecovery = createIDARestartRecoveryState(
      acceptedConstraintCheckpoint.stepSize, timeStep,
      getOptions().initialStepSize, getOptions().minStepSize,
      getOptions().maxStepSize);
  if (!restartRecovery.active ||
      IDASetInitStep(idaMemory, restartRecovery.initialStep) != IDA_SUCCESS ||
      IDASetMaxStep(idaMemory, restartRecovery.currentStepLimit) !=
          IDA_SUCCESS ||
      IDASetMaxOrd(idaMemory, restartRecovery.currentMaximumOrder) !=
          IDA_SUCCESS) {
    std::cerr << "stage7e_ida_restart_recovery_setup_failed time=" << time
              << " accepted_step="
              << acceptedConstraintCheckpoint.stepSize << std::endl;
    return false;
  }
  currentTime = time;
  constraintProbeInternalTime = time;
  ++idaReInitCount;
  verifyFirstOrderAfterReInit = true;
  return true;
}

bool IDAInstance::advanceIDAReInitRecovery(realtype time) {
  // 中文：重启后的 step/order ramp 是显式状态机；每个 accepted step 才推进，
  // 防止一次切换后立刻恢复过大的步长和高阶历史。
  // English: Post-restart step/order ramping is an explicit state machine and
  // advances only on accepted steps, preventing an immediate return to large
  // steps and high order after a switch.
  if (!restartRecovery.active) {
    return true;
  }
  bool complete = advanceIDARestartRecoveryState(restartRecovery);
  double maximumStep = complete ? getOptions().maxStepSize
                                : restartRecovery.currentStepLimit;
  int maximumOrder = complete ? restartRecovery.targetMaximumOrder
                              : restartRecovery.currentMaximumOrder;
  if (IDASetMaxStep(idaMemory, maximumStep) != IDA_SUCCESS ||
      IDASetMaxOrd(idaMemory, maximumOrder) != IDA_SUCCESS) {
    std::cerr << "stage7e_ida_restart_recovery_failed time=" << time
              << " accepted_steps=" << restartRecovery.acceptedSteps
              << std::endl;
    return false;
  }
  if (isStage6GRuntimeProbeEnabled()) {
    std::cerr << std::setprecision(17)
              << "[stage6g-runtime-probe] basis_epoch=" << basisEpoch
              << " event=reinit_recovery time=" << time
              << " accepted_steps=" << restartRecovery.acceptedSteps
              << " maximum_order=" << maximumOrder
              << " maximum_step=" << maximumStep
              << " recovery_complete=" << (complete ? 1 : 0)
              << std::endl;
  }
  return true;
}

bool IDAInstance::performPendingBasisSwitch() {
  // 中文：切换事务固定从 accepted checkpoint 评估所有预认证候选，经 condition、
  // residual 与连续性门选出安全集合；失败候选不会污染 active layout。
  // English: The switch transaction evaluates all pre-certified candidates
  // from the accepted checkpoint and selects a safe set through condition,
  // residual, and continuity gates. Failed candidates cannot pollute the active
  // layout.
  if (!pendingBasisSwitch ||
      (constraintStateSets.empty() && constraintExecutionEpochs.empty()) ||
      isConstraintBasisSwitchDisabled()) {
    return false;
  }
  PendingBasisSwitch request = *pendingBasisSwitch;
  pendingBasisSwitch.reset();
  const bool executionEpoch =
      request.scope.kind == ConstraintSwitchScopeKind::ExecutionDomain;
  const uint64_t switchId = request.scope.id;
  const char *switchScopeName =
      executionEpoch ? "execution_domain" : "semantic_component";
  if (executionEpoch != !constraintExecutionEpochs.empty()) {
    expectedConstraintFailure = true;
    std::cerr << "stage6gd_switch_rollback_failed switch_scope="
              << switchScopeName << " switch_id=" << switchId
              << " reason=switch_scope_registry_mismatch" << std::endl;
    return false;
  }
  if (!acceptedConstraintCheckpoint.valid) {
    std::cerr << "dynamic_state_selection_required switch_scope="
              << switchScopeName << " switch_id=" << switchId
              << " time=" << request.trialTime
              << " condition=" << request.condition
              << " reason=missing_accepted_checkpoint"
              << std::endl;
    return false;
  }
  if (!restoreAcceptedConstraintCheckpoint()) {
    expectedConstraintFailure = true;
    std::cerr << "stage6gd_switch_rollback_failed switch_scope="
              << switchScopeName << " switch_id=" << switchId
              << " checkpoint_time=" << acceptedConstraintCheckpoint.time
              << " reason=initial_checkpoint_restore_failed" << std::endl;
    return false;
  }

  auto active = activeConstraintStateSets.find(switchId);
  if (active == activeConstraintStateSets.end()) {
    expectedConstraintFailure = true;
    std::cerr << "stage6gd_switch_rollback_failed switch_scope="
              << switchScopeName << " switch_id=" << switchId
              << " checkpoint_time=" << acceptedConstraintCheckpoint.time
              << " reason=active_state_set_missing" << std::endl;
    return false;
  }
  uint64_t previousStateSet = active->second;
  const std::string previousConfiguration =
      getConstraintConfigurationSignature();
  discontinuityCoordinator.beginPhysicalTime(
      acceptedConstraintCheckpoint.time, previousConfiguration);
  double checkpointDifference = std::numeric_limits<double>::infinity();
  if (!constraintStateSetRegistrationsCertified ||
      !verifyAcceptedConstraintCheckpointRestored(
          request.scope, previousStateSet, checkpointDifference)) {
    expectedConstraintFailure = true;
    std::cerr << "stage6gd_switch_rollback_failed switch_scope="
              << switchScopeName << " switch_id=" << switchId
              << " checkpoint_time=" << acceptedConstraintCheckpoint.time
              << " reason=checkpoint_verification_failed"
              << " maximum_difference=" << checkpointDifference << std::endl;
    return false;
  }

  // 中文：候选必须重新评估完整 provenance residual roles；只看 local Newton
  // 成功或 condition 不足以证明新状态基满足原 DAE。
  // English: A candidate must re-evaluate every provenance residual role.
  // Local-Newton success or conditioning alone does not prove that the new
  // basis satisfies the original DAE.
  struct CandidateResidualCertificate {
    double position{0};
    double tangent{0};
    double highest{0};
    double support{0};
    double maximum{0};
    uint64_t positionScalars{0};
    uint64_t tangentScalars{0};
    uint64_t highestScalars{0};
    uint64_t supportScalars{0};
    bool complete{false};
  };

  auto evaluateCandidateResiduals =
      [&](uint64_t stateSetId,
          CandidateResidualCertificate &certificate) -> bool {
    certificate = {};
    std::map<Equation, std::string> ownedEquations;
    auto evaluate = [&](Equation equation, const std::string &role) {
      auto inserted = ownedEquations.emplace(equation, role);
      if (!inserted.second) {
        return inserted.first->second == role;
      }
      std::vector<int64_t> indices;
      getEquationBeginIndices(equation, indices);
      do {
        double value = std::abs(residualFunctions[equation](
            acceptedConstraintCheckpoint.time, indices.data()));
        if (!std::isfinite(value)) {
          return false;
        }
        if (role == "position") {
          certificate.position = std::max(certificate.position, value);
          ++certificate.positionScalars;
        } else if (role == "tangent") {
          certificate.tangent = std::max(certificate.tangent, value);
          ++certificate.tangentScalars;
        } else if (role == "highest") {
          certificate.highest = std::max(certificate.highest, value);
          ++certificate.highestScalars;
        } else if (role == "auxiliary_support") {
          certificate.support = std::max(certificate.support, value);
          ++certificate.supportScalars;
        } else {
          return false;
        }
      } while (advanceEquationIndices(indices, equationRanges[equation]));
      return true;
    };

    if (executionEpoch) {
      for (const ConstraintExecutionEpochEquationDescriptor &descriptor :
           constraintExecutionEpochEquations) {
        if (descriptor.domainId == switchId &&
            descriptor.epochId == stateSetId &&
            !evaluate(descriptor.equation, descriptor.role)) {
          return false;
        }
      }
    } else {
      for (const ConstraintStateSetEquationDescriptor &descriptor :
           constraintStateSetEquations) {
        if (descriptor.componentId == switchId &&
            descriptor.stateSetId == stateSetId &&
            !evaluate(descriptor.equation, descriptor.role)) {
          return false;
        }
      }
    }

    ConstraintResidualRoleCounts expected;
    if (executionEpoch) {
      auto expectedIt = constraintExecutionEpochResidualCertificates.find(
          {switchId, stateSetId});
      if (expectedIt == constraintExecutionEpochResidualCertificates.end()) {
        return false;
      }
      expected = expectedIt->second;
    } else {
      auto component = constraintComponents.find(switchId);
      if (component == constraintComponents.end()) {
        return false;
      }
      expected.position = component->second.residualPositionScalars;
      expected.tangent = component->second.residualTangentScalars;
      expected.highest = component->second.residualHighestScalars;
      expected.support = component->second.residualSupportScalars;
    }
    ConstraintResidualRoleCounts observed{
        certificate.positionScalars, certificate.tangentScalars,
        certificate.highestScalars, certificate.supportScalars};
    certificate.complete =
        hasCompleteConstraintResidualCertificate(expected, observed);
    certificate.maximum =
        std::max({certificate.position, certificate.tangent,
                  certificate.highest, certificate.support});
    return certificate.complete;
  };

  // 中文：execution-domain 路径遍历编译期 transition graph 中的直接邻接
  // epochs，逐个在同一 accepted checkpoint 上评估并在候选间完整回滚。
  // English: The execution-domain path visits direct neighbors in the compiled
  // transition graph, evaluates each at the same accepted checkpoint, and
  // fully rolls back between candidates.
  if (executionEpoch) {
    auto currentEpoch = constraintExecutionEpochs.find(
        {switchId, previousStateSet});
    if (currentEpoch == constraintExecutionEpochs.end()) {
      expectedConstraintFailure = true;
      std::cerr << "stage6gd_switch_rollback_failed switch_scope="
                << switchScopeName << " switch_id=" << switchId
                << " reason=active_execution_epoch_missing" << std::endl;
      return false;
    }

    struct CandidateResult {
      uint64_t epochId{0};
      uint64_t preferDummyScalars{0};
      uint64_t defaultDummyScalars{0};
      uint64_t avoidDummyScalars{0};
      double condition{std::numeric_limits<double>::infinity()};
      double residual{std::numeric_limits<double>::infinity()};
      CandidateResidualCertificate residualCertificate;
      double stateJump{std::numeric_limits<double>::infinity()};
      double modelAdjustment{std::numeric_limits<double>::infinity()};
      Variable modelAdjustmentVariable{0};
      bool safe{false};
    };

    auto evaluateEpoch = [&](uint64_t epochId,
                             CandidateResult &result) -> bool {
      auto candidate = constraintExecutionEpochs.find(
          {switchId, epochId});
      if (candidate == constraintExecutionEpochs.end()) {
        return false;
      }
      result.epochId = epochId;
      result.preferDummyScalars = candidate->second.preferDummyScalars;
      result.defaultDummyScalars = candidate->second.defaultDummyScalars;
      result.avoidDummyScalars = candidate->second.avoidDummyScalars;

      activeConstraintStateSets[switchId] = epochId;
      if (!prepareConstraintComponentExecutions() ||
          !restoreAcceptedConstraintCheckpoint()) {
        return false;
      }

      evaluatingBasisCandidate = true;
      int localResult =
          solveConstraintComponents(acceptedConstraintCheckpoint.time);
      evaluatingBasisCandidate = false;
      if (localResult != 0) {
        return true;
      }

      std::vector<ConstraintSwitchHealthSample> healthSamples;
      for (const auto &execution : constraintComponentExecutions) {
        if (!execution->switchScope) {
          continue;
        }
        healthSamples.push_back(
            {*execution->switchScope,
             std::max(
                 execution->lastHealthCertificate.chart.oneNormCondition,
                 execution->lastHealthCertificate.localClosure
                     .oneNormCondition)});
      }
      result.condition = getMaximumConstraintSwitchScopeCondition(
                             healthSamples, request.scope)
                             .value_or(
                                 std::numeric_limits<double>::infinity());
      if (!evaluateCandidateResiduals(epochId,
                                      result.residualCertificate)) {
        return true;
      }
      result.residual = result.residualCertificate.maximum;

      result.stateJump = 0;
      std::set<Variable> outerVariables;
      for (const ConstraintExecutionEpochVariableDescriptor &descriptor :
           constraintExecutionEpochVariables) {
        if (descriptor.domainId != switchId ||
            descriptor.epochId != epochId || descriptor.owner != "outer" ||
            !outerVariables.insert(descriptor.variable).second) {
          continue;
        }
        std::vector<uint64_t> indices;
        getVariableBeginIndices(descriptor.variable, indices);
        uint64_t scalar = 0;
        do {
          result.stateJump = std::max(
              result.stateJump,
              std::abs(algebraicAndStateVariablesGetters[descriptor.variable](
                           indices.data()) -
                       acceptedConstraintCheckpoint
                           .modelVariables[descriptor.variable][scalar++]));
        } while (advanceVariableIndices(
            indices, variablesDimensions[descriptor.variable]));
      }

      result.modelAdjustment = 0;
      for (Variable variable = 0; variable < getNumOfArrayVariables();
           ++variable) {
        std::vector<uint64_t> indices;
        getVariableBeginIndices(variable, indices);
        uint64_t scalar = 0;
        do {
          double adjustment = std::abs(
              algebraicAndStateVariablesGetters[variable](indices.data()) -
              acceptedConstraintCheckpoint.modelVariables[variable]
                                                       [scalar++]);
          if (adjustment > result.modelAdjustment) {
            result.modelAdjustment = adjustment;
            result.modelAdjustmentVariable = variable;
          }
        } while (advanceVariableIndices(indices,
                                        variablesDimensions[variable]));
      }
      result.safe = isSafeConstraintStateSetCandidate(
          true, result.condition, result.residual, result.stateJump);
      return true;
    };

    std::vector<CandidateResult> evaluatedCandidates;
    for (uint64_t candidateEpoch : currentEpoch->second.transitions) {
      SolverDiscontinuityTransaction transaction{
          SolverDiscontinuityKind::ConstraintBasisSwitch,
          switchId, previousStateSet, candidateEpoch,
          request.phase};
      if (discontinuityCoordinator.isRejected(transaction)) {
        continue;
      }
      CandidateResult candidate;
      bool evaluated = evaluateEpoch(candidateEpoch, candidate);
      if (isStage6GRuntimeProbeEnabled()) {
        std::cerr << std::setprecision(17)
                  << "[stage6g-runtime-probe] basis_epoch=" << basisEpoch
                  << " event=epoch_evaluated domain=" << switchId
                  << " checkpoint_time="
                  << acceptedConstraintCheckpoint.time
                  << " candidate_epoch=" << candidateEpoch
                  << " evaluated=" << (evaluated ? 1 : 0)
                  << " safe=" << (candidate.safe ? 1 : 0)
                  << " condition=" << candidate.condition
                  << " residual=" << candidate.residual
                  << " position_residual="
                  << candidate.residualCertificate.position
                  << " tangent_residual="
                  << candidate.residualCertificate.tangent
                  << " highest_residual="
                  << candidate.residualCertificate.highest
                  << " support_residual="
                  << candidate.residualCertificate.support
                  << " residual_certificate_complete="
                  << (candidate.residualCertificate.complete ? 1 : 0)
                  << " state_jump_max=" << candidate.stateJump
                  << " prefer_dummy=" << candidate.preferDummyScalars
                  << " default_dummy=" << candidate.defaultDummyScalars
                  << " avoid_dummy=" << candidate.avoidDummyScalars
                  << std::endl;
      }
      if (!evaluated || !candidate.safe) {
        discontinuityCoordinator.reject(transaction);
        if (!rollbackConstraintStateSetSwitch(request.scope,
                                              previousStateSet)) {
          return false;
        }
        continue;
      }
      evaluatedCandidates.push_back(candidate);
      if (!rollbackConstraintStateSetSwitch(request.scope,
                                            previousStateSet)) {
        return false;
      }
    }
    std::vector<ConstraintExecutionEpochCandidate> candidateScores;
    candidateScores.reserve(evaluatedCandidates.size());
    for (const CandidateResult &candidate : evaluatedCandidates) {
      candidateScores.push_back(
          {candidate.epochId, candidate.preferDummyScalars,
           candidate.defaultDummyScalars, candidate.avoidDummyScalars,
           candidate.condition, candidate.safe});
    }
    std::optional<uint64_t> selectedEpoch =
        selectConstraintExecutionEpoch(candidateScores);
    if (!selectedEpoch) {
      std::cerr << "dynamic_state_selection_required switch_scope="
                << switchScopeName << " switch_id=" << switchId
                << " time=" << request.trialTime
                << " condition=" << request.condition
                << " reason=no_safe_execution_epoch" << std::endl;
      return false;
    }

    CandidateResult selected;
    if (!evaluateEpoch(*selectedEpoch, selected) || !selected.safe) {
      rollbackConstraintStateSetSwitch(request.scope,
                                       previousStateSet);
      std::cerr << "dynamic_state_selection_required switch_scope="
                << switchScopeName << " switch_id=" << switchId
                << " time=" << request.trialTime
                << " reason=selected_epoch_not_reproducible" << std::endl;
      return false;
    }
    SolverDiscontinuityTransaction selectedTransaction{
        SolverDiscontinuityKind::ConstraintBasisSwitch,
        switchId, previousStateSet, selected.epochId,
        request.phase};
    const std::string selectedConfiguration =
        getConstraintConfigurationSignature();
    if (discontinuityCoordinator.wouldRepeatConfiguration(
            selectedConfiguration)) {
      discontinuityCoordinator.reject(selectedTransaction);
      rollbackConstraintStateSetSwitch(request.scope,
                                       previousStateSet);
      std::cerr << "dynamic_state_selection_required switch_scope="
                << switchScopeName << " switch_id=" << switchId
                << " time=" << request.trialTime
                << " reason=discontinuity_configuration_cycle"
                << " superdense_index="
                << discontinuityCoordinator.getSuperdenseIndex()
                << std::endl;
      return false;
    }
    if (!rebuildIDAAfterBasisSwitch(acceptedConstraintCheckpoint.time)) {
      rollbackConstraintStateSetSwitch(request.scope,
                                       previousStateSet);
      return false;
    }
    if (discontinuityCoordinator.commit(selectedTransaction,
                                        selectedConfiguration) !=
        SolverDiscontinuityCoordinator::CommitResult::Committed) {
      rollbackConstraintStateSetSwitch(request.scope,
                                       previousStateSet);
      std::cerr << "stage7e_discontinuity_commit_failed switch_scope="
                << switchScopeName << " switch_id=" << switchId
                << " reason=configuration_cycle_after_reinit" << std::endl;
      return false;
    }
    ++basisSwitchCount;
    ++basisEpoch;
    saveAcceptedConstraintCheckpoint(acceptedConstraintCheckpoint.time);
    std::cerr << std::setprecision(17)
              << "[stage6g-runtime-probe] basis_epoch=" << basisEpoch
              << " event=execution_epoch_switched domain="
              << switchId
              << " checkpoint_time=" << currentTime
              << " from_epoch=" << previousStateSet
              << " to_epoch=" << selected.epochId
              << " condition=" << selected.condition
              << " state_jump_max=" << selected.stateJump
              << " model_adjustment_max=" << selected.modelAdjustment
              << " model_adjustment_variable="
              << selected.modelAdjustmentVariable
              << " ida_reinit_count=" << idaReInitCount
              << " superdense_index="
              << discontinuityCoordinator.getSuperdenseIndex()
              << " checkpoint_restored=1 registration_certified=1"
              << " switch_epoch_certified=1" << std::endl;
    return true;
  }

  // 中文：没有 epoch map 的旧单-component 路径仍只在已注册 standby set 中
  // 切换，并使用相同 residual/continuity/rollback 门。
  // English: Without an epoch map, the legacy single-component path still
  // switches only to a registered standby set and uses the same residual,
  // continuity, and rollback gates.
  const ConstraintStateSetDescriptor *standby = nullptr;
  for (const ConstraintStateSetDescriptor &candidate : constraintStateSets) {
    SolverDiscontinuityTransaction transaction{
        SolverDiscontinuityKind::ConstraintBasisSwitch,
        switchId, previousStateSet, candidate.stateSetId,
        request.phase};
    if (candidate.componentId != switchId ||
        candidate.stateSetId == previousStateSet ||
        discontinuityCoordinator.isRejected(transaction)) {
      continue;
    }
    standby = &candidate;
    break;
  }
  if (!standby) {
    std::cerr << "dynamic_state_selection_required switch_scope="
              << switchScopeName << " switch_id=" << switchId
              << " time=" << request.trialTime
              << " condition=" << request.condition
              << " reason=no_safe_standby" << std::endl;
    return false;
  }

  active->second = standby->stateSetId;
  if (!prepareConstraintComponentExecutions() ||
      !restoreAcceptedConstraintCheckpoint()) {
    rollbackConstraintStateSetSwitch(request.scope, previousStateSet);
    return false;
  }

  struct OuterStateValue {
    Variable variable;
    std::vector<uint64_t> indices;
    double value;
    std::optional<double> derivative;
    std::string name;
  };
  std::vector<OuterStateValue> standbyOuterValues;
  std::set<Variable> capturedOuterVariables;
  for (const ConstraintStateSetVariableDescriptor &descriptor :
       constraintStateSetVariables) {
    if (descriptor.componentId != switchId ||
        descriptor.stateSetId != standby->stateSetId ||
        descriptor.owner != "outer" ||
        !capturedOuterVariables.insert(descriptor.variable).second) {
      continue;
    }
    const VariableDimensions &dimensions =
        variablesDimensions[descriptor.variable];
    std::vector<uint64_t> indices;
    getVariableBeginIndices(descriptor.variable, indices);
    do {
      OuterStateValue value{descriptor.variable, indices,
                            algebraicAndStateVariablesGetters
                                [descriptor.variable](indices.data()),
                            std::nullopt, descriptor.name};
      auto state = stateVariablesMapping.find(descriptor.variable);
      if (state != stateVariablesMapping.end()) {
        value.derivative = derivativeVariablesGetters[state->second](
            indices.data());
      }
      standbyOuterValues.push_back(std::move(value));
    } while (advanceVariableIndices(indices, dimensions));
  }

  evaluatingBasisCandidate = true;
  int localResult = solveConstraintComponents(acceptedConstraintCheckpoint.time);
  evaluatingBasisCandidate = false;
  double condition = std::numeric_limits<double>::infinity();
  double residual = std::numeric_limits<double>::infinity();
  CandidateResidualCertificate residualCertificate;
  if (localResult == 0) {
    std::vector<ConstraintSwitchHealthSample> healthSamples;
    for (const auto &execution : constraintComponentExecutions) {
      if (!execution->switchScope) {
        continue;
      }
      healthSamples.push_back(
          {*execution->switchScope,
           std::max(execution->lastHealthCertificate.chart.oneNormCondition,
                    execution->lastHealthCertificate.localClosure
                        .oneNormCondition)});
    }
    condition = getMaximumConstraintSwitchScopeCondition(healthSamples,
                                                         request.scope)
                    .value_or(std::numeric_limits<double>::infinity());
    if (std::isfinite(condition) &&
        evaluateCandidateResiduals(standby->stateSetId,
                                   residualCertificate)) {
      residual = residualCertificate.maximum;
    }
  }

  double maximumStateJump = 0;
  std::string maximumStateJumpVariable = "none";
  for (const OuterStateValue &value : standbyOuterValues) {
    double stateJump = std::abs(
        algebraicAndStateVariablesGetters[value.variable](value.indices.data()) -
        value.value);
    if (stateJump > maximumStateJump) {
      maximumStateJump = stateJump;
      maximumStateJumpVariable = value.name;
    }
    if (value.derivative) {
      auto state = stateVariablesMapping.find(value.variable);
      if (state == stateVariablesMapping.end()) {
        maximumStateJump = std::numeric_limits<double>::infinity();
        maximumStateJumpVariable = value.name + ".derivative";
        break;
      }
      double derivativeJump = std::abs(
          derivativeVariablesGetters[state->second](value.indices.data()) -
          *value.derivative);
      if (derivativeJump > maximumStateJump) {
        maximumStateJump = derivativeJump;
        maximumStateJumpVariable = value.name + ".derivative";
      }
    }
  }

  double maximumModelAdjustment = 0;
  Variable maximumModelAdjustmentVariable = 0;
  for (Variable variable = 0; variable < getNumOfArrayVariables(); ++variable) {
    std::vector<uint64_t> indices;
    getVariableBeginIndices(variable, indices);
    uint64_t scalar = 0;
    do {
      double adjustment = std::abs(
          algebraicAndStateVariablesGetters[variable](indices.data()) -
          acceptedConstraintCheckpoint.modelVariables[variable][scalar++]);
      if (adjustment > maximumModelAdjustment) {
        maximumModelAdjustment = adjustment;
        maximumModelAdjustmentVariable = variable;
      }
    } while (advanceVariableIndices(indices, variablesDimensions[variable]));
  }

  if (isStage6GRuntimeProbeEnabled()) {
    std::cerr << std::setprecision(17)
              << "[stage6g-runtime-probe] basis_epoch=" << basisEpoch
              << " event=standby_evaluated component=" << switchId
              << " checkpoint_time=" << acceptedConstraintCheckpoint.time
              << " candidate_state_set=" << standby->stateSetId
              << " condition=" << condition << " residual=" << residual
              << " position_residual=" << residualCertificate.position
              << " tangent_residual=" << residualCertificate.tangent
              << " highest_residual=" << residualCertificate.highest
              << " support_residual=" << residualCertificate.support
              << " residual_certificate_complete="
              << (residualCertificate.complete ? 1 : 0)
              << " state_jump_max=" << maximumStateJump
              << " state_jump_variable=" << maximumStateJumpVariable
              << " model_adjustment_max=" << maximumModelAdjustment
              << " model_adjustment_variable="
              << maximumModelAdjustmentVariable
              << std::endl;
  }
  if (!isSafeConstraintStateSetCandidate(localResult == 0, condition,
                                         residual, maximumStateJump)) {
    SolverDiscontinuityTransaction transaction{
        SolverDiscontinuityKind::ConstraintBasisSwitch,
        switchId, previousStateSet, standby->stateSetId,
        request.phase};
    discontinuityCoordinator.reject(transaction);
    bool rollbackCertified =
        rollbackConstraintStateSetSwitch(request.scope,
                                         previousStateSet);
    std::cerr << "dynamic_state_selection_required switch_scope="
              << switchScopeName << " switch_id=" << switchId
              << " time=" << request.trialTime
              << " condition=" << request.condition
              << " standby_condition=" << condition
              << " standby_residual=" << residual
              << " standby_state_jump=" << maximumStateJump
              << " checkpoint_restored=" << (rollbackCertified ? 1 : 0)
              << " reason=no_safe_standby" << std::endl;
    return false;
  }

  SolverDiscontinuityTransaction selectedTransaction{
      SolverDiscontinuityKind::ConstraintBasisSwitch,
      switchId, previousStateSet, standby->stateSetId,
      request.phase};
  const std::string selectedConfiguration =
      getConstraintConfigurationSignature();
  if (discontinuityCoordinator.wouldRepeatConfiguration(
          selectedConfiguration)) {
    discontinuityCoordinator.reject(selectedTransaction);
    bool rollbackCertified = rollbackConstraintStateSetSwitch(
        request.scope, previousStateSet);
    std::cerr << "dynamic_state_selection_required switch_scope="
              << switchScopeName << " switch_id=" << switchId
              << " time=" << request.trialTime
              << " reason=discontinuity_configuration_cycle"
              << " superdense_index="
              << discontinuityCoordinator.getSuperdenseIndex()
              << " checkpoint_restored="
              << (rollbackCertified ? 1 : 0) << std::endl;
    return false;
  }

  if (!rebuildIDAAfterBasisSwitch(acceptedConstraintCheckpoint.time)) {
    return false;
  }
  if (discontinuityCoordinator.commit(selectedTransaction,
                                      selectedConfiguration) !=
      SolverDiscontinuityCoordinator::CommitResult::Committed) {
    rollbackConstraintStateSetSwitch(request.scope, previousStateSet);
    std::cerr << "stage7e_discontinuity_commit_failed switch_scope="
              << switchScopeName << " switch_id=" << switchId
              << " reason=configuration_cycle_after_reinit" << std::endl;
    return false;
  }
  ++basisSwitchCount;
  ++basisEpoch;
  saveAcceptedConstraintCheckpoint(acceptedConstraintCheckpoint.time);
  std::cerr << std::setprecision(17)
            << "[stage6g-runtime-probe] basis_epoch=" << basisEpoch
            << " event=basis_switched component=" << switchId
            << " checkpoint_time=" << currentTime
            << " from_state_set=" << previousStateSet
            << " to_state_set=" << standby->stateSetId
            << " condition=" << condition
            << " state_jump_max=" << maximumStateJump
            << " state_jump_variable=" << maximumStateJumpVariable
            << " model_adjustment_max=" << maximumModelAdjustment
            << " model_adjustment_variable="
            << maximumModelAdjustmentVariable
            << " ida_reinit_count=" << idaReInitCount
            << " superdense_index="
            << discontinuityCoordinator.getSuperdenseIndex()
            << " checkpoint_restored=1"
            << " registration_certified="
            << (constraintStateSetRegistrationsCertified ? 1 : 0)
            << " switch_epoch_certified=1" << std::endl;
  return true;
}

void IDAInstance::idaErrorHandler(int errorCode, const char *module,
                                  const char *function, char *message,
                                  void *userData) {
  auto *instance = static_cast<IDAInstance *>(userData);
  if (instance &&
      shouldSuppressIDAError(errorCode, instance->pendingBasisSwitch.has_value(),
                             instance->expectedConstraintFailure)) {
    return;
  }
  std::cerr << "[IDA ERROR] " << (module ? module : "IDA") << "::"
            << (function ? function : "unknown") << " code=" << errorCode
            << " " << (message ? message : "") << std::endl;
}

bool IDAInstance::applyConstraintSchur(realtype time, realtype alpha) {
  // 中文：按 execution-group DAG 传播 dz/ds，并把完整
  // R_s - R_z J_zz^{-1} G_s 写回 outer sparse Jacobian；每组每次只分解一次
  // J_zz，禁止把有依赖的 groups 当成独立 Schur blocks。
  // English: Propagate dz/ds through the execution-group DAG and write the full
  // R_s - R_z J_zz^{-1} G_s correction into the outer sparse Jacobian. Each
  // group factorizes J_zz once per call; dependent groups are never treated as
  // independent Schur blocks.
  std::vector<double> outerErrorWeights;
  if (idaMemory != nullptr && variablesVector != nullptr &&
      scalarVariablesNumber > 0) {
    N_Vector errorWeights = N_VClone(variablesVector);
    if (errorWeights != nullptr) {
      if (IDAGetErrWeights(idaMemory, errorWeights) == IDA_SUCCESS) {
        const realtype *data = N_VGetArrayPointer(errorWeights);
        outerErrorWeights.assign(data, data + scalarVariablesNumber);
      }
      N_VDestroy(errorWeights);
    }
  }
  std::map<JacobianColumn,
           std::pair<ConstraintComponentExecutionRuntime *, uint64_t>>
      localColumnOwners;
  for (const auto &execution : constraintComponentExecutions) {
    for (const auto &[column, localIndex] : execution->localColumnIndices) {
      localColumnOwners.emplace(
          column, std::make_pair(execution.get(), localIndex));
    }
  }

  // 中文：第一遍按拓扑序求每组 total dz/ds；若 coupling column 来自上游
  // local group，则通过已传播 response 合成 G_s，而不是当作独立外部列。
  // English: The first pass computes each group's total dz/ds in topological
  // order. If a coupling column belongs to an upstream local group, compose it
  // through the propagated response instead of treating it as independent.
  for (const auto &executionPtr : constraintComponentExecutions) {
    ConstraintComponentExecutionRuntime &execution = *executionPtr;
    if (!execution.solver->factorizeCurrentJacobian(time)) {
      std::cerr << "stage6gb_component_requires_c3 execution_group="
                << execution.executionGroupId
                << " reason=local_jacobian_singular"
                << " time=" << time << std::endl;
      return false;
    }
    ++execution.schurFactorizations;

    execution.totalResponses.clear();
    for (const JacobianColumn &outerColumn :
         execution.outerCouplingColumns) {
      std::vector<double> localRightHandSide(execution.localScalars, 0);
      for (uint64_t row = 0; row < execution.localScalarRows.size(); ++row) {
        const auto &[equation, equationIndices] =
            execution.localScalarRows[row];
        std::vector<JacobianColumn> accesses =
            computeAccessedJacobianColumns(equation, equationIndices.data(),
                                           true);
        double directDerivative = 0;
        if (std::find(accesses.begin(), accesses.end(), outerColumn) !=
            accesses.end()) {
          double direct = 0;
          if (!evaluateConstraintJacobian(
                  execution, equation, equationIndices, outerColumn, time,
                  alpha, direct)) {
            std::cerr << "stage7c_component_requires_c3 group="
                      << execution.executionGroupId
                      << " reason=missing_direct_gs_jacobian_callback"
                      << std::endl;
            return false;
          }
          directDerivative += direct;
        }

        std::vector<double> crossDerivatives;
        std::vector<double> dependencyResponses;
        for (const JacobianColumn &access : accesses) {
          auto owner = localColumnOwners.find(access);
          if (owner == localColumnOwners.end() ||
              owner->second.first == &execution) {
            continue;
          }
          ConstraintComponentExecutionRuntime &dependency =
              *owner->second.first;
          auto response = dependency.totalResponses.find(outerColumn);
          if (response == dependency.totalResponses.end()) {
            continue;
          }
          double crossDerivative = 0;
          if (!evaluateConstraintJacobian(
                  execution, equation, equationIndices, access, time, 0.0,
                  crossDerivative)) {
            std::cerr << "stage7c_component_requires_c3 group="
                      << execution.executionGroupId
                      << " dependency=" << dependency.executionGroupId
                      << " reason=missing_cross_group_jacobian_callback"
                      << std::endl;
            return false;
          }
          crossDerivatives.push_back(crossDerivative);
          dependencyResponses.push_back(
              response->second[owner->second.second]);
        }
        double totalDerivative = 0;
        if (!composeExecutionGroupDerivative(
                directDerivative, crossDerivatives, dependencyResponses,
                totalDerivative)) {
          std::cerr << "stage7c_component_requires_c3 group="
                    << execution.executionGroupId
                    << " reason=invalid_cross_group_sensitivity"
                    << std::endl;
          return false;
        }
        localRightHandSide[row] = totalDerivative;
      }

      std::vector<double> response;
      if (!execution.solver->solveCurrentJacobian(localRightHandSide,
                                                   response)) {
        std::cerr << "stage6gb_component_requires_c3 component="
                  << execution.executionGroupId
                  << " reason=schur_right_hand_side_solve_failed"
                  << std::endl;
        return false;
      }
      ++execution.schurRightHandSides;
      execution.totalResponses.emplace(outerColumn, std::move(response));
    }

    double maximumResponseAmplification = 0;
    bool responseAvailable = !execution.totalResponses.empty();
    for (const auto &[outerColumn, response] : execution.totalResponses) {
      uint64_t flatColumn =
          variableOffsets[outerColumn.first] +
          getVariableFlatIndex(variablesDimensions[outerColumn.first],
                               outerColumn.second);
      double outerPerturbation = 0;
      if (flatColumn < outerErrorWeights.size() &&
          outerErrorWeights[flatColumn] > 0 &&
          std::isfinite(outerErrorWeights[flatColumn])) {
        outerPerturbation = 1.0 / outerErrorWeights[flatColumn];
      } else if (outerColumn.first < variableNominalGetters.size() &&
                 variableNominalGetters[outerColumn.first] != nullptr) {
        outerPerturbation = variableNominalGetters[outerColumn.first](
            outerColumn.second.data());
      } else {
        double value = algebraicAndStateVariablesGetters[outerColumn.first](
            outerColumn.second.data());
        outerPerturbation = std::max(1.0, std::abs(value));
      }
      if (!(outerPerturbation > 0) || !std::isfinite(outerPerturbation) ||
          response.size() != execution.localScalars) {
        maximumResponseAmplification =
            std::numeric_limits<double>::infinity();
        break;
      }
      for (uint64_t localScalar = 0; localScalar < response.size();
           ++localScalar) {
        double localNominal =
            execution.solver->getScalarVariableNominal(localScalar);
        if (!(localNominal > 0) || !std::isfinite(localNominal) ||
            !std::isfinite(response[localScalar])) {
          maximumResponseAmplification =
              std::numeric_limits<double>::infinity();
          break;
        }
        maximumResponseAmplification =
            std::max(maximumResponseAmplification,
                     std::abs(response[localScalar]) * outerPerturbation /
                         localNominal);
      }
    }
    execution.lastHealthCertificate.responseAvailable = responseAvailable;
    execution.lastHealthCertificate.responseAmplification =
        maximumResponseAmplification;
    if (responseAvailable &&
        maximumResponseAmplification >=
            execution.maximumResponseAmplification) {
      execution.maximumResponseAmplification = maximumResponseAmplification;
      execution.maximumResponseAmplificationTime = time;
    }
    ConstraintBasisHealthAction responseHealth =
        classifyConstraintBasisHealth(execution.lastHealthCertificate, false);
    if (responseHealth != ConstraintBasisHealthAction::Accept &&
        execution.switchScope &&
        !isConstraintBasisSwitchDisabled() && !evaluatingBasisCandidate &&
        !pendingBasisSwitch) {
      pendingBasisSwitch = PendingBasisSwitch{
          *execution.switchScope, execution.stateSetId, time,
          std::max(execution.lastHealthCertificate.chart.oneNormCondition,
                   execution.lastHealthCertificate.localClosure
                       .oneNormCondition),
          "schur_response"};
      if (isStage6GRuntimeProbeEnabled()) {
        std::cerr << std::setprecision(17)
                  << "[stage6g-runtime-probe] basis_epoch=" << basisEpoch
                  << " event=switch_requested switch_scope="
                  << (execution.switchScope->kind ==
                              ConstraintSwitchScopeKind::ExecutionDomain
                          ? "execution_domain"
                          : "semantic_component")
                  << " switch_id=" << execution.switchScope->id
                  << " execution_group=" << execution.executionGroupId
                  << " state_set=" << execution.stateSetId
                  << " trial_time=" << time
                  << " response_amplification="
                  << maximumResponseAmplification
                  << " phase=schur_response reason="
                  << (responseHealth == ConstraintBasisHealthAction::Unsafe
                          ? "unsafe_domain_response"
                          : "domain_response_hysteresis")
                  << std::endl;
      }
    }

    for (Equation equation : execution.outerEquations) {
      std::vector<int64_t> equationIndices;
      getEquationBeginIndices(equation, equationIndices);
      do {
        std::vector<JacobianColumn> allAccesses =
            computeAccessedJacobianColumns(equation, equationIndices.data(),
                                           true);
        std::set<uint64_t> localComponents;
        for (const JacobianColumn &column : allAccesses) {
          auto component = execution.localColumnComponents.find(column);
          if (component != execution.localColumnComponents.end()) {
            localComponents.insert(component->second);
          }
        }
        if (localComponents.empty()) {
          continue;
        }

        std::set<JacobianColumn> rowCouplingColumns;
        for (uint64_t localComponent : localComponents) {
          const auto &columns =
              execution.componentOuterCouplingColumns[localComponent];
          rowCouplingColumns.insert(columns.begin(), columns.end());
        }

        std::vector<double> rz(execution.localScalars, 0);
        for (uint64_t localColumn = 0;
             localColumn < execution.localScalarColumns.size();
             ++localColumn) {
          const JacobianColumn &column =
              execution.localScalarColumns[localColumn];
          if (std::find(allAccesses.begin(), allAccesses.end(), column) ==
              allAccesses.end()) {
            continue;
          }
          double value = 0;
          if (!evaluateConstraintJacobian(execution, equation,
                                          equationIndices, column, time, 0.0,
                                          value)) {
            std::cerr << "stage6gb_component_requires_c3 component="
                      << execution.executionGroupId
                      << " reason=missing_rz_jacobian_callback" << std::endl;
            return false;
          }
          rz[localColumn] = value;
        }

        uint64_t scalarEquationIndex =
            equationOffsets[equation] +
            getEquationFlatIndex(equationIndices, equationRanges[equation]);
        auto &jacobianRow = jacobianMatrixData[scalarEquationIndex];
        for (const JacobianColumn &column : rowCouplingColumns) {
          auto response = execution.totalResponses.find(column);
          if (response == execution.totalResponses.end()) {
            std::cerr << "stage7c_component_requires_c3 group="
                      << execution.executionGroupId
                      << " reason=missing_schur_local_response" << std::endl;
            return false;
          }
          double correction = 0;
          for (uint64_t localColumn = 0;
               localColumn < execution.localScalars; ++localColumn) {
            correction += rz[localColumn] * response->second[localColumn];
          }
          sunindextype flatColumn = static_cast<sunindextype>(
              variableOffsets[column.first] +
              getVariableFlatIndex(variablesDimensions[column.first],
                                   column.second));
          auto entry = std::find_if(
              jacobianRow.begin(), jacobianRow.end(),
              [&](const std::pair<sunindextype, double> &candidate) {
                return candidate.first == flatColumn;
              });
          if (entry == jacobianRow.end()) {
            std::cerr << "stage6gb_component_requires_c3 component="
                      << execution.executionGroupId
                      << " reason=missing_schur_sparsity_entry" << std::endl;
            return false;
          }
          entry->second -= correction;
          ++execution.schurFillEntries;
        }
      } while (advanceEquationIndices(equationIndices,
                                      equationRanges[equation]));
    }

    if (getStage6GRuntimeProbeMode() == Stage6GRuntimeProbeMode::Trace) {
      std::cerr << "[stage6g-runtime-probe] sample=schur"
                << " execution_group=" << execution.executionGroupId
                << " dependencies=" << execution.dependencies.size()
                << " jzz_rows=" << execution.localScalars
                << " gs_columns=" << execution.outerCouplingColumns.size()
                << " rz_rows=" << execution.outerEquations.size()
                << " factorizations=" << execution.schurFactorizations
                << " rhs_solves=" << execution.schurRightHandSides
                << " fill_entries=" << execution.schurFillEntries
                << " response_amplification="
                << maximumResponseAmplification
                << std::endl;
    }
  }
  return true;
}
Variable IDAInstance::addAlgebraicVariable(uint64_t rank,
                                           const uint64_t *dimensions,
                                           VariableGetter getterFunction,
                                           VariableSetter setterFunction,
                                           const char *name) {
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[IDA] Adding algebraic variable";

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
  algebraicAndStateVariablesGetters.push_back(getterFunction);
  algebraicAndStateVariablesSetters.push_back(setterFunction);
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

bool IDAInstance::setVariableNominal(
    Variable variable, VariableGetter nominalGetterFunction) {
  if (nominalGetterFunction == nullptr ||
      variable >= variableNominalGetters.size()) {
    return false;
  }
  variableNominalGetters[variable] = nominalGetterFunction;
  return true;
}

Variable IDAInstance::addStateVariable(uint64_t rank,
                                       const uint64_t *dimensions,
                                       VariableGetter stateGetterFunction,
                                       VariableSetter stateSetterFunction,
                                       VariableGetter derivativeGetterFunction,
                                       VariableSetter derivativeSetterFunction,
                                       const char *name) {
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[IDA] Adding state variable";

    if (name != nullptr) {
      std::cerr << " \"" << name << "\"";
    }

    std::cerr << std::endl;
  }

  assert(variableOffsets.size() == getNumOfArrayVariables() + 1);

  // Add variable offset and dimensions.
  VariableDimensions variableDimensions(rank);
  uint64_t flatSize = 1;

  for (uint64_t i = 0; i < rank; ++i) {
    flatSize *= dimensions[i];
    variableDimensions[i] = dimensions[i];
  }

  variablesDimensions.push_back(variableDimensions);

  // Store the position of the start of the flattened array.
  uint64_t offset = variableOffsets.back();
  variableOffsets.push_back(offset + flatSize);

  // Store the getter and setter functions for the state variable.
  algebraicAndStateVariablesGetters.push_back(stateGetterFunction);
  algebraicAndStateVariablesSetters.push_back(stateSetterFunction);
  variableNominalGetters.push_back(nullptr);

  // Store the getter and setter functions for the derivative variable.
  derivativeVariablesGetters.push_back(derivativeGetterFunction);
  derivativeVariablesSetters.push_back(derivativeSetterFunction);

  // Return the index of the variable.
  Variable id = getNumOfArrayVariables() - 1;
  stateVariablesMapping[id] = derivativeVariablesGetters.size() - 1;

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
    std::cerr << "  - State variable getter function address: "
              << reinterpret_cast<void *>(stateGetterFunction) << std::endl;
    std::cerr << "  - State variable setter function address: "
              << reinterpret_cast<void *>(stateSetterFunction) << std::endl;
    std::cerr << "  - Derivative variable getter function address: "
              << reinterpret_cast<void *>(derivativeGetterFunction)
              << std::endl;
    std::cerr << "  - Derivative variable setter function address: "
              << reinterpret_cast<void *>(derivativeSetterFunction)
              << std::endl;
  }

  return id;
}

Equation IDAInstance::addEquation(const int64_t *ranges, uint64_t equationRank,
                                  const char *stringRepresentation) {
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[IDA] Adding equation";

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

void IDAInstance::addVariableAccess(Equation equation, Variable variable,
                                    AccessFunction accessFunction) {
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[IDA] Adding access information" << std::endl;
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

void IDAInstance::addVariableRangeAccess(Equation equation, Variable variable,
                                         const int64_t *ranges,
                                         uint64_t rank) {
  assert(equation < getNumOfVectorizedEquations());
  assert(variable < getNumOfArrayVariables());

  MultidimensionalRange accessRange;
  accessRange.reserve(rank);
  for (uint64_t dimension = 0; dimension < rank; ++dimension) {
    accessRange.push_back(
        Range{ranges[dimension * 2], ranges[dimension * 2 + 1]});
  }

  precomputedAccesses = true;
  if (variableRangeAccesses.size() <= static_cast<size_t>(equation)) {
    variableRangeAccesses.resize(equation + 1);
  }
  variableRangeAccesses[equation].emplace_back(variable,
                                               std::move(accessRange));
}

void IDAInstance::setResidualFunction(Equation equation,
                                      ResidualFunction residualFunction) {
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[IDA] Setting residual function for equation " << equation
              << ". Address: " << reinterpret_cast<void *>(residualFunction)
              << std::endl;
  }

  if (residualFunctions.size() <= equation) {
    residualFunctions.resize(equation + 1, nullptr);
  }

  residualFunctions[equation] = residualFunction;
}

void IDAInstance::addJacobianFunction(Equation equation, Variable variable,
                                      JacobianFunction jacobianFunction,
                                      uint64_t numOfSeeds,
                                      uint64_t *seedSizes) {
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[IDA] Setting jacobian function for equation " << equation
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

bool IDAInstance::initialize() {
  // 中文：初始化顺序是 registration contract、execution layout、IDA vectors/
  // callbacks；任何证书失败都发生在可执行 solver 状态被部分创建之前。
  // English: Initialization orders registration contract, execution layout,
  // and IDA vectors/callbacks so certificate failures occur before a partially
  // executable solver state is exposed.
  assert(!initialized && "The IDA instance has already been initialized");

  if (getStage6GRuntimeProbeMode() == Stage6GRuntimeProbeMode::Invalid) {
    std::cerr << "[stage6g-runtime-probe] error=invalid_probe_mode expected="
                 "summary|trace"
              << std::endl;
    return false;
  }

  for (const ConstraintEquationDescriptor &descriptor : constraintEquations) {
    if (constraintComponents.find(descriptor.componentId) ==
            constraintComponents.end() ||
        descriptor.equation >= getNumOfVectorizedEquations()) {
      std::cerr << "[stage6g-runtime-probe] error=invalid_equation_descriptor"
                << std::endl;
      return false;
    }
  }
  for (const ConstraintVariableDescriptor &descriptor : constraintVariables) {
    if (constraintComponents.find(descriptor.componentId) ==
            constraintComponents.end() ||
        descriptor.variable >= getNumOfArrayVariables()) {
      std::cerr << "[stage6g-runtime-probe] error=invalid_variable_descriptor"
                << std::endl;
      return false;
    }
  }
  if (!validateConstraintStateSetRegistrations()) {
    return false;
  }

  const bool runtimeProbe = isRuntimeIDAProbeEnabled();
  auto initializationStart = std::chrono::steady_clock::now();

  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[IDA] Performing initialization" << std::endl;
  }

  // 中文：只有 registration contract 成功后才创建 memory pool、active
  // execution runtimes 和 SUNDIALS objects，避免错误模型留下半初始化资源。
  // English: Create the memory pool, active execution runtimes, and SUNDIALS
  // objects only after the registration contract succeeds, avoiding partially
  // initialized resources for malformed models.
  memoryPoolId = MemoryPoolManager::getInstance().create();
  currentTime = startTime;
  constraintProbeInternalTime = startTime;

  if (!prepareConstraintComponentExecutions()) {
    return false;
  }

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

  if (runtimeProbe) {
    std::cerr << "[ida-runtime-probe] initialize begin scalar_equations="
              << scalarEquationsNumber
              << " scalar_variables=" << scalarVariablesNumber
              << " vectorized_equations=" << getNumOfVectorizedEquations()
              << std::endl;
  }

  if (scalarEquationsNumber == 0) {
    // IDA has nothing to solve.
    initialized = true;
    return true;
  }

  // Create the SUNDIALS context.
#if SUNDIALS_VERSION_MAJOR >= 7
  if (SUNContext_Create(comm, &ctx) != 0) {
    return false;
  }
#elif SUNDIALS_VERSION_MAJOR >= 6
  if (SUNContext_Create(nullptr, &ctx) != 0) {
    return false;
  }
#endif

  // Create and initialize the variables vector.
#if SUNDIALS_VERSION_MAJOR >= 6
  variablesVector =
      N_VNew_Serial(static_cast<sunindextype>(scalarVariablesNumber), ctx);
#else
  variablesVector =
      N_VNew_Serial(static_cast<sunindextype>(scalarVariablesNumber));
#endif

  assert(
      checkAllocation(static_cast<void *>(variablesVector), "N_VNew_Serial"));

  for (uint64_t i = 0; i < scalarVariablesNumber; ++i) {
    N_VGetArrayPointer(variablesVector)[i] = 0;
  }

  // Create and initialize the derivatives vector.
#if SUNDIALS_VERSION_MAJOR >= 6
  derivativesVector =
      N_VNew_Serial(static_cast<sunindextype>(scalarVariablesNumber), ctx);
#else
  derivativesVector =
      N_VNew_Serial(static_cast<sunindextype>(scalarVariablesNumber));
#endif

  assert(
      checkAllocation(static_cast<void *>(derivativesVector), "N_VNew_Serial"));

  for (uint64_t i = 0; i < scalarVariablesNumber; ++i) {
    N_VGetArrayPointer(derivativesVector)[i] = 0;
  }

  // Create and initialize the IDs vector.
#if SUNDIALS_VERSION_MAJOR >= 6
  idVector =
      N_VNew_Serial(static_cast<sunindextype>(scalarVariablesNumber), ctx);
#else
  idVector = N_VNew_Serial(static_cast<sunindextype>(scalarVariablesNumber));
#endif

  assert(checkAllocation(static_cast<void *>(idVector), "N_VNew_Serial"));

  for (Variable var = 0; var < getNumOfArrayVariables(); ++var) {
    if (isLocalVariable(var)) {
      continue;
    }
    VariableKind variableKind = getVariableKind(var);
    uint64_t arrayOffset = variableOffsets[var];
    uint64_t flatSize = getVariableFlatSize(var);

    for (uint64_t scalarOffset = 0; scalarOffset < flatSize; ++scalarOffset) {
      uint64_t offset = arrayOffset + scalarOffset;

      if (variableKind == VariableKind::ALGEBRAIC) {
        N_VGetArrayPointer(idVector)[offset] = 0;
      } else if (variableKind == VariableKind::STATE) {
        N_VGetArrayPointer(idVector)[offset] = 1;
      }
    }
  }

  // Create and initialize the tolerances vector.
#if SUNDIALS_VERSION_MAJOR >= 6
  tolerancesVector =
      N_VNew_Serial(static_cast<sunindextype>(scalarVariablesNumber), ctx);
#else
  tolerancesVector =
      N_VNew_Serial(static_cast<sunindextype>(scalarVariablesNumber));
#endif

  assert(
      checkAllocation(static_cast<void *>(tolerancesVector), "N_VNew_Serial"));

  for (Variable var = 0; var < getNumOfArrayVariables(); ++var) {
    VariableKind variableKind = getVariableKind(var);
    uint64_t arrayOffset = variableOffsets[var];
    uint64_t flatSize = getVariableFlatSize(var);

    for (uint64_t scalarOffset = 0; scalarOffset < flatSize; ++scalarOffset) {
      uint64_t offset = arrayOffset + scalarOffset;

      if (variableKind == VariableKind::ALGEBRAIC) {
        N_VGetArrayPointer(tolerancesVector)[offset] =
            std::min(getOptions().maxAlgebraicAbsoluteTolerance,
                     getOptions().absoluteTolerance);
      } else if (variableKind == VariableKind::STATE) {
        N_VGetArrayPointer(tolerancesVector)[offset] =
            getOptions().absoluteTolerance;
      }
    }
  }

  // Check that all the residual functions have been set.
  assert(residualFunctions.size() == getNumOfVectorizedEquations());

  assert(std::all_of(
      residualFunctions.begin(), residualFunctions.end(),
      [](const ResidualFunction &function) { return function != nullptr; }));

  // Check if the IDA instance is not informed about the accesses that all
  // the jacobian functions have been set.
  assert(precomputedAccesses ||
         jacobianFunctions.size() == getNumOfVectorizedEquations());

  assert(precomputedAccesses ||
         std::all_of(jacobianFunctions.begin(), jacobianFunctions.end(),
                     [&](std::vector<JacobianFunctionDescriptor> functions) {
                       if (functions.size() !=
                           algebraicAndStateVariablesGetters.size()) {
                         return false;
                       }

                       return std::all_of(
                           functions.begin(), functions.end(),
                           [](const JacobianFunctionDescriptor &function) {
                             return function.first != nullptr;
                           });
                     }));

  // Check that all the getters and setters have been set.
  assert(
      std::none_of(algebraicAndStateVariablesGetters.begin(),
                   algebraicAndStateVariablesGetters.end(),
                   [](VariableGetter getter) { return getter == nullptr; }) &&
      "Not all the variable getters have been set");

  assert(
      std::none_of(algebraicAndStateVariablesSetters.begin(),
                   algebraicAndStateVariablesSetters.end(),
                   [](VariableSetter setter) { return setter == nullptr; }) &&
      "Not all the variable setters have been set");

  assert(
      std::none_of(derivativeVariablesGetters.begin(),
                   derivativeVariablesGetters.end(),
                   [](VariableGetter getter) { return getter == nullptr; }) &&
      "Not all the derivative getters have been set");

  assert(
      std::none_of(derivativeVariablesSetters.begin(),
                   derivativeVariablesSetters.end(),
                   [](VariableSetter setter) { return setter == nullptr; }) &&
      "Not all the derivative setters have been set");

  // Reserve the space for data of the jacobian matrix.
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[IDA] Reserving space for the data of the Jacobian matrix"
              << std::endl;
  }

  jacobianMatrixData.resize(scalarEquationsNumber);

  uint64_t numOfVectorizedEquations = getNumOfVectorizedEquations();
  uint64_t reservedScalarEquations = 0;
  uint64_t reservedJacobianColumns = 0;
  auto reservationStart = std::chrono::steady_clock::now();

  for (Equation eq = 0; eq < numOfVectorizedEquations; ++eq) {
    if (isLocalEquation(eq)) {
      continue;
    }

    std::vector<int64_t> equationIndices;
    getEquationBeginIndices(eq, equationIndices);
    uint64_t equationScalarEquations = 0;
    uint64_t equationJacobianColumns = 0;

    do {
      uint64_t equationArrayOffset = equationOffsets[eq];

      uint64_t equationScalarOffset =
          getEquationFlatIndex(equationIndices, equationRanges[eq]);

      uint64_t scalarEquationIndex = equationArrayOffset + equationScalarOffset;

      // Compute the column indexes that may be non-zeros.
      std::vector<JacobianColumn> jacobianColumns =
          computeJacobianColumns(eq, equationIndices.data());

      jacobianMatrixData[scalarEquationIndex].resize(jacobianColumns.size());
      ++reservedScalarEquations;
      ++equationScalarEquations;
      reservedJacobianColumns += jacobianColumns.size();
      equationJacobianColumns += jacobianColumns.size();

      // 运行时探针只在显式启用时打印，用于区分初始化建图和求解回调耗时。
      if (runtimeProbe && reservedScalarEquations % 1024 == 0) {
        std::cerr << "[ida-runtime-probe] reserve progress scalar_equations="
                  << reservedScalarEquations
                  << " jacobian_columns=" << reservedJacobianColumns
                  << " elapsed_ms="
                  << elapsedMillisecondsSince(reservationStart) << std::endl;
      }

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

    if (runtimeProbe) {
      std::cerr << "[ida-runtime-probe] reserve equation_done equation=" << eq
                << " scalar_equations=" << equationScalarEquations
                << " jacobian_columns=" << equationJacobianColumns
                << " total_elapsed_ms="
                << elapsedMillisecondsSince(reservationStart) << std::endl;
    }
  }

  if (runtimeProbe) {
    std::cerr << "[ida-runtime-probe] reserve done scalar_equations="
              << reservedScalarEquations
              << " jacobian_columns=" << reservedJacobianColumns
              << " elapsed_ms=" << elapsedMillisecondsSince(reservationStart)
              << std::endl;
  }

  // Compute the total amount of non-zero values in the Jacobian Matrix.
  computeNNZ();

  if (runtimeProbe) {
    std::cerr << "[ida-runtime-probe] computeNNZ done non_zero_values="
              << nonZeroValuesNumber
              << " elapsed_ms=" << elapsedMillisecondsSince(initializationStart)
              << std::endl;
  }

  // Compute the workload for each thread.
  computeThreadChunks();

  if (runtimeProbe) {
    std::cerr << "[ida-runtime-probe] computeThreadChunks done chunks="
              << threadEquationsChunks.size()
              << " elapsed_ms=" << elapsedMillisecondsSince(initializationStart)
              << std::endl;
  }

  // Initialize the values of the variables living inside IDA.
  copyVariablesFromMARCO(variablesVector, derivativesVector);

  if (runtimeProbe) {
    std::cerr << "[ida-runtime-probe] copyVariablesFromMARCO done elapsed_ms="
              << elapsedMillisecondsSince(initializationStart) << std::endl;
  }

  // Create and initialize the memory for IDA.
#if SUNDIALS_VERSION_MAJOR >= 6
  idaMemory = IDACreate(ctx);
#else
  idaMemory = IDACreate();
#endif

  if (!checkAllocation(idaMemory, "IDACreate")) {
    return false;
  }

  if (IDASetErrHandlerFn(idaMemory, IDAInstance::idaErrorHandler, this) !=
      IDA_SUCCESS) {
    return false;
  }

  if (!idaInit()) {
    return false;
  }

  if (!idaSVTolerances()) {
    return false;
  }

  // Create sparse SUNMatrix for use in linear solver.
#if SUNDIALS_VERSION_MAJOR >= 6
  sparseMatrix = SUNSparseMatrix(
      static_cast<sunindextype>(scalarEquationsNumber),
      static_cast<sunindextype>(scalarEquationsNumber),
      static_cast<sunindextype>(nonZeroValuesNumber), CSR_MAT, ctx);
#else
  sparseMatrix =
      SUNSparseMatrix(static_cast<sunindextype>(scalarEquationsNumber),
                      static_cast<sunindextype>(scalarEquationsNumber),
                      static_cast<sunindextype>(nonZeroValuesNumber), CSR_MAT);
#endif

  if (!checkAllocation(static_cast<void *>(sparseMatrix), "SUNSparseMatrix")) {
    return false;
  }

  // Create and attach a KLU SUNLinearSolver object.
#if SUNDIALS_VERSION_MAJOR >= 6
  linearSolver = SUNLinSol_KLU(variablesVector, sparseMatrix, ctx);
#else
  linearSolver = SUNLinSol_KLU(variablesVector, sparseMatrix);
#endif

  if (!checkAllocation(static_cast<void *>(linearSolver), "SUNLinSol_KLU")) {
    return false;
  }

  if (!idaSetLinearSolver()) {
    return false;
  }

  if (!idaSetUserData() || !idaSetMaxNumSteps() || !idaSetInitialStepSize() ||
      !idaSetMinStepSize() || !idaSetMaxStepSize() ||
      !idaSetMaxErrTestFails() || !idaSetSuppressAlg() || !idaSetId() ||
      !idaSetJacobianFunction() || !idaSetMaxNonlinIters() ||
      !idaSetMaxConvFails() || !idaSetNonlinConvCoef() ||
      !idaSetNonlinConvCoefIC() || !idaSetMaxNumStepsIC() ||
      !idaSetMaxNumJacsIC() || !idaSetMaxNumItersIC() ||
      !idaSetLineSearchOffIC()) {
    return false;
  }

  initialized = true;

  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[IDA] Initialization completed" << std::endl;
  }

  if (runtimeProbe) {
    std::cerr << "[ida-runtime-probe] initialize done elapsed_ms="
              << elapsedMillisecondsSince(initializationStart) << std::endl;
  }

  return true;
}

bool IDAInstance::calcIC() {
  if (!initialized) {
    if (!initialize()) {
      return false;
    }
  }

  if (getNumOfScalarEquations() == 0) {
    // IDA has nothing to solve
    return true;
  }

  realtype firstOutTime =
      (endTime - startTime) / getOptions().timeScalingFactorInit;

  const bool runtimeProbe = isRuntimeIDAProbeEnabled();
  auto calcICStart = std::chrono::steady_clock::now();

  if (runtimeProbe) {
    std::cerr << "[ida-runtime-probe] calcIC begin first_out_time="
              << firstOutTime << std::endl;
  }

  IDA_PROFILER_IC_START
  auto calcICRetVal = IDACalcIC(idaMemory, IDA_YA_YDP_INIT, firstOutTime);
  IDA_PROFILER_IC_STOP

  if (runtimeProbe) {
    std::cerr << "[ida-runtime-probe] calcIC done ret=" << calcICRetVal
              << " elapsed_ms=" << elapsedMillisecondsSince(calcICStart)
              << std::endl;
  }

  if (calcICRetVal != IDA_SUCCESS) {
    if (calcICRetVal == IDALS_MEM_NULL) {
      std::cerr << "IDACalcIC - The ida_mem pointer is NULL" << std::endl;
    } else if (calcICRetVal == IDA_NO_MALLOC) {
      std::cerr
          << "IDACalcIC - The allocation function IDAInit has not been called"
          << std::endl;
    } else if (calcICRetVal == IDA_ILL_INPUT) {
      std::cerr << "IDACalcIC - One of the input arguments was illegal"
                << std::endl;
    } else if (calcICRetVal == IDA_LSETUP_FAIL) {
      std::cerr << "IDACalcIC - The linear solver’s setup function failed in "
                   "an unrecoverable manner"
                << std::endl;
    } else if (calcICRetVal == IDA_LINIT_FAIL) {
      std::cerr
          << "IDACalcIC - The linear solver’s initialization function failed"
          << std::endl;
    } else if (calcICRetVal == IDA_LSOLVE_FAIL) {
      std::cerr << "IDACalcIC - The linear solver’s solve function failed in "
                   "an unrecoverable manner"
                << std::endl;
    } else if (calcICRetVal == IDA_BAD_EWT) {
      std::cerr
          << "IDACalcIC - Some component of the error weight vector is zero "
             "(illegal), either for the input value of y0 or a corrected value"
          << std::endl;
    } else if (calcICRetVal == IDA_FIRST_RES_FAIL) {
      std::cerr
          << "IDACalcIC - The user’s residual function returned a recoverable "
             "error flag on the first call, but IDACalcIC was unable to recover"
          << std::endl;
    } else if (calcICRetVal == IDA_RES_FAIL) {
      std::cerr << "IDACalcIC - The user’s residual function returned a "
                   "nonrecoverable error flag"
                << std::endl;
    } else if (calcICRetVal == IDA_NO_RECOVERY) {
      std::cerr << "IDACalcIC - The user’s residual function, or the linear "
                   "solver’s setup or solve function had a recoverable error, "
                   "but IDACalcIC was unable to recover"
                << std::endl;
    } else if (calcICRetVal == IDA_CONSTR_FAIL) {
      std::cerr << "IDACalcIC - IDACalcIC was unable to find a solution "
                   "satisfying the inequality constraints"
                << std::endl;
    } else if (calcICRetVal == IDA_LINESEARCH_FAIL) {
      std::cerr << "IDACalcIC - The linesearch algorithm failed to find a "
                   "solution with a step larger than steptol in weighted RMS "
                   "norm, and within the allowed number of backtracks"
                << std::endl;
    } else if (calcICRetVal == IDA_CONV_FAIL) {
      std::cerr << "IDACalcIC - IDACalcIC failed to get convergence of the "
                   "Newton iterations"
                << std::endl;
    }

    return false;
  }

  // 中文：IDA 的 consistent outer y/y' 写回模型后必须再次求解 local closure；
  // 初始 accepted checkpoint 因此包含完整 C2 state，而非仅 outer vector。
  // English: After IDA's consistent outer y/y' is copied to the model, solve
  // local closure once more. The initial accepted checkpoint therefore stores
  // the complete C2 state rather than only the outer vector.
  auto getConsistentIcRetVal =
      IDAGetConsistentIC(idaMemory, variablesVector, derivativesVector);

  if (getConsistentIcRetVal != IDA_SUCCESS) {
    if (getConsistentIcRetVal == IDA_ILL_INPUT) {
      std::cerr << "IDAGetConsistentIC - Called before the first IDASolve"
                << std::endl;
    } else if (getConsistentIcRetVal == IDA_MEM_NULL) {
      std::cerr << "IDAGetConsistentIC - The ida_mem pointer is NULL"
                << std::endl;
    }

    return false;
  }

  copyVariablesIntoMARCO(variablesVector, derivativesVector);
  if (!refreshConstraintComponents(startTime)) {
    return false;
  }
  saveAcceptedConstraintCheckpoint(startTime);
  return true;
}

bool IDAInstance::recordConstraintProbeSample(const char *sampleKind,
                                              realtype time,
                                              N_Vector variables,
                                              N_Vector derivatives) {
  if (constraintComponents.empty()) {
    return true;
  }

  // 中文：probe 用当前状态重新计算 fresh residual，并与 IDA 最近 nonlinear
  // residual、error weights 和 estimated local errors 并列报告；不复用陈旧值。
  // English: The probe recomputes fresh residuals at the current state and
  // reports them alongside IDA's latest nonlinear residual, error weights, and
  // estimated local errors; stale values are never reused as evidence.
  N_Vector freshResiduals = N_VClone(variablesVector);
  N_Vector errorWeights = N_VClone(variablesVector);
  N_Vector estimatedErrors = N_VClone(variablesVector);
  if (!freshResiduals || !errorWeights || !estimatedErrors) {
    std::cerr << "[stage6g-runtime-probe] error=vector_allocation_failed"
              << std::endl;
    if (freshResiduals) {
      N_VDestroy(freshResiduals);
    }
    if (errorWeights) {
      N_VDestroy(errorWeights);
    }
    if (estimatedErrors) {
      N_VDestroy(estimatedErrors);
    }
    return false;
  }

  N_VConst(0, freshResiduals);
  if (residualFunction(time, variables, derivatives, freshResiduals, this) !=
      IDA_SUCCESS) {
    N_VDestroy(freshResiduals);
    N_VDestroy(errorWeights);
    N_VDestroy(estimatedErrors);
    return false;
  }

  realtype nonlinearTime = 0;
  realtype cj = 0;
  N_Vector predictedVariables = nullptr;
  N_Vector predictedDerivatives = nullptr;
  N_Vector nonlinearVariables = nullptr;
  N_Vector nonlinearDerivatives = nullptr;
  N_Vector nonlinearResiduals = nullptr;
  void *userData = nullptr;
  int nonlinearDataResult = IDAGetNonlinearSystemData(
      idaMemory, &nonlinearTime, &predictedVariables, &predictedDerivatives,
      &nonlinearVariables, &nonlinearDerivatives, &nonlinearResiduals, &cj,
      &userData);

  int order = 0;
  realtype stepSize = 0;
  if (nonlinearDataResult != IDA_SUCCESS ||
      IDAGetCurrentOrder(idaMemory, &order) != IDA_SUCCESS ||
      IDAGetCurrentStep(idaMemory, &stepSize) != IDA_SUCCESS ||
      IDAGetCurrentCj(idaMemory, &cj) != IDA_SUCCESS ||
      IDAGetErrWeights(idaMemory, errorWeights) != IDA_SUCCESS ||
      IDAGetEstLocalErrors(idaMemory, estimatedErrors) != IDA_SUCCESS) {
    std::cerr << "[stage6g-runtime-probe] error=sundials_probe_query_failed"
              << std::endl;
    N_VDestroy(freshResiduals);
    N_VDestroy(errorWeights);
    N_VDestroy(estimatedErrors);
    return false;
  }

  long int nonlinearIterations = 0;
  long int convergenceFailures = 0;
  long int errorTestFailures = 0;
  if (IDAGetNonlinSolvStats(idaMemory, &nonlinearIterations,
                            &convergenceFailures) != IDA_SUCCESS ||
      IDAGetNumErrTestFails(idaMemory, &errorTestFailures) != IDA_SUCCESS) {
    std::cerr << "[stage6g-runtime-probe] error=sundials_stats_query_failed"
              << std::endl;
    N_VDestroy(freshResiduals);
    N_VDestroy(errorWeights);
    N_VDestroy(estimatedErrors);
    return false;
  }

  long int nonlinearDelta =
      nonlinearIterations - constraintProbePreviousNonlinearIterations;
  long int errorTestDelta =
      errorTestFailures - constraintProbePreviousErrorTestFailures;
  long int convergenceDelta =
      convergenceFailures - constraintProbePreviousConvergenceFailures;
  constraintProbePreviousNonlinearIterations = nonlinearIterations;
  constraintProbePreviousErrorTestFailures = errorTestFailures;
  constraintProbePreviousConvergenceFailures = convergenceFailures;

  if (jacobianMatrix(time, cj, variables, derivatives, freshResiduals,
                     sparseMatrix, this, nullptr, nullptr, nullptr) !=
      IDA_SUCCESS) {
    N_VDestroy(freshResiduals);
    N_VDestroy(errorWeights);
    N_VDestroy(estimatedErrors);
    return false;
  }

  const realtype *freshData = N_VGetArrayPointer(freshResiduals);
  const realtype *nonlinearData = N_VGetArrayPointer(nonlinearResiduals);
  const realtype *weightData = N_VGetArrayPointer(errorWeights);
  const realtype *errorData = N_VGetArrayPointer(estimatedErrors);
  const realtype *idData = N_VGetArrayPointer(idVector);

  double stateErrorSum = 0;
  double algebraicErrorSum = 0;
  uint64_t stateErrorCount = 0;
  uint64_t algebraicErrorCount = 0;
  double stateErrorMax = 0;
  double algebraicErrorMax = 0;
  for (uint64_t i = 0; i < getNumOfScalarVariables(); ++i) {
    double scaledError = std::abs(errorData[i] * weightData[i]);
    if (idData[i] == 1) {
      stateErrorSum += scaledError * scaledError;
      stateErrorMax = std::max(stateErrorMax, scaledError);
      ++stateErrorCount;
    } else {
      algebraicErrorSum += scaledError * scaledError;
      algebraicErrorMax = std::max(algebraicErrorMax, scaledError);
      ++algebraicErrorCount;
    }
  }
  double stateErrorWrms =
      stateErrorCount ? std::sqrt(stateErrorSum / stateErrorCount) : 0;
  double algebraicErrorWrms = algebraicErrorCount
                                  ? std::sqrt(algebraicErrorSum /
                                              algebraicErrorCount)
                                  : 0;

  auto isFinite = [](double value) { return std::isfinite(value); };
  bool trace = getStage6GRuntimeProbeMode() ==
               Stage6GRuntimeProbeMode::Trace;

  for (const auto &[componentId, componentDescriptor] :
       constraintComponents) {
    double positionRaw = 0;
    double tangentRaw = 0;
    double highestRaw = 0;
    double supportRaw = 0;
    double positionNonlinear = 0;
    double tangentNonlinear = 0;
    double highestNonlinear = 0;
    double supportNonlinear = 0;
    std::vector<uint64_t> conditionRows;
    std::vector<std::pair<uint64_t, uint64_t>> conditionRowColumns;

    for (const ConstraintEquationDescriptor &descriptor :
         constraintEquations) {
      if (descriptor.componentId != componentId ||
          descriptor.equation >= equationOffsets.size() - 1) {
        continue;
      }
      uint64_t begin = equationOffsets[descriptor.equation];
      uint64_t end = equationOffsets[descriptor.equation + 1];
      bool conditionEquation = descriptor.role == "position" ||
                               descriptor.role == "tangent";
      uint64_t matchedColumnBegin = 0;
      uint64_t matchedColumnEnd = 0;
      bool matchedVariableFound = false;
      if (conditionEquation) {
        for (const ConstraintVariableDescriptor &variableDescriptor :
             constraintVariables) {
          if (variableDescriptor.componentId != componentId ||
              variableDescriptor.role != "dependent" ||
              variableDescriptor.name != descriptor.matchedVariableName ||
              variableDescriptor.variable >= variableOffsets.size() - 1) {
            continue;
          }
          matchedColumnBegin = variableOffsets[variableDescriptor.variable];
          matchedColumnEnd =
              variableOffsets[variableDescriptor.variable + 1];
          matchedVariableFound = true;
          break;
        }
        if (!matchedVariableFound ||
            end - begin != matchedColumnEnd - matchedColumnBegin) {
          std::cerr
              << "[stage6g-runtime-probe] error=matched_variable_ownership_"
                 "mismatch component="
              << componentId << " equation=" << descriptor.equation
              << " matched_variable=" << descriptor.matchedVariableName
              << std::endl;
          N_VDestroy(freshResiduals);
          N_VDestroy(errorWeights);
          N_VDestroy(estimatedErrors);
          return false;
        }
      }
      for (uint64_t row = begin; row < end; ++row) {
        double raw = std::abs(freshData[row]);
        double nonlinear = std::abs(nonlinearData[row]);
        if (descriptor.role == "position") {
          positionRaw = std::max(positionRaw, raw);
          positionNonlinear = std::max(positionNonlinear, nonlinear);
          conditionRows.push_back(row);
          conditionRowColumns.emplace_back(
              row, matchedColumnBegin + (row - begin));
        } else if (descriptor.role == "tangent") {
          tangentRaw = std::max(tangentRaw, raw);
          tangentNonlinear = std::max(tangentNonlinear, nonlinear);
          conditionRows.push_back(row);
          conditionRowColumns.emplace_back(
              row, matchedColumnBegin + (row - begin));
        } else if (descriptor.role == "highest") {
          highestRaw = std::max(highestRaw, raw);
          highestNonlinear = std::max(highestNonlinear, nonlinear);
        } else {
          supportRaw = std::max(supportRaw, raw);
          supportNonlinear = std::max(supportNonlinear, nonlinear);
        }
      }
    }

    std::set<uint64_t> dependentColumns;
    for (const ConstraintVariableDescriptor &descriptor :
         constraintVariables) {
      if (descriptor.componentId != componentId ||
          descriptor.role != "dependent" ||
          descriptor.variable >= variableOffsets.size() - 1) {
        continue;
      }
      uint64_t begin = variableOffsets[descriptor.variable];
      uint64_t end = variableOffsets[descriptor.variable + 1];
      for (uint64_t column = begin; column < end; ++column) {
        bool structurallyOwned = false;
        for (uint64_t row : conditionRows) {
          for (const auto &[candidate, value] : jacobianMatrixData[row]) {
            (void)value;
            structurallyOwned |=
                static_cast<uint64_t>(candidate) == column;
          }
        }
        if (structurallyOwned) {
          dependentColumns.insert(column);
        }
      }
    }

    double scaledResidualSum = 0;
    uint64_t scaledResidualCount = 0;
    for (const auto &[row, column] : conditionRowColumns) {
      double scaled = freshData[row] * weightData[column];
      scaledResidualSum += scaled * scaled;
      ++scaledResidualCount;
    }
    double scaledResidualWrms =
        scaledResidualCount
            ? std::sqrt(scaledResidualSum / scaledResidualCount)
            : 0;

    ConstraintConditionEstimate condition;
    bool conditionFromExecutionCertificate = false;
    double localClosureCondition = 0;
    double responseAmplification = 0;
    for (const auto &execution : constraintComponentExecutions) {
      if (execution->semanticComponentIds.count(componentId) == 0) {
        continue;
      }
      condition = execution->lastHealthCertificate.chart;
      localClosureCondition =
          execution->lastHealthCertificate.localClosure.oneNormCondition;
      responseAmplification =
          execution->lastHealthCertificate.responseAmplification;
      conditionFromExecutionCertificate = true;
      break;
    }
    if (!conditionFromExecutionCertificate &&
        conditionRows.size() <= 64 &&
        conditionRows.size() == dependentColumns.size()) {
      std::vector<double> matrix(conditionRows.size() *
                                 dependentColumns.size());
      for (uint64_t rowIndex = 0; rowIndex < conditionRows.size();
           ++rowIndex) {
        uint64_t columnIndex = 0;
        for (uint64_t column : dependentColumns) {
          for (const auto &[candidate, value] :
               jacobianMatrixData[conditionRows[rowIndex]]) {
            if (static_cast<uint64_t>(candidate) == column) {
              matrix[rowIndex * dependentColumns.size() + columnIndex] =
                  value;
              break;
            }
          }
          ++columnIndex;
        }
      }
      condition = estimateConstraintCondition(matrix, conditionRows.size());
    } else {
      if (!conditionFromExecutionCertificate) {
        condition.rank = 0;
        condition.rankDeficient = true;
        condition.oneNormCondition =
            std::numeric_limits<double>::infinity();
      }
    }

    if (!isFinite(time) || !isFinite(positionRaw) ||
        !isFinite(tangentRaw) || !isFinite(highestRaw) ||
        !isFinite(supportRaw) || !isFinite(scaledResidualWrms) ||
        !isFinite(stateErrorWrms) || !isFinite(algebraicErrorWrms)) {
      std::cerr << "[stage6g-runtime-probe] error=non_finite_sample"
                << " component=" << componentId << " time=" << time
                << std::endl;
      N_VDestroy(freshResiduals);
      N_VDestroy(errorWeights);
      N_VDestroy(estimatedErrors);
      return false;
    }

    ConstraintProbeSummary &summary = constraintProbeSummaries[componentId];
    if (std::string(sampleKind) == "accepted") {
      ++summary.acceptedSamples;
    } else {
      ++summary.outputSamples;
    }
    if (positionRaw > summary.maximumPositionResidual) {
      summary.maximumPositionResidual = positionRaw;
      summary.maximumPositionTime = time;
    }
    if (tangentRaw > summary.maximumTangentResidual) {
      summary.maximumTangentResidual = tangentRaw;
      summary.maximumTangentTime = time;
    }
    summary.maximumHighestResidual =
        std::max(summary.maximumHighestResidual, highestRaw);
    summary.maximumSupportResidual =
        std::max(summary.maximumSupportResidual, supportRaw);
    if (std::isfinite(condition.oneNormCondition) &&
        condition.oneNormCondition > summary.maximumCondition) {
      summary.maximumCondition = condition.oneNormCondition;
      summary.maximumConditionTime = time;
    }

    if (trace) {
      std::cerr << std::setprecision(17)
                << "[stage6g-runtime-probe] sample=" << sampleKind
                << " component=" << componentId << " time=" << time
                << " nonlinear_time=" << nonlinearTime
                << " step_size=" << stepSize << " order=" << order
                << " cj=" << cj
                << " position_owned_scalars="
                << componentDescriptor.positionScalars
                << " position_residual_scalars="
                << componentDescriptor.residualPositionScalars
                << " position_raw_max=" << positionRaw
                << " position_last_nonlinear_max=" << positionNonlinear
                << " tangent_owned_scalars="
                << componentDescriptor.tangentScalars
                << " tangent_residual_scalars="
                << componentDescriptor.residualTangentScalars
                << " tangent_raw_max=" << tangentRaw
                << " tangent_last_nonlinear_max=" << tangentNonlinear
                << " highest_owned_scalars="
                << componentDescriptor.highestScalars
                << " highest_residual_scalars="
                << componentDescriptor.residualHighestScalars
                << " highest_raw_max=" << highestRaw
                << " highest_last_nonlinear_max=" << highestNonlinear
                << " support_owned_scalars="
                << componentDescriptor.supportScalars
                << " support_residual_scalars="
                << componentDescriptor.residualSupportScalars
                << " support_raw_max=" << supportRaw
                << " support_last_nonlinear_max=" << supportNonlinear
                << " matched_weighted_wrms=" << scaledResidualWrms
                << " state_local_error_wrms=" << stateErrorWrms
                << " state_local_error_max=" << stateErrorMax
                << " algebraic_local_error_wrms=" << algebraicErrorWrms
                << " algebraic_local_error_max=" << algebraicErrorMax
                << " nonlinear_iterations_delta=" << nonlinearDelta
                << " error_test_failures_delta=" << errorTestDelta
                << " convergence_failures_delta=" << convergenceDelta
                << " suppress_algebraic="
                << (effectiveSuppressAlgebraicErrorTest == SUNTRUE ? 1 : 0)
                << " suppress_source="
                << effectiveSuppressAlgebraicErrorTestSource
                << " condition_rows=" << conditionRows.size()
                << " condition_columns=" << dependentColumns.size()
                << " condition_rank=" << condition.rank
                << " condition_min_pivot=" << condition.minimumPivot
                << " condition_max_pivot=" << condition.maximumPivot
                << " condition_estimate=" << condition.oneNormCondition
                << " condition_scaled=" << (condition.scaled ? 1 : 0)
                << " condition_estimator="
                << (condition.sparseEstimator ? "sparse_klu" : "dense_lu")
                << " local_closure_condition=" << localClosureCondition
                << " response_amplification=" << responseAmplification
                << " condition_warning="
                << (condition.oneNormCondition >= 1e8 ? 1 : 0)
                << " rank_deficient=" << (condition.rankDeficient ? 1 : 0)
                << std::endl;
    }
  }

  N_VDestroy(freshResiduals);
  N_VDestroy(errorWeights);
  N_VDestroy(estimatedErrors);
  return true;
}

void IDAInstance::printConstraintProbeSummary() const {
  for (const auto &[componentId, summary] : constraintProbeSummaries) {
    auto componentIt = constraintComponents.find(componentId);
    if (componentIt == constraintComponents.end()) {
      continue;
    }
    const ConstraintComponentDescriptor &component = componentIt->second;
    std::cerr << std::setprecision(17)
              << "[stage6g-runtime-probe] summary component=" << componentId
              << " accepted_samples=" << summary.acceptedSamples
              << " output_samples=" << summary.outputSamples
              << " position_owned_scalars=" << component.positionScalars
              << " position_residual_scalars="
              << component.residualPositionScalars
              << " position_raw_peak=" << summary.maximumPositionResidual
              << " position_peak_time=" << summary.maximumPositionTime
              << " tangent_owned_scalars=" << component.tangentScalars
              << " tangent_residual_scalars="
              << component.residualTangentScalars
              << " tangent_raw_peak=" << summary.maximumTangentResidual
              << " tangent_peak_time=" << summary.maximumTangentTime
              << " highest_owned_scalars=" << component.highestScalars
              << " highest_residual_scalars="
              << component.residualHighestScalars
              << " highest_raw_peak=" << summary.maximumHighestResidual
              << " support_owned_scalars=" << component.supportScalars
              << " support_residual_scalars="
              << component.residualSupportScalars
              << " support_raw_peak=" << summary.maximumSupportResidual
              << " condition_peak=" << summary.maximumCondition
              << " condition_peak_time=" << summary.maximumConditionTime
              << " suppress_algebraic="
              << (effectiveSuppressAlgebraicErrorTest == SUNTRUE ? 1 : 0)
              << " suppress_source="
              << effectiveSuppressAlgebraicErrorTestSource << std::endl;
  }

  for (const auto &execution : constraintComponentExecutions) {
    std::cerr << std::setprecision(17)
              << "[stage6g-runtime-probe] summary execution_group="
              << execution->executionGroupId
              << " semantic_components=";
    bool firstSemanticComponent = true;
    for (uint64_t semanticComponent : execution->semanticComponentIds) {
      if (!firstSemanticComponent) {
        std::cerr << ",";
      }
      firstSemanticComponent = false;
      std::cerr << semanticComponent;
    }
    std::cerr
              << " local_scalars=" << execution->localScalars
              << " local_solve_attempts=" << execution->localSolveAttempts
              << " local_solve_successes=" << execution->localSolveSuccesses
              << " local_solve_recoverable_failures="
              << execution->localSolveRecoverableFailures
              << " local_solve_fatal_failures="
              << execution->localSolveFatalFailures
              << " local_newton_iterations="
              << execution->localNonlinearIterations
              << " local_residual_peak="
              << execution->maximumLocalResidual
              << " local_residual_peak_time="
              << execution->maximumLocalResidualTime
              << " condition_peak=" << execution->maximumCondition
              << " condition_peak_time="
              << execution->maximumConditionTime
              << " chart_condition_peak="
              << execution->maximumChartCondition
              << " local_closure_condition_peak="
              << execution->maximumLocalClosureCondition
              << " response_amplification_peak="
              << execution->maximumResponseAmplification
              << " response_amplification_peak_time="
              << execution->maximumResponseAmplificationTime
              << " explicit_nominal_scalars="
              << execution->solver->getExplicitNominalScalars()
              << " anchor_nominal_scalars="
              << execution->solver->getAnchorNominalScalars()
              << " jzz_rows=" << execution->localScalars
              << " gs_columns=" << execution->outerCouplingColumns.size()
              << " rz_equation_blocks=" << execution->outerEquations.size()
              << " schur_factorizations="
              << execution->schurFactorizations
              << " schur_rhs_solves=" << execution->schurRightHandSides
              << " schur_fill_entries=" << execution->schurFillEntries
              << std::endl;
  }
}

bool IDAInstance::step() {
  // 中文：无可切换 component 时保留 IDA_NORMAL 快路径；需要 probe 或状态切换时
  // 使用 IDA_ONE_STEP 保存 accepted checkpoints，并用 IDAGetDky 恢复原输出网格。
  // English: Models without switchable components retain the IDA_NORMAL fast
  // path. Probe/switch runs use IDA_ONE_STEP for accepted checkpoints and
  // IDAGetDky to preserve the original output grid.
  if (!initialized) {
    if (!initialize()) {
      return false;
    }
  }

  if (getNumOfScalarEquations() == 0) {
    // IDA has nothing to solve. Just increment the time.

    if (getOptions().equidistantTimeGrid) {
      currentTime += timeStep;
    } else {
      currentTime = endTime;
    }

    return true;
  }

  // Execute one step.
  expectedConstraintFailure = false;
  IDA_PROFILER_STEPS_COUNTER_INCREMENT
  IDA_PROFILER_STEP_START

  ++stepsNumber;

  realtype tout =
      getOptions().equidistantTimeGrid ? (stepsNumber * timeStep) : endTime;

  int solveRetVal = IDA_SUCCESS;
  bool constraintProbe = isStage6GRuntimeProbeEnabled() &&
                         !constraintComponents.empty();
  bool stateSetStepping =
      !constraintStateSets.empty() || !constraintExecutionEpochs.empty();
  if (!constraintProbe && !stateSetStepping) {
    solveRetVal = IDASolve(idaMemory, tout, &currentTime, variablesVector,
                           derivativesVector, IDA_NORMAL);
  } else {
    realtype internalTime =
        constraintProbeInternalTime.value_or(currentTime);
    std::vector<std::pair<JacobianColumn, double>> outputLocalWarmStart;
    while (internalTime < tout) {
      realtype previousTime = internalTime;
      outputLocalWarmStart.clear();
      for (const auto &execution : constraintComponentExecutions) {
        for (const JacobianColumn &column : execution->localScalarColumns) {
          outputLocalWarmStart.emplace_back(
              column,
              algebraicAndStateVariablesGetters[column.first](
                  column.second.data()));
        }
      }
      solveRetVal = IDASolve(idaMemory, tout, &internalTime, variablesVector,
                             derivativesVector, IDA_ONE_STEP);
      if (solveRetVal != IDA_SUCCESS && solveRetVal != IDA_TSTOP_RETURN &&
          solveRetVal != IDA_ROOT_RETURN) {
        if (pendingBasisSwitch && performPendingBasisSwitch()) {
          internalTime = currentTime;
          solveRetVal = IDA_SUCCESS;
          continue;
        }
        break;
      }
      if (verifyFirstOrderAfterReInit) {
        int currentOrder = 0;
        if (IDAGetCurrentOrder(idaMemory, &currentOrder) != IDA_SUCCESS ||
            currentOrder != 1) {
          std::cerr << "stage6gc_ida_reinit_order_mismatch expected=1 actual="
                    << currentOrder << std::endl;
          return false;
        }
        if (isStage6GRuntimeProbeEnabled()) {
          std::cerr << "[stage6g-runtime-probe] basis_epoch=" << basisEpoch
                    << " event=first_step_after_reinit order=" << currentOrder
                    << " time=" << internalTime << std::endl;
        }
        verifyFirstOrderAfterReInit = false;
      }
      if (!advanceIDAReInitRecovery(internalTime)) {
        return false;
      }
      copyVariablesIntoMARCO(variablesVector, derivativesVector);
      if (!refreshConstraintComponents(internalTime)) {
        if (pendingBasisSwitch && performPendingBasisSwitch()) {
          internalTime = currentTime;
          solveRetVal = IDA_SUCCESS;
          continue;
        }
        return false;
      }
      saveAcceptedConstraintCheckpoint(internalTime);
      if (constraintProbe &&
          !recordConstraintProbeSample("accepted", internalTime,
                                       variablesVector,
                                       derivativesVector)) {
        return false;
      }
      if (internalTime <= previousTime || solveRetVal == IDA_TSTOP_RETURN ||
          solveRetVal == IDA_ROOT_RETURN) {
        break;
      }
    }

    constraintProbeInternalTime = internalTime;
    if (solveRetVal == IDA_SUCCESS && internalTime >= tout) {
      if (IDAGetDky(idaMemory, tout, 0, variablesVector) != IDA_SUCCESS ||
          IDAGetDky(idaMemory, tout, 1, derivativesVector) != IDA_SUCCESS) {
        std::cerr << "[stage6g-runtime-probe] error=output_interpolation_failed"
                  << " output_time=" << tout
                  << " internal_time=" << internalTime << std::endl;
        return false;
      }
      currentTime = tout;
      // 中文：IDA 只插值 outer state；输出 callback 读取共享模型数组前，必须
      // 在同一插值状态重建 component-local variables。probe 也会算 residual，
      // 但数学输出契约不能依赖诊断是否开启。
      // English: IDA interpolates only outer state. Reconstruct component-local
      // variables at that interpolated state before output callbacks read the
      // shared arrays. Probe sampling also evaluates residuals, but the
      // mathematical output contract must not depend on diagnostics.
      copyVariablesIntoMARCO(variablesVector, derivativesVector);
      for (const auto &[column, value] : outputLocalWarmStart) {
        algebraicAndStateVariablesSetters[column.first](
            value, column.second.data());
      }
      if (!refreshConstraintComponents(currentTime)) {
        return false;
      }
      if (constraintProbe &&
          !recordConstraintProbeSample("output", currentTime,
                                       variablesVector,
                                       derivativesVector)) {
        return false;
      }
    } else {
      currentTime = internalTime;
    }
  }

  IDA_PROFILER_STEP_STOP

  if (solveRetVal != IDA_SUCCESS) {
    if (expectedConstraintFailure) {
      return false;
    }
    if (solveRetVal == IDA_TSTOP_RETURN) {
      return true;
    }

    if (solveRetVal == IDA_ROOT_RETURN) {
      return true;
    }

    if (solveRetVal == IDA_MEM_NULL) {
      std::cerr << "IDASolve - The ida_mem pointer is NULL" << std::endl;
    } else if (solveRetVal == IDA_ILL_INPUT) {
      std::cerr
          << "IDASolve - One of the inputs to IDASolve was illegal, or some "
             "other input to the solver was either illegal or missing"
          << std::endl;
    } else if (solveRetVal == IDA_TOO_MUCH_WORK) {
      std::cerr << "IDASolve - The solver took mxstep internal steps but could "
                   "not reach tout"
                << std::endl;
    } else if (solveRetVal == IDA_TOO_MUCH_ACC) {
      std::cerr << "IDASolve - The solver could not satisfy the accuracy "
                   "demanded by the user for some internal step"
                << std::endl;
    } else if (solveRetVal == IDA_ERR_FAIL) {
      std::cerr << "IDASolve - Error test failures occurred too many times "
                   "during one internal time step or occurred with |h| = hmin"
                << std::endl;
    } else if (solveRetVal == IDA_CONV_FAIL) {
      std::cerr
          << "IDASolve - Convergence test failures occurred too many times "
             "during one internal time step or occurred with |h| = hmin"
          << std::endl;
    } else if (solveRetVal == IDA_LINIT_FAIL) {
      std::cerr
          << "IDASolve - The linear solver’s initialization function failed"
          << std::endl;
    } else if (solveRetVal == IDA_LSETUP_FAIL) {
      std::cerr << "IDASolve - The linear solver’s setup function failed in an "
                   "unrecoverable manner"
                << std::endl;
    } else if (solveRetVal == IDA_LSOLVE_FAIL) {
      std::cerr << "IDASolve - The linear solver’s solve function failed in an "
                   "unrecoverable manner"
                << std::endl;
    } else if (solveRetVal == IDA_CONSTR_FAIL) {
      std::cerr << "IDASolve - The inequality constraints were violated and "
                   "the solver was unable to recover"
                << std::endl;
    } else if (solveRetVal == IDA_REP_RES_ERR) {
      std::cerr
          << "IDASolve - The user’s residual function repeatedly returned a "
             "recoverable error flag, but the solver was unable to recover"
          << std::endl;
    } else if (solveRetVal == IDA_RES_FAIL) {
      std::cerr << "IDASolve - The user’s residual function returned a "
                   "nonrecoverable error flag"
                << std::endl;
    } else if (solveRetVal == IDA_RTFUNC_FAIL) {
      std::cerr << "IDASolve - The rootfinding function failed" << std::endl;
    }

    return false;
  }

  copyVariablesIntoMARCO(variablesVector, derivativesVector);
  if (!refreshConstraintComponents(currentTime)) {
    return false;
  }

  return true;
}

realtype IDAInstance::getCurrentTime() const { return currentTime; }

int IDAInstance::residualFunction(realtype time, N_Vector variables,
                                  N_Vector derivatives, N_Vector residuals,
                                  void *userData) {
  IDA_PROFILER_RESIDUALS_CALL_COUNTER_INCREMENT
  static std::atomic<uint64_t> residualCallCounter{0};
  uint64_t residualCall = ++residualCallCounter;
  const bool runtimeProbe = isRuntimeIDAProbeEnabled();
  auto residualStart = std::chrono::steady_clock::now();

  if (runtimeProbe) {
    std::cerr << "[ida-runtime-probe] residual begin call=" << residualCall
              << " time=" << time << std::endl;
  }

  realtype *rval = N_VGetArrayPointer(residuals);
  auto *instance = static_cast<IDAInstance *>(userData);

  // 中文：先把 IDA 试探值写回 MARCO 数组并求解 local closure，随后 outer
  // residual 才读取完整一致状态。
  // English: Copy IDA trial values into MARCO arrays and solve local closure
  // first, so outer residuals observe a complete consistent state.
  instance->copyVariablesIntoMARCO(variables, derivatives);
  int localSolveResult = instance->solveConstraintComponents(time);
  if (localSolveResult != 0) {
    return localSolveResult < 0 ? -1 : 1;
  }

  // For every vectorized equation, set the residual values of the variables
  // it writes into.
  IDA_PROFILER_RESIDUALS_START

  instance->equationsParallelIteration(
      EquationsParallelIterationKind::Residuals,
      [&](Equation eq, const std::vector<int64_t> &equationIndices,
          const JacobianSeedsMap &jacobianSeedsMap) {
        assert(equationIndices.size() == instance->getEquationRank(eq));

        uint64_t equationArrayOffset = instance->equationOffsets[eq];

        uint64_t equationScalarOffset =
            getEquationFlatIndex(equationIndices, instance->equationRanges[eq]);

        uint64_t offset = equationArrayOffset + equationScalarOffset;

        auto residualFn = instance->residualFunctions[eq];
        auto *eqIndicesPtr = equationIndices.data();

        auto residualFunctionResult = residualFn(time, eqIndicesPtr);
        *(rval + offset) = residualFunctionResult;
      });

  IDA_PROFILER_RESIDUALS_STOP

  if (runtimeProbe) {
    std::cerr << "[ida-runtime-probe] residual done call=" << residualCall
              << " elapsed_ms=" << elapsedMillisecondsSince(residualStart)
              << std::endl;
  }

  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[IDA] Residuals function called" << std::endl;
    std::cerr << "Variables:" << std::endl;
    instance->printVariablesVector(variables);
    std::cerr << "Derivatives:" << std::endl;
    instance->printDerivativesVector(derivatives);
    std::cerr << "Residuals vector:" << std::endl;
    instance->printResidualsVector(residuals);
  }

  return IDA_SUCCESS;
}

int IDAInstance::jacobianMatrix(realtype time, realtype alpha,
                                N_Vector variables, N_Vector derivatives,
                                N_Vector residuals, SUNMatrix jacobianMatrix,
                                void *userData, N_Vector tempv1,
                                N_Vector tempv2, N_Vector tempv3) {
  IDA_PROFILER_PARTIAL_DERIVATIVES_CALL_COUNTER_INCREMENT
  static std::atomic<uint64_t> jacobianCallCounter{0};
  uint64_t jacobianCall = ++jacobianCallCounter;
  const bool runtimeProbe = isRuntimeIDAProbeEnabled();
  auto jacobianStart = std::chrono::steady_clock::now();

  if (runtimeProbe) {
    std::cerr << "[ida-runtime-probe] jacobian begin call=" << jacobianCall
              << " time=" << time << " alpha=" << alpha << std::endl;
  }

  realtype *jacobian = SUNSparseMatrix_Data(jacobianMatrix);
  auto *instance = static_cast<IDAInstance *>(userData);

  // 中文：Jacobian 使用同一 local closure 状态，普通 AD 先生成 direct outer
  // 项，再由 applyConstraintSchur 加入局部变量的隐式响应。
  // English: Jacobian evaluation uses the same local-closure state. Ordinary AD
  // produces direct outer terms before applyConstraintSchur adds the implicit
  // response of local variables.
  instance->copyVariablesIntoMARCO(variables, derivatives);
  int localSolveResult = instance->solveConstraintComponents(time);
  if (localSolveResult != 0) {
    return localSolveResult < 0 ? -1 : 1;
  }

  // For every vectorized equation, compute its row within the Jacobian
  // matrix.
  IDA_PROFILER_PARTIAL_DERIVATIVES_START

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
        std::vector<JacobianColumn> directJacobianColumns =
            instance->computeAccessedJacobianColumns(
                eq, equationIndices.data(), false);
        std::set<JacobianColumn> directColumns(directJacobianColumns.begin(),
                                               directJacobianColumns.end());

        // For every scalar variable with respect to which the equation must be
        // partially differentiated.
        for (size_t i = 0, e = jacobianColumns.size(); i < e; ++i) {
          const JacobianColumn &column = jacobianColumns[i];
          Variable variable = column.first;
          const auto &variableIndices = column.second;

          uint64_t variableArrayOffset = instance->variableOffsets[variable];

          uint64_t variableScalarOffset = getVariableFlatIndex(
              instance->variablesDimensions[variable], column.second);

          double jacobianFunctionResult = 0;
          if (directColumns.find(column) != directColumns.end()) {
            auto jacobianFunction =
                instance->jacobianFunctions[eq][variable].first;
            auto seedsMapIt = jacobianSeedsMap.find(jacobianFunction);
            assert(seedsMapIt != jacobianSeedsMap.end());
            assert(jacobianFunction != nullptr);

            jacobianFunctionResult = jacobianFunction(
                time, equationIndices.data(), variableIndices.data(), alpha,
                instance->memoryPoolId, seedsMapIt->second.data());
          }

          instance->jacobianMatrixData[scalarEquationIndex][i].second =
              jacobianFunctionResult;

          auto index = static_cast<sunindextype>(variableArrayOffset +
                                                 variableScalarOffset);

          instance->jacobianMatrixData[scalarEquationIndex][i].first = index;
        }
      });

  if (!instance->applyConstraintSchur(time, alpha)) {
    return -1;
  }

  // Move the partial derivatives into the SUNDIALS sparse matrix.
  sunindextype *rowPtrs = SUNSparseMatrix_IndexPointers(jacobianMatrix);
  sunindextype *columnIndices = SUNSparseMatrix_IndexValues(jacobianMatrix);

  sunindextype offset = 0;
  *rowPtrs++ = offset;

  for (const auto &row : instance->jacobianMatrixData) {
    offset += static_cast<sunindextype>(row.size());
    *rowPtrs++ = offset;

    for (const auto &column : row) {
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

  IDA_PROFILER_PARTIAL_DERIVATIVES_STOP

  if (runtimeProbe) {
    std::cerr << "[ida-runtime-probe] jacobian done call=" << jacobianCall
              << " elapsed_ms=" << elapsedMillisecondsSince(jacobianStart)
              << std::endl;
  }

  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[IDA] Jacobian matrix function called" << std::endl;
    std::cerr << "Time: " << time << std::endl;
    std::cerr << "Alpha: " << alpha << std::endl;
    std::cerr << "Variables:" << std::endl;
    instance->printVariablesVector(variables);
    std::cerr << "Derivatives:" << std::endl;
    instance->printDerivativesVector(derivatives);
    std::cerr << "Residuals vector:" << std::endl;
    instance->printResidualsVector(residuals);
    std::cerr << "Jacobian matrix:" << std::endl;
    instance->printJacobianMatrix(jacobianMatrix);
  }

  return IDA_SUCCESS;
}

uint64_t IDAInstance::getNumOfArrayVariables() const {
  return variablesDimensions.size();
}

uint64_t IDAInstance::getNumOfScalarVariables() const {
  return scalarVariablesNumber;
}

VariableKind IDAInstance::getVariableKind(Variable variable) const {
  auto it = stateVariablesMapping.find(variable);
  auto endIt = stateVariablesMapping.end();
  return it == endIt ? VariableKind::ALGEBRAIC : VariableKind::STATE;
}

uint64_t IDAInstance::getVariableFlatSize(Variable variable) const {
  if (isLocalVariable(variable)) {
    return 0;
  }
  uint64_t result = 1;

  for (uint64_t dimension : variablesDimensions[variable]) {
    result *= dimension;
  }

  return result;
}

uint64_t IDAInstance::getNumOfVectorizedEquations() const {
  return equationRanges.size();
}

uint64_t IDAInstance::getNumOfScalarEquations() const {
  return scalarEquationsNumber;
}

uint64_t IDAInstance::getEquationRank(Equation equation) const {
  return equationRanges[equation].size();
}

uint64_t IDAInstance::getEquationFlatSize(Equation equation) const {
  assert(equation < getNumOfVectorizedEquations());
  if (isLocalEquation(equation)) {
    return 0;
  }
  uint64_t result = 1;

  for (const Range &range : equationRanges[equation]) {
    result *= range.end - range.begin;
  }

  return result;
}

uint64_t IDAInstance::getVariableRank(Variable variable) const {
  return variablesDimensions[variable].rank();
}

void IDAInstance::iterateAccessedArrayVariables(
    Equation equation, std::function<void(Variable)> callback) const {
  if (precomputedAccesses) {
    std::set<Variable> visited;
    if (equation < variableAccesses.size()) {
      for (const auto &access : variableAccesses[equation]) {
        if (isLocalVariable(access.first)) {
          continue;
        }
        if (visited.insert(access.first).second) {
          callback(access.first);
        }
      }
    }
    if (equation < variableRangeAccesses.size()) {
      for (const auto &access : variableRangeAccesses[equation]) {
        if (isLocalVariable(access.first)) {
          continue;
        }
        if (visited.insert(access.first).second) {
          callback(access.first);
        }
      }
    }
  } else {
    uint64_t numOfArrayVariables = getNumOfArrayVariables();

    for (Variable variable = 0; variable < numOfArrayVariables; ++variable) {
      if (isLocalVariable(variable)) {
        continue;
      }
      callback(variable);
    }
  }
}

/// Determine which of the columns of the current Jacobian row has to be
/// populated, and with respect to which variable the partial derivative has
/// to be performed. The row is determined by the indices of the equation.
std::vector<JacobianColumn>
IDAInstance::computeAccessedJacobianColumns(
    Equation eq, const int64_t *equationIndices,
    bool includeLocalVariables) const {
  std::set<JacobianColumn> uniqueColumns;

  if (precomputedAccesses) {
    if (eq < variableAccesses.size()) {
      for (const auto &access : variableAccesses[eq]) {
        Variable variable = access.first;
        if (!includeLocalVariables && isLocalVariable(variable)) {
          continue;
        }
        AccessFunction accessFunction = access.second;

        uint64_t variableRank = getVariableRank(variable);

        std::vector<uint64_t> variableIndices;
        variableIndices.resize(variableRank, 0);
        accessFunction(equationIndices, variableIndices.data());

        bool accessOutOfBounds = false;

        for (uint64_t i = 0; i < variableRank; ++i) {
          if (variableIndices[i] >= variablesDimensions[variable][i]) {
            accessOutOfBounds = true;
            break;
          }
        }

        if (accessOutOfBounds) {
          // Precomputed affine accesses may describe boundary stencils. On
          // boundary rows, out-of-range neighbors cannot be Jacobian nonzeros.
          continue;
        }

        uniqueColumns.insert({variable, variableIndices});
      }
    }

    if (eq < variableRangeAccesses.size()) {
      for (const auto &[variable, range] : variableRangeAccesses[eq]) {
        if (!includeLocalVariables && isLocalVariable(variable)) {
          continue;
        }
        if (range.size() != getVariableRank(variable)) {
          continue;
        }
        for (auto indices = MultidimensionalRangeIterator::begin(range),
                  end = MultidimensionalRangeIterator::end(range);
             indices != end; ++indices) {
          std::vector<uint64_t> variableIndices(range.size(), 0);
          const int64_t *current = *indices;
          bool outOfBounds = false;
          for (size_t dimension = 0; dimension < range.size(); ++dimension) {
            if (current[dimension] < 0 ||
                static_cast<uint64_t>(current[dimension]) >=
                    variablesDimensions[variable][dimension]) {
              outOfBounds = true;
              break;
            }
            variableIndices[dimension] =
                static_cast<uint64_t>(current[dimension]);
          }
          if (!outOfBounds) {
            uniqueColumns.insert({variable, std::move(variableIndices)});
          }
        }
      }
    }
  } else {
    uint64_t numOfArrayVariables = getNumOfArrayVariables();

    for (Variable variable = 0; variable < numOfArrayVariables; ++variable) {
      if (!includeLocalVariables && isLocalVariable(variable)) {
        continue;
      }
      const auto &dimensions = variablesDimensions[variable];

      for (auto indices = dimensions.indicesBegin(),
                end = dimensions.indicesEnd();
           indices != end; ++indices) {
        JacobianColumn column(variable, {});

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

std::vector<JacobianColumn>
IDAInstance::computeJacobianColumns(Equation eq,
                                    const int64_t *equationIndices) const {
  std::set<JacobianColumn> columns;
  for (const JacobianColumn &column :
       computeAccessedJacobianColumns(eq, equationIndices, false)) {
    columns.insert(column);
  }

  std::vector<JacobianColumn> allAccesses =
      computeAccessedJacobianColumns(eq, equationIndices, true);
  for (const auto &execution : constraintComponentExecutions) {
    if (std::find(execution->outerEquations.begin(),
                  execution->outerEquations.end(),
                  eq) == execution->outerEquations.end()) {
      continue;
    }

    std::set<uint64_t> localComponents;
    for (const JacobianColumn &column : allAccesses) {
      auto component = execution->localColumnComponents.find(column);
      if (component != execution->localColumnComponents.end()) {
        localComponents.insert(component->second);
      }
    }
    for (uint64_t localComponent : localComponents) {
      auto coupling =
          execution->componentOuterCouplingColumns.find(localComponent);
      if (coupling != execution->componentOuterCouplingColumns.end()) {
        columns.insert(coupling->second.begin(), coupling->second.end());
      }
    }
  }

  return {columns.begin(), columns.end()};
}

/// Compute the number of non-zero values in the Jacobian Matrix. Also
/// compute the column indexes of all non-zero values in the Jacobian Matrix.
/// This allows to avoid the recomputation of such indexes during the
/// Jacobian evaluation.
void IDAInstance::computeNNZ() {
  nonZeroValuesNumber = 0;
  std::vector<int64_t> equationIndices;

  for (size_t eq = 0; eq < getNumOfVectorizedEquations(); ++eq) {
    if (isLocalEquation(eq)) {
      continue;
    }

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

void IDAInstance::rebuildJacobianMatrixStorage() {
  // 中文：basis/epoch 切换会改变 active outer rows 与 columns；重建只遍历 outer
  // descriptors，local closure 的 J_zz 仍由各 execution-group KINSOL 独立持有。
  // English: A basis/epoch switch changes active outer rows and columns. This
  // rebuild visits only outer descriptors; each execution-group KINSOL keeps
  // its local J_zz storage independently.
  jacobianMatrixData.clear();
  jacobianMatrixData.resize(scalarEquationsNumber);
  nonZeroValuesNumber = 0;

  std::vector<int64_t> equationIndices;
  for (Equation equation = 0; equation < getNumOfVectorizedEquations();
       ++equation) {
    if (isLocalEquation(equation)) {
      continue;
    }

    getEquationBeginIndices(equation, equationIndices);
    do {
      uint64_t scalarEquationIndex =
          equationOffsets[equation] +
          getEquationFlatIndex(equationIndices, equationRanges[equation]);
      std::vector<JacobianColumn> columns =
          computeJacobianColumns(equation, equationIndices.data());
      jacobianMatrixData[scalarEquationIndex].resize(columns.size());
      nonZeroValuesNumber += columns.size();
    } while (advanceEquationIndices(equationIndices,
                                    equationRanges[equation]));
  }
}

void IDAInstance::computeThreadChunks() {
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

      iterateAccessedArrayVariables(equation, [&](Variable variable) {
        auto jacobianFunction = jacobianFunctions[equation][variable].first;
        const auto &seedSizes = jacobianFunctions[equation][variable].second;

        for (const auto &seedSize : seedSizes) {
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

void IDAInstance::copyVariablesFromMARCO(
    N_Vector algebraicAndStateVariablesVector,
    N_Vector derivativeVariablesVector) {
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[IDA] Copying variables from MARCO" << std::endl;
  }

  IDA_PROFILER_COPY_VARS_FROM_MARCO_START

  realtype *varsPtr = N_VGetArrayPointer(algebraicAndStateVariablesVector);
  realtype *dersPtr = N_VGetArrayPointer(derivativeVariablesVector);
  uint64_t numOfArrayVariables = getNumOfArrayVariables();

  for (Variable var = 0; var < numOfArrayVariables; ++var) {
    // 中文：component-local blocks 直接读写 MARCO 原数组并由 local solver 管理，
    // 不能再次复制到 outer IDA vector。
    // English: Component-local blocks access original MARCO arrays and belong
    // to the local solver, so they must not be copied into the outer IDA
    // vector again.
    if (isLocalVariable(var)) {
      continue;
    }

    uint64_t variableArrayOffset = variableOffsets[var];
    const auto &dimensions = variablesDimensions[var];

    std::vector<uint64_t> varIndices;
    getVariableBeginIndices(var, varIndices);

    do {
      uint64_t variableScalarOffset =
          getVariableFlatIndex(dimensions, varIndices.data());

      uint64_t offset = variableArrayOffset + variableScalarOffset;

      // Get the state / algebraic variable.
      auto getterFn = algebraicAndStateVariablesGetters[var];
      auto value = static_cast<realtype>(getterFn(varIndices.data()));
      varsPtr[offset] = value;

      if (marco::runtime::simulation::getOptions().debug) {
        std::cerr << "Got var " << var << " ";
        printIndices(varIndices);
        std::cerr << " with value " << std::fixed << std::setprecision(9)
                  << value << std::endl;
      }

      // Get the derivative variable, if the variable was a state.
      auto derivativeVariablePositionIt = stateVariablesMapping.find(var);

      if (derivativeVariablePositionIt != stateVariablesMapping.end()) {
        auto derGetterFn =
            derivativeVariablesGetters[derivativeVariablePositionIt->second];

        auto derValue = static_cast<realtype>(derGetterFn(varIndices.data()));

        dersPtr[offset] = derValue;

        if (marco::runtime::simulation::getOptions().debug) {
          std::cerr << "Got der(var " << var << ") ";
          printIndices(varIndices);
          std::cerr << " with value " << std::fixed << std::setprecision(9)
                    << derValue << std::endl;
        }
      }
    } while (advanceVariableIndices(varIndices, variablesDimensions[var]));
  }

  IDA_PROFILER_COPY_VARS_FROM_MARCO_STOP
}

void IDAInstance::copyVariablesIntoMARCO(
    N_Vector algebraicAndStateVariablesVector,
    N_Vector derivativeVariablesVector) {
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[IDA] Copying variables into MARCO" << std::endl;
  }

  IDA_PROFILER_COPY_VARS_INTO_MARCO_START

  realtype *varsPtr = N_VGetArrayPointer(algebraicAndStateVariablesVector);
  realtype *dersPtr = N_VGetArrayPointer(derivativeVariablesVector);
  uint64_t numOfArrayVariables = getNumOfArrayVariables();

  for (Variable var = 0; var < numOfArrayVariables; ++var) {
    // 中文：与 copy-from 对称跳过 local blocks，避免 outer accepted state 覆盖
    // 刚由 constraint closure 重建的 dependent values。
    // English: Symmetrically skip local blocks so the outer accepted state
    // cannot overwrite dependent values just reconstructed by the constraint
    // closure.
    if (isLocalVariable(var)) {
      continue;
    }

    uint64_t variableArrayOffset = variableOffsets[var];
    const auto &dimensions = variablesDimensions[var];

    std::vector<uint64_t> varIndices;
    getVariableBeginIndices(var, varIndices);

    do {
      uint64_t variableScalarOffset =
          getVariableFlatIndex(dimensions, varIndices.data());

      uint64_t offset = variableArrayOffset + variableScalarOffset;

      // Set the state / algebraic variable.
      auto setterFn = algebraicAndStateVariablesSetters[var];
      auto value = static_cast<double>(varsPtr[offset]);

      if (marco::runtime::simulation::getOptions().debug) {
        std::cerr << "Setting var " << var << " ";
        printIndices(varIndices);
        std::cerr << " to " << value << std::endl;
      }

      setterFn(value, varIndices.data());

      assert([&]() -> bool {
        auto getterFn = algebraicAndStateVariablesGetters[var];
        return getterFn(varIndices.data()) == value;
      }() && "Variable value not set correctly");

      // Set the derivative variable, if the variable was a state.
      auto derivativeVariablePositionIt = stateVariablesMapping.find(var);

      if (derivativeVariablePositionIt != stateVariablesMapping.end()) {
        auto derSetterFn =
            derivativeVariablesSetters[derivativeVariablePositionIt->second];

        auto derValue = static_cast<double>(dersPtr[offset]);

        if (marco::runtime::simulation::getOptions().debug) {
          std::cerr << "Setting der(var " << var << ") ";
          printIndices(varIndices);
          std::cerr << " to " << derValue << std::endl;
        }

        derSetterFn(derValue, varIndices.data());

        assert([&]() -> bool {
          auto derGetterFn =
              derivativeVariablesGetters[derivativeVariablePositionIt->second];

          return derGetterFn(varIndices.data()) == derValue;
        }() && "Derivative value not set correctly");
      }
    } while (advanceVariableIndices(varIndices, variablesDimensions[var]));
  }

  IDA_PROFILER_COPY_VARS_INTO_MARCO_STOP
}

void IDAInstance::equationsParallelIteration(
    EquationsParallelIterationKind kind,
  std::function<void(Equation equation,
                     const std::vector<int64_t> &equationIndices,
                     const JacobianSeedsMap &jacobianSeedsMap)>
      processFn) {
  // 中文：按 chunk 把 vectorized equation 的标量点分给线程；profiling/debug
  // 模式会记录每个 worker 的实际负载。
  // English: Shard scalar points of vectorized equations across worker chunks;
  // profiling/debug mode records the actual work handled by each worker.
  unsigned int numOfThreads = threadPool.getNumOfThreads();
  std::atomic_size_t chunkIndex = 0;
  const auto &simulationOptions = marco::runtime::simulation::getOptions();
  bool collectStats = simulationOptions.debug || simulationOptions.profiling;
  std::vector<profiling::ParallelThreadWorkStats> threadWork(numOfThreads);
  std::vector<std::thread::id> threadIds(numOfThreads);
  const bool runtimeProbe = isRuntimeIDAProbeEnabled();
  auto iterationStart = std::chrono::steady_clock::now();
  std::atomic<uint64_t> processedScalarEquations{0};

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

          if (runtimeProbe) {
            uint64_t processed = ++processedScalarEquations;
            if (processed % 1024 == 0) {
              std::cerr << "[ida-runtime-probe] "
                        << getParallelIterationKindName(kind)
                        << " progress scalar_equations=" << processed
                        << " elapsed_ms="
                        << elapsedMillisecondsSince(iterationStart)
                        << std::endl;
            }
          }

          if (collectStats) {
            ++threadWork[thread].scalarEquations;
          }
        } while (advanceEquationIndicesUntil(
            equationIndices, equationRanges[equation], std::get<2>(chunk)));
      }
    });
  }

  threadPool.wait();

  if (runtimeProbe) {
    std::cerr << "[ida-runtime-probe] " << getParallelIterationKindName(kind)
              << " iteration done scalar_equations="
              << processedScalarEquations.load()
              << " elapsed_ms=" << elapsedMillisecondsSince(iterationStart)
              << std::endl;
  }

  if (simulationOptions.debug) {
    printParallelIterationStats(kind, threadWork, threadIds);
  }

  if (kind == EquationsParallelIterationKind::Residuals) {
    IDA_PROFILER_RESIDUALS_PARALLEL_WORK_RECORD(threadWork)
  } else {
    IDA_PROFILER_PARTIAL_DERIVATIVES_PARALLEL_WORK_RECORD(threadWork)
  }
}

void IDAInstance::getVariableBeginIndices(
    Variable variable, std::vector<uint64_t> &indices) const {
  uint64_t variableRank = getVariableRank(variable);
  indices.resize(variableRank);

  for (uint64_t i = 0; i < variableRank; ++i) {
    indices[i] = 0;
  }
}

void IDAInstance::getVariableEndIndices(Variable variable,
                                        std::vector<uint64_t> &indices) const {
  uint64_t variableRank = getVariableRank(variable);
  indices.resize(variableRank);

  for (uint64_t i = 0; i < variableRank; ++i) {
    indices[i] = variablesDimensions[variable][i];
  }
}

void IDAInstance::getEquationBeginIndices(Equation equation,
                                          std::vector<int64_t> &indices) const {
  uint64_t equationRank = getEquationRank(equation);
  indices.resize(equationRank);

  for (uint64_t i = 0; i < equationRank; ++i) {
    indices[i] = equationRanges[equation][i].begin;
  }
}

void IDAInstance::getEquationEndIndices(Equation equation,
                                        std::vector<int64_t> &indices) const {
  uint64_t equationRank = getEquationRank(equation);
  indices.resize(equationRank);

  for (uint64_t i = 0; i < equationRank; ++i) {
    indices[i] = equationRanges[equation][i].end;
  }
}

void IDAInstance::printStatistics() const {
  if (getNumOfScalarEquations() == 0) {
    return;
  }

  long nst, nre, nje, nni, nli, netf, nncf;
  realtype ais, ls;

  IDAGetNumSteps(idaMemory, &nst);
  IDAGetNumResEvals(idaMemory, &nre);
  IDAGetNumJacEvals(idaMemory, &nje);
  IDAGetNumNonlinSolvIters(idaMemory, &nni);
  IDAGetNumLinIters(idaMemory, &nli);
  IDAGetNumErrTestFails(idaMemory, &netf);
  IDAGetNumNonlinSolvConvFails(idaMemory, &nncf);
  IDAGetActualInitStep(idaMemory, &ais);
  IDAGetLastStep(idaMemory, &ls);

  std::cerr << std::endl << "Final Run Statistics:" << std::endl;

  std::cerr << "Number of vector equations       = ";
  std::cerr << getNumOfVectorizedEquations() << std::endl;
  std::cerr << "Number of scalar equations       = ";
  std::cerr << getNumOfScalarEquations() << std::endl;
  std::cerr << "Number of non-zero values        = ";
  std::cerr << nonZeroValuesNumber << std::endl;

  std::cerr << "Number of steps                  = " << nst << std::endl;
  std::cerr << "Number of residual evaluations   = " << nre << std::endl;
  std::cerr << "Number of Jacobian evaluations   = " << nje << std::endl;

  std::cerr << "Number of nonlinear iterations   = " << nni << std::endl;
  std::cerr << "Number of linear iterations      = " << nli << std::endl;
  std::cerr << "Number of error test failures    = " << netf << std::endl;
  std::cerr << "Number of nonlin. conv. failures = " << nncf << std::endl;

  std::cerr << "Actual initial step size used    = " << ais << std::endl;
  std::cerr << "Step size used for the last step = " << ls << std::endl;
}

bool IDAInstance::idaInit() {
  auto retVal = IDAInit(idaMemory, residualFunction, startTime, variablesVector,
                        derivativesVector);

  if (retVal == IDA_MEM_NULL) {
    std::cerr << "IDAInit - The ida_mem pointer is NULL" << std::endl;
    return false;
  }

  if (retVal == IDA_MEM_FAIL) {
    std::cerr << "IDAInit - A memory allocation request has failed"
              << std::endl;
    return false;
  }

  if (retVal == IDA_ILL_INPUT) {
    std::cerr << "IDAInit - An input argument to IDAInit has an illegal value"
              << std::endl;
    return false;
  }

  return retVal == IDA_SUCCESS;
}

bool IDAInstance::idaSVTolerances() {
  auto retVal = IDASVtolerances(idaMemory, getOptions().relativeTolerance,
                                tolerancesVector);

  if (retVal == IDA_MEM_NULL) {
    std::cerr << "IDASVtolerances - The ida_mem pointer is NULL" << std::endl;
    return false;
  }

  if (retVal == IDA_NO_MALLOC) {
    std::cerr << "IDASVtolerances - The allocation function IDAInit(has not "
                 "been called"
              << std::endl;
    return false;
  }

  if (retVal == IDA_ILL_INPUT) {
    std::cerr << "IDASVtolerances - The relative error tolerance was negative "
                 "or the absolute tolerance vector had a negative component"
              << std::endl;
    return false;
  }

  return retVal == IDA_SUCCESS;
}

bool IDAInstance::idaSetLinearSolver() {
  auto retVal = IDASetLinearSolver(idaMemory, linearSolver, sparseMatrix);

  if (retVal == IDALS_MEM_NULL) {
    std::cerr << "IDASetLinearSolver - The ida_mem pointer is NULL"
              << std::endl;
    return false;
  }

  if (retVal == IDALS_ILL_INPUT) {
    std::cerr << "IDASetLinearSolver - The IDALS interface is not compatible "
                 "with the LS or J input objects or is incompatible with the "
                 "N_Vector object passed to IDAInit"
              << std::endl;
    return false;
  }

  if (retVal == IDALS_SUNLS_FAIL) {
    std::cerr << "IDASetLinearSolver - A call to the LS object failed"
              << std::endl;
    return false;
  }

  if (retVal == IDALS_MEM_FAIL) {
    std::cerr << "IDASetLinearSolver - A memory allocation request failed"
              << std::endl;
    return false;
  }

  return retVal == IDALS_SUCCESS;
}

bool IDAInstance::idaSetUserData() {
  auto retVal = IDASetUserData(idaMemory, this);

  if (retVal == IDA_MEM_NULL) {
    std::cerr << "IDASetUserData - The ida_mem pointer is NULL" << std::endl;
    return false;
  }

  return retVal == IDA_SUCCESS;
}

bool IDAInstance::idaSetMaxNumSteps() {
  auto retVal = IDASetMaxNumSteps(idaMemory, getOptions().maxSteps);

  if (retVal == IDA_MEM_NULL) {
    std::cerr << "IDASetMaxNumSteps - The ida_mem pointer is NULL" << std::endl;
    return false;
  }

  if (retVal == IDA_ILL_INPUT) {
    std::cerr << "IDASetMaxNumSteps - Either hmax is not positive or it is "
                 "smaller than the minimum allowable step"
              << std::endl;
    return false;
  }

  return retVal == IDA_SUCCESS;
}

bool IDAInstance::idaSetInitialStepSize() {
  auto retVal = IDASetInitStep(idaMemory, getOptions().initialStepSize);

  if (retVal == IDA_MEM_NULL) {
    std::cerr << "IDASetInitStep - The ida_mem pointer is NULL" << std::endl;
    return false;
  }

  return retVal == IDA_SUCCESS;
}

bool IDAInstance::idaSetMinStepSize() {
#if SUNDIALS_VERSION_MAJOR >= 6 && SUNDIALS_VERSION_MINOR >= 2
  auto retVal = IDASetMinStep(idaMemory, getOptions().minStepSize);

  if (retVal == IDA_MEM_NULL) {
    std::cerr << "IDASetMinStep - The ida_mem pointer is NULL" << std::endl;
    return false;
  }

  if (retVal == IDA_ILL_INPUT) {
    std::cerr << "IDASetMinStep - hmin is negative" << std::endl;
    return false;
  }

  return retVal == IDA_SUCCESS;
#else
  return true;
#endif
}

bool IDAInstance::idaSetMaxStepSize() {
  auto retVal = IDASetMaxStep(idaMemory, getOptions().maxStepSize);

  if (retVal == IDA_MEM_NULL) {
    std::cerr << "IDASetMaxStep - The ida_mem pointer is NULL" << std::endl;
    return false;
  }

  if (retVal == IDA_ILL_INPUT) {
    std::cerr << "IDASetMaxStep - Either hmax is not positive or it is smaller "
                 "than the minimum allowable step"
              << std::endl;
    return false;
  }

  return retVal == IDA_SUCCESS;
}

bool IDAInstance::idaSetMaxErrTestFails() {
  auto retVal = IDASetMaxErrTestFails(idaMemory, getOptions().maxErrTestFails);

  if (retVal == IDA_MEM_NULL) {
    std::cerr << "IDASetMaxErrTestFails - The ida_mem pointer is NULL"
              << std::endl;
    return false;
  }

  return retVal == IDA_SUCCESS;
}

bool IDAInstance::idaSetSuppressAlg() {
  // 中文：优先级固定为 CLI、model default、legacy false，并缓存 effective 值与
  // 来源，保证 probe、重初始化和 basis switch 复用同一策略。
  // English: Precedence is CLI, model default, then legacy false. Cache both
  // the effective value and source so probes, reinitialization, and basis
  // switches reuse one policy.
  booleantype effectiveValue = SUNFALSE;
  const char *source = "legacy";

  if (getOptions().suppressAlgOverride.has_value()) {
    effectiveValue = *getOptions().suppressAlgOverride;
    source = "cli";
  } else if (modelSuppressAlgebraicErrorTest.has_value()) {
    effectiveValue = *modelSuppressAlgebraicErrorTest;
    source = "model-default";
  }

  effectiveSuppressAlgebraicErrorTest = effectiveValue;
  effectiveSuppressAlgebraicErrorTestSource = source;

  if (marco::runtime::simulation::getOptions().debug ||
      isRuntimeIDAProbeEnabled() || isStage6GRuntimeProbeEnabled()) {
    std::cerr << "[ida-runtime-probe] algebraic_error_test suppress="
              << (effectiveValue == SUNTRUE ? 1 : 0)
              << " source=" << source << std::endl;
  }

  auto retVal = IDASetSuppressAlg(idaMemory, effectiveValue);

  if (retVal == IDA_MEM_NULL) {
    std::cerr << "IDASetSuppressAlg - The ida_mem pointer is NULL" << std::endl;
    return false;
  }

  return retVal == IDA_SUCCESS;
}

bool IDAInstance::idaSetId() {
  auto retVal = IDASetId(idaMemory, idVector);

  if (retVal == IDA_MEM_NULL) {
    std::cerr << "IDASetId - The ida_mem pointer is NULL" << std::endl;
    return false;
  }

  return retVal == IDA_SUCCESS;
}

bool IDAInstance::idaSetJacobianFunction() {
  auto retVal = IDASetJacFn(idaMemory, jacobianMatrix);

  if (retVal == IDA_MEM_NULL) {
    std::cerr << "IDASetJacFn - The ida_mem pointer is NULL" << std::endl;
    return false;
  }

  if (retVal == IDALS_LMEM_NULL) {
    std::cerr << "IDASetJacFn - The IDALS linear solver interface has not been "
                 "initialized"
              << std::endl;
    return false;
  }

  return retVal == IDA_SUCCESS;
}

bool IDAInstance::idaSetMaxNonlinIters() {
  auto retVal = IDASetMaxNonlinIters(idaMemory, getOptions().maxNonlinIters);

  if (retVal == IDA_MEM_NULL) {
    std::cerr << "IDASetMaxNonlinIters - The ida_mem pointer is NULL"
              << std::endl;
    return false;
  }

  if (retVal == IDA_MEM_FAIL) {
    std::cerr << "IDASetMaxNonlinIters - The SUNNonlinearSolver object is NULL"
              << std::endl;
    return false;
  }

  return retVal == IDA_SUCCESS;
}

bool IDAInstance::idaSetMaxConvFails() {
  auto retVal = IDASetMaxConvFails(idaMemory, getOptions().maxConvFails);

  if (retVal == IDA_MEM_NULL) {
    std::cerr << "IDASetMaxConvFails - The ida_mem pointer is NULL"
              << std::endl;
    return false;
  }

  return retVal == IDA_SUCCESS;
}

bool IDAInstance::idaSetNonlinConvCoef() {
  auto retVal = IDASetNonlinConvCoef(idaMemory, getOptions().nonlinConvCoef);

  if (retVal == IDA_MEM_NULL) {
    std::cerr << "IDASetNonlinConvCoef - The ida_mem pointer is NULL"
              << std::endl;
    return false;
  }

  if (retVal == IDA_ILL_INPUT) {
    std::cerr << "IDASetNonlinConvCoef - The value of nlscoef is <= 0"
              << std::endl;
    return false;
  }

  return retVal == IDA_SUCCESS;
}

bool IDAInstance::idaSetNonlinConvCoefIC() {
  auto retVal =
      IDASetNonlinConvCoefIC(idaMemory, getOptions().nonlinConvCoefIC);

  if (retVal == IDA_MEM_NULL) {
    std::cerr << "IDASetNonlinConvCoefIC - The ida_mem pointer is NULL"
              << std::endl;
    return false;
  }

  if (retVal == IDA_ILL_INPUT) {
    std::cerr << "IDASetNonlinConvCoefIC - The epiccon factor is <= 0"
              << std::endl;
    return false;
  }

  return retVal == IDA_SUCCESS;
}

bool IDAInstance::idaSetMaxNumStepsIC() {
  auto retVal =
      IDASetMaxNumStepsIC(idaMemory, static_cast<int>(getOptions().maxStepsIC));

  if (retVal == IDA_MEM_NULL) {
    std::cerr << "IDASetMaxNumStepsIC - The ida_mem pointer is NULL"
              << std::endl;
    return false;
  }

  if (retVal == IDA_ILL_INPUT) {
    std::cerr << "IDASetMaxNumStepsIC - maxnh is non-positive" << std::endl;
    return false;
  }

  return retVal == IDA_SUCCESS;
}

bool IDAInstance::idaSetMaxNumJacsIC() {
  auto retVal = IDASetMaxNumJacsIC(idaMemory, getOptions().maxNumJacsIC);

  if (retVal == IDA_MEM_NULL) {
    std::cerr << "IDASetMaxNumJacsIC - The ida_mem pointer is NULL"
              << std::endl;
    return false;
  }

  if (retVal == IDA_ILL_INPUT) {
    std::cerr << "IDASetMaxNumJacsIC - maxnj is non-positive" << std::endl;
    return false;
  }

  return retVal == IDA_SUCCESS;
}

bool IDAInstance::idaSetMaxNumItersIC() {
  auto retVal = IDASetMaxNumItersIC(idaMemory, getOptions().maxNumItersIC);

  if (retVal == IDA_MEM_NULL) {
    std::cerr << "IDASetMaxNumItersIC - The ida_mem pointer is NULL"
              << std::endl;
    return false;
  }

  if (retVal == IDA_ILL_INPUT) {
    std::cerr << "IDASetMaxNumItersIC - maxnit is non-positive" << std::endl;
    return false;
  }

  return retVal == IDA_SUCCESS;
}

bool IDAInstance::idaSetLineSearchOffIC() {
  auto retVal = IDASetLineSearchOffIC(idaMemory, getOptions().lineSearchOff);

  if (retVal == IDA_MEM_NULL) {
    std::cerr << "IDASetLineSearchOffIC - The ida_mem pointer is NULL"
              << std::endl;
    return false;
  }

  return retVal == IDA_SUCCESS;
}

void IDAInstance::printVariablesVector(N_Vector variables) const {
  realtype *data = N_VGetArrayPointer(variables);
  uint64_t numOfArrayVariables = getNumOfArrayVariables();

  for (Variable var = 0; var < numOfArrayVariables; ++var) {
    if (isLocalVariable(var)) {
      continue;
    }

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

void IDAInstance::printDerivativesVector(N_Vector derivatives) const {
  realtype *data = N_VGetArrayPointer(derivatives);
  uint64_t numOfArrayVariables = getNumOfArrayVariables();

  for (Variable var = 0; var < numOfArrayVariables; ++var) {
    if (isLocalVariable(var)) {
      continue;
    }

    auto it = stateVariablesMapping.find(var);

    if (it != stateVariablesMapping.end()) {
      std::vector<uint64_t> indices;
      getVariableBeginIndices(var, indices);

      do {
        std::cerr << "der(var " << var << ") ";
        printIndices(indices);
        std::cerr << "\t" << std::fixed << std::setprecision(9) << *data
                  << std::endl;
        ++data;
      } while (advanceVariableIndices(indices, variablesDimensions[var]));
    }
  }
}

void IDAInstance::printResidualsVector(N_Vector residuals) const {
  realtype *data = N_VGetArrayPointer(residuals);
  uint64_t numOfVectorizedEquations = getNumOfVectorizedEquations();

  for (Equation eq = 0; eq < numOfVectorizedEquations; ++eq) {
    if (isLocalEquation(eq)) {
      continue;
    }

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

// Highly inefficient, use only for debug purposes.
static double getCellFromSparseMatrix(SUNMatrix matrix, uint64_t rowIndex,
                                      uint64_t columnIndex) {
  realtype *data = SUNSparseMatrix_Data(matrix);

  sunindextype *rowPtrs = SUNSparseMatrix_IndexPointers(matrix);
  sunindextype *columnIndices = SUNSparseMatrix_IndexValues(matrix);

  sunindextype beginIndex = rowPtrs[rowIndex];
  sunindextype endIndex = rowPtrs[rowIndex + 1];

  for (sunindextype i = beginIndex; i < endIndex; ++i) {
    if (columnIndices[i] == static_cast<sunindextype>(columnIndex)) {
      return data[i];
    }
  }

  return 0;
}

void IDAInstance::printJacobianMatrix(SUNMatrix jacobianMatrix) const {
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
} // namespace marco::runtime::sundials::ida

//===---------------------------------------------------------------------===//
// Exported functions
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// idaCreate

static void *idaCreate_pvoid() {
  auto *instance = new IDAInstance();
  return static_cast<void *>(instance);
}

RUNTIME_FUNC_DEF(idaCreate, PTR(void))

//===---------------------------------------------------------------------===//
// idaCalcIC

static void idaCalcIC_void(void *instance) {
  bool result = static_cast<IDAInstance *>(instance)->calcIC();
  if (!result) {
    std::exit(EXIT_FAILURE);
  }
}

RUNTIME_FUNC_DEF(idaCalcIC, void, PTR(void))

//===---------------------------------------------------------------------===//
// idaStep

static void idaStep_void(void *instance) {
  bool result = static_cast<IDAInstance *>(instance)->step();
  if (!result) {
    std::exit(EXIT_FAILURE);
  }
}

RUNTIME_FUNC_DEF(idaStep, void, PTR(void))

//===---------------------------------------------------------------------===//
// idaFree

static void idaFree_void(void *instance) {
  delete static_cast<IDAInstance *>(instance);
}

RUNTIME_FUNC_DEF(idaFree, void, PTR(void))

//===---------------------------------------------------------------------===//
// idaSetStartTime

static void idaSetStartTime_void(void *instance, double startTime) {
  static_cast<IDAInstance *>(instance)->setStartTime(startTime);
}

RUNTIME_FUNC_DEF(idaSetStartTime, void, PTR(void), double)

//===---------------------------------------------------------------------===//
// idaSetEndTime

static void idaSetEndTime_void(void *instance, double endTime) {
  static_cast<IDAInstance *>(instance)->setEndTime(endTime);
}

RUNTIME_FUNC_DEF(idaSetEndTime, void, PTR(void), double)

//===---------------------------------------------------------------------===//
// idaSetSuppressAlgebraicErrorTest

static void idaSetSuppressAlgebraicErrorTest_void(void *instance,
                                                  bool suppress) {
  static_cast<IDAInstance *>(instance)->setSuppressAlgebraicErrorTest(suppress);
}

RUNTIME_FUNC_DEF(idaSetSuppressAlgebraicErrorTest, void, PTR(void), bool)

//===---------------------------------------------------------------------===//
// idaRegisterConstraintComponent

// 中文：以下 additive C ABI 仅把编译期认证 descriptor 注册到现有 IDA instance；
// 它们不做 matching、owner 推断或数组拆分。
// English: The additive C ABI below only registers compile-time-certified
// descriptors with an existing IDA instance. It performs no matching,
// ownership inference, or array splitting.

static void idaRegisterConstraintComponent_void(void *instance,
                                                uint64_t componentId) {
  static_cast<IDAInstance *>(instance)->registerConstraintComponent(
      componentId);
}

RUNTIME_FUNC_DEF(idaRegisterConstraintComponent, void, PTR(void), uint64_t)

//===---------------------------------------------------------------------===//
// idaRegisterConstraintComponentRoleOwnership

static void idaRegisterConstraintComponentRoleOwnership_void(
    void *instance, uint64_t componentId, void *role, uint64_t ownedScalars,
    uint64_t residualScalars) {
  static_cast<IDAInstance *>(instance)
      ->registerConstraintComponentRoleOwnership(
          componentId, static_cast<const char *>(role), ownedScalars,
          residualScalars);
}

RUNTIME_FUNC_DEF(idaRegisterConstraintComponentRoleOwnership, void, PTR(void),
                 uint64_t, PTR(void), uint64_t, uint64_t)

//===---------------------------------------------------------------------===//
// idaRegisterConstraintEquation

static void idaRegisterConstraintEquation_void(
    void *instance, uint64_t equation, uint64_t componentId, void *role,
    uint64_t order, void *matchedVariableName) {
  static_cast<IDAInstance *>(instance)->registerConstraintEquation(
      componentId, equation, static_cast<const char *>(role), order,
      static_cast<const char *>(matchedVariableName));
}

RUNTIME_FUNC_DEF(idaRegisterConstraintEquation, void, PTR(void), uint64_t,
                 uint64_t, PTR(void), uint64_t, PTR(void))

//===---------------------------------------------------------------------===//
// idaRegisterConstraintVariable

static void idaRegisterConstraintVariable_void(void *instance,
                                               uint64_t variable,
                                               uint64_t componentId,
                                               void *role, void *name) {
  static_cast<IDAInstance *>(instance)->registerConstraintVariable(
      componentId, variable, static_cast<const char *>(role),
      static_cast<const char *>(name));
}

RUNTIME_FUNC_DEF(idaRegisterConstraintVariable, void, PTR(void), uint64_t,
                 uint64_t, PTR(void), PTR(void))

//===---------------------------------------------------------------------===//
// idaRegisterSolverExecutionGroup

static void idaRegisterSolverExecutionGroup_void(
    void *instance, uint64_t groupId, void *kind, void *mode,
    uint64_t topologicalOrder) {
  static_cast<IDAInstance *>(instance)->registerSolverExecutionGroup(
      groupId, static_cast<const char *>(kind), static_cast<const char *>(mode),
      topologicalOrder);
}

RUNTIME_FUNC_DEF(idaRegisterSolverExecutionGroup, void, PTR(void), uint64_t,
                 PTR(void), PTR(void), uint64_t)

//===---------------------------------------------------------------------===//
// idaRegisterSolverExecutionGroupCertificate

static void idaRegisterSolverExecutionGroupCertificate_void(
    void *instance, uint64_t groupId, uint64_t outerEquationScalars,
    uint64_t outerVariableScalars, uint64_t localEquationScalars,
    uint64_t localVariableScalars) {
  static_cast<IDAInstance *>(instance)
      ->registerSolverExecutionGroupCertificate(
          groupId, outerEquationScalars, outerVariableScalars,
          localEquationScalars, localVariableScalars);
}

RUNTIME_FUNC_DEF(idaRegisterSolverExecutionGroupCertificate, void, PTR(void),
                 uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)

//===---------------------------------------------------------------------===//
// idaRegisterSolverExecutionGroupMember

static void idaRegisterSolverExecutionGroupMember_void(
    void *instance, uint64_t groupId, uint64_t componentId) {
  static_cast<IDAInstance *>(instance)->registerSolverExecutionGroupMember(
      groupId, componentId);
}

RUNTIME_FUNC_DEF(idaRegisterSolverExecutionGroupMember, void, PTR(void),
                 uint64_t, uint64_t)

//===---------------------------------------------------------------------===//
// idaRegisterSolverExecutionGroupDependency

static void idaRegisterSolverExecutionGroupDependency_void(
    void *instance, uint64_t groupId, uint64_t dependencyGroupId) {
  static_cast<IDAInstance *>(instance)
      ->registerSolverExecutionGroupDependency(groupId, dependencyGroupId);
}

RUNTIME_FUNC_DEF(idaRegisterSolverExecutionGroupDependency, void, PTR(void),
                 uint64_t, uint64_t)

//===---------------------------------------------------------------------===//
// idaEnableConstraintComponentExecution

static void idaEnableConstraintComponentExecution_void(void *instance,
                                                        uint64_t componentId) {
  static_cast<IDAInstance *>(instance)->enableConstraintComponentExecution(
      componentId);
}

RUNTIME_FUNC_DEF(idaEnableConstraintComponentExecution, void, PTR(void),
                 uint64_t)

//===---------------------------------------------------------------------===//
// idaRegisterConstraintVariableExecution

static void idaRegisterConstraintVariableExecution_void(
    void *instance, uint64_t variable, uint64_t componentId, void *owner) {
  static_cast<IDAInstance *>(instance)->registerConstraintVariableExecution(
      componentId, variable, static_cast<const char *>(owner));
}

RUNTIME_FUNC_DEF(idaRegisterConstraintVariableExecution, void, PTR(void),
                 uint64_t, uint64_t, PTR(void))

//===---------------------------------------------------------------------===//
// idaRegisterConstraintEquationExecution

static void idaRegisterConstraintEquationExecution_void(
    void *instance, uint64_t equation, uint64_t componentId, void *owner) {
  static_cast<IDAInstance *>(instance)->registerConstraintEquationExecution(
      componentId, equation, static_cast<const char *>(owner));
}

RUNTIME_FUNC_DEF(idaRegisterConstraintEquationExecution, void, PTR(void),
                 uint64_t, uint64_t, PTR(void))

//===---------------------------------------------------------------------===//
// idaRegisterConstraintStateSet

static void idaRegisterConstraintStateSet_void(
    void *instance, uint64_t componentId, uint64_t stateSetId, uint64_t active,
    void *canonicalKey) {
  static_cast<IDAInstance *>(instance)->registerConstraintStateSet(
      componentId, stateSetId, active != 0,
      static_cast<const char *>(canonicalKey));
}

RUNTIME_FUNC_DEF(idaRegisterConstraintStateSet, void, PTR(void), uint64_t,
                 uint64_t, uint64_t, PTR(void))

//===---------------------------------------------------------------------===//
// idaRegisterConstraintStateSetVariable

static void idaRegisterConstraintStateSetVariable_void(
    void *instance, uint64_t variable, uint64_t componentId,
    uint64_t stateSetId, void *owner, void *name) {
  static_cast<IDAInstance *>(instance)->registerConstraintStateSetVariable(
      componentId, stateSetId, variable, static_cast<const char *>(owner),
      static_cast<const char *>(name));
}

RUNTIME_FUNC_DEF(idaRegisterConstraintStateSetVariable, void, PTR(void),
                 uint64_t, uint64_t, uint64_t, PTR(void), PTR(void))

//===---------------------------------------------------------------------===//
// idaRegisterConstraintStateSetEquation

static void idaRegisterConstraintStateSetEquation_void(
    void *instance, uint64_t equation, uint64_t componentId,
    uint64_t stateSetId, void *owner, void *role, uint64_t order,
    void *matchedVariableName) {
  static_cast<IDAInstance *>(instance)->registerConstraintStateSetEquation(
      componentId, stateSetId, equation, static_cast<const char *>(owner),
      static_cast<const char *>(role), order,
      static_cast<const char *>(matchedVariableName));
}

RUNTIME_FUNC_DEF(idaRegisterConstraintStateSetEquation, void, PTR(void),
                 uint64_t, uint64_t, uint64_t, PTR(void), PTR(void), uint64_t,
                 PTR(void))

//===---------------------------------------------------------------------===//
// Constraint execution epochs

static void idaRegisterConstraintExecutionEpoch_void(
    void *instance, uint64_t domainId, uint64_t epochId, uint64_t active,
    void *canonicalKey, uint64_t preferDummyScalars,
    uint64_t defaultDummyScalars, uint64_t avoidDummyScalars) {
  static_cast<IDAInstance *>(instance)->registerConstraintExecutionEpoch(
      domainId, epochId, active != 0, static_cast<const char *>(canonicalKey),
      preferDummyScalars, defaultDummyScalars, avoidDummyScalars);
}

RUNTIME_FUNC_DEF(idaRegisterConstraintExecutionEpoch, void, PTR(void),
                 uint64_t, uint64_t, uint64_t, PTR(void), uint64_t, uint64_t,
                 uint64_t)

static void idaRegisterConstraintExecutionEpochTransition_void(
    void *instance, uint64_t domainId, uint64_t fromEpochId,
    uint64_t toEpochId) {
  static_cast<IDAInstance *>(instance)
      ->registerConstraintExecutionEpochTransition(domainId, fromEpochId,
                                                   toEpochId);
}

RUNTIME_FUNC_DEF(idaRegisterConstraintExecutionEpochTransition, void,
                 PTR(void), uint64_t, uint64_t, uint64_t)

static void idaRegisterConstraintExecutionEpochGroup_void(
    void *instance, uint64_t domainId, uint64_t epochId,
    uint64_t executionGroupId, void *kind, void *mode,
    uint64_t topologicalOrder) {
  static_cast<IDAInstance *>(instance)->registerConstraintExecutionEpochGroup(
      domainId, epochId, executionGroupId, static_cast<const char *>(kind),
      static_cast<const char *>(mode), topologicalOrder, 0, 0, 0, 0);
}

RUNTIME_FUNC_DEF(idaRegisterConstraintExecutionEpochGroup, void, PTR(void),
                 uint64_t, uint64_t, uint64_t, PTR(void), PTR(void), uint64_t)

static void idaRegisterConstraintExecutionEpochGroupCertificate_void(
    void *instance, uint64_t domainId, uint64_t epochId,
    uint64_t executionGroupId, uint64_t outerEquationScalars,
    uint64_t outerVariableScalars, uint64_t localEquationScalars,
    uint64_t localVariableScalars) {
  static_cast<IDAInstance *>(instance)
      ->registerConstraintExecutionEpochGroupCertificate(
          domainId, epochId, executionGroupId, outerEquationScalars,
          outerVariableScalars, localEquationScalars, localVariableScalars);
}

RUNTIME_FUNC_DEF(idaRegisterConstraintExecutionEpochGroupCertificate, void,
                 PTR(void), uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                 uint64_t, uint64_t)

static void idaRegisterConstraintExecutionEpochGroupMember_void(
    void *instance, uint64_t domainId, uint64_t epochId,
    uint64_t executionGroupId, uint64_t componentId) {
  static_cast<IDAInstance *>(instance)
      ->registerConstraintExecutionEpochGroupMember(
          domainId, epochId, executionGroupId, componentId);
}

RUNTIME_FUNC_DEF(idaRegisterConstraintExecutionEpochGroupMember, void,
                 PTR(void), uint64_t, uint64_t, uint64_t, uint64_t)

static void idaRegisterConstraintExecutionEpochGroupDependency_void(
    void *instance, uint64_t domainId, uint64_t epochId,
    uint64_t executionGroupId, uint64_t dependencyGroupId) {
  static_cast<IDAInstance *>(instance)
      ->registerConstraintExecutionEpochGroupDependency(
          domainId, epochId, executionGroupId, dependencyGroupId);
}

RUNTIME_FUNC_DEF(idaRegisterConstraintExecutionEpochGroupDependency, void,
                 PTR(void), uint64_t, uint64_t, uint64_t, uint64_t)

static void idaRegisterConstraintExecutionEpochVariable_void(
    void *instance, uint64_t variable, uint64_t domainId, uint64_t epochId,
    uint64_t executionGroupId, void *owner, void *name) {
  static_cast<IDAInstance *>(instance)->registerConstraintExecutionEpochVariable(
      domainId, epochId, executionGroupId, variable,
      static_cast<const char *>(owner), static_cast<const char *>(name));
}

RUNTIME_FUNC_DEF(idaRegisterConstraintExecutionEpochVariable, void,
                 PTR(void), uint64_t, uint64_t, uint64_t, uint64_t, PTR(void),
                 PTR(void))

static void idaRegisterConstraintExecutionEpochEquation_void(
    void *instance, uint64_t equation, uint64_t domainId, uint64_t epochId,
    uint64_t executionGroupId, void *owner) {
  static_cast<IDAInstance *>(instance)->registerConstraintExecutionEpochEquation(
      domainId, epochId, executionGroupId, equation,
      static_cast<const char *>(owner));
}

RUNTIME_FUNC_DEF(idaRegisterConstraintExecutionEpochEquation, void,
                 PTR(void), uint64_t, uint64_t, uint64_t, uint64_t, PTR(void))

static void idaRegisterConstraintExecutionEpochEquationMetadata_void(
    void *instance, uint64_t equation, uint64_t domainId, uint64_t epochId,
    void *role, uint64_t order, void *matchedVariableName) {
  static_cast<IDAInstance *>(instance)
      ->registerConstraintExecutionEpochEquationMetadata(
          domainId, epochId, equation, static_cast<const char *>(role), order,
          static_cast<const char *>(matchedVariableName));
}

RUNTIME_FUNC_DEF(idaRegisterConstraintExecutionEpochEquationMetadata, void,
                 PTR(void), uint64_t, uint64_t, uint64_t, PTR(void), uint64_t,
                 PTR(void))

//===---------------------------------------------------------------------===//
// idaSetTimeStep

static void idaSetTimeStep_void(void *instance, double timeStep) {
  static_cast<IDAInstance *>(instance)->setTimeStep(timeStep);
}

RUNTIME_FUNC_DEF(idaSetTimeStep, void, PTR(void), double)

//===---------------------------------------------------------------------===//
// idaGetCurrentTime

static double idaGetCurrentTime_f64(void *instance) {
  return static_cast<double>(
      static_cast<IDAInstance *>(instance)->getCurrentTime());
}

RUNTIME_FUNC_DEF(idaGetCurrentTime, double, PTR(void))

//===---------------------------------------------------------------------===//
// idaAddAlgebraicVariable

static uint64_t idaAddAlgebraicVariable_i64(void *instance, uint64_t rank,
                                            uint64_t *dimensions, void *getter,
                                            void *setter, void *name) {
  return static_cast<IDAInstance *>(instance)->addAlgebraicVariable(
      rank, dimensions, reinterpret_cast<VariableGetter>(getter),
      reinterpret_cast<VariableSetter>(setter),
      static_cast<const char *>(name));
}

RUNTIME_FUNC_DEF(idaAddAlgebraicVariable, uint64_t, PTR(void), uint64_t,
                 PTR(uint64_t), PTR(void), PTR(void), PTR(void))

//===---------------------------------------------------------------------===//
// idaAddStateVariable

static uint64_t idaAddStateVariable_i64(void *instance, uint64_t rank,
                                        uint64_t *dimensions, void *stateGetter,
                                        void *stateSetter,
                                        void *derivativeGetter,
                                        void *derivativeSetter, void *name) {
  return static_cast<IDAInstance *>(instance)->addStateVariable(
      rank, dimensions, reinterpret_cast<VariableGetter>(stateGetter),
      reinterpret_cast<VariableSetter>(stateSetter),
      reinterpret_cast<VariableGetter>(derivativeGetter),
      reinterpret_cast<VariableSetter>(derivativeSetter),
      static_cast<const char *>(name));
}

RUNTIME_FUNC_DEF(idaAddStateVariable, uint64_t, PTR(void), uint64_t,
                 PTR(uint64_t), PTR(void), PTR(void), PTR(void), PTR(void),
                 PTR(void))

//===---------------------------------------------------------------------===//
// idaSetVariableNominal

static void idaSetVariableNominal_void(void *instance, uint64_t variable,
                                       void *getter) {
  // 中文：nominal getter 必须在 solver 初始化前绑定到准确的 compact block；
  // 注册失败保留结构化诊断，不能退回随 trial point 变化的隐式尺度。
  // English: Bind the nominal getter to the exact compact block before solver
  // initialization. Registration failure is diagnosed rather than falling
  // back to a trial-point-dependent scale.
  if (!static_cast<IDAInstance *>(instance)->setVariableNominal(
          variable, reinterpret_cast<VariableGetter>(getter))) {
    std::cerr << "stage7e_invalid_nominal_registration variable=" << variable
              << std::endl;
  }
}

RUNTIME_FUNC_DEF(idaSetVariableNominal, void, PTR(void), uint64_t, PTR(void))

//===---------------------------------------------------------------------===//
// idaAddVariableAccess

static void idaAddVariableAccess_void(void *instance, uint64_t equationIndex,
                                      uint64_t variableIndex,
                                      void *accessFunction) {
  static_cast<IDAInstance *>(instance)->addVariableAccess(
      equationIndex, variableIndex,
      reinterpret_cast<AccessFunction>(accessFunction));
}

RUNTIME_FUNC_DEF(idaAddVariableAccess, void, PTR(void), uint64_t, uint64_t,
                 PTR(void))

//===---------------------------------------------------------------------===//
// idaAddVariableRangeAccess

static void idaAddVariableRangeAccess_void(void *instance,
                                           uint64_t equationIndex,
                                           uint64_t variableIndex,
                                           int64_t *ranges, uint64_t rank) {
  // 中文：range access 表达 equation-independent compact slice，主要供 helper
  // closure 的保守 sparsity 使用；它不改变 variable block 的 storage owner。
  // English: A range access denotes an equation-independent compact slice,
  // primarily for conservative helper-closure sparsity; it does not change
  // the variable block's storage owner.
  static_cast<IDAInstance *>(instance)->addVariableRangeAccess(
      equationIndex, variableIndex, ranges, rank);
}

RUNTIME_FUNC_DEF(idaAddVariableRangeAccess, void, PTR(void), uint64_t,
                 uint64_t, PTR(int64_t), uint64_t)

//===---------------------------------------------------------------------===//
// idaAddEquation

static uint64_t idaAddEquation_i64(void *instance, int64_t *ranges,
                                   uint64_t rank, void *stringRepresentation) {
  return static_cast<IDAInstance *>(instance)->addEquation(
      ranges, rank, static_cast<const char *>(stringRepresentation));
}

RUNTIME_FUNC_DEF(idaAddEquation, uint64_t, PTR(void), PTR(int64_t), uint64_t,
                 PTR(void))

//===---------------------------------------------------------------------===//
// idaSetResidual

static void idaSetResidual_void(void *instance, uint64_t equationIndex,
                                void *residualFunction) {
  static_cast<IDAInstance *>(instance)->setResidualFunction(
      equationIndex, reinterpret_cast<ResidualFunction>(residualFunction));
}

RUNTIME_FUNC_DEF(idaSetResidual, void, PTR(void), uint64_t, PTR(void))

//===---------------------------------------------------------------------===//
// idaAddJacobian

static void idaAddJacobian_void(void *instance, uint64_t equationIndex,
                                uint64_t variableIndex, void *jacobianFunction,
                                uint64_t numOfSeeds, uint64_t *seedSizes) {
  static_cast<IDAInstance *>(instance)->addJacobianFunction(
      equationIndex, variableIndex,
      reinterpret_cast<JacobianFunction>(jacobianFunction), numOfSeeds,
      seedSizes);
}

RUNTIME_FUNC_DEF(idaAddJacobian, void, PTR(void), uint64_t, uint64_t, PTR(void),
                 uint64_t, PTR(uint64_t))

//===---------------------------------------------------------------------===//
// idaPrintStatistics

static void printStatistics_void(void *instance) {
  static_cast<IDAInstance *>(instance)->printStatistics();
}

RUNTIME_FUNC_DEF(printStatistics, void, PTR(void))

#endif // SUNDIALS_ENABLE
