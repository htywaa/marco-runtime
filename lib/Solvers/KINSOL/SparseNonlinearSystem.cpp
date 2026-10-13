#ifdef SUNDIALS_ENABLE

#include "marco/Runtime/Solvers/KINSOL/SparseNonlinearSystem.h"
#include "marco/Runtime/Support/MemoryManagement.h"
#include "marco/Runtime/Simulation/Options.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

#if SUNDIALS_VERSION_MAJOR >= 7
#ifdef MPI_ENABLE
#include <mpi.h>
#endif
#endif

namespace marco::runtime::sundials::kinsol {

namespace {

// 中文：只观察本来发生的调用；profile 关闭时不增加回调、分配或读时钟，实例仍保持原非并发使用契约。
// Instrument only calls that already occur. No callback, allocation or clock
// read is introduced when profiling is disabled. A solver instance retains
// its existing non-concurrent-use contract.
class PassiveScope {
public:
  PassiveScope(uint64_t &calls, uint64_t &nanoseconds)
      : enabled(simulation::getOptions().profiling), total(nanoseconds) {
    if (enabled) {
      ++calls;
      start = std::chrono::steady_clock::now();
    }
  }
  ~PassiveScope() {
    if (enabled)
      total += std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - start).count();
  }
private:
  bool enabled;
  uint64_t &total;
  std::chrono::steady_clock::time_point start;
};

// 中文：小型 condition block 使用局部 partial-pivot LU；该 dense 存储严格受
// denseLimit 限制，绝不扩展成全模型 dense Jacobian。
// English: Small condition blocks use local partial-pivot LU. Dense storage is
// strictly bounded by denseLimit and is never expanded into a full-model dense
// Jacobian.
SparseNonlinearSystem::ConditionEstimate estimateDenseCondition(
    const std::vector<double> &matrix, uint64_t size,
    std::vector<double> &lu, std::vector<uint64_t> &permutation,
    std::vector<double> &rhs, std::vector<double> &y,
    std::vector<double> &solution) {
  SparseNonlinearSystem::ConditionEstimate result;
  if (size == 0) {
    result.oneNormCondition = 1;
    result.reciprocalCondition = 1;
    return result;
  }

  lu.assign(matrix.begin(), matrix.end());
  permutation.resize(size);
  std::iota(permutation.begin(), permutation.end(), 0);
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
  double rankTolerance = 100 * std::numeric_limits<double>::epsilon() *
                         std::max(1.0, infinityNorm);
  result.minimumPivot = std::numeric_limits<double>::infinity();

  for (uint64_t column = 0; column < size; ++column) {
    uint64_t pivot = column;
    double pivotMagnitude = 0;
    for (uint64_t row = column; row < size; ++row) {
      double candidate = std::abs(lu[row * size + column]);
      if (candidate > pivotMagnitude) {
        pivot = row;
        pivotMagnitude = candidate;
      }
    }
    if (pivotMagnitude <= rankTolerance) {
      result.rankDeficient = true;
      result.oneNormCondition = std::numeric_limits<double>::infinity();
      result.reciprocalCondition = 0;
      result.minimumPivot = result.rank == 0 ? 0 : result.minimumPivot;
      return result;
    }
    if (pivot != column) {
      for (uint64_t j = 0; j < size; ++j) {
        std::swap(lu[column * size + j], lu[pivot * size + j]);
      }
      std::swap(permutation[column], permutation[pivot]);
    }
    result.minimumPivot = std::min(result.minimumPivot, pivotMagnitude);
    result.maximumPivot = std::max(result.maximumPivot, pivotMagnitude);
    ++result.rank;
    for (uint64_t row = column + 1; row < size; ++row) {
      lu[row * size + column] /= lu[column * size + column];
      for (uint64_t j = column + 1; j < size; ++j) {
        lu[row * size + j] -=
            lu[row * size + column] * lu[column * size + j];
      }
    }
  }

  double inverseOneNorm = 0;
  if (size <= 16) {
    // 中文：共同求解独立逆矩阵列：每列前代/回代与求和顺序仍和原实现一致，只交错独立右端项。
    //       不同查询间不复用 condition、主元阈值或数值结果，实例额外 scratch 限为 2×16×16 个 double。
    // Solve the independent inverse columns together. Each individual column
    // retains exactly the original forward/backward substitution and summation
    // order; only independent right-hand sides are interleaved. No condition
    // check, pivot threshold or numerical result is reused between queries.
    // Bound the extra scratch to 2 * 16 * 16 doubles per instance.
    y.resize(size * size);
    solution.resize(size * size);
    for (uint64_t row = 0; row < size; ++row) {
      double *target = y.data() + row * size;
      for (uint64_t column = 0; column < size; ++column)
        target[column] = permutation[row] == column ? 1.0 : 0.0;
      for (uint64_t j = 0; j < row; ++j) {
        const double coefficient = lu[row * size + j];
        const double *source = y.data() + j * size;
        for (uint64_t column = 0; column < size; ++column)
          target[column] -= coefficient * source[column];
      }
    }
    for (uint64_t reverse = size; reverse > 0; --reverse) {
      const uint64_t row = reverse - 1;
      double *target = solution.data() + row * size;
      std::copy_n(y.data() + row * size, size, target);
      for (uint64_t j = row + 1; j < size; ++j) {
        const double coefficient = lu[row * size + j];
        const double *source = solution.data() + j * size;
        for (uint64_t column = 0; column < size; ++column)
          target[column] -= coefficient * source[column];
      }
      for (uint64_t column = 0; column < size; ++column)
        target[column] /= lu[row * size + row];
    }
    for (uint64_t column = 0; column < size; ++column) {
      double sum = 0;
      for (uint64_t row = 0; row < size; ++row)
        sum += std::abs(solution[row * size + column]);
      inverseOneNorm = std::max(inverseOneNorm, sum);
    }
    result.oneNormCondition = oneNorm * inverseOneNorm;
    result.reciprocalCondition =
        result.oneNormCondition > 0 ? 1.0 / result.oneNormCondition : 1;
    return result;
  }
  rhs.assign(size, 0);
  y.assign(size, 0);
  solution.assign(size, 0);
  for (uint64_t inverseColumn = 0; inverseColumn < size; ++inverseColumn) {
    std::fill(rhs.begin(), rhs.end(), 0);
    rhs[inverseColumn] = 1;
    for (uint64_t row = 0; row < size; ++row) {
      y[row] = rhs[permutation[row]];
      for (uint64_t j = 0; j < row; ++j) {
        y[row] -= lu[row * size + j] * y[j];
      }
    }
    for (uint64_t reverse = size; reverse > 0; --reverse) {
      uint64_t row = reverse - 1;
      solution[row] = y[row];
      for (uint64_t j = row + 1; j < size; ++j) {
        solution[row] -= lu[row * size + j] * solution[j];
      }
      solution[row] /= lu[row * size + row];
    }
    double columnSum = 0;
    for (double value : solution) {
      columnSum += std::abs(value);
    }
    inverseOneNorm = std::max(inverseOneNorm, columnSum);
  }
  result.oneNormCondition = oneNorm * inverseOneNorm;
  result.reciprocalCondition =
      result.oneNormCondition > 0 ? 1.0 / result.oneNormCondition : 1;
  return result;
}

} // namespace

SparseNonlinearSystem::SparseNonlinearSystem(
    Configuration configuration, ResidualCallback residualCallback,
    JacobianCallback jacobianCallback)
    : configuration(configuration),
      residualCallback(std::move(residualCallback)),
      jacobianCallback(std::move(jacobianCallback)) {
  if (simulation::getOptions().profiling) {
    static std::atomic<uint64_t> nextStatisticsId{0};
    statistics.id = ++nextStatisticsId;
  }
#if SUNDIALS_VERSION_MAJOR >= 7
#ifdef MPI_ENABLE
  comm = MPI_COMM_WORLD;
#else
  comm = SUN_COMM_NULL;
#endif
#endif
}

SparseNonlinearSystem::~SparseNonlinearSystem() {
  if (simulation::getOptions().profiling)
    printStatistics("final");
  if (memory != nullptr) {
    KINFree(&memory);
  }
  if (linearSolver != nullptr) {
    SUNLinSolFree(linearSolver);
  }
  if (sparseMatrix != nullptr) {
    SUNMatDestroy(sparseMatrix);
  }
  if (variablesVector != nullptr) {
    N_VDestroy(variablesVector);
  }
  if (variableScaleVector != nullptr) {
    N_VDestroy(variableScaleVector);
  }
  if (residualScaleVector != nullptr) {
    N_VDestroy(residualScaleVector);
  }
  if (factorResidualVector != nullptr) {
    N_VDestroy(factorResidualVector);
  }
  if (linearRightHandSideVector != nullptr) {
    N_VDestroy(linearRightHandSideVector);
  }
  if (linearSolutionVector != nullptr) {
    N_VDestroy(linearSolutionVector);
  }
#if SUNDIALS_VERSION_MAJOR >= 6
  if (context != nullptr) {
    SUNContext_Free(&context);
  }
#endif
}

bool SparseNonlinearSystem::initialize() {
  // 中文：该内核封装独立的 KINSOL/KLU 生命周期，使普通 KINSOL 和 component
  // closure 共享相同稀疏求解语义而不共享全局 CLI 状态。
  // English: This kernel owns an independent KINSOL/KLU lifecycle so ordinary
  // KINSOL and component closure share sparse-solve semantics without sharing
  // global CLI state.
  if (initialized) {
    return true;
  }
  if (configuration.size == 0) {
    initialized = true;
    return true;
  }

#if SUNDIALS_VERSION_MAJOR >= 7
  if (SUNContext_Create(comm, &context) != 0) {
    return false;
  }
#elif SUNDIALS_VERSION_MAJOR >= 6
  if (SUNContext_Create(nullptr, &context) != 0) {
    return false;
  }
#endif

#if SUNDIALS_VERSION_MAJOR >= 6
  variablesVector = N_VNew_Serial(configuration.size, context);
  variableScaleVector = N_VNew_Serial(configuration.size, context);
  residualScaleVector = N_VNew_Serial(configuration.size, context);
#else
  variablesVector = N_VNew_Serial(configuration.size);
  variableScaleVector = N_VNew_Serial(configuration.size);
  residualScaleVector = N_VNew_Serial(configuration.size);
#endif
  if (variablesVector == nullptr || variableScaleVector == nullptr ||
      residualScaleVector == nullptr) {
    return false;
  }
  N_VConst(0, variablesVector);
  N_VConst(1, variableScaleVector);
  N_VConst(1, residualScaleVector);
  currentVariableScale.assign(configuration.size, 1);
  currentResidualScale.assign(configuration.size, 1);

#if SUNDIALS_VERSION_MAJOR >= 6
  memory = KINCreate(context);
#else
  memory = KINCreate();
#endif
  if (memory == nullptr ||
      KINInit(memory, residualBridge, variablesVector) != KIN_SUCCESS ||
      KINSetUserData(memory, this) != KIN_SUCCESS) {
    return false;
  }

  if (configuration.quietErrors &&
      KINSetErrHandlerFn(memory, quietErrorHandler, this) != KIN_SUCCESS) {
    return false;
  }
  if (KINSetFuncNormTol(memory, configuration.functionNormTolerance) !=
          KIN_SUCCESS ||
      KINSetScaledStepTol(memory, configuration.scaledStepTolerance) !=
          KIN_SUCCESS ||
      KINSetMaxNewtonStep(memory, configuration.maximumNewtonStep) !=
          KIN_SUCCESS) {
    return false;
  }

#if SUNDIALS_VERSION_MAJOR >= 6
  sparseMatrix = SUNSparseMatrix(configuration.size, configuration.size,
                                 configuration.nonZeros, CSR_MAT, context);
#else
  sparseMatrix = SUNSparseMatrix(configuration.size, configuration.size,
                                 configuration.nonZeros, CSR_MAT);
#endif
  if (sparseMatrix == nullptr) {
    return false;
  }
#if SUNDIALS_VERSION_MAJOR >= 6
  linearSolver = SUNLinSol_KLU(variablesVector, sparseMatrix, context);
#else
  linearSolver = SUNLinSol_KLU(variablesVector, sparseMatrix);
#endif
  if (linearSolver == nullptr ||
      KINSetLinearSolver(memory, linearSolver, sparseMatrix) != KINLS_SUCCESS ||
      KINSetJacFn(memory, jacobianBridge) != KIN_SUCCESS) {
    return false;
  }

  initialized = true;
  return true;
}

SparseNonlinearSystem::SolveStatus
// 中文：原始停止码与每次调用迭代数分别保存；步长停滞不能冒充函数范数已达标。
// English: Retain raw stop flags and per-call iteration counts separately; small-step
// stagnation does not imply function-norm convergence.
SparseNonlinearSystem::solve(realtype time) {
  PassiveScope observation(statistics.solves, statistics.solveNanoseconds);
  explicitFactorizationValid = false;
  lastNonlinearIterations = 0;
  // 中文：仅 callback 明确失败或 KINSOL 报告结构/API 错误时才升级为 fatal；
  // 普通 Newton/线搜索失败保留为 recoverable，交由 IDA 的试探步策略处理。
  // English: Only callback failures or structural/API KINSOL errors become
  // fatal. Ordinary Newton or line-search failures remain recoverable for the
  // enclosing IDA trial-step policy.
  if (!initialize()) {
    return SolveStatus::FatalFailure;
  }
  if (configuration.size == 0) {
    return SolveStatus::Success;
  }
  currentTime = time;
  callbackFailed = false;
  lastSolveFlag = KINSol(memory, variablesVector,
                         configuration.lineSearch ? KIN_LINESEARCH : KIN_NONE,
                         variableScaleVector, residualScaleVector);
  // 中文：KINSol 每次调用重置该计数；减去前一次求解值会漏记重复局部闭包的迭代。
  // KINSol resets this counter on each invocation. Subtracting the previous
  // solve's count silently loses iterations in repeated local closures.
  long int iterationsAfter = 0;
  if (KINGetNumNonlinSolvIters(memory, &iterationsAfter) != KIN_SUCCESS) {
    return SolveStatus::FatalFailure;
  }
  lastNonlinearIterations = static_cast<uint64_t>(
      std::max<long int>(0, iterationsAfter));
  if (simulation::getOptions().profiling) {
    statistics.nonlinearIterations += lastNonlinearIterations;
    ++statistics.solveFlags[lastSolveFlag];
  }
  return classifySolveFlag(lastSolveFlag, callbackFailed);
}

SparseNonlinearSystem::FactorizationStatus
SparseNonlinearSystem::evaluateJacobian(realtype time) {
  return evaluateJacobianImpl(time, nullptr);
}

SparseNonlinearSystem::FactorizationStatus
SparseNonlinearSystem::evaluateJacobianImpl(
    realtype time, const std::vector<double> *residuals) {
  explicitFactorizationValid = false;
  // 中文：factorization 路径显式刷新 residual 后再刷新 Jacobian，确保 AD 与
  // component 依赖使用同一个当前试探状态。
  // English: The factorization path explicitly refreshes residuals before the
  // Jacobian so AD and component dependencies observe the same trial state.
  if (!initialize())
    return FactorizationStatus::SolverFailure;
  if (configuration.size == 0)
    return FactorizationStatus::Success;
  currentTime = time;
  callbackFailed = false;
  if (factorResidualVector == nullptr) {
    factorResidualVector = N_VClone(variablesVector);
  }
  if (factorResidualVector == nullptr) {
    return FactorizationStatus::ResourceFailure;
  }
  int residualResult;
  if (residuals) {
    if (residuals->size() != configuration.size)
      return FactorizationStatus::EvaluationFailure;
    std::copy(residuals->begin(), residuals->end(),
              N_VGetArrayPointer(factorResidualVector));
    residualResult = std::all_of(residuals->begin(), residuals->end(),
                                 [](double x) { return std::isfinite(x); })
                         ? KIN_SUCCESS : KIN_SYSFUNC_FAIL;
    if (simulation::getOptions().profiling)
      ++statistics.reusedResiduals;
    observeResidual(factorResidualVector, residualResult);
  } else {
    residualResult = residualBridge(variablesVector, factorResidualVector, this);
  }
  int jacobianResult = residualResult == KIN_SUCCESS
                           ? jacobianBridge(variablesVector, factorResidualVector,
                                            sparseMatrix, this, nullptr,
                                            nullptr)
                           : KIN_SYSFUNC_FAIL;
  return jacobianResult == KIN_SUCCESS && !callbackFailed
             ? FactorizationStatus::Success
             : FactorizationStatus::EvaluationFailure;
}

bool SparseNonlinearSystem::factorize(realtype time) {
  return factorizeWithStatus(time) == FactorizationStatus::Success;
}

SparseNonlinearSystem::FactorizationStatus
SparseNonlinearSystem::factorizeWithStatus(realtype time) {
  return factorizeImpl(time, nullptr);
}

SparseNonlinearSystem::FactorizationStatus
SparseNonlinearSystem::factorizeWithResidual(
    realtype time, const std::vector<double> &residuals) {
  return factorizeImpl(time, &residuals);
}

SparseNonlinearSystem::FactorizationStatus
SparseNonlinearSystem::factorizeImpl(
    realtype time, const std::vector<double> *residuals) {
  PassiveScope observation(statistics.factorizations,
                           statistics.factorNanoseconds);
  auto evaluated = evaluateJacobianImpl(time, residuals);
  if (evaluated != FactorizationStatus::Success || configuration.size == 0)
    return evaluated;
  if (SUNLinSolSetup(linearSolver, sparseMatrix) == SUNLS_SUCCESS) {
    explicitFactorizationValid = true;
    return FactorizationStatus::Success;
  }
  sun_klu_common *common = SUNLinSol_KLUGetCommon(linearSolver);
  const auto status = common ? common->status : KLU_INVALID;
  // 中文：KLU refactor 失败可能留下部分数值分解；保留当前 trial 的失败，但让下一次明确 setup 或非线性求解重新分解。
  //       不能在此重试奇异 trial。
  // A failed KLU refactor may leave a partial numeric factorization. Keep the
  // current trial's failure, but force a fresh factorization for the next
  // explicit setup or nonlinear solve. This does not retry a singular trial.
  if (SUNLinSolInitialize(linearSolver) != SUNLS_SUCCESS)
    return FactorizationStatus::SolverFailure;
  if (status == KLU_SINGULAR)
    return FactorizationStatus::Singular;
  if (status == KLU_OUT_OF_MEMORY || status == KLU_TOO_LARGE)
    return FactorizationStatus::ResourceFailure;
  return FactorizationStatus::SolverFailure;
}

bool SparseNonlinearSystem::solveFactorized(
    const std::vector<double> &rhs, std::vector<double> &solution) {
  PassiveScope observation(statistics.rightHandSides, statistics.rhsNanoseconds);
  // 中文：复用最近一次 SUNLinSolSetup 的 KLU 数值分解；这里不重建矩阵，
  // 因而一次 outer Jacobian 可高效求解多个 Schur sensitivity RHS。
  // English: Reuse the most recent KLU numeric factorization from
  // SUNLinSolSetup. The matrix is not rebuilt, enabling multiple Schur
  // sensitivity right-hand sides per outer Jacobian evaluation.
  if (!explicitFactorizationValid || rhs.size() != configuration.size ||
      variablesVector == nullptr) {
    return false;
  }
  if (linearRightHandSideVector == nullptr) {
    linearRightHandSideVector = N_VClone(variablesVector);
  }
  if (linearSolutionVector == nullptr) {
    linearSolutionVector = N_VClone(variablesVector);
  }
  if (linearRightHandSideVector == nullptr || linearSolutionVector == nullptr) {
    return false;
  }
  std::copy(rhs.begin(), rhs.end(),
            N_VGetArrayPointer(linearRightHandSideVector));
  N_VConst(0, linearSolutionVector);
  int result = SUNLinSolSolve(linearSolver, sparseMatrix, linearSolutionVector,
                              linearRightHandSideVector, 0.0);
  if (result == SUNLS_SUCCESS) {
    const double *values = N_VGetArrayPointer(linearSolutionVector);
    solution.assign(values, values + configuration.size);
  }
  return result == SUNLS_SUCCESS;
}

bool SparseNonlinearSystem::setScaling(
    const std::vector<double> &variableScale,
    const std::vector<double> &residualScale) {
  // 中文：在写入 SUNDIALS vector 前完整验证尺度，避免部分更新造成 condition
  // probe 与 nonlinear solve 使用不同的权重。
  // English: Validate every scale before touching SUNDIALS vectors so a
  // partial update cannot make condition probes and nonlinear solves use
  // different weights.
  if (!initialize() || variableScale.size() != configuration.size ||
      residualScale.size() != configuration.size) {
    return false;
  }
  for (uint64_t i = 0; i < configuration.size; ++i) {
    if (!(variableScale[i] > 0) || !std::isfinite(variableScale[i]) ||
        !(residualScale[i] > 0) || !std::isfinite(residualScale[i])) {
      return false;
    }
  }
  std::copy(variableScale.begin(), variableScale.end(),
            N_VGetArrayPointer(variableScaleVector));
  std::copy(residualScale.begin(), residualScale.end(),
            N_VGetArrayPointer(residualScaleVector));
  currentVariableScale = variableScale;
  currentResidualScale = residualScale;
  return true;
}

bool SparseNonlinearSystem::setScalingFromVariableNominals(
    const std::vector<double> &variableNominals) {
  // 中文：变量按 nominal 缩放，残差按初始化 Jacobian 的加权行范数缩放；尺度
  // 一经建立便保持固定，便于 condition 与收敛证书跨试探点比较。
  // English: Variables are scaled by nominal and residuals by weighted row
  // norms of the initialization Jacobian. The resulting scales remain fixed so
  // condition and convergence certificates are comparable across trial points.
  if (variableNominals.size() != configuration.size || sparseMatrix == nullptr) {
    return false;
  }
  std::vector<double> variableScale(configuration.size, 1);
  std::vector<double> residualScale(configuration.size, 1);
  for (uint64_t column = 0; column < configuration.size; ++column) {
    if (!(variableNominals[column] > 0) ||
        !std::isfinite(variableNominals[column])) {
      return false;
    }
    variableScale[column] = 1.0 / variableNominals[column];
  }

  const sunindextype *rowPointers =
      SUNSparseMatrix_IndexPointers(sparseMatrix);
  const sunindextype *columnIndices =
      SUNSparseMatrix_IndexValues(sparseMatrix);
  const realtype *matrixValues = SUNSparseMatrix_Data(sparseMatrix);
  for (uint64_t row = 0; row < configuration.size; ++row) {
    double rowMagnitude = 0;
    for (sunindextype entry = rowPointers[row];
         entry < rowPointers[row + 1]; ++entry) {
      uint64_t column = static_cast<uint64_t>(columnIndices[entry]);
      rowMagnitude += std::abs(matrixValues[entry]) * variableNominals[column];
    }
    residualScale[row] = 1.0 / std::max(1.0, rowMagnitude);
  }
  return setScaling(variableScale, residualScale);
}

bool SparseNonlinearSystem::extractEntries(
    const std::vector<uint64_t> &rows,
    const std::vector<uint64_t> &columns,
    std::vector<double> &values) const {
  // 中文：调用者以(row,column)配对请求局部条目；未显式存储的稀疏项按结构零返回，
  // 不把局部 probe 扩展成 dense 全矩阵。
  // English: Callers request paired (row,column) entries. Structurally absent
  // sparse entries return zero without expanding the probe into a full dense
  // matrix.
  if (rows.size() != columns.size() || sparseMatrix == nullptr) {
    return false;
  }
  const sunindextype *rowPointers =
      SUNSparseMatrix_IndexPointers(sparseMatrix);
  const sunindextype *columnIndices =
      SUNSparseMatrix_IndexValues(sparseMatrix);
  const realtype *matrixValues = SUNSparseMatrix_Data(sparseMatrix);
  values.assign(rows.size(), 0);
  for (uint64_t i = 0; i < rows.size(); ++i) {
    if (rows[i] >= configuration.size || columns[i] >= configuration.size) {
      return false;
    }
    for (sunindextype entry = rowPointers[rows[i]];
         entry < rowPointers[rows[i] + 1]; ++entry) {
      if (static_cast<uint64_t>(columnIndices[entry]) == columns[i]) {
        values[i] = matrixValues[entry];
        break;
      }
    }
  }
  return true;
}

bool SparseNonlinearSystem::estimateFlatCondition(
    const double *matrix, const std::vector<uint64_t> &rows,
    const std::vector<uint64_t> &columns, ConditionEstimate &estimate) const {
  PassiveScope observation(statistics.conditions,statistics.conditionNanoseconds);
  const size_t n=rows.size();
  if (!matrix || !n || n>16 || columns.size()!=n ||
      currentVariableScale.size()!=configuration.size ||
      currentResidualScale.size()!=configuration.size) return false;
  auto &w=conditionWorkspace;
  w.dense.resize(n*n);
  uint64_t seenRows=0,seenColumns=0;
  for(size_t i=0;i<n;++i) {
    if(rows[i]>=configuration.size || columns[i]>=configuration.size ||
       rows[i]>=64 || columns[i]>=64 || (seenRows&(uint64_t(1)<<rows[i])) ||
       (seenColumns&(uint64_t(1)<<columns[i]))) return false;
    seenRows|=uint64_t(1)<<rows[i];seenColumns|=uint64_t(1)<<columns[i];
    for(size_t j=0;j<n;++j) {
      if(columns[j]>=configuration.size) return false;
      double value=currentResidualScale[rows[i]]*
          matrix[rows[i]*configuration.size+columns[j]]/currentVariableScale[columns[j]];
      if(!std::isfinite(value)) return false;
      w.dense[i*n+j]=value;
    }
  }
  estimate=estimateDenseCondition(w.dense,n,w.lu,w.permutation,w.rhs,w.forward,w.solution);
  return true;
}


bool SparseNonlinearSystem::estimateScaledSubmatrixCondition(
    const std::vector<uint64_t> &rows,
    const std::vector<uint64_t> &columns, ConditionEstimate &estimate,
    uint64_t denseLimit) const {
  PassiveScope observation(statistics.conditions,
                           statistics.conditionNanoseconds);
  estimate = {};
  if (rows.size() != columns.size() || sparseMatrix == nullptr ||
      currentVariableScale.size() != configuration.size ||
      currentResidualScale.size() != configuration.size) {
    return false;
  }
  const uint64_t size = rows.size();
  if (size == 0) {
    estimate.oneNormCondition = 1;
    estimate.reciprocalCondition = 1;
    return true;
  }

  auto &workspace = conditionWorkspace;
  auto &localColumns = workspace.localColumns;
  auto &localRows = workspace.localRows;
  localColumns.assign(configuration.size, -1);
  localRows.assign(configuration.size, false);
  for (uint64_t i = 0; i < size; ++i) {
    if (rows[i] >= configuration.size || columns[i] >= configuration.size ||
        localRows[rows[i]] || localColumns[columns[i]] >= 0 ||
        !(currentVariableScale[columns[i]] > 0) ||
        !(currentResidualScale[rows[i]] > 0)) {
      return false;
    }
    localRows[rows[i]] = true;
    localColumns[columns[i]] = static_cast<int64_t>(i);
  }

  const sunindextype *rowPointers =
      SUNSparseMatrix_IndexPointers(sparseMatrix);
  const sunindextype *columnIndices =
      SUNSparseMatrix_IndexValues(sparseMatrix);
  const realtype *matrixValues = SUNSparseMatrix_Data(sparseMatrix);

  // 中文：小块精确估计，大块直接在抽取出的稀疏 CSC 上使用 KLU 的 rcond/condest；
  // 两条路径都只观察请求的 component 子矩阵。
  // English: Small blocks use the exact local estimate, while large blocks use
  // KLU rcond/condest on an extracted sparse CSC matrix. Both paths inspect only
  // the requested component submatrix.
  if (size <= denseLimit) {
    auto &dense = workspace.dense;
    dense.assign(size * size, 0);
    for (uint64_t localRow = 0; localRow < size; ++localRow) {
      uint64_t row = rows[localRow];
      for (sunindextype entry = rowPointers[row];
           entry < rowPointers[row + 1]; ++entry) {
        uint64_t column = static_cast<uint64_t>(columnIndices[entry]);
        int64_t localColumn = localColumns[column];
        if (localColumn < 0) {
          continue;
        }
        dense[localRow * size + static_cast<uint64_t>(localColumn)] =
            currentResidualScale[row] * matrixValues[entry] /
            currentVariableScale[column];
      }
    }
    estimate = estimateDenseCondition(dense, size, workspace.lu,
                                      workspace.permutation, workspace.rhs,
                                      workspace.forward, workspace.solution);
    return true;
  }

  struct Entry {
    sunindextype row;
    double value;
  };
  std::vector<std::vector<Entry>> columnEntries(size);
  double infinityNorm = 0;
  for (uint64_t localRow = 0; localRow < size; ++localRow) {
    uint64_t row = rows[localRow];
    double rowMagnitude = 0;
    for (sunindextype entry = rowPointers[row];
         entry < rowPointers[row + 1]; ++entry) {
      uint64_t column = static_cast<uint64_t>(columnIndices[entry]);
      int64_t localColumn = localColumns[column];
      if (localColumn < 0) {
        continue;
      }
      double value = currentResidualScale[row] * matrixValues[entry] /
                     currentVariableScale[column];
      if (value == 0) {
        continue;
      }
      columnEntries[static_cast<uint64_t>(localColumn)].push_back(
          {static_cast<sunindextype>(localRow), value});
      rowMagnitude += std::abs(value);
    }
    infinityNorm = std::max(infinityNorm, rowMagnitude);
  }
  uint64_t nonZeros = 0;
  for (auto &entries : columnEntries) {
    std::sort(entries.begin(), entries.end(),
              [](const Entry &left, const Entry &right) {
                return left.row < right.row;
              });
    nonZeros += entries.size();
  }
  if (nonZeros == 0) {
    estimate.rankDeficient = true;
    estimate.oneNormCondition = std::numeric_limits<double>::infinity();
    estimate.sparseEstimator = true;
    return true;
  }

#if SUNDIALS_VERSION_MAJOR >= 6
  SUNMatrix conditionMatrix =
      SUNSparseMatrix(size, size, nonZeros, CSC_MAT, context);
  N_Vector workVector = N_VNew_Serial(size, context);
#else
  SUNMatrix conditionMatrix = SUNSparseMatrix(size, size, nonZeros, CSC_MAT);
  N_Vector workVector = N_VNew_Serial(size);
#endif
  if (conditionMatrix == nullptr || workVector == nullptr) {
    if (conditionMatrix != nullptr) {
      SUNMatDestroy(conditionMatrix);
    }
    if (workVector != nullptr) {
      N_VDestroy(workVector);
    }
    return false;
  }
  sunindextype *conditionPointers =
      SUNSparseMatrix_IndexPointers(conditionMatrix);
  sunindextype *conditionRows = SUNSparseMatrix_IndexValues(conditionMatrix);
  realtype *conditionValues = SUNSparseMatrix_Data(conditionMatrix);
  sunindextype offset = 0;
  conditionPointers[0] = 0;
  for (uint64_t column = 0; column < size; ++column) {
    for (const Entry &entry : columnEntries[column]) {
      conditionRows[offset] = entry.row;
      conditionValues[offset] = entry.value;
      ++offset;
    }
    conditionPointers[column + 1] = offset;
  }

#if SUNDIALS_VERSION_MAJOR >= 6
  SUNLinearSolver conditionSolver =
      SUNLinSol_KLU(workVector, conditionMatrix, context);
#else
  SUNLinearSolver conditionSolver = SUNLinSol_KLU(workVector, conditionMatrix);
#endif
  if (conditionSolver == nullptr) {
    N_VDestroy(workVector);
    SUNMatDestroy(conditionMatrix);
    return false;
  }

  estimate.sparseEstimator = true;
  int setupResult = SUNLinSolSetup(conditionSolver, conditionMatrix);
  sun_klu_common *common = SUNLinSol_KLUGetCommon(conditionSolver);
  bool certificateAvailable = true;
  if (setupResult != SUNLS_SUCCESS || common == nullptr) {
    certificateAvailable = common && common->status == KLU_SINGULAR;
    if (certificateAvailable) {
      estimate.rank = common->numerical_rank < 0
                          ? 0
                          : static_cast<uint64_t>(common->numerical_rank);
      estimate.rankDeficient = true;
      estimate.oneNormCondition = std::numeric_limits<double>::infinity();
    }
  } else {
    sun_klu_symbolic *symbolic = SUNLinSol_KLUGetSymbolic(conditionSolver);
    sun_klu_numeric *numeric = SUNLinSol_KLUGetNumeric(conditionSolver);
    int reciprocalResult =
        symbolic != nullptr && numeric != nullptr
            ? sun_klu_rcond(symbolic, numeric, common)
            : 0;
    int conditionResult =
        symbolic != nullptr && numeric != nullptr
            ? sun_klu_condest(conditionPointers, conditionValues, symbolic,
                              numeric, common)
            : 0;
    estimate.rank = common->numerical_rank < 0
                        ? size
                        : static_cast<uint64_t>(common->numerical_rank);
    estimate.reciprocalCondition =
        reciprocalResult != 0 ? common->rcond : 0;
    estimate.oneNormCondition =
        conditionResult != 0 && std::isfinite(common->condest)
            ? common->condest
            : (common->rcond > 0 ? 1.0 / common->rcond
                                 : std::numeric_limits<double>::infinity());
    double rankTolerance = 100 * std::numeric_limits<double>::epsilon() *
                           std::max(1.0, infinityNorm);
    estimate.rankDeficient =
        estimate.rank < size || !(estimate.reciprocalCondition > rankTolerance);
  }
  SUNLinSolFree(conditionSolver);
  N_VDestroy(workVector);
  SUNMatDestroy(conditionMatrix);
  return certificateAvailable;
}

int SparseNonlinearSystem::residualBridge(N_Vector variables,
                                          N_Vector residuals,
                                          void *userData) {
  // 中文：单独记录 callbackFailed，因为若只查看 KINSOL 最终 flag，会把用户
  // callback 的结构错误误判为可恢复的 nonlinear convergence failure。
  // English: Track callbackFailed separately because the final KINSOL flag
  // alone could misclassify a user callback's structural error as a
  // recoverable nonlinear convergence failure.
  auto *system = static_cast<SparseNonlinearSystem *>(userData);
  if (simulation::getOptions().profiling)
    ++system->statistics.residuals;
  int result = system->residualCallback(variables, residuals);
  system->observeResidual(residuals, result);
  return result;
}

void SparseNonlinearSystem::observeResidual(N_Vector residuals, int result) {
  lastResidualObservation.available = false;
  if (simulation::getOptions().profiling && result == KIN_SUCCESS) {
    auto &observation = lastResidualObservation;
    observation = {};
    observation.available = true;
    const realtype *values = N_VGetArrayPointer(residuals);
    for (uint64_t i = 0; i < configuration.size; ++i) {
      double absolute = std::abs(values[i]);
      double scaled = absolute * currentResidualScale[i];
      if (!std::isfinite(absolute) || !std::isfinite(scaled)) {
        observation.finite = false;
        observation.maximumNorm = observation.scaledMaximumNorm =
            std::numeric_limits<double>::infinity();
        break;
      }
      observation.maximumNorm = std::max(observation.maximumNorm, absolute);
      observation.scaledMaximumNorm =
          std::max(observation.scaledMaximumNorm, scaled);
    }
  }
  if (result != KIN_SUCCESS) {
    callbackFailed = true;
  }
}

int SparseNonlinearSystem::jacobianBridge(
    N_Vector variables, N_Vector residuals, SUNMatrix jacobian,
    void *userData, N_Vector temp1, N_Vector temp2) {
  auto *system = static_cast<SparseNonlinearSystem *>(userData);
  if (simulation::getOptions().profiling)
    ++system->statistics.jacobians;
  int result = system->jacobianCallback(variables, residuals, jacobian);
  if (result != KIN_SUCCESS) {
    system->callbackFailed = true;
  }
  return result;
}

void SparseNonlinearSystem::printStatistics(const char *phase) const {
  const auto &s = statistics;
  std::cerr << "[sparse-solver-statistics] phase=" << phase
            << " id=" << s.id << " scope=lifetime_cumulative"
            << " scalar_unknowns=" << configuration.size
            << " nnz=" << configuration.nonZeros
            << " solves=" << s.solves << " residuals=" << s.residuals
            << " jacobians=" << s.jacobians
            << " explicit_factorization_attempts=" << s.factorizations
            << " rhs_attempts=" << s.rightHandSides
            << " condition_attempts=" << s.conditions
            << " reused_residuals=" << s.reusedResiduals
            << " solve_inclusive_ns=" << s.solveNanoseconds
            << " factor_inclusive_ns=" << s.factorNanoseconds
            << " rhs_inclusive_ns=" << s.rhsNanoseconds
            << " condition_inclusive_ns=" << s.conditionNanoseconds
            << " nonlinear_iterations=" << s.nonlinearIterations
            << " raw_solve_flags=";
  bool firstFlag = true;
  for (const auto &[flag, count] : s.solveFlags) {
    std::cerr << (firstFlag ? "" : ",") << flag << ":" << count;
    firstFlag = false;
  }
  std::cerr << '\n';
}

void SparseNonlinearSystem::quietErrorHandler(int errorCode,
                                              const char *module,
                                              const char *function,
                                              char *message,
                                              void *userData) {}

SparseNonlinearSystem::SolveStatus
SparseNonlinearSystem::classifySolveFlag(int flag, bool callbackFailed) {
  // 中文：白名单式 fatal 分类刻意保留普通收敛失败的 recoverable 语义，
  // 但内存、上下文、非法输入和 callback 失败必须立即停止。
  // English: Fatal classification is explicit so ordinary convergence
  // failures remain recoverable, while memory, context, illegal-input, and
  // callback failures stop immediately.
  if (!callbackFailed &&
      (flag == KIN_SUCCESS || flag == KIN_INITIAL_GUESS_OK)) {
    return SolveStatus::Success;
  }
  // 中文：KIN_STEP_LT_STPTOL 只证明步长小，不证明残差收敛；保留失败迭代点给 IDA 健康检查和回滚。
  // KIN_STEP_LT_STPTOL only certifies a small step, not a converged residual.
  // Preserve the failed iterate for IDA's health inspection and rollback.
  if (callbackFailed || flag == KIN_SYSFUNC_FAIL ||
      flag == KIN_FIRST_SYSFUNC_ERR || flag == KIN_REPTD_SYSFUNC_ERR ||
      flag == KIN_ILL_INPUT || flag == KIN_MEM_FAIL ||
      flag == KIN_VECTOROP_ERR || flag == KIN_CONTEXT_ERR) {
    return SolveStatus::FatalFailure;
  }
  return SolveStatus::RecoverableFailure;
}

} // namespace marco::runtime::sundials::kinsol

#endif // SUNDIALS_ENABLE
