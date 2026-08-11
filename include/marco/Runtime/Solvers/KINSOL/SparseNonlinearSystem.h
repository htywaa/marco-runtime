#ifndef MARCO_RUNTIME_SOLVERS_KINSOL_SPARSENONLINEARSYSTEM_H
#define MARCO_RUNTIME_SOLVERS_KINSOL_SPARSENONLINEARSYSTEM_H

#ifdef SUNDIALS_ENABLE

#include "kinsol/kinsol.h"
#include "nvector/nvector_serial.h"
#include "sunlinsol/sunlinsol_klu.h"
#include "sunmatrix/sunmatrix_sparse.h"
#include <cstdint>
#include <functional>
#include <vector>

namespace marco::runtime::sundials::kinsol {

/// 中文：KINSOL 与 IDA local closure 共用的稀疏非线性内核；所有 scaling、
/// recoverable failure 与 KLU 生命周期都由实例局部配置控制。
/// English: Sparse nonlinear kernel shared by KINSOL and IDA local closures;
/// scaling, recoverable failures, and KLU lifetime are controlled by the
/// instance-local configuration.
class SparseNonlinearSystem {
public:
  /// 中文：区分局部试探点可恢复失败与结构/回调致命失败，IDA 据此决定缩步重试
  /// 还是终止当前执行计划。
  /// English: Distinguishes recoverable trial-point failures from structural
  /// or callback failures so IDA can retry with a smaller step or terminate.
  enum class SolveStatus { Success, RecoverableFailure, FatalFailure };

  /// 中文：条件证书同时承载小型 dense 精确估计和大型 sparse KLU 估计；
  /// sparseEstimator 标明采用的路径，避免把近似证书误读成完整 dense 分析。
  /// English: The condition certificate represents both exact small dense
  /// estimates and large sparse KLU estimates; sparseEstimator identifies the
  /// path so callers do not mistake it for a full dense analysis.
  struct ConditionEstimate {
    uint64_t rank{0};
    double minimumPivot{0};
    double maximumPivot{0};
    double reciprocalCondition{0};
    double oneNormCondition{0};
    bool rankDeficient{false};
    bool sparseEstimator{false};
  };

  using ResidualCallback =
      std::function<int(N_Vector variables, N_Vector residuals)>;
  using JacobianCallback =
      std::function<int(N_Vector variables, N_Vector residuals,
                        SUNMatrix jacobian)>;

  /// 中文：每个局部系统独立携带容差、Newton 步长和错误输出策略，不继承
  /// 全局 KINSOL CLI 状态。
  /// English: Each local system carries its own tolerances, Newton-step bound,
  /// and error policy instead of inheriting global KINSOL CLI state.
  struct Configuration {
    uint64_t size{0};
    uint64_t nonZeros{0};
    realtype functionNormTolerance{0};
    realtype scaledStepTolerance{0};
    realtype maximumNewtonStep{0};
    bool lineSearch{true};
    bool quietErrors{false};
  };

  SparseNonlinearSystem(Configuration configuration,
                        ResidualCallback residualCallback,
                        JacobianCallback jacobianCallback);
  ~SparseNonlinearSystem();

  SparseNonlinearSystem(const SparseNonlinearSystem &) = delete;
  SparseNonlinearSystem &operator=(const SparseNonlinearSystem &) = delete;

  bool initialize();

  /// 中文：solve 只推进非线性迭代；factorize/solveFactorized 则暴露同一
  /// 稀疏 Jacobian 分解，供 component Schur 多右端项重复使用。
  /// English: solve performs nonlinear iteration, while
  /// factorize/solveFactorized expose the same sparse Jacobian factorization
  /// for repeated component Schur right-hand sides.
  SolveStatus solve(realtype time);
  bool factorize(realtype time);
  bool solveFactorized(const std::vector<double> &rhs,
                       std::vector<double> &solution);

  bool setScaling(const std::vector<double> &variableScale,
                  const std::vector<double> &residualScale);

  /// 中文：nominal/初始化锚点建立后，变量与残差尺度在后续试探点保持固定，
  /// 使收敛判据和 chart condition 可比较。
  /// English: Once nominal/initialization anchors establish the scales, they
  /// remain fixed across trial points so convergence and chart conditions are
  /// comparable.
  bool setScalingFromVariableNominals(
      const std::vector<double> &variableNominals);
  void setLineSearchEnabled(bool enabled) {
    configuration.lineSearch = enabled;
  }

  bool extractEntries(const std::vector<uint64_t> &rows,
                      const std::vector<uint64_t> &columns,
                      std::vector<double> &values) const;

  /// 中文：condition probe 只提取请求的局部子块；大系统保持 sparse estimate，
  /// 不分配全局 dense matrix。
  /// English: Condition probing extracts only the requested local submatrix;
  /// large systems retain a sparse estimate without allocating a global dense
  /// matrix.
  bool estimateScaledSubmatrixCondition(
      const std::vector<uint64_t> &rows,
      const std::vector<uint64_t> &columns, ConditionEstimate &estimate,
      uint64_t denseLimit = 64) const;

  const std::vector<double> &getVariableScale() const {
    return currentVariableScale;
  }

  const std::vector<double> &getResidualScale() const {
    return currentResidualScale;
  }

  double getVariableNominal(uint64_t scalarIndex) const {
    return scalarIndex < currentVariableScale.size() &&
                   currentVariableScale[scalarIndex] > 0
               ? 1.0 / currentVariableScale[scalarIndex]
               : 0;
  }

  N_Vector getVariablesVector() const { return variablesVector; }
  SUNMatrix getJacobianMatrix() const { return sparseMatrix; }
  int getLastSolveFlag() const { return lastSolveFlag; }
  uint64_t getLastNonlinearIterations() const {
    return lastNonlinearIterations;
  }

private:
  /// 中文：bridge 将用户 callback 状态归一化为 KINSOL 返回码，并记录 callback
  /// 是否真正失败，供 classifySolveFlag 区分数值重试与结构错误。
  /// English: The bridges normalize user callback status to KINSOL return
  /// codes and remember callback failure so classifySolveFlag can distinguish
  /// numerical retries from structural errors.
  static int residualBridge(N_Vector variables, N_Vector residuals,
                            void *userData);
  static int jacobianBridge(N_Vector variables, N_Vector residuals,
                            SUNMatrix jacobian, void *userData,
                            N_Vector temp1, N_Vector temp2);
  static void quietErrorHandler(int errorCode, const char *module,
                                const char *function, char *message,
                                void *userData);

  bool evaluateJacobian(realtype time);
  static SolveStatus classifySolveFlag(int flag, bool callbackFailed);

private:
  /// 中文：configuration 和 callbacks 是不可变描述；currentTime、solve flag
  /// 与 scaling 是每次试探/初始化更新的运行状态。
  /// English: Configuration and callbacks describe the immutable system;
  /// current time, solve flags, and scales are mutable trial/initialization
  /// state.
  Configuration configuration;
  ResidualCallback residualCallback;
  JacobianCallback jacobianCallback;
  realtype currentTime{0};
  bool initialized{false};
  bool callbackFailed{false};
  int lastSolveFlag{KIN_SUCCESS};
  uint64_t lastNonlinearIterations{0};

#if SUNDIALS_VERSION_MAJOR >= 7
  SUNComm comm{0};
#endif
#if SUNDIALS_VERSION_MAJOR >= 6
  SUNContext context{nullptr};
#endif

  void *memory{nullptr};
  N_Vector variablesVector{nullptr};
  N_Vector variableScaleVector{nullptr};
  N_Vector residualScaleVector{nullptr};
  SUNMatrix sparseMatrix{nullptr};
  SUNLinearSolver linearSolver{nullptr};
  std::vector<double> currentVariableScale;
  std::vector<double> currentResidualScale;
};

} // namespace marco::runtime::sundials::kinsol

#endif // SUNDIALS_ENABLE

#endif // MARCO_RUNTIME_SOLVERS_KINSOL_SPARSENONLINEARSYSTEM_H
