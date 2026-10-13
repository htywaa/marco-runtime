#ifndef MARCO_RUNTIME_SOLVERS_KINSOL_INSTANCE_H
#define MARCO_RUNTIME_SOLVERS_KINSOL_INSTANCE_H

#ifdef SUNDIALS_ENABLE

#include "kinsol/kinsol.h"
#include "marco/Runtime/Solvers/KINSOL/SparseNonlinearSystem.h"
#include "marco/Runtime/Solvers/KINSOL/ScalarBlockSolver.h"
#include "marco/Runtime/Solvers/IDA/C2KernelV1.h"
#include "marco/Runtime/Solvers/SUNDIALS/Instance.h"
#include "nvector/nvector_serial.h"
#include "sundials/sundials_config.h"
#include "sundials/sundials_types.h"
#include "sunlinsol/sunlinsol_klu.h"
#include "sunmatrix/sunmatrix_sparse.h"
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>

namespace marco::runtime::sundials::ida {
class IDAInstance;
}

namespace marco::runtime::sundials::kinsol {

/// Signature of residual functions.
/// The 1st argument is a pointer to the list of equation indices.
/// The result is the residual value.
using ResidualFunction = double (*)(const int64_t *);

/// Signature of Jacobian functions.
/// The 1st argument is a pointer to the list of equation indices.
/// The 2nd argument is a pointer to the list of variable indices.
/// The result is the Jacobian value.
/// The 3rd argument is the identifier of the memory pool owning the AD seeds.
/// The 4th argument is a pointer to the list of AD seed identifiers
/// The result is the Jacobian value.
using JacobianFunction = double (*)(const int64_t *, const uint64_t *, uint64_t,
                                    const uint64_t *);

/// 中文：带时间 callback 是 IDA component-local 闭包的适配层；普通 KINSOL
/// 仍使用无时间签名。postprocessor 在返回 residual 前刷新依赖 execution group。
/// English: Time-aware callbacks adapt IDA component-local closures while
/// ordinary KINSOL keeps its time-independent ABI. The postprocessor refreshes
/// dependent execution groups before residuals are returned.
using TimedResidualFunction = double (*)(double, const int64_t *);
using TimedJacobianFunction = double (*)(double, const int64_t *,
                                         const uint64_t *, double, uint64_t,
                                         const uint64_t *);
using TimedResidualPostprocessor =
    std::function<bool(double, double *, uint64_t)>;

/// A descriptor of a Jacobian function is a pair of value consisting in:
///  - the function pointer
///  - the number of elements of each AD seed
using JacobianFunctionDescriptor =
    std::pair<JacobianFunction, std::vector<uint64_t>>;

/// A map indicating the IDs of the buffers living inside the memory pool to
/// be used as AD seeds for each Jacobian function.
using JacobianSeedsMap = std::map<uintptr_t, std::vector<uint64_t>>;

/// A chunk of equations to be processed by a thread while computing the
/// residual values or partial derivatives.
/// A chunk is composed of:
///   - the identifier (position) of the equation.
///   - the begin indices (included)
///   - the end indices (excluded)
///   - the map indicating the buffer IDs to be used when computing the
///     partial derivatives
using ThreadEquationsChunk = std::tuple<Equation, std::vector<int64_t>,
                                        std::vector<int64_t>, JacobianSeedsMap>;

enum class EquationsParallelIterationKind { Residuals, Jacobian };

/// 中文：有效 worker 为一时，方程回调在调用线程执行并读取其 TLS；多 worker 继续在线程池执行。
///       这不会使同一 solver 实例支持并发调用或重入。
/// With one effective worker, equation callbacks execute on the caller thread
/// and observe its TLS. Multiple workers keep pool-thread execution. This does
/// not make one solver instance concurrently callable or reentrant.
class KINSOLInstance {
public:
  /// 中文：RecoverableFailure 仅表示当前 IDA 试探点不可解；FatalFailure 表示
  /// callback、结构或资源错误，调用者不得继续缩步掩盖。
  /// English: RecoverableFailure means only the current IDA trial point could
  /// not be solved; FatalFailure denotes callback, structural, or resource
  /// errors that must not be hidden by step-size retries.
  enum class SolveStatus { Success, RecoverableFailure, FatalFailure };

  KINSOLInstance();

  ~KINSOLInstance();

  Variable addVariable(uint64_t rank, const uint64_t *dimensions,
                       VariableGetter getterFunction,
                       VariableSetter setterFunction, const char *name);

  bool setVariableNominalGetter(Variable variable,
                                VariableGetter nominalGetter);

  /// Add the information about an equation that is handled by KINSOL.
  Equation addEquation(const int64_t *ranges, uint64_t rank,
                       const char *stringRepresentation);

  void addVariableAccess(Equation equation, Variable variableIndex,
                         AccessFunction accessFunction);

  /// Add the function pointer that computes the residual value of an
  /// equation.
  void setResidualFunction(Equation equationIndex,
                           ResidualFunction residualFunction);

  void setTimedResidualFunction(Equation equationIndex,
                                TimedResidualFunction residualFunction);

  void setTimedResidualPostprocessor(TimedResidualPostprocessor callback);

  /// Add the function pointer that computes a partial derivative of an
  /// equation.
  void addJacobianFunction(Equation equationIndex, Variable variableIndex,
                           JacobianFunction jacobianFunction,
                           uint64_t numOfSeeds, uint64_t *seedSizes);

  void addTimedJacobianFunction(Equation equationIndex, Variable variableIndex,
                                TimedJacobianFunction jacobianFunction,
                                uint64_t numOfSeeds, uint64_t *seedSizes);

  /// Instantiate and initialize all the classes needed by KINSOL in order to
  /// solve the given system of equations. It also sets optional simulation
  /// parameters for KINSOL.
  bool initialize();

  /// 中文：实例可请求有界复用索引几何；调用方须认证访问映射只依赖方程索引，且直到销毁保持不变，IDA 使用已验证 epoch 布局。
  ///       只缓存几何及固定回调/seed-buffer 绑定，不缓存数值或求解决策；新注册使计划失效，seed 存储属于固定 chunk。
  ///       纯度未知或预算不足保留普通路径，限额只能降低，并须在注册完成后调用。
  /// Request bounded reuse of index geometry for this instance. The caller
  /// must certify that access maps are pure functions of equation indices and
  /// remain fixed until destruction (IDA uses its validated epoch layout).
  /// Only index geometry and fixed callback/seed-buffer bindings are cached,
  /// never numerical values or solver decisions. New registration invalidates
  /// the plan; bound seed storage belongs to this instance's fixed chunks.
  /// Unknown access purity uses the ordinary path. Limits may only decrease;
  /// insufficient budget leaves that path available. Call after registration.
  bool requestImmutableAccessLayout(uint64_t workLimit = 1000000,
                                    uint64_t storageLimit = 16000000);

  /// 中文：内部可选契约要求活动 IDA owner 认证原标量配对及完整纯多项式 support。
  ///       累计预算由 owner 拥有且必须长于实例生命周期；initialize() 再核对实际访问列，新的注册使认证失效。
  /// Internal optional contract: original scalar pairing and complete pure
  /// polynomial supports, certified by the active IDA owner. The cumulative
  /// budget belongs to that owner and must outlive this instance. Actual access
  /// columns are checked again at initialize(). Registration invalidates it.
  bool requestScalarBlockPlan(std::vector<ScalarBlockPlan::Row> rows,
                             ScalarBlockPlan::Budget &budget);
  bool requestScalarBlockPlan(std::vector<ScalarBlockPlan::Row> rows,
                             std::vector<TimedResidualFunction> explicitValues,
                             ScalarBlockPlan::Budget &budget);

  bool solve();

  bool solve(realtype time);

  bool solve(realtype time, bool enableLineSearch);

  /// 中文：外层 IDA 事务负责回滚，失败回调的 trial 值先保留给 condition 检查；独立 solve() 则在返回 false 前恢复注册未知量。
  /// The enclosing IDA transaction owns rollback: failed callback trial values
  /// remain available for its condition inspection. Standalone solve() rolls
  /// back registered unknowns before returning false.
  SolveStatus solveWithStatus(realtype time);

  void setFunctionNormTolerance(realtype tolerance);

  void setScaledStepTolerance(realtype tolerance);

  void setMaximumNewtonStep(realtype maximumStep);

  void setLineSearchEnabled(bool enabled);

  void setQuietErrors(bool enabled);

  bool configureInitializationAnchorScaling(realtype time);

  /// 中文：以下接口复用当前 sparse Jacobian factorization，供 Schur 补、
  /// condition 证书和多 RHS sensitivity 求解使用。
  /// English: The following APIs reuse the current sparse Jacobian
  /// factorization for Schur complements, condition certificates, and
  /// multiple sensitivity right-hand sides.
  bool factorizeCurrentJacobian(realtype time);
  bool bindC2KernelV1(const MarcoC2KernelV1 &descriptor);
  bool hasC2KernelV1() const { return c2Kernel != nullptr; }
  bool certifyC2KernelPoint(realtype time,
      const std::vector<uint64_t> &rows, const std::vector<uint64_t> &columns,
      SparseNonlinearSystem::ConditionEstimate &chart,
      SparseNonlinearSystem::ConditionEstimate &closure);
  SparseNonlinearSystem::FactorizationStatus
  factorizeCurrentJacobianWithStatus(realtype time);

  bool solveCurrentJacobian(const std::vector<double> &rhs,
                            std::vector<double> &solution);

  bool getCurrentJacobianEntries(const std::vector<uint64_t> &rows,
                                 const std::vector<uint64_t> &columns,
                                 std::vector<double> &values) const;

  bool estimateScaledJacobianCondition(
      const std::vector<uint64_t> &rows,
      const std::vector<uint64_t> &columns,
      SparseNonlinearSystem::ConditionEstimate &estimate) const;

  double getScalarVariableNominal(uint64_t scalarIndex) const;

  uint64_t getLastNonlinearIterations() const;
  uint64_t getScalarBlockAcceptedCalls() const { return scalarBlockAccepted; }
  uint64_t getScalarBlockDeclinedCalls() const { return scalarBlockDeclined; }

  SparseNonlinearSystem::ResidualObservation getLastResidualObservation() const {
    return nonlinearSystem ? nonlinearSystem->getLastResidualObservation()
                           : SparseNonlinearSystem::ResidualObservation{};
  }

  void printInitializationScaling() const;

  uint64_t getStatisticsId() const {
    return nonlinearSystem ? nonlinearSystem->getStatistics().id : 0;
  }

  uint64_t getExplicitNominalScalars() const {
    return explicitNominalScalars;
  }

  uint64_t getAnchorNominalScalars() const { return anchorNominalScalars; }

  /// KINSOLResFn user-defined residual function, passed to KINSOL through
  /// KINSOLInit. It contains how to compute the Residual Function of the
  /// system, starting from the provided UserData struct, iterating through
  /// every equation.
  static int residualFunction(N_Vector variables, N_Vector residuals,
                              void *userData);

  static int residualFunction(realtype time, N_Vector variables,
                              N_Vector residuals, void *userData);

  static int jacobianMatrix(N_Vector variables, N_Vector residuals,
                            SUNMatrix jacobianMatrix, void *userData,
                            N_Vector tempv1, N_Vector tempv2);

private:
  friend class marco::runtime::sundials::ida::IDAInstance;

  // 中文：只有活动 IDA owner 可在同一调用作用域内检查本 epoch 的成功点；认证纯标量回调不能修改参数、外层输入或 basis。
  //       该检查是既有完整 post-solve 健康检查，不是任意模型回调；回执不能跨 trial、失败求解、嵌套求解或函数返回存活。
  // Only the active IDA owner may inspect this epoch's successful point in
  // the same lexical invocation. Its certified pure scalar callbacks cannot
  // modify parameters, outer inputs or the basis. The inspection is the
  // existing full post-solve health check, never an arbitrary model callback.
  // No receipt survives a trial, a failed solve, nested solve or this return.
  SolveStatus solveAndCheck(realtype time,
                           const std::function<void()> &inspection);
  struct SuccessfulPoint {
    realtype time;
    const std::vector<double> &variables;
    const std::vector<double> &residuals;
  };
  const SuccessfulPoint *successfulPoint{nullptr};

  SolveStatus solveImpl(realtype time, bool rollbackOnFailure);
  template <bool Generated>
  SolveStatus solveBody(realtype time, bool rollbackOnFailure);
  using SolveEntry = SolveStatus (KINSOLInstance::*)(realtype, bool);
  SolveEntry solveEntry = &KINSOLInstance::solveBody<false>;

  void invalidateAccessLayout();
  bool prepareScalarBlockSolver();

  [[nodiscard]] uint64_t getNumOfArrayVariables() const;

  [[nodiscard]] uint64_t getNumOfScalarVariables() const;

  [[nodiscard]] uint64_t getVariableFlatSize(Variable variable) const;

  [[nodiscard]] uint64_t getNumOfVectorizedEquations() const;

  [[nodiscard]] uint64_t getNumOfScalarEquations() const;

  [[nodiscard]] uint64_t getEquationRank(Equation equation) const;

  [[nodiscard]] uint64_t getEquationFlatSize(Equation equation) const;

  [[nodiscard]] uint64_t getVariableRank(Variable variable) const;

  void
  iterateAccessedArrayVariables(Equation equation,
                                std::function<void(Variable)> callback) const;

  std::vector<JacobianColumn>
  computeJacobianColumns(Equation eq, const int64_t *equationIndices) const;

  void computeNNZ();

  void computeThreadChunks();

  void copyVariablesFromMARCO(N_Vector variables);

  void copyVariablesIntoMARCO(N_Vector variables);

  void equationsParallelIteration(
      EquationsParallelIterationKind kind,
      std::function<void(Equation equation,
                         const std::vector<int64_t> &equationIndices,
                         const JacobianSeedsMap &jacobianSeedsMap,
                         uint64_t scalarEquationIndex)>
          processFn);

  void getVariableBeginIndices(Variable variable,
                               std::vector<uint64_t> &indices) const;

  void getVariableEndIndices(Variable variable,
                             std::vector<uint64_t> &indices) const;

  void getEquationBeginIndices(Equation equation,
                               std::vector<int64_t> &indices) const;

  void getEquationEndIndices(Equation equation,
                             std::vector<int64_t> &indices) const;

private:
  /// @name Debug functions
  /// {
  void printVariablesVector(N_Vector variables) const;

  void printResidualsVector(N_Vector residuals) const;

  void printJacobianMatrix(SUNMatrix jacobianMatrix) const;

  /// }

private:
  // Whether the instance has been inizialized or not.
  bool initialized{false};

  struct ImmutableAccessLayout {
    struct VariableEntry {
      Variable variable;
      std::vector<uint64_t> indices;
      uint64_t offset;
    };
    struct RowEntry {
      size_t chunk;
      std::vector<int64_t> indices;
      uint64_t offset;
    };
    struct JacobianEntry {
      sunindextype column;
      JacobianFunction callback;
      TimedJacobianFunction timedCallback;
      const uint64_t *seeds;
    };
    std::vector<std::vector<JacobianColumn>> columns;
    std::vector<std::vector<JacobianEntry>> evaluations;
    std::vector<VariableEntry> variables;
    std::vector<RowEntry> rows;
  };
  bool immutableAccessLayoutRequested{false};
  uint64_t layoutWorkLimit{0}, layoutStorageLimit{0};
  std::unique_ptr<ImmutableAccessLayout> accessLayout;
  std::vector<ScalarBlockPlan::Row> scalarBlockRows;
  std::vector<TimedResidualFunction> scalarExplicitValues;
  std::unique_ptr<const ScalarBlockPlan> scalarBlockPlan;
  ScalarBlockPlan::Budget *scalarBlockBudget{nullptr};
  std::unique_ptr<ScalarBlockSolver> scalarBlockSolver;
  bool scalarBlockPrepared{false}, lastSolveWasBlocked{false};
  uint64_t scalarBlockIterations{0}, scalarBlockAccepted{0}, scalarBlockDeclined{0};
  std::vector<double> scalarBlockValues;
  struct C2KernelState;
  std::unique_ptr<C2KernelState> c2Kernel;

  // Model size.
  uint64_t scalarVariablesNumber{0};
  uint64_t scalarEquationsNumber{0};
  uint64_t nonZeroValuesNumber{0};

  // The iteration ranges of the vectorized equations.
  std::vector<MultidimensionalRange> equationRanges;

  // The residual functions associated with the equations.
  // The i-th position contains the pointer to the residual function of the
  // i-th equation.
  /// 中文：普通与 timed callback 使用平行描述表；每个 equation 至少拥有一种，
  /// 但不要求为了 timed-only component 伪造普通 callback。
  /// English: Ordinary and timed callbacks use parallel descriptor tables.
  /// Every equation owns at least one form, without fabricating an ordinary
  /// callback for timed-only components.
  std::vector<ResidualFunction> residualFunctions;
  std::vector<TimedResidualFunction> timedResidualFunctions;
  TimedResidualPostprocessor timedResidualPostprocessor;

  // The jacobian functions associated with the equations.
  // The i-th position contains the list of partial derivative functions of
  // the i-th equation. The j-th function represents the function to
  // compute the derivative with respect to the j-th variable.
  std::vector<std::vector<JacobianFunctionDescriptor>> jacobianFunctions;
  std::vector<std::vector<
      std::pair<TimedJacobianFunction, std::vector<uint64_t>>>>
      timedJacobianFunctions;

  /// 中文：这些覆盖项在 initialize 前冻结，使普通 KINSOL 与 IDA 局部闭包
  /// 可共享实现而保持实例级数值策略隔离。
  /// English: These overrides are frozen before initialization, allowing
  /// ordinary KINSOL and IDA local closures to share implementation while
  /// keeping numerical policy instance-local.
  realtype currentTime{0};
  std::optional<realtype> functionNormTolerance;
  std::optional<realtype> scaledStepTolerance;
  std::optional<realtype> maximumNewtonStep;
  bool lineSearchEnabled{true};
  bool quietErrors{false};
  bool initializationAnchorScalingConfigured{false};

  // Whether the IDA instance is informed about the accesses to the
  // variables.
  bool precomputedAccesses{false};

  std::vector<VarAccessList> variableAccesses;

  // The offset of each array variable inside the flattened variables
  // vector.
  std::vector<uint64_t> variableOffsets;

  // The dimensions list of each array variable.
  std::vector<VariableDimensions> variablesDimensions;

  // The offset of each array equation inside the flattened equations
  // vector.
  std::vector<uint64_t> equationOffsets;

  // Support structure for the computation of the jacobian matrix.
  // The outer vector has a number of elements equal to the scalar number
  // of equations. Each of them represents a row of the matrix and consists
  // in a vector of paired elements. The first element of each pair
  // represents the index of the column (that is, the independent scalar
  // variable for the partial derivative) while the second one is the
  // value of the partial derivative.
  std::vector<std::vector<std::pair<sunindextype, double>>> jacobianMatrixData;

  /// 中文：底层 SUNDIALS vectors、KLU 与 scaling 的所有权集中在共享内核；
  /// facade 只维护 MARCO 数组布局、callback 与稀疏列描述。
  /// English: The shared kernel owns SUNDIALS vectors, KLU, and scaling; this
  /// facade retains MARCO array layout, callbacks, and sparse-column metadata.
  std::unique_ptr<SparseNonlinearSystem> nonlinearSystem;
  std::atomic<uint64_t> scalarResidualCalls{0}, scalarJacobianCalls{0};

  std::vector<VariableGetter> variableGetters;
  std::vector<VariableSetter> variableSetters;
  /// 中文：显式 nominal 与初始化值锚点分开计数，runtime probe 可审计实际尺度来源。
  /// English: Explicit nominals and initialization-value anchors are counted
  /// separately so runtime probes can audit the actual scale source.
  std::vector<VariableGetter> variableNominalGetters;
  uint64_t explicitNominalScalars{0};
  uint64_t anchorNominalScalars{0};

  // Thread pool.
  ThreadPool threadPool;

  // Memory pool ID.
  uint64_t memoryPoolId;

  // The list of chunks the threads will process. Each thread elaborates
  // one chunk at a time.
  // The information is computed only once during the initialization to
  // save time during the actual simulation.
  std::vector<ThreadEquationsChunk> threadEquationsChunks;
};
} // namespace marco::runtime::sundials::kinsol

//===---------------------------------------------------------------------===//
// Exported functions
//===---------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(kinsolCreate, PTR(void))

RUNTIME_FUNC_DECL(kinsolSolve, void, PTR(void))

RUNTIME_FUNC_DECL(kinsolFree, void, PTR(void))

RUNTIME_FUNC_DECL(kinsolAddVariable, uint64_t, PTR(void), uint64_t,
                  PTR(uint64_t), PTR(void), PTR(void), PTR(void))

RUNTIME_FUNC_DECL(kinsolAddVariableAccess, void, PTR(void), uint64_t, uint64_t,
                  PTR(void))

RUNTIME_FUNC_DECL(kinsolAddEquation, uint64_t, PTR(void), PTR(int64_t),
                  uint64_t, PTR(void))

RUNTIME_FUNC_DECL(kinsolSetResidual, void, PTR(void), uint64_t, PTR(void))

RUNTIME_FUNC_DECL(kinsolAddJacobian, void, PTR(void), uint64_t, uint64_t,
                  PTR(void), uint64_t, PTR(uint64_t))

#endif // SUNDIALS_ENABLE

#endif // MARCO_RUNTIME_SOLVERS_KINSOL_INSTANCE_H
