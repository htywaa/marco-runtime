#ifndef MARCO_RUNTIME_SOLVERS_KINSOL_INSTANCE_H
#define MARCO_RUNTIME_SOLVERS_KINSOL_INSTANCE_H

#ifdef SUNDIALS_ENABLE

#include "kinsol/kinsol.h"
#include "marco/Runtime/Solvers/KINSOL/SparseNonlinearSystem.h"
#include "marco/Runtime/Solvers/SUNDIALS/Instance.h"
#include "nvector/nvector_serial.h"
#include "sundials/sundials_config.h"
#include "sundials/sundials_types.h"
#include "sunlinsol/sunlinsol_klu.h"
#include "sunmatrix/sunmatrix_sparse.h"
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>

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

  bool solve();

  bool solve(realtype time);

  bool solve(realtype time, bool enableLineSearch);

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
                         const JacobianSeedsMap &jacobianSeedsMap)>
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
