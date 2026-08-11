#ifndef MARCO_RUNTIME_SOLVERS_IDA_INSTANCE_H
#define MARCO_RUNTIME_SOLVERS_IDA_INSTANCE_H

#ifdef SUNDIALS_ENABLE

#include "ida/ida.h"
#include "marco/Runtime/Solvers/SUNDIALS/DiscontinuityCoordinator.h"
#include "marco/Runtime/Solvers/SUNDIALS/Instance.h"
#include "nvector/nvector_serial.h"
#include "sundials/sundials_config.h"
#include "sundials/sundials_types.h"
#include "sunlinsol/sunlinsol_klu.h"
#include "sunmatrix/sunmatrix_sparse.h"
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace marco::runtime::sundials::ida {
enum class VariableKind { ALGEBRAIC, STATE };

/// Signature of residual functions.
/// The 1st argument is the current time.
/// The 2nd argument is a pointer to the list of equation indices.
/// The result is the residual value.
using ResidualFunction = double (*)(double, const int64_t *);

/// Signature of Jacobian functions.
/// The 1st argument is the current time.
/// The 2nd argument is a pointer to the list of equation indices.
/// The 3rd argument is a pointer to the list of variable indices.
/// The 4th argument is the 'alpha' value.
/// The 5th argument is the identifier of the memory pool owning the AD seeds.
/// The 6th argument is a pointer to the list of AD seed identifiers
/// The result is the Jacobian value.
using JacobianFunction = double (*)(double, const int64_t *, const uint64_t *,
                                    double, uint64_t, const uint64_t *);

/// A descriptor of a Jacobian function is a pair of value consisting in:
///  - the function pointer
///  - the number of elements of each AD seed
using JacobianFunctionDescriptor =
    std::pair<JacobianFunction, std::vector<uint64_t>>;

/// A map indicating the IDs of the buffers living inside the memory pool to
/// be used as AD seeds for each Jacobian function.
using JacobianSeedsMap = std::map<JacobianFunction, std::vector<uint64_t>>;

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

/// 中文：以下 descriptors 是编译期 compact execution 证书的 runtime 镜像；
/// 它们引用已有 variable/equation handles，不复制原数组或重新执行 Matching。
/// English: The descriptors below mirror the compile-time compact execution
/// certificate at runtime. They reference existing variable/equation handles
/// without copying arrays or rerunning Matching.
struct ConstraintComponentDescriptor {
  uint64_t componentId;
  uint64_t positionScalars;
  uint64_t residualPositionScalars;
  uint64_t tangentScalars;
  uint64_t residualTangentScalars;
  uint64_t highestScalars;
  uint64_t residualHighestScalars;
  uint64_t supportScalars;
  uint64_t residualSupportScalars;
};

struct ConstraintEquationDescriptor {
  uint64_t componentId;
  Equation equation;
  std::string role;
  uint64_t order;
  std::string matchedVariableName;
};

struct ConstraintVariableDescriptor {
  uint64_t componentId;
  Variable variable;
  std::string role;
  std::string name;
};

/// 中文：execution descriptors 指向实际 outer/local runtime owner；它们与
/// 上面的数学 provenance 分离，因此多个 semantic components 可合并到同一
/// solver group。
/// English: Execution descriptors identify actual outer/local runtime owners.
/// They are separate from semantic provenance, so multiple semantic components
/// may be merged into one solver group.
struct ConstraintVariableExecutionDescriptor {
  uint64_t executionGroupId;
  Variable variable;
  std::string owner;
};

struct ConstraintEquationExecutionDescriptor {
  uint64_t executionGroupId;
  Equation equation;
  std::string owner;
};

struct SolverExecutionGroupDescriptor {
  uint64_t groupId;
  std::string kind;
  std::string mode;
  uint64_t topologicalOrder;
  uint64_t outerEquationScalars;
  uint64_t outerVariableScalars;
  uint64_t localEquationScalars;
  uint64_t localVariableScalars;
  std::set<uint64_t> semanticComponentIds;
  std::set<uint64_t> dependencies;
};

/// 中文：state-set descriptors 描述单 component 的候选基；epoch descriptors
/// 则把一个或多个 component 候选组合成可原子切换的执行域。
/// English: State-set descriptors describe candidate bases for one component;
/// epoch descriptors combine one or more component candidates into an
/// atomically switchable execution domain.
struct ConstraintStateSetDescriptor {
  uint64_t componentId;
  uint64_t stateSetId;
  bool active;
  std::string canonicalKey;
};

struct ConstraintStateSetVariableDescriptor {
  uint64_t componentId;
  uint64_t stateSetId;
  Variable variable;
  std::string owner;
  std::string name;
};

struct ConstraintStateSetEquationDescriptor {
  uint64_t componentId;
  uint64_t stateSetId;
  Equation equation;
  std::string owner;
  std::string role;
  uint64_t order;
  std::string matchedVariableName;
};

struct ConstraintExecutionEpochDescriptor {
  uint64_t domainId;
  uint64_t epochId;
  bool active;
  std::string canonicalKey;
  uint64_t preferDummyScalars;
  uint64_t defaultDummyScalars;
  uint64_t avoidDummyScalars;
  std::set<uint64_t> transitions;
};

struct ConstraintExecutionEpochGroupDescriptor {
  uint64_t domainId;
  uint64_t epochId;
  SolverExecutionGroupDescriptor group;
};

struct ConstraintExecutionEpochVariableDescriptor {
  uint64_t domainId;
  uint64_t epochId;
  uint64_t executionGroupId;
  Variable variable;
  std::string owner;
  std::string name;
};

struct ConstraintExecutionEpochEquationDescriptor {
  uint64_t domainId;
  uint64_t epochId;
  uint64_t executionGroupId;
  Equation equation;
  std::string owner;
  std::string role;
  uint64_t order;
  std::string matchedVariableName;
};

struct ConstraintExecutionEpochCandidate {
  uint64_t epochId;
  uint64_t preferDummyScalars;
  uint64_t defaultDummyScalars;
  uint64_t avoidDummyScalars;
  double condition;
  bool safe;
};

struct ConstraintConditionEstimate {
  uint64_t rank{0};
  double minimumPivot{0};
  double maximumPivot{0};
  double reciprocalCondition{0};
  double oneNormCondition{0};
  bool rankDeficient{false};
  bool scaled{false};
  bool sparseEstimator{false};
};

/// 中文：chart 与 local closure 的联合健康证书决定接受、请求预认证切换或
/// fail-stop；单一未经缩放的 condition 不得独立触发切换。
/// English: The joint chart/local-closure health certificate decides whether
/// to accept, request a pre-certified switch, or fail-stop; an unscaled
/// condition estimate alone must not drive switching.
enum class ConstraintBasisHealthAction {
  Accept,
  RequestSwitch,
  Unsafe
};

struct ConstraintBasisHealthCertificate {
  ConstraintConditionEstimate chart;
  ConstraintConditionEstimate localClosure;
  double responseAmplification{0};
  bool responseAvailable{false};
};

ConstraintConditionEstimate
estimateConstraintCondition(const std::vector<double> &matrix, uint64_t size);

bool requiresDynamicStateSelection(
    const ConstraintConditionEstimate &estimate, bool failedLocalTrial);

bool shouldRequestConstraintStateSetSwitch(
    const ConstraintConditionEstimate &estimate, bool failedLocalTrial);

ConstraintBasisHealthAction classifyConstraintBasisHealth(
    const ConstraintBasisHealthCertificate &certificate,
    bool failedLocalTrial);

/// 中文：IDAReInit 后只通过公开 API 以 order 1 重建历史，并受控恢复步长与阶数。
/// English: After IDAReInit, rebuild history through public APIs at order one
/// and recover step size and order under explicit limits.
struct IDARestartRecoveryState {
  bool active{false};
  uint64_t acceptedSteps{0};
  double initialStep{0};
  double targetStep{0};
  double currentStepLimit{0};
  int currentMaximumOrder{1};
  int targetMaximumOrder{5};
};

IDARestartRecoveryState createIDARestartRecoveryState(
    double acceptedStep, double outputStep, double configuredInitialStep,
    double configuredMinimumStep, double configuredMaximumStep);

bool advanceIDARestartRecoveryState(IDARestartRecoveryState &state);

bool isSafeConstraintStateSetCandidate(bool localSolveSucceeded,
                                       double maximumCondition,
                                       double maximumConstraintResidual,
                                       double maximumOuterStateJump);

struct ConstraintResidualRoleCounts {
  uint64_t position{0};
  uint64_t tangent{0};
  uint64_t highest{0};
  uint64_t support{0};
};

bool hasCompleteConstraintResidualCertificate(
    const ConstraintResidualRoleCounts &expected,
    const ConstraintResidualRoleCounts &observed);

enum class ConstraintSwitchScopeKind {
  SemanticComponent,
  ExecutionDomain,
};

/// 中文：切换 scope 可以是单一语义 component 或预认证 execution domain；
/// runtime 只能在编译期登记的 epoch 图中移动。
/// English: A switch scope may cover one semantic component or a certified
/// execution domain; runtime transitions are restricted to the compiled epoch
/// graph.
struct ConstraintSwitchScope {
  ConstraintSwitchScopeKind kind{ConstraintSwitchScopeKind::SemanticComponent};
  uint64_t id{0};

  bool operator==(const ConstraintSwitchScope &other) const {
    return kind == other.kind && id == other.id;
  }

  bool operator!=(const ConstraintSwitchScope &other) const {
    return !(*this == other);
  }
};

struct ConstraintSwitchHealthSample {
  ConstraintSwitchScope scope;
  double condition{0};
};

std::optional<double> getMaximumConstraintSwitchScopeCondition(
    const std::vector<ConstraintSwitchHealthSample> &samples,
    ConstraintSwitchScope scope);

std::optional<uint64_t> selectConstraintExecutionEpoch(
    const std::vector<ConstraintExecutionEpochCandidate> &candidates);

bool shouldSuppressIDAError(int errorCode, bool pendingBasisSwitch,
                            bool expectedConstraintFailure);

bool composeExecutionGroupDerivative(
    double directDerivative,
    const std::vector<double> &crossDerivatives,
    const std::vector<double> &dependencyResponses, double &result);

/// 中文：IDA 实例统一管理 outer integration、component-local sparse closure、
/// accepted checkpoint 以及事件/状态基共享的 discontinuity transaction。
/// English: An IDA instance coordinates outer integration, component-local
/// sparse closures, accepted checkpoints, and discontinuity transactions
/// shared by model events and state-basis switches.
class IDAInstance {
public:
  IDAInstance();

  ~IDAInstance();

  void setStartTime(double time);
  void setEndTime(double time);
  void setTimeStep(double time);
  void setSuppressAlgebraicErrorTest(bool suppress);

  /// 中文：以下注册函数只装配编译期证书，不执行运行时 Matching。initialize()
  /// 会在分配 IDA 向量前统一校验 handles、owners、维数和活动状态集。
  /// English: These registration methods only assemble compile-time
  /// certificates; they do not run Matching. initialize() validates handles,
  /// owners, dimensions, and active state sets before allocating IDA vectors.
  void registerConstraintComponent(uint64_t componentId);
  void registerConstraintComponentRoleOwnership(uint64_t componentId,
                                                const char *role,
                                                uint64_t ownedScalars,
                                                uint64_t residualScalars);
  void registerConstraintEquation(uint64_t componentId, Equation equation,
                                  const char *role, uint64_t order,
                                  const char *matchedVariableName);
  void registerConstraintVariable(uint64_t componentId, Variable variable,
                                  const char *role, const char *name);
  void registerSolverExecutionGroup(uint64_t groupId, const char *kind,
                                    const char *mode,
                                    uint64_t topologicalOrder);
  void registerSolverExecutionGroupCertificate(
      uint64_t groupId, uint64_t outerEquationScalars,
      uint64_t outerVariableScalars, uint64_t localEquationScalars,
      uint64_t localVariableScalars);
  void registerSolverExecutionGroupMember(uint64_t groupId,
                                          uint64_t componentId);
  void registerSolverExecutionGroupDependency(uint64_t groupId,
                                              uint64_t dependencyGroupId);
  void enableConstraintComponentExecution(uint64_t executionGroupId);
  void registerConstraintVariableExecution(uint64_t executionGroupId,
                                           Variable variable,
                                           const char *owner);
  void registerConstraintEquationExecution(uint64_t executionGroupId,
                                           Equation equation,
                                           const char *owner);
  void registerConstraintStateSet(uint64_t componentId, uint64_t stateSetId,
                                  bool active, const char *canonicalKey);
  void registerConstraintStateSetVariable(uint64_t componentId,
                                          uint64_t stateSetId,
                                          Variable variable,
                                          const char *owner,
                                          const char *name);
  void registerConstraintStateSetEquation(uint64_t componentId,
                                          uint64_t stateSetId,
                                          Equation equation,
                                          const char *owner,
                                          const char *role, uint64_t order,
                                          const char *matchedVariableName);
  void registerConstraintExecutionEpoch(
      uint64_t domainId, uint64_t epochId, bool active,
      const char *canonicalKey, uint64_t preferDummyScalars,
      uint64_t defaultDummyScalars, uint64_t avoidDummyScalars);
  void registerConstraintExecutionEpochTransition(uint64_t domainId,
                                                  uint64_t fromEpochId,
                                                  uint64_t toEpochId);
  void registerConstraintExecutionEpochGroup(
      uint64_t domainId, uint64_t epochId, uint64_t executionGroupId,
      const char *kind, const char *mode, uint64_t topologicalOrder,
      uint64_t outerEquationScalars, uint64_t outerVariableScalars,
      uint64_t localEquationScalars, uint64_t localVariableScalars);
  void registerConstraintExecutionEpochGroupCertificate(
      uint64_t domainId, uint64_t epochId, uint64_t executionGroupId,
      uint64_t outerEquationScalars, uint64_t outerVariableScalars,
      uint64_t localEquationScalars, uint64_t localVariableScalars);
  void registerConstraintExecutionEpochGroupMember(
      uint64_t domainId, uint64_t epochId, uint64_t executionGroupId,
      uint64_t componentId);
  void registerConstraintExecutionEpochGroupDependency(
      uint64_t domainId, uint64_t epochId, uint64_t executionGroupId,
      uint64_t dependencyGroupId);
  void registerConstraintExecutionEpochVariable(
      uint64_t domainId, uint64_t epochId, uint64_t executionGroupId,
      Variable variable, const char *owner, const char *name);
  void registerConstraintExecutionEpochEquation(
      uint64_t domainId, uint64_t epochId, uint64_t executionGroupId,
      Equation equation, const char *owner);
  void registerConstraintExecutionEpochEquationMetadata(
      uint64_t domainId, uint64_t epochId, Equation equation,
      const char *role, uint64_t order, const char *matchedVariableName);

  /// 中文：variable/equation handles 是扁平求解向量与原数组 getter/setter
  /// 之间的桥梁；一个 handle 始终对应一个已认证 compact block。
  /// English: Variable/equation handles bridge flattened solver vectors and
  /// original-array getters/setters. Each handle represents one certified
  /// compact block.
  Variable addAlgebraicVariable(uint64_t rank, const uint64_t *dimensions,
                                VariableGetter getterFunction,
                                VariableSetter setterFunction,
                                const char *name);

  Variable addStateVariable(uint64_t rank, const uint64_t *dimensions,
                            VariableGetter stateGetterFunction,
                            VariableSetter stateSetterFunction,
                            VariableGetter derivativeGetterFunction,
                            VariableSetter derivativeSetterFunction,
                            const char *name);

  bool setVariableNominal(Variable variable,
                          VariableGetter nominalGetterFunction);

  /// Add the information about an equation that is handled by IDA.
  Equation addEquation(const int64_t *ranges, uint64_t rank,
                       const char *stringRepresentation);

  void addVariableAccess(Equation equation, Variable variableIndex,
                         AccessFunction accessFunction);

  void addVariableRangeAccess(Equation equation, Variable variable,
                              const int64_t *ranges, uint64_t rank);

  /// Add the function pointer that computes the residual value of an
  /// equation.
  void setResidualFunction(Equation equationIndex,
                           ResidualFunction residualFunction);

  /// Add the function pointer that computes a partial derivative of an
  /// equation.
  void addJacobianFunction(Equation equationIndex, Variable variableIndex,
                           JacobianFunction jacobianFunction,
                           uint64_t numOfSeeds, uint64_t *seedSizes);

  /// Instantiate and initialize all the classes needed by IDA in order to
  /// solve the given system of equations. It also sets optional simulation
  /// parameters for IDA. It must be called before the first usage of
  /// idaStep() and after a call to idaAllocData(). It may fail in case of
  /// malformed model.
  bool initialize();

  /// Invoke IDA to perform the computation of the initial values of the
  /// variables. Returns true if the computation was successful, false
  /// otherwise.
  bool calcIC();

  /// Invoke IDA to perform one step of the computation. If a time step is
  /// given, the output will show the variables in an equidistant time grid
  /// based on the step time parameter. Otherwise, the output will show the
  /// variables at every step of the computation. Returns true if the
  /// computation was successful, false otherwise.
  bool step();

  /// Returns the time reached by the solver after the last step.
  realtype getCurrentTime() const;

  /// Prints statistics regarding the computation of the system.
  void printStatistics() const;

  /// IDAResFn user-defined residual function, passed to IDA through
  /// IDAInit. It contains how to compute the Residual Function of the
  /// system, starting from the provided UserData struct, iterating through
  /// every equation.
  /// 中文：这两个 SUNDIALS callbacks 先刷新 component-local closures，再计算
  /// outer residual 或带 Schur 修正的 Jacobian；普通模型直接走原有路径。
  /// English: These SUNDIALS callbacks refresh component-local closures before
  /// computing outer residuals or Schur-corrected Jacobians. Ordinary models
  /// retain the original path.
  static int residualFunction(realtype time, N_Vector variables,
                              N_Vector derivatives, N_Vector residuals,
                              void *userData);

  /// IDALsJacFn user-defined Jacobian approximation function, passed to
  /// IDA through IDASetJacFn. It contains how to compute the Jacobian
  /// Matrix of the system, starting from the provided UserData struct,
  /// iterating through every equation and variable. The matrix is
  /// represented in CSR format.
  static int jacobianMatrix(realtype time, realtype alpha, N_Vector variables,
                            N_Vector derivatives, N_Vector residuals,
                            SUNMatrix jacobianMatrix, void *userData,
                            N_Vector tempv1, N_Vector tempv2, N_Vector tempv3);

private:
  [[nodiscard]] uint64_t getNumOfArrayVariables() const;

  [[nodiscard]] uint64_t getNumOfScalarVariables() const;

  [[nodiscard]] VariableKind getVariableKind(Variable variable) const;

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

  std::vector<JacobianColumn>
  computeAccessedJacobianColumns(Equation eq, const int64_t *equationIndices,
                                 bool includeLocalVariables) const;

  void computeNNZ();

  void rebuildJacobianMatrixStorage();

  void computeThreadChunks();

  void copyVariablesFromMARCO(N_Vector algebraicAndStateVariablesVector,
                              N_Vector derivativeVariablesVector);

  void copyVariablesIntoMARCO(N_Vector algebraicAndStateVariablesVector,
                              N_Vector derivativeVariablesVector);

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

  /// 中文：constraint runtime 子系统分为证书验证、local solve/Schur、accepted
  /// checkpoint 和 basis-switch transaction；这些 helper 不改变公开 ABI。
  /// English: The constraint runtime subsystem is divided into certificate
  /// validation, local solve/Schur, accepted checkpoints, and the basis-switch
  /// transaction. These helpers do not alter the public ABI.
  bool recordConstraintProbeSample(const char *sampleKind, realtype time,
                                   N_Vector variables,
                                   N_Vector derivatives);
  void printConstraintProbeSummary() const;

  struct ConstraintComponentExecutionRuntime;
  bool validateConstraintStateSetRegistrations();
  bool prepareConstraintComponentExecutions();
  int solveConstraintComponents(realtype time);
  bool refreshConstraintComponents(realtype time);
  bool evaluateConstraintJacobian(
      ConstraintComponentExecutionRuntime &execution, Equation equation,
      const std::vector<int64_t> &equationIndices,
      const JacobianColumn &column, realtype time, realtype alpha,
      double &result);
  bool applyConstraintSchur(realtype time, realtype alpha);
  bool isLocalVariable(Variable variable) const;
  bool isLocalEquation(Equation equation) const;
  void saveAcceptedConstraintCheckpoint(realtype time);
  bool restoreAcceptedConstraintCheckpoint();
  bool verifyAcceptedConstraintCheckpointRestored(
      ConstraintSwitchScope scope, uint64_t expectedStateSet,
      double &maximumDifference);
  bool rollbackConstraintStateSetSwitch(ConstraintSwitchScope scope,
                                        uint64_t previousStateSet);
  std::string getConstraintConfigurationSignature() const;
  bool performPendingBasisSwitch();
  bool rebuildIDAAfterBasisSwitch(realtype time);
  bool advanceIDAReInitRecovery(realtype time);
  static void idaErrorHandler(int errorCode, const char *module,
                              const char *function, char *message,
                              void *userData);

private:
  /// @name Forwarded methods
  /// {

  bool idaInit();
  bool idaSVTolerances();
  bool idaSetLinearSolver();
  bool idaSetUserData();
  bool idaSetMaxNumSteps();
  bool idaSetInitialStepSize();
  bool idaSetMinStepSize();
  bool idaSetMaxStepSize();
  bool idaSetMaxErrTestFails();
  bool idaSetSuppressAlg();
  bool idaSetId();
  bool idaSetJacobianFunction();
  bool idaSetMaxNonlinIters();
  bool idaSetMaxConvFails();
  bool idaSetNonlinConvCoef();
  bool idaSetNonlinConvCoefIC();
  bool idaSetMaxNumStepsIC();
  bool idaSetMaxNumJacsIC();
  bool idaSetMaxNumItersIC();
  bool idaSetLineSearchOffIC();

  /// }
  /// @name Debug functions
  /// {
  void printVariablesVector(N_Vector variables) const;

  void printDerivativesVector(N_Vector derivatives) const;

  void printResidualsVector(N_Vector residuals) const;

  void printJacobianMatrix(SUNMatrix jacobianMatrix) const;

  /// }

private:
#if SUNDIALS_VERSION_MAJOR >= 7
  SUNComm comm{0};
#endif

#if SUNDIALS_VERSION_MAJOR >= 6
  // SUNDIALS context.
  SUNContext ctx{nullptr};
#endif

  // Whether the instance has been inizialized or not.
  bool initialized{false};

  // 中文：模型级默认只在 CLI 未显式指定时生效，最终值和来源一并保留供 probe。
  // English: The model-level default applies only without an explicit CLI
  // choice; retain both the effective value and its source for diagnostics.
  std::optional<booleantype> modelSuppressAlgebraicErrorTest;
  booleantype effectiveSuppressAlgebraicErrorTest{SUNFALSE};
  std::string effectiveSuppressAlgebraicErrorTestSource{"legacy"};

  // 中文：注册期容器保存不可变的语义与执行描述；initialize() 认证完成后，
  // constraintComponentExecutions 才持有可变 KINSOL/KLU runtime 状态。
  // English: Registration containers hold immutable semantic and execution
  // descriptors. Mutable KINSOL/KLU state is created in
  // constraintComponentExecutions only after initialize() certifies them.
  std::map<uint64_t, ConstraintComponentDescriptor> constraintComponents;
  std::vector<ConstraintEquationDescriptor> constraintEquations;
  std::vector<ConstraintVariableDescriptor> constraintVariables;
  std::map<uint64_t, SolverExecutionGroupDescriptor> solverExecutionGroups;
  std::set<uint64_t> executableConstraintComponents;
  std::vector<ConstraintVariableExecutionDescriptor>
      constraintVariableExecutions;
  std::vector<ConstraintEquationExecutionDescriptor>
      constraintEquationExecutions;
  std::vector<ConstraintStateSetDescriptor> constraintStateSets;
  std::vector<ConstraintStateSetVariableDescriptor>
      constraintStateSetVariables;
  std::vector<ConstraintStateSetEquationDescriptor>
      constraintStateSetEquations;
  std::map<std::pair<uint64_t, uint64_t>, ConstraintExecutionEpochDescriptor>
      constraintExecutionEpochs;
  std::map<std::pair<uint64_t, uint64_t>, ConstraintResidualRoleCounts>
      constraintExecutionEpochResidualCertificates;
  std::map<std::tuple<uint64_t, uint64_t, uint64_t>,
           ConstraintExecutionEpochGroupDescriptor>
      constraintExecutionEpochGroups;
  std::vector<ConstraintExecutionEpochVariableDescriptor>
      constraintExecutionEpochVariables;
  std::vector<ConstraintExecutionEpochEquationDescriptor>
      constraintExecutionEpochEquations;
  std::vector<bool> localConstraintVariables;
  std::vector<bool> localConstraintEquations;
  std::vector<std::unique_ptr<ConstraintComponentExecutionRuntime>>
      constraintComponentExecutions;
  std::map<uint64_t, uint64_t> activeConstraintStateSets;
  std::map<std::tuple<uint64_t, uint64_t, uint64_t>, uint64_t>
      epochRuntimeExecutionGroupIds;
  std::set<uint64_t> activeEpochRuntimeExecutionGroupIds;
  bool constraintStateSetRegistrationsCertified{false};

  // 中文：切换请求来自失败 trial，但事务只能从最近 accepted checkpoint
  // 开始；pending、checkpoint 和 recovery 分别保存触发、回滚及重启阶段状态。
  // English: A failed trial may request a switch, but the transaction starts
  // only from the latest accepted checkpoint. Pending, checkpoint, and recovery
  // objects respectively hold trigger, rollback, and restart state.
  struct PendingBasisSwitch {
    ConstraintSwitchScope scope;
    uint64_t stateSetId{0};
    realtype trialTime{0};
    double condition{0};
    std::string phase;
  };

  struct AcceptedConstraintCheckpoint {
    bool valid{false};
    realtype time{0};
    std::vector<realtype> variables;
    std::vector<realtype> derivatives;
    std::vector<std::vector<realtype>> modelVariables;
    realtype stepSize{0};
    int order{1};
  };

  std::optional<PendingBasisSwitch> pendingBasisSwitch;
  AcceptedConstraintCheckpoint acceptedConstraintCheckpoint;
  bool evaluatingBasisCandidate{false};
  bool expectedConstraintFailure{false};
  uint64_t basisEpoch{0};
  uint64_t basisSwitchCount{0};
  uint64_t idaReInitCount{0};
  bool verifyFirstOrderAfterReInit{false};
  IDARestartRecoveryState restartRecovery;
  SolverDiscontinuityCoordinator discontinuityCoordinator;

  // 中文：probe 汇总与求解控制解耦，关闭 probe 时不改变 IDA stepping 模式或
  // constraint 数学路径。
  // English: Probe summaries are decoupled from solver control; disabling the
  // probe does not alter IDA stepping or constraint mathematics.
  struct ConstraintProbeSummary {
    uint64_t acceptedSamples{0};
    uint64_t outputSamples{0};
    double maximumPositionResidual{0};
    double maximumTangentResidual{0};
    double maximumHighestResidual{0};
    double maximumSupportResidual{0};
    double maximumCondition{0};
    realtype maximumPositionTime{0};
    realtype maximumTangentTime{0};
    realtype maximumConditionTime{0};
  };

  std::map<uint64_t, ConstraintProbeSummary> constraintProbeSummaries;
  std::optional<realtype> constraintProbeInternalTime;
  long int constraintProbePreviousNonlinearIterations{0};
  long int constraintProbePreviousErrorTestFailures{0};
  long int constraintProbePreviousConvergenceFailures{0};

  // Model size.
  uint64_t scalarVariablesNumber{0};
  uint64_t scalarEquationsNumber{0};
  uint64_t nonZeroValuesNumber{0};

  // The iteration ranges of the vectorized equations.
  std::vector<MultidimensionalRange> equationRanges;

  // The residual functions associated with the equations.
  // The i-th position contains the pointer to the residual function of the
  // i-th equation.
  std::vector<ResidualFunction> residualFunctions;

  // The jacobian functions associated with the equations.
  // The i-th position contains the list of partial derivative functions of
  // the i-th equation. The j-th function represents the function to
  // compute the derivative with respect to the j-th variable.
  std::vector<std::vector<JacobianFunctionDescriptor>> jacobianFunctions;

  // Whether the IDA instance is informed about the accesses to the
  // variables.
  bool precomputedAccesses{false};

  std::vector<VarAccessList> variableAccesses;

  using VariableRangeAccess =
      std::pair<Variable, MultidimensionalRange>;
  std::vector<std::vector<VariableRangeAccess>> variableRangeAccesses;

  // The offset of each array variable inside the flattened variables
  // vector.
  std::vector<uint64_t> variableOffsets;

  // The dimensions list of each array variable.
  std::vector<VariableDimensions> variablesDimensions;

  // The offset of each array equation inside the flattened equations
  // vector.
  std::vector<uint64_t> equationOffsets;

  // Simulation times.
  int64_t stepsNumber{0};
  realtype startTime;
  realtype endTime;
  realtype timeStep;
  realtype currentTime = 0;

  // Variables vectors and values.
  N_Vector variablesVector;
  N_Vector derivativesVector;

  // The vector stores whether each scalar variable is an algebraic or a
  // state one.
  // 0 = algebraic
  // 1 = state
  N_Vector idVector;

  // The tolerance for each scalar variable.
  N_Vector tolerancesVector;

  // IDA classes.
  void *idaMemory;

  SUNMatrix sparseMatrix;

  // Support structure for the computation of the jacobian matrix.
  // The outer vector has a number of elements equal to the scalar number
  // of equations. Each of them represents a row of the matrix and consists
  // in a vector of paired elements. The first element of each pair
  // represents the index of the column (that is, the independent scalar
  // variable for the partial derivative) while the second one is the
  // value of the partial derivative.
  std::vector<std::vector<std::pair<sunindextype, double>>> jacobianMatrixData;

  SUNLinearSolver linearSolver;

  std::vector<VariableGetter> algebraicAndStateVariablesGetters;
  std::vector<VariableSetter> algebraicAndStateVariablesSetters;
  std::vector<VariableGetter> variableNominalGetters;

  std::vector<VariableGetter> derivativeVariablesGetters;
  std::vector<VariableSetter> derivativeVariablesSetters;

  // Mapping from the IDA variable position to state variables position.
  std::map<Variable, size_t> stateVariablesMapping;

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
} // namespace marco::runtime::sundials::ida

//===---------------------------------------------------------------------===//
// 中文：以下 additive C ABI 由 IDAToLLVM 生成代码调用；现有构造/step ABI
// 保持兼容，constraint/state-set 注册只增加可选描述能力。
// English: The additive C ABI below is called by IDAToLLVM-generated code.
// Existing construction/step ABI remains compatible; constraint/state-set
// registrations only add optional descriptor capabilities.
//===---------------------------------------------------------------------===//

RUNTIME_FUNC_DECL(idaCreate, PTR(void))

RUNTIME_FUNC_DECL(idaCalcIC, void, PTR(void))

RUNTIME_FUNC_DECL(idaStep, void, PTR(void))

RUNTIME_FUNC_DECL(idaFree, void, PTR(void))

RUNTIME_FUNC_DECL(idaSetStartTime, void, PTR(void), double)

RUNTIME_FUNC_DECL(idaSetEndTime, void, PTR(void), double)

RUNTIME_FUNC_DECL(idaSetSuppressAlgebraicErrorTest, void, PTR(void), bool)

RUNTIME_FUNC_DECL(idaRegisterConstraintComponent, void, PTR(void), uint64_t)
RUNTIME_FUNC_DECL(idaRegisterConstraintComponentRoleOwnership, void,
                  PTR(void), uint64_t, PTR(void), uint64_t, uint64_t)

RUNTIME_FUNC_DECL(idaRegisterConstraintEquation, void, PTR(void), uint64_t,
                  uint64_t, PTR(void), uint64_t, PTR(void))

RUNTIME_FUNC_DECL(idaRegisterConstraintVariable, void, PTR(void), uint64_t,
                  uint64_t, PTR(void), PTR(void))

RUNTIME_FUNC_DECL(idaEnableConstraintComponentExecution, void, PTR(void),
                  uint64_t)

RUNTIME_FUNC_DECL(idaRegisterSolverExecutionGroup, void, PTR(void), uint64_t,
                  PTR(void), PTR(void), uint64_t)

RUNTIME_FUNC_DECL(idaRegisterSolverExecutionGroupCertificate, void, PTR(void),
                  uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)

RUNTIME_FUNC_DECL(idaRegisterSolverExecutionGroupMember, void, PTR(void),
                  uint64_t, uint64_t)

RUNTIME_FUNC_DECL(idaRegisterSolverExecutionGroupDependency, void, PTR(void),
                  uint64_t, uint64_t)

RUNTIME_FUNC_DECL(idaRegisterConstraintVariableExecution, void, PTR(void),
                  uint64_t, uint64_t, PTR(void))

RUNTIME_FUNC_DECL(idaRegisterConstraintEquationExecution, void, PTR(void),
                  uint64_t, uint64_t, PTR(void))

RUNTIME_FUNC_DECL(idaRegisterConstraintStateSet, void, PTR(void), uint64_t,
                  uint64_t, uint64_t, PTR(void))

RUNTIME_FUNC_DECL(idaRegisterConstraintStateSetVariable, void, PTR(void),
                  uint64_t, uint64_t, uint64_t, PTR(void), PTR(void))

RUNTIME_FUNC_DECL(idaRegisterConstraintStateSetEquation, void, PTR(void),
                  uint64_t, uint64_t, uint64_t, PTR(void), PTR(void),
                  uint64_t, PTR(void))

RUNTIME_FUNC_DECL(idaRegisterConstraintExecutionEpoch, void, PTR(void),
                  uint64_t, uint64_t, uint64_t, PTR(void), uint64_t,
                  uint64_t, uint64_t)

RUNTIME_FUNC_DECL(idaRegisterConstraintExecutionEpochTransition, void,
                  PTR(void), uint64_t, uint64_t, uint64_t)

RUNTIME_FUNC_DECL(idaRegisterConstraintExecutionEpochGroup, void, PTR(void),
                  uint64_t, uint64_t, uint64_t, PTR(void), PTR(void), uint64_t)

RUNTIME_FUNC_DECL(idaRegisterConstraintExecutionEpochGroupCertificate, void,
                  PTR(void), uint64_t, uint64_t, uint64_t, uint64_t,
                  uint64_t, uint64_t, uint64_t)

RUNTIME_FUNC_DECL(idaRegisterConstraintExecutionEpochGroupMember, void,
                  PTR(void), uint64_t, uint64_t, uint64_t, uint64_t)

RUNTIME_FUNC_DECL(idaRegisterConstraintExecutionEpochGroupDependency, void,
                  PTR(void), uint64_t, uint64_t, uint64_t, uint64_t)

RUNTIME_FUNC_DECL(idaRegisterConstraintExecutionEpochVariable, void,
                  PTR(void), uint64_t, uint64_t, uint64_t, uint64_t,
                  PTR(void), PTR(void))

RUNTIME_FUNC_DECL(idaRegisterConstraintExecutionEpochEquation, void,
                  PTR(void), uint64_t, uint64_t, uint64_t, uint64_t,
                  PTR(void))

RUNTIME_FUNC_DECL(idaRegisterConstraintExecutionEpochEquationMetadata, void,
                  PTR(void), uint64_t, uint64_t, uint64_t, PTR(void),
                  uint64_t, PTR(void))

RUNTIME_FUNC_DECL(idaSetTimeStep, void, PTR(void), double)

RUNTIME_FUNC_DECL(idaGetCurrentTime, double, PTR(void))

RUNTIME_FUNC_DECL(idaAddAlgebraicVariable, uint64_t, PTR(void), uint64_t,
                  PTR(uint64_t), PTR(void), PTR(void), PTR(void))

RUNTIME_FUNC_DECL(idaAddStateVariable, uint64_t, PTR(void), uint64_t,
                  PTR(uint64_t), PTR(void), PTR(void), PTR(void), PTR(void),
                  PTR(void))

RUNTIME_FUNC_DECL(idaSetVariableNominal, void, PTR(void), uint64_t, PTR(void))

RUNTIME_FUNC_DECL(idaAddVariableAccess, void, PTR(void), uint64_t, uint64_t,
                  PTR(void))

RUNTIME_FUNC_DECL(idaAddVariableRangeAccess, void, PTR(void), uint64_t,
                  uint64_t, PTR(int64_t), uint64_t)

RUNTIME_FUNC_DECL(idaAddEquation, uint64_t, PTR(void), PTR(int64_t), uint64_t,
                  PTR(void))

RUNTIME_FUNC_DECL(idaSetResidual, void, PTR(void), uint64_t, PTR(void))

RUNTIME_FUNC_DECL(idaAddJacobian, void, PTR(void), uint64_t, uint64_t,
                  PTR(void), uint64_t, PTR(uint64_t))

RUNTIME_FUNC_DECL(printStatistics, void, PTR(void))

#endif // SUNDIALS_ENABLE

#endif // MARCO_RUNTIME_SOLVERS_IDA_INSTANCE_H
