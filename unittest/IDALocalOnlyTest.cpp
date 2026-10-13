#include "marco/Runtime/Solvers/IDA/Instance.h"
#include "marco/Runtime/Solvers/IDA/Options.h"
#include "marco/Runtime/Multithreading/Options.h"
#include "gtest/gtest.h"
#include <cmath>
#include <limits>
#include <string>

using namespace marco::runtime::sundials;
using namespace marco::runtime::sundials::ida;

namespace {
double firstValue;
double secondValue;
bool failSecond;
double lastConditionValue;
struct OptionsGuard {
  Options saved = getOptions();
  ~OptionsGuard() { getOptions() = saved; }
};

double getFirst(const uint64_t *) { return firstValue; }
double getSecond(const uint64_t *) { return secondValue; }
void setFirst(double value, const uint64_t *) { firstValue = value; }
void setSecond(double value, const uint64_t *) { secondValue = value; }
void pointAccess(const int64_t *, uint64_t *result) { result[0] = 0; }
double firstResidual(double time, const int64_t *) {
  return firstValue - time;
}
double secondResidual(double time, const int64_t *) {
  return failSecond && time > 2
             ? std::numeric_limits<double>::quiet_NaN()
             : secondValue - 2 * time;
}
double unitJacobian(double, const int64_t *, const uint64_t *, double,
                    uint64_t, const uint64_t *) {
  return 1;
}
double disappearingRootResidual(double time, const int64_t *) {
  return secondValue * secondValue + (time > 2 ? 1 : -1);
}
double disappearingRootJacobian(double time, const int64_t *, const uint64_t *,
                                double, uint64_t, const uint64_t *) {
  if (time > 2)
    lastConditionValue = secondValue;
  return 2 * secondValue;
}

std::pair<Variable, Equation> addLocalGroup(IDAInstance &instance, uint64_t id, VariableGetter getter,
                   VariableSetter setter, ResidualFunction residual,
                   JacobianFunction jacobian = unitJacobian) {
  uint64_t dimensions[] = {1};
  int64_t ranges[] = {0, 1};
  Variable variable = instance.addAlgebraicVariable(1, dimensions, getter,
                                                    setter, "local");
  Equation equation = instance.addEquation(ranges, 1, "local time constraint");
  instance.addVariableAccess(equation, variable, pointAccess);
  instance.setResidualFunction(equation, residual);
  uint64_t unusedSeedSize = 1;
  instance.addJacobianFunction(equation, variable, jacobian, 0,
                                &unusedSeedSize);
  instance.registerConstraintComponent(id);
  instance.registerConstraintComponentRoleOwnership(id, "position", 1, 1);
  instance.registerConstraintVariable(id, variable, "dependent", "local");
  instance.registerConstraintEquation(id, equation, "position", 0, "local");
  instance.registerSolverExecutionGroup(id, "constraint", "implicit_closure", id);
  instance.registerSolverExecutionGroupCertificate(id, 0, 0, 1, 1);
  instance.registerSolverExecutionGroupMember(id, id);
  instance.enableConstraintComponentExecution(id);
  instance.registerConstraintVariableExecution(id, variable, "local");
  instance.registerConstraintEquationExecution(id, equation, "local");
  return {variable, equation};
}
} // namespace

TEST(IDALocalOnlyTest, InitializesAndAdvancesTimeDependentLocalConstraints) {
  OptionsGuard optionsGuard;
  getOptions().equidistantTimeGrid = true;
  firstValue = -10;
  secondValue = -20;
  failSecond = false;
  IDAInstance instance;
  instance.setStartTime(2);
  instance.setEndTime(3);
  instance.setTimeStep(0.25);
  addLocalGroup(instance, 1, getFirst, setFirst, firstResidual);
  addLocalGroup(instance, 2, getSecond, setSecond, secondResidual);
  ASSERT_TRUE(instance.initialize());
  EXPECT_NEAR(firstValue, 2, 1e-12);
  EXPECT_NEAR(secondValue, 4, 1e-12);
  ASSERT_TRUE(instance.step());
  EXPECT_DOUBLE_EQ(instance.getCurrentTime(), 2.25);
  EXPECT_NEAR(firstValue, 2.25, 1e-12);
  EXPECT_NEAR(secondValue, 4.5, 1e-12);
}

TEST(IDALocalOnlyTest, LateLocalFailurePreservesAcceptedValuesAndTime) {
  OptionsGuard optionsGuard;
  getOptions().equidistantTimeGrid = true;
  firstValue = 2;
  secondValue = 4;
  failSecond = true;
  IDAInstance instance;
  instance.setStartTime(2);
  instance.setEndTime(3);
  instance.setTimeStep(0.25);
  addLocalGroup(instance, 1, getFirst, setFirst, firstResidual);
  addLocalGroup(instance, 2, getSecond, setSecond, secondResidual);
  ASSERT_TRUE(instance.initialize());
  testing::internal::CaptureStderr();
  EXPECT_FALSE(instance.step());
  std::string diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_NE(diagnostic.find("kind=jacobian_evaluation_failed"), std::string::npos);
  EXPECT_EQ(diagnostic.find("dynamic_state_selection_required"), std::string::npos);
  EXPECT_DOUBLE_EQ(instance.getCurrentTime(), 2);
  EXPECT_DOUBLE_EQ(firstValue, 2);
  EXPECT_DOUBLE_EQ(secondValue, 4);
}

TEST(IDALocalOnlyTest, FailedTrialConditionIsInspectedBeforeOwnerRollback) {
  OptionsGuard optionsGuard;
  getOptions().equidistantTimeGrid = true;
  firstValue = 2;
  secondValue = 1;
  lastConditionValue = std::numeric_limits<double>::quiet_NaN();
  IDAInstance instance;
  instance.setStartTime(2);
  instance.setEndTime(3);
  instance.setTimeStep(0.25);
  addLocalGroup(instance, 1, getFirst, setFirst, firstResidual);
  addLocalGroup(instance, 2, getSecond, setSecond, disappearingRootResidual,
                 disappearingRootJacobian);
  ASSERT_TRUE(instance.initialize());
  EXPECT_FALSE(instance.step());
  EXPECT_TRUE(std::isfinite(lastConditionValue));
  EXPECT_NE(lastConditionValue, 1);
  EXPECT_DOUBLE_EQ(instance.getCurrentTime(), 2);
  EXPECT_DOUBLE_EQ(firstValue, 2);
  EXPECT_DOUBLE_EQ(secondValue, 1);
}

TEST(IDALocalOnlyTest, ClipsLocalOutputGridToFinalTime) {
  OptionsGuard optionsGuard;
  getOptions().equidistantTimeGrid = true;
  firstValue = -10;
  IDAInstance instance;
  instance.setStartTime(2);
  instance.setEndTime(2.25);
  instance.setTimeStep(0.1);
  addLocalGroup(instance, 1, getFirst, setFirst, firstResidual);
  ASSERT_TRUE(instance.calcIC());
  for (double expected : {2.1, 2.2, 2.25}) {
    ASSERT_TRUE(instance.step());
    EXPECT_DOUBLE_EQ(instance.getCurrentTime(), expected);
    EXPECT_NEAR(firstValue, expected, 1e-12);
  }
  ASSERT_TRUE(instance.step());
  EXPECT_DOUBLE_EQ(instance.getCurrentTime(), 2.25);
  EXPECT_NEAR(firstValue, 2.25, 1e-12);
}

TEST(IDALocalOnlyTest, OuterSolveUsesStartTimeAndClipsFinalOutput) {
  OptionsGuard optionsGuard;
  getOptions().equidistantTimeGrid = true;
  for (double start : {0.0, 2.0}) {
    SCOPED_TRACE(start);
    firstValue = -10;
    IDAInstance instance;
    instance.setStartTime(start);
    instance.setEndTime(start + 0.25);
    instance.setTimeStep(0.1);
    uint64_t dimensions[] = {1};
    int64_t ranges[] = {0, 1};
    Variable variable = instance.addAlgebraicVariable(
        1, dimensions, getFirst, setFirst, "outer");
    Equation equation = instance.addEquation(ranges, 1, "x = time");
    instance.addVariableAccess(equation, variable, pointAccess);
    instance.setResidualFunction(equation, firstResidual);
    uint64_t unusedSeedSize = 1;
    instance.addJacobianFunction(equation, variable, unitJacobian, 0,
                                 &unusedSeedSize);
    ASSERT_TRUE(instance.calcIC());
    EXPECT_NEAR(firstValue, start, 1e-10);
    for (double offset : {0.1, 0.2, 0.25}) {
      ASSERT_TRUE(instance.step());
      EXPECT_DOUBLE_EQ(instance.getCurrentTime(), start + offset);
      EXPECT_NEAR(firstValue, start + offset, 1e-10);
    }
    ASSERT_TRUE(instance.step());
    EXPECT_DOUBLE_EQ(instance.getCurrentTime(), start + 0.25);
  }
}

TEST(IDALocalOnlyTest, PolynomialMetadataPreservesFailureInspectionAndRollback) {
  OptionsGuard optionsGuard;
  getOptions().equidistantTimeGrid = true;
  firstValue = 2;
  secondValue = 1;
  lastConditionValue = std::numeric_limits<double>::quiet_NaN();
  IDAInstance instance;
  instance.setStartTime(2);
  instance.setEndTime(3);
  instance.setTimeStep(0.25);
  auto a = addLocalGroup(instance, 1, getFirst, setFirst, firstResidual);
  auto b = addLocalGroup(instance, 2, getSecond, setSecond,
                         disappearingRootResidual, disappearingRootJacobian);
  uint64_t linear[] = {1}, zero[] = {0}, square[] = {1};
  ASSERT_TRUE(instance.setScalarPolynomialV1(a.second, a.first, 1, &a.first,
                                             1, linear, zero));
  ASSERT_TRUE(instance.setScalarPolynomialV1(b.second, b.first, 1, &b.first,
                                             1, zero, square));
  ASSERT_TRUE(instance.initialize());
  EXPECT_FALSE(instance.step());
  EXPECT_TRUE(std::isfinite(lastConditionValue));
  EXPECT_NE(lastConditionValue, 1);
  EXPECT_DOUBLE_EQ(instance.getCurrentTime(), 2);
  EXPECT_DOUBLE_EQ(firstValue, 2);
  EXPECT_DOUBLE_EQ(secondValue, 1);
}

TEST(IDALocalOnlyTest, InvalidPolynomialRegistrationsFailBeforeExecution) {
  for (unsigned which = 0; which < 4; ++which) {
    IDAInstance instance;
    auto a = addLocalGroup(instance, 1, getFirst, setFirst, firstResidual);
    uint64_t linear[] = {1}, squared[] = {0};
    if (which == 0) linear[0] = 2;
    if (which == 1) squared[0] = 1;
    if (which == 2) a.first = 900;
    if (which == 3)
      ASSERT_TRUE(instance.setScalarPolynomialV1(a.second, a.first, 1, &a.first,
                                                 1, linear, squared));
    EXPECT_FALSE(instance.setScalarPolynomialV1(a.second, a.first, 1, &a.first,
                                                1, linear, squared));
    testing::internal::CaptureStderr();
    EXPECT_FALSE(instance.initialize());
    auto diagnostic = testing::internal::GetCapturedStderr();
    EXPECT_NE(diagnostic.find("Invalid V1 scalar polynomial"), std::string::npos);
  }
}

TEST(IDALocalOnlyTest, PolynomialPairingCannotCrossExecutionOwners) {
  OptionsGuard guard;
  IDAInstance instance;
  instance.setStartTime(2);
  instance.setEndTime(3);
  instance.setTimeStep(0.25);
  auto a = addLocalGroup(instance, 1, getFirst, setFirst, firstResidual);
  auto b = addLocalGroup(instance, 2, getSecond, setSecond, secondResidual);
  uint64_t variables[] = {a.first, b.first}, linear[] = {1}, squared[] = {0};
  ASSERT_TRUE(instance.setScalarPolynomialV1(a.second, b.first, 2, variables,
                                             1, linear, squared));
  testing::internal::CaptureStderr();
  EXPECT_FALSE(instance.initialize());
  auto diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_NE(diagnostic.find("pairing certificate mismatch"), std::string::npos);
}

TEST(IDALocalOnlyTest, CallbackReplacementDiscardsUninitializedPolynomialProof) {
  OptionsGuard guard;
  for (unsigned which = 0; which != 4; ++which) {
    SCOPED_TRACE(which);
    firstValue = secondValue = 0;
    failSecond = false;
    IDAInstance instance;
    instance.setStartTime(2);
    instance.setEndTime(3);
    instance.setTimeStep(0.25);
    auto a = addLocalGroup(instance, 1, getFirst, setFirst, firstResidual);
    auto b = addLocalGroup(instance, 2, getSecond, setSecond, secondResidual);
    uint64_t variables[] = {a.first, b.first}, linear[] = {1}, squared[] = {0};
    // This stale pairing would be rejected if it survived the definition change.
    ASSERT_TRUE(instance.setScalarPolynomialV1(a.second, b.first, 2, variables,
                                               1, linear, squared));
    if (which == 0) instance.setResidualFunction(a.second, firstResidual);
    if (which == 1) instance.addJacobianFunction(a.second, a.first,
                                                 unitJacobian, 0, nullptr);
    if (which == 2) instance.addVariableAccess(a.second, a.first, pointAccess);
    if (which == 3) {
      int64_t range[] = {0, 1};
      instance.addVariableRangeAccess(a.second, a.first, range, 1);
    }
    ASSERT_TRUE(instance.initialize());
    EXPECT_NEAR(firstValue, 2, 1e-12);
    EXPECT_NEAR(secondValue, 4, 1e-12);
  }
}

TEST(IDALocalOnlyTest, StaleActivePolynomialCannotExecuteOldCallbacks) {
  OptionsGuard guard;
  for (unsigned which = 0; which != 3; ++which) {
    SCOPED_TRACE(which);
    firstValue = 2;
    IDAInstance instance;
    instance.setStartTime(2);
    instance.setEndTime(3);
    instance.setTimeStep(0.25);
    auto a = addLocalGroup(instance, 1, getFirst, setFirst, firstResidual);
    uint64_t linear[] = {1}, squared[] = {0};
    ASSERT_TRUE(instance.setScalarPolynomialV1(a.second, a.first, 1, &a.first,
                                               1, linear, squared));
    ASSERT_TRUE(instance.initialize());
    if (which == 0) instance.setResidualFunction(a.second, firstResidual);
    if (which == 1) instance.addJacobianFunction(a.second, a.first,
                                                 unitJacobian, 0, nullptr);
    if (which == 2) instance.addVariableAccess(a.second, a.first, pointAccess);
    testing::internal::CaptureStderr();
    EXPECT_FALSE(instance.calcIC());
    EXPECT_FALSE(instance.step());
    auto diagnostic = testing::internal::GetCapturedStderr();
    EXPECT_NE(diagnostic.find("stale V1 scalar polynomial"), std::string::npos);
    EXPECT_DOUBLE_EQ(firstValue, 2);
    EXPECT_DOUBLE_EQ(instance.getCurrentTime(), 2);
  }
}

namespace {
double inspectionParameter = 1;
uint64_t inspectionResiduals = 0, inspectionJacobians = 0;
double inspectionResidual(double time, const int64_t *) {
  ++inspectionResiduals; // Passive observation, not an input to the equation.
  return firstValue - inspectionParameter * time;
}
double inspectionJacobian(double, const int64_t *, const uint64_t *, double,
                           uint64_t, const uint64_t *) {
  ++inspectionJacobians;
  return 1;
}
double inspectionExplicit(double time, const int64_t *) {
  return inspectionParameter * time;
}
}

TEST(IDALocalOnlyTest, SuccessfulPointReuseKeepsAllHealthChecksAndNewInputs) {
  OptionsGuard guard;
  getOptions().equidistantTimeGrid = true;
#ifdef THREADS_ENABLE
  auto &threadOptions = marco::runtime::multithreading::multithreadingOptions();
  struct RestoreThreads {
    decltype(threadOptions) target;
    marco::runtime::multithreading::MultithreadingOptions saved;
    ~RestoreThreads() { target = saved; }
  } restore{threadOptions, threadOptions};
  threadOptions.numOfThreads = 1;
#endif
  firstValue = 2;
  inspectionParameter = 1;
  IDAInstance instance;
  instance.setStartTime(2);
  instance.setEndTime(3);
  instance.setTimeStep(.25);
  auto a = addLocalGroup(instance, 1, getFirst, setFirst,
                         inspectionResidual, inspectionJacobian);
  uint64_t linear[] = {1}, squared[] = {0};
  ASSERT_TRUE(instance.setScalarPolynomialV1(a.second, a.first, 1, &a.first,
                                             1, linear, squared));
  ASSERT_TRUE(instance.initialize());
  for (double parameter : {3., -2., 1.}) {
    inspectionParameter = parameter; // Outside the previous lexical scope.
    firstValue = 100;               // No old local solution is reusable.
    inspectionResiduals = inspectionJacobians = 0;
    ASSERT_TRUE(instance.step());
    EXPECT_DOUBLE_EQ(firstValue, parameter * instance.getCurrentTime());
    // Pre-health F/J, affine-block F/J, complete final F, post-health J.
    // Only the duplicate final F disappears; both independent health checks
    // and their current-point Jacobians must remain.
    EXPECT_EQ(inspectionResiduals, 3u);
    EXPECT_EQ(inspectionJacobians, 3u);
  }
}

TEST(IDALocalOnlyTest, ExplicitValueKeepsHealthChecksAndDefinitionInvalidation) {
  OptionsGuard guard;
  getOptions().equidistantTimeGrid = true;
#ifdef THREADS_ENABLE
  auto &threadOptions = marco::runtime::multithreading::multithreadingOptions();
  struct RestoreThreads {
    decltype(threadOptions) target;
    marco::runtime::multithreading::MultithreadingOptions saved;
    ~RestoreThreads() { target = saved; }
  } restore{threadOptions, threadOptions};
  threadOptions.numOfThreads = 1;
#endif
  firstValue = 2;
  inspectionParameter = 1;
  IDAInstance instance;
  instance.setStartTime(2);
  instance.setEndTime(3);
  instance.setTimeStep(.25);
  auto a = addLocalGroup(instance, 1, getFirst, setFirst,
                         inspectionResidual, inspectionJacobian);
  uint64_t linear[] = {1}, squared[] = {0};
  ASSERT_TRUE(instance.setScalarPolynomialV1(a.second, a.first, 1, &a.first,
                                             1, linear, squared));
  ASSERT_TRUE(instance.setScalarExplicitV1(a.second, a.first, inspectionExplicit));
  ASSERT_TRUE(instance.initialize());
  for (double parameter : {3., -2., 1.}) {
    inspectionParameter = parameter;
    firstValue = 100;
    inspectionResiduals = inspectionJacobians = 0;
    ASSERT_TRUE(instance.step());
    EXPECT_DOUBLE_EQ(firstValue, parameter * instance.getCurrentTime());
    // Pre-health F/J, current coefficient J, full final F, post-health J.
    EXPECT_EQ(inspectionResiduals, 2u);
    EXPECT_EQ(inspectionJacobians, 3u);
  }
  instance.setResidualFunction(a.second, inspectionResidual);
  testing::internal::CaptureStderr();
  EXPECT_FALSE(instance.step());
  EXPECT_NE(testing::internal::GetCapturedStderr().find("stale V1"),
            std::string::npos);
}

TEST(IDALocalOnlyTest, InvalidExplicitBindingsFailBeforeExecution) {
  for (unsigned which = 0; which != 4; ++which) {
    IDAInstance instance;
    auto a = addLocalGroup(instance, 1, getFirst, setFirst, firstResidual);
    uint64_t linear[] = {which == 2 ? 0u : 1u};
    uint64_t squared[] = {which == 2 ? 1u : 0u};
    ASSERT_TRUE(instance.setScalarPolynomialV1(a.second, a.first, 1, &a.first,
                                               1, linear, squared));
    if (which == 3)
      ASSERT_TRUE(instance.setScalarExplicitV1(a.second, a.first, inspectionExplicit));
    EXPECT_FALSE(instance.setScalarExplicitV1(a.second,
        which == 0 ? 900 : a.first, which == 1 ? nullptr : inspectionExplicit));
    testing::internal::CaptureStderr();
    EXPECT_FALSE(instance.initialize());
    EXPECT_NE(testing::internal::GetCapturedStderr().find("Invalid V1"),
              std::string::npos);
  }
}
