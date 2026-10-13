#include "marco/Runtime/Drivers/KINSOL/CLI.h"
#include "marco/Runtime/Solvers/KINSOL/Instance.h"
#include "marco/Runtime/Solvers/KINSOL/Options.h"
#include "gtest/gtest.h"

using namespace marco::runtime::sundials::kinsol;

namespace {
double value = 0;
double target = 0;
double getValue(const uint64_t *) { return value; }
void setValue(double newValue, const uint64_t *) { value = newValue; }
void singletonAccess(const int64_t *, uint64_t *result) { result[0] = 0; }
double residual(double, const int64_t *) { return value - target; }
double jacobian(double, const int64_t *, const uint64_t *, double, uint64_t,
                const uint64_t *) {
  return 1;
}

class KINSOLCommandLineTest : public ::testing::Test {
protected:
  Options savedOptions = getOptions();
  void SetUp() override { getOptions() = Options{}; }
  void TearDown() override { getOptions() = savedOptions; }

  void parse(std::initializer_list<const char *> arguments) {
    argh::parser parser(static_cast<int>(arguments.size()), arguments.begin());
    CommandLineOptions{}.parseCommandLineOptions(parser);
  }

  void configure(KINSOLInstance &instance) {
    instance.setQuietErrors(true);
    uint64_t dimensions[] = {1};
    int64_t ranges[] = {0, 1};
    auto variable = instance.addVariable(1, dimensions, getValue, setValue,
                                         "cli_value");
    auto equation = instance.addEquation(ranges, 1, "x - target = 0");
    instance.addVariableAccess(equation, variable, singletonAccess);
    instance.setTimedResidualFunction(equation, residual);
    uint64_t unusedSeedSize = 1;
    instance.addTimedJacobianFunction(equation, variable, jacobian, 0,
                                      &unusedSeedSize);
  }

  bool solve(double initial, double root) {
    value = initial;
    target = root;
    KINSOLInstance instance;
    configure(instance);
    return instance.solve(0);
  }
};
} // namespace

TEST_F(KINSOLCommandLineTest, DefaultsAndExplicitNamesTakePrecedence) {
  parse({"model"});
  EXPECT_DOUBLE_EQ(getOptions().fnormtol, 1e-6);
  EXPECT_DOUBLE_EQ(getOptions().scsteptol, 1e-6);
  EXPECT_DOUBLE_EQ(getOptions().maxNewtonStep, 0);
  parse({"model", "--kinsol-absolute-tolerance=1e-8",
         "--kinsol-relative-tolerance=1e-9"});
  EXPECT_DOUBLE_EQ(getOptions().fnormtol, 1e-8);
  EXPECT_DOUBLE_EQ(getOptions().scsteptol, 1e-9);
  parse({"model", "--kinsol-absolute-tolerance=1e-5",
         "--kinsol-relative-tolerance=1e-4",
         "--kinsol-function-norm-tolerance=1e-12",
         "--kinsol-scaled-step-tolerance=1e-11"});
  EXPECT_DOUBLE_EQ(getOptions().fnormtol, 1e-12);
  EXPECT_DOUBLE_EQ(getOptions().scsteptol, 1e-11);
}

TEST_F(KINSOLCommandLineTest, ResidualToleranceChangesActualConvergence) {
  EXPECT_TRUE(solve(1, 1 + 5e-9));
  EXPECT_DOUBLE_EQ(value, 1);
  parse({"model", "--kinsol-function-norm-tolerance=1e-12",
         "--kinsol-scaled-step-tolerance=1e-12"});
  EXPECT_TRUE(solve(1, 1 + 5e-9));
  EXPECT_NEAR(value, 1 + 5e-9, 1e-14);
}

TEST_F(KINSOLCommandLineTest, MaximumStepReachesDistantRootWithoutDefaultChange) {
  EXPECT_FALSE(solve(0, 125000));
  EXPECT_DOUBLE_EQ(value, 0); // Failed trials must not be committed.
  parse({"model", "--kinsol-max-newton-step=1e9"});
  EXPECT_TRUE(solve(0, 125000));
  EXPECT_NEAR(value, 125000, 1e-9);
}

TEST_F(KINSOLCommandLineTest, OwnedFailureRetainsTrialForConditionInspection) {
  value = 0;
  target = 125000;
  KINSOLInstance instance;
  configure(instance);
  ASSERT_EQ(instance.solveWithStatus(0),
            KINSOLInstance::SolveStatus::RecoverableFailure);
  // IDA owns this transaction. It must inspect the actual failed iterate
  // before restoring its saved local state, unlike standalone solve().
  double failedTrial = value;
  ASSERT_GT(failedTrial, 0);
  ASSERT_LT(failedTrial, target);
  ASSERT_TRUE(instance.factorizeCurrentJacobian(0));
  EXPECT_DOUBLE_EQ(value, failedTrial);
  setValue(0, nullptr);
  EXPECT_DOUBLE_EQ(value, 0);
}
