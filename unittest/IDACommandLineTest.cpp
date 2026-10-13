#include "marco/Runtime/Drivers/IDA/CLI.h"
#include "marco/Runtime/Solvers/IDA/Options.h"
#include "marco/Runtime/Solvers/KINSOL/Options.h"
#include "gtest/gtest.h"

namespace {
namespace ida = marco::runtime::sundials::ida;
namespace kinsol = marco::runtime::sundials::kinsol;

class IDACommandLineTest : public ::testing::Test {
protected:
  ida::Options saved = ida::getOptions();
  void SetUp() override { ida::getOptions() = ida::Options{}; }
  void TearDown() override { ida::getOptions() = saved; }
  void parse(std::initializer_list<const char *> arguments) {
    argh::parser parser(static_cast<int>(arguments.size()), arguments.begin());
    ida::CommandLineOptions{}.parseCommandLineOptions(parser);
  }
};

TEST_F(IDACommandLineTest, C2DefaultsAndOverridesAreIndependent) {
  auto ordinary = kinsol::getOptions();
  parse({"model"});
  EXPECT_DOUBLE_EQ(ida::getOptions().c2FunctionNormTolerance, 1e-12);
  EXPECT_DOUBLE_EQ(ida::getOptions().c2ScaledStepTolerance, 1e-14);
  parse({"model", "--ida-relative-tolerance=1e-9",
         "--ida-c2-function-norm-tolerance=1e-8",
         "--ida-c2-scaled-step-tolerance=1e-10"});
  EXPECT_DOUBLE_EQ(ida::getOptions().relativeTolerance, 1e-9);
  EXPECT_DOUBLE_EQ(ida::getOptions().c2FunctionNormTolerance, 1e-8);
  EXPECT_DOUBLE_EQ(ida::getOptions().c2ScaledStepTolerance, 1e-10);
  EXPECT_DOUBLE_EQ(kinsol::getOptions().fnormtol, ordinary.fnormtol);
  EXPECT_DOUBLE_EQ(kinsol::getOptions().scsteptol, ordinary.scsteptol);
}

TEST_F(IDACommandLineTest, InvalidC2TolerancesAreRejected) {
  for (const char *argument : {
           "--ida-c2-function-norm-tolerance=0",
           "--ida-c2-function-norm-tolerance=-1",
           "--ida-c2-function-norm-tolerance=nan",
           "--ida-c2-function-norm-tolerance=inf",
           "--ida-c2-function-norm-tolerance=1e999",
           "--ida-c2-function-norm-tolerance=1e-999",
           "--ida-c2-function-norm-tolerance=1e-8junk",
           "--ida-c2-function-norm-tolerance=",
           "--ida-c2-function-norm-tolerance",
           "--ida-c2-scaled-step-tolerance=0",
           "--ida-c2-scaled-step-tolerance=nan",
           "--ida-c2-scaled-step-tolerance="}) {
    EXPECT_EXIT(parse({"model", argument}), ::testing::ExitedWithCode(1),
                "expected one finite positive value");
  }
  EXPECT_EXIT(parse({"model", "--ida-c2-scaled-step-tolerance=1e-14",
                     "--ida-c2-scaled-step-tolerance=1e-8"}),
              ::testing::ExitedWithCode(1), "expected one finite positive value");
}

TEST_F(IDACommandLineTest, PositiveFiniteSubnormalIsAccepted) {
  parse({"model", "--ida-c2-function-norm-tolerance=1e-310"});
  EXPECT_GT(ida::getOptions().c2FunctionNormTolerance, 0);
  EXPECT_LT(ida::getOptions().c2FunctionNormTolerance, 1e-308);
}
} // namespace
