#include "marco/Runtime/CLI/CLI.h"
#include "marco/Runtime/Printers/CSV/Options.h"
#include "marco/Runtime/Simulation/Options.h"
#include "marco/Runtime/Simulation/Runtime.h"
#ifdef TEST_RUNGE_KUTTA
#include "marco/Runtime/Solvers/RungeKutta/Options.h"
#else
#include "marco/Runtime/Solvers/EulerForward/Options.h"
#endif
#include "gtest/gtest.h"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace marco::runtime;

namespace {
double modelTime, state, algebraic, trial;
double largestAcceptedStep = std::numeric_limits<double>::infinity();
double trialStep;
bool invalidError = false;
std::vector<double> triedSteps, acceptedTimes;

class DriverEndTimeTest : public ::testing::Test {
protected:
  printing::PrintOptions savedPrint = printing::printOptions();
  simulation::Options savedSimulation = simulation::getOptions();
#ifdef TEST_RUNGE_KUTTA
  rungekutta::Options savedSolver = rungekutta::getOptions();
#else
  eulerforward::Options savedSolver = eulerforward::getOptions();
#endif
  CLI savedCLI = std::move(getCLI());
  std::filesystem::path previousDirectory = std::filesystem::current_path();
  std::filesystem::path directory;

  void SetUp() override {
    directory = std::filesystem::temp_directory_path() /
        ("marco-end-time-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    ASSERT_TRUE(std::filesystem::create_directory(directory));
    std::filesystem::current_path(directory);
    printing::printOptions() = {};
    simulation::getOptions() = {};
#ifdef TEST_RUNGE_KUTTA
    rungekutta::getOptions() = {};
#else
    eulerforward::getOptions() = {};
#endif
    getCLI() = CLI{};
    modelTime = state = algebraic = trial = trialStep = 0;
    largestAcceptedStep = std::numeric_limits<double>::infinity();
    invalidError = false;
    triedSteps.clear();
    acceptedTimes.clear();
  }
  void TearDown() override {
    std::filesystem::current_path(previousDirectory);
    std::filesystem::remove_all(directory);
    printing::printOptions() = savedPrint;
    simulation::getOptions() = savedSimulation;
#ifdef TEST_RUNGE_KUTTA
    rungekutta::getOptions() = savedSolver;
#else
    eulerforward::getOptions() = savedSolver;
#endif
    getCLI() = std::move(savedCLI);
  }
  int run(std::initializer_list<std::string> arguments, std::string &output) {
    std::vector<std::string> strings{"end-time-contract", "--precision=17"};
    strings.insert(strings.end(), arguments);
    std::vector<char *> argv;
    for (auto &s : strings) argv.push_back(s.data());
    testing::internal::CaptureStdout();
    int result = runSimulation(static_cast<int>(argv.size()), argv.data());
    output = testing::internal::GetCapturedStdout();
    return result;
  }
  void expectTrajectory(const std::string &output,
                        const std::vector<double> &times) {
    std::istringstream stream(output);
    std::string line;
    ASSERT_TRUE(static_cast<bool>(std::getline(stream, line)));
    ASSERT_EQ(line, "\"time\",\"x\",\"z\"");
    for (double time : times) {
      ASSERT_TRUE(static_cast<bool>(std::getline(stream, line)));
      std::istringstream row(line);
      std::string column;
      ASSERT_TRUE(static_cast<bool>(std::getline(row, column, ',')));
      EXPECT_NEAR(std::stod(column), time, 1e-14);
      ASSERT_TRUE(static_cast<bool>(std::getline(row, column, ',')));
      EXPECT_NEAR(std::stod(column), 1 + 2 * time, 1e-13);
      ASSERT_TRUE(static_cast<bool>(std::getline(row, column, ',')));
      EXPECT_NEAR(std::stod(column), 3 + 7 * time, 1e-13);
    }
    EXPECT_FALSE(static_cast<bool>(std::getline(stream, line)));
    EXPECT_DOUBLE_EQ(modelTime, times.back());
    EXPECT_NEAR(state, 1 + 2 * modelTime, 1e-13);
    EXPECT_NEAR(algebraic, 3 + 7 * modelTime, 1e-13);
  }
};
} // namespace

// 中文：链接真实驱动、CLI、CSV和runSimulation；仅模型ABI使用解析ODE x'=2。
// English: The real driver, CLI, CSV and runSimulation are linked. Only the
// generated model ABI is a fixture for the independently known ODE x'=2.
extern "C" {
bool hasExperimentStartTime() { return false; }
bool hasExperimentEndTime() { return false; }
double getExperimentStartTime() { return 0; }
double getExperimentEndTime() { return 1; }
double getTime() { return modelTime; }
void setTime(double time) { modelTime = time; }
char *getModelName() { static char name[] = "EndTimeContract"; return name; }
int64_t getNumOfVariables() { return 2; }
char *getVariableName(int64_t variable) {
  static char x[] = "x", z[] = "z";
  return variable == 0 ? x : z;
}
int64_t getVariableRank(int64_t) { return 0; }
bool isPrintable(int64_t) { return true; }
int64_t getVariableNumOfPrintableRanges(int64_t) { return 0; }
int64_t getVariablePrintableRangeBegin(int64_t, int64_t, int64_t) { return 0; }
int64_t getVariablePrintableRangeEnd(int64_t, int64_t, int64_t) { return 0; }
double getVariableValue(int64_t variable, const int64_t *) {
  return variable == 0 ? state : algebraic;
}
int64_t getDerivative(int64_t) { return -1; }
void init() {}
void deinit() {}
void icModelBegin() {}
void icModelEnd() {}
void solveICModel() { state = 1 + 2 * modelTime; algebraic = 3 + 7 * modelTime; }
void dynamicModelBegin() {}
void dynamicModelEnd() {}
void updateStateVariables(double step) {
  triedSteps.push_back(step);
  state += 2 * step;
}
void tryStep(double step) {
  triedSteps.push_back(step);
  trialStep = step;
  trial = state + 2 * step;
}
double estimateError() {
  return invalidError ? std::numeric_limits<double>::quiet_NaN() :
      trialStep > largestAcceptedStep ? 1 : 0;
}
void acceptStep() { state = trial; }
void updateNonStateVariables() {
  algebraic = 3 * state + modelTime;
  acceptedTimes.push_back(modelTime);
}
}

TEST_F(DriverEndTimeTest, StopsAtExactEndpoint) {
  std::string output;
  ASSERT_EQ(run({"--end-time=0.2", "--time-step=0.1"}, output), 0);
  expectTrajectory(output, {0, .1, .2});
}
TEST_F(DriverEndTimeTest, ShortensLastStepToEndpoint) {
  std::string output;
  ASSERT_EQ(run({"--end-time=0.25", "--time-step=0.1"}, output), 0);
  expectTrajectory(output, {0, .1, .2, .25});
}
TEST_F(DriverEndTimeTest, RepeatedStepsDoNotAddDuplicateEndpoint) {
  std::string output;
  ASSERT_EQ(run({"--end-time=1", "--time-step=0.1"}, output), 0);
  std::vector<double> times;
  for (int i = 0; i <= 10; ++i) times.push_back(i / 10.0);
  expectTrajectory(output, times);
  EXPECT_EQ(acceptedTimes.size(), 10u);
}
TEST_F(DriverEndTimeTest, IntervalShorterThanStep) {
  std::string output;
  ASSERT_EQ(run({"--end-time=0.03", "--time-step=0.1"}, output), 0);
  expectTrajectory(output, {0, .03});
}
TEST_F(DriverEndTimeTest, StartsFromRequestedNonzeroTime) {
  std::string output;
  ASSERT_EQ(run({"--start-time=2", "--end-time=2.25", "--time-step=0.1"}, output), 0);
  expectTrajectory(output, {2, 2.1, 2.2, 2.25});
}
TEST_F(DriverEndTimeTest, EmptyIntervalDoesNotAdvanceState) {
  std::string output;
  ASSERT_EQ(run({"--end-time=0", "--time-step=0.1"}, output), 0);
  expectTrajectory(output, {0});
  EXPECT_TRUE(triedSteps.empty());
}
TEST_F(DriverEndTimeTest, NoOutputPreservesEndpointAndAlgebraicUpdate) {
  std::string output;
  ASSERT_EQ(run({"--end-time=0.25", "--time-step=0.1", "--no-output"}, output), 0);
  EXPECT_TRUE(output.empty());
  EXPECT_TRUE(std::filesystem::is_empty(directory));
  EXPECT_DOUBLE_EQ(modelTime, .25);
  EXPECT_NEAR(state, 1.5, 1e-13);
  EXPECT_NEAR(algebraic, 4.75, 1e-13);
}
TEST_F(DriverEndTimeTest, RejectsZeroTimeStep) {
  std::string output;
  EXPECT_NE(run({"--end-time=0.25", "--time-step=0", "--no-output"}, output), 0);
  EXPECT_TRUE(triedSteps.empty());
}
TEST_F(DriverEndTimeTest, RejectsNegativeTimeStep) {
  std::string output;
  EXPECT_NE(run({"--end-time=0.25", "--time-step=-0.1", "--no-output"}, output), 0);
  EXPECT_TRUE(triedSteps.empty());
}
TEST_F(DriverEndTimeTest, RejectsStepThatCannotAdvanceFloatingPointTime) {
  std::string output;
  EXPECT_NE(run({"--start-time=1e20", "--end-time=100000000000000016384",
                 "--time-step=1", "--no-output"}, output), 0);
  EXPECT_TRUE(triedSteps.empty());
}
#ifdef TEST_RUNGE_KUTTA
TEST_F(DriverEndTimeTest, RejectedTrialsNearEndpointKeepAcceptedStateConsistent) {
  largestAcceptedStep = .04;
  std::string output;
  ASSERT_EQ(run({"--end-time=0.12", "--time-step=0.1", "--no-output"}, output), 0);
  EXPECT_GT(triedSteps.size(), acceptedTimes.size());
  EXPECT_DOUBLE_EQ(modelTime, .12);
  EXPECT_NEAR(state, 1.24, 1e-13);
  EXPECT_NEAR(algebraic, 3.84, 1e-13);
  for (double time : acceptedTimes) EXPECT_LE(time, .12);
}
TEST_F(DriverEndTimeTest, RejectsNonfiniteErrorBeforeAcceptingTrial) {
  invalidError = true;
  std::string output;
  EXPECT_NE(run({"--end-time=0.1", "--time-step=0.1", "--no-output"}, output), 0);
  EXPECT_TRUE(acceptedTimes.empty());
  EXPECT_DOUBLE_EQ(modelTime, 0);
  EXPECT_DOUBLE_EQ(state, 1);
}
#endif
