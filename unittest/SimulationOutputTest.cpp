#include "marco/Runtime/CLI/CLI.h"
#include "marco/Runtime/Drivers/Driver.h"
#include "marco/Runtime/Printers/CSV/Options.h"
#include "marco/Runtime/Simulation/CLI.h"
#include "marco/Runtime/Simulation/Options.h"
#include "marco/Runtime/Simulation/Runtime.h"
#include "gtest/gtest.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace marco::runtime;

namespace {
double outputTestTime = 0;
double outputTestState = 0;
double outputTestAlgebraic = 0;
unsigned outputTestGetterCalls = 0;
int outputTestDriverStatus = 0;
std::vector<std::string> outputTestEvents;

class OutputContractDriver : public Driver {
public:
  using Driver::Driver;
  std::unique_ptr<cli::Category> getCLIOptions() override {
    return simulation::getCLIOptions();
  }
  int run() override {
    outputTestEvents.emplace_back("driver");
    // Both state advancement and non-solver algebraics precede printing,
    // exactly as in the production drivers.
    for (int i = 1; i <= 2; ++i) {
      setTime(i);
      outputTestState += 3;
      outputTestAlgebraic = outputTestState * outputTestState;
      getSimulation()->getPrinter()->printValues();
    }
    return outputTestDriverStatus;
  }
};

class SimulationOutputTest : public ::testing::Test {
protected:
  printing::PrintOptions savedPrint = printing::printOptions();
  simulation::Options savedSimulation = simulation::getOptions();
  CLI savedCLI = std::move(getCLI());
  std::filesystem::path previousDirectory = std::filesystem::current_path();
  std::filesystem::path directory;

  void SetUp() override {
    directory = std::filesystem::temp_directory_path() /
                ("marco-output-" + std::to_string(
                    std::chrono::steady_clock::now().time_since_epoch().count()));
    ASSERT_TRUE(std::filesystem::create_directory(directory));
    std::filesystem::current_path(directory);
    printing::printOptions() = {};
    simulation::getOptions() = {};
    getCLI() = CLI{};
    outputTestTime = outputTestState = outputTestAlgebraic = 0;
    outputTestGetterCalls = 0;
    outputTestDriverStatus = 0;
    outputTestEvents.clear();
  }
  void TearDown() override {
    std::filesystem::current_path(previousDirectory);
    std::filesystem::remove_all(directory);
    printing::printOptions() = savedPrint;
    simulation::getOptions() = savedSimulation;
    getCLI() = std::move(savedCLI);
  }
  int run(std::initializer_list<std::string> extra, std::string &output) {
    std::vector<std::string> arguments{"output-contract", "--end-time=2"};
    arguments.insert(arguments.end(), extra);
    std::vector<char *> argv;
    for (std::string &argument : arguments) {
      argv.push_back(argument.data());
    }
    testing::internal::CaptureStdout();
    int status = runSimulation(static_cast<int>(argv.size()), argv.data());
    output = testing::internal::GetCapturedStdout();
    return status;
  }
  void expectCompletedComputation() {
    EXPECT_DOUBLE_EQ(outputTestTime, 2);
    EXPECT_DOUBLE_EQ(outputTestState, 7);
    EXPECT_DOUBLE_EQ(outputTestAlgebraic, 49);
    EXPECT_EQ(outputTestEvents,
              (std::vector<std::string>{"init", "ic-begin", "ic-solve",
                                        "ic-end", "dynamic-begin", "driver",
                                        "dynamic-end", "deinit"}));
  }
};
} // namespace

// A model ABI fixture, not a replacement solver. It exercises the real
// runSimulation lifecycle, CLI and CSV printer together.
namespace marco::runtime {
std::unique_ptr<Driver> getDriver(Simulation *simulation) {
  return std::make_unique<OutputContractDriver>(simulation);
}
} // namespace marco::runtime

extern "C" {
bool hasExperimentStartTime() { return false; }
bool hasExperimentEndTime() { return false; }
double getExperimentStartTime() { return 0; }
double getExperimentEndTime() { return 2; }
double getTime() { return outputTestTime; }
void setTime(double time) { outputTestTime = time; }
char *getModelName() { static char name[] = "OutputContract"; return name; }
int64_t getNumOfVariables() { return 2; }
char *getVariableName(int64_t var) {
  static char state[] = "x";
  static char algebraic[] = "y";
  return var == 0 ? state : algebraic;
}
int64_t getVariableRank(int64_t) { return 0; }
bool isPrintable(int64_t) { return true; }
int64_t getVariableNumOfPrintableRanges(int64_t) { return 0; }
int64_t getVariablePrintableRangeBegin(int64_t, int64_t, int64_t) { return 0; }
int64_t getVariablePrintableRangeEnd(int64_t, int64_t, int64_t) { return 0; }
double getVariableValue(int64_t var, const int64_t *) {
  ++outputTestGetterCalls;
  return var == 0 ? outputTestState : outputTestAlgebraic;
}
int64_t getDerivative(int64_t) { return -1; }
void init() { outputTestEvents.emplace_back("init"); }
void deinit() { outputTestEvents.emplace_back("deinit"); }
void icModelBegin() { outputTestEvents.emplace_back("ic-begin"); }
void icModelEnd() { outputTestEvents.emplace_back("ic-end"); }
void solveICModel() {
  outputTestEvents.emplace_back("ic-solve");
  outputTestState = outputTestAlgebraic = 1;
}
void dynamicModelBegin() { outputTestEvents.emplace_back("dynamic-begin"); }
void dynamicModelEnd() { outputTestEvents.emplace_back("dynamic-end"); }
}

TEST_F(SimulationOutputTest, NoOutputPreservesWholeLifecycleAndValues) {
  std::string output;
  EXPECT_EQ(run({"--no-output", "--csv-output=missing/path/result.csv"}, output), 0);
  expectCompletedComputation();
  EXPECT_TRUE(output.empty());
  EXPECT_EQ(outputTestGetterCalls, 0u);
  EXPECT_TRUE(std::filesystem::is_empty(directory));
}

TEST_F(SimulationOutputTest, DefaultOutputPreservesFileAndStdout) {
  std::string output;
  EXPECT_EQ(run({}, output), 0);
  expectCompletedComputation();
  EXPECT_EQ(outputTestGetterCalls, 6u);
  std::ifstream file(directory / "OutputContract_res.csv");
  ASSERT_TRUE(file);
  std::ostringstream contents;
  contents << file.rdbuf();
  EXPECT_EQ(contents.str(), output);
  EXPECT_NE(output.find("2.000000000,7.000000000,49.000000000"), std::string::npos);
}

TEST_F(SimulationOutputTest, NoOutputPreservesDriverFailureStatus) {
  outputTestDriverStatus = 42;
  std::string output;
  EXPECT_EQ(run({"--no-output"}, output), 42);
  expectCompletedComputation();
  EXPECT_TRUE(output.empty());
  EXPECT_TRUE(std::filesystem::is_empty(directory));
}
