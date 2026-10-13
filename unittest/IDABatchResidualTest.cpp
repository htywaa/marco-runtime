#include "marco/Runtime/Multithreading/Options.h"
#include "marco/Runtime/Solvers/IDA/Instance.h"
#include "marco/Runtime/Solvers/IDA/Options.h"
#include "gtest/gtest.h"
#include <array>
#include <atomic>
#include <cmath>
#include <limits>

using marco::runtime::sundials::ida::IDAInstance;
namespace {
std::array<double, 6> values;
std::atomic<unsigned> scalarCalls{0}, batchCalls{0};
int failure = 0;
thread_local bool caller = false;
bool observedCaller = false;

struct Guard {
  marco::runtime::sundials::ida::Options saved =
      marco::runtime::sundials::ida::getOptions();
#ifdef THREADS_ENABLE
  marco::runtime::multithreading::MultithreadingOptions threads =
      marco::runtime::multithreading::multithreadingOptions();
#endif
  explicit Guard(unsigned workers = 1) {
#ifdef THREADS_ENABLE
    auto &options = marco::runtime::multithreading::multithreadingOptions();
    options.enableMultithreading = true;
    options.numOfThreads = workers;
#endif
    marco::runtime::sundials::ida::getOptions().equidistantTimeGrid = true;
    for (unsigned i = 0; i < values.size(); ++i) values[i] = i;
    scalarCalls = batchCalls = 0;
    failure = 0;
    caller = true;
    observedCaller = false;
  }
  ~Guard() {
    marco::runtime::sundials::ida::getOptions() = saved;
#ifdef THREADS_ENABLE
    marco::runtime::multithreading::multithreadingOptions() = threads;
#endif
    caller = false;
  }
};
double get(const uint64_t *index) { return values[index[0]]; }
void set(double value, const uint64_t *index) { values[index[0]] = value; }
void access(const int64_t *row, uint64_t *column) { column[0] = 3 - row[0]; }
double residual(double time, const int64_t *row) {
  ++scalarCalls;
  auto index = 3 - row[0];
  return values[index] - time - index;
}
int32_t batch(double time, double *out, uint64_t count) {
  ++batchCalls;
  observedCaller = caller;
  if (failure && time > 0) {
    if (failure == 2) {
      out[0] = std::numeric_limits<double>::quiet_NaN();
      return 0;
    }
    return failure;
  }
  if (count != values.size() && count != 3) return -1;
  for (unsigned row = 0; row < count; ++row) {
    auto index = 5 - row;
    out[row] = values[index] - time - index;
  }
  return 0;
}
double jacobian(double, const int64_t *, const uint64_t *, double, uint64_t,
                const uint64_t *) { return 1; }
auto prepare(IDAInstance &instance, bool split = false) {
  uint64_t shape[] = {6};
  int64_t range[] = {-2, split ? 1 : 4};
  auto variable = instance.addAlgebraicVariable(1, shape, get, set, "x");
  auto equation = instance.addEquation(range, 1, "x[3-i] = time+3-i");
  instance.addVariableAccess(equation, variable, access);
  instance.setResidualFunction(equation, residual);
  instance.addJacobianFunction(equation, variable, jacobian, 0, nullptr);
  if (split) {
    int64_t tail[] = {1, 4};
    auto scalarEquation = instance.addEquation(tail, 1, "scalar tail");
    instance.addVariableAccess(scalarEquation, variable, access);
    instance.setResidualFunction(scalarEquation, residual);
    instance.addJacobianFunction(scalarEquation, variable, jacobian, 0, nullptr);
  }
  instance.setEndTime(.4);
  instance.setTimeStep(.1);
  return equation;
}
} // namespace

TEST(IDABatchResidualTest, FreshWholeRangePreservesReverseMappingAndMultiWorker) {
  for (unsigned workers : {1u, 2u}) {
    Guard guard(workers);
    IDAInstance instance;
    auto equation = prepare(instance);
    ASSERT_TRUE(instance.setBatchResidualFunctionV1(equation, 6, batch));
    ASSERT_TRUE(instance.calcIC());
    for (unsigned step = 1; step <= 4; ++step) {
      ASSERT_TRUE(instance.step());
      for (unsigned i = 0; i < values.size(); ++i)
        EXPECT_NEAR(values[i], .1 * step + i, 1e-8);
    }
#ifdef THREADS_ENABLE
    if (workers != 1) {
      EXPECT_EQ(batchCalls, 0u);
      EXPECT_GT(scalarCalls, 0u);
      continue;
    }
#endif
    EXPECT_GT(batchCalls, 4u);
    EXPECT_EQ(scalarCalls, 0u);
    EXPECT_TRUE(observedCaller);
  }
}

TEST(IDABatchResidualTest, InvalidRegistrationFailsBeforeSolverInitialization) {
  for (unsigned variant = 0; variant < 4; ++variant) {
    Guard guard;
    IDAInstance instance;
    auto equation = prepare(instance);
    if (variant == 3) {
      ASSERT_TRUE(instance.setBatchResidualFunctionV1(equation, 6, batch));
    }
    EXPECT_FALSE(instance.setBatchResidualFunctionV1(
        variant == 0 ? equation + 1 : equation, variant == 1 ? 5 : 6,
        variant == 2 ? nullptr : batch));
    EXPECT_FALSE(instance.initialize());
    EXPECT_EQ(batchCalls, 0u);
    EXPECT_EQ(scalarCalls, 0u);
  }
}

TEST(IDABatchResidualTest, ReplacingScalarDefinitionInvalidatesBatch) {
  Guard guard;
  IDAInstance instance;
  auto equation = prepare(instance);
  ASSERT_TRUE(instance.setBatchResidualFunctionV1(equation, 6, batch));
  instance.setResidualFunction(equation, residual);
  ASSERT_TRUE(instance.calcIC());
  ASSERT_TRUE(instance.step());
  EXPECT_EQ(batchCalls, 0u);
  EXPECT_GT(scalarCalls, 0u);
}

TEST(IDABatchResidualTest, FailurePropagatesWithoutScalarFallback) {
  for (int code : {-1, 1, 2}) {
    Guard guard;
    IDAInstance instance;
    auto equation = prepare(instance);
    ASSERT_TRUE(instance.setBatchResidualFunctionV1(equation, 6, batch));
    ASSERT_TRUE(instance.calcIC());
    failure = code;
    EXPECT_FALSE(instance.step());
    EXPECT_EQ(scalarCalls, 0u);
  }
}

TEST(IDABatchResidualTest, DisjointBatchAndScalarFamiliesCoverEveryRow) {
  Guard guard;
  IDAInstance instance;
  auto equation = prepare(instance, true);
  ASSERT_TRUE(instance.setBatchResidualFunctionV1(equation, 3, batch));
  ASSERT_TRUE(instance.calcIC());
  for (unsigned step = 1; step <= 4; ++step) {
    ASSERT_TRUE(instance.step());
    for (unsigned i = 0; i < values.size(); ++i)
      EXPECT_NEAR(values[i], .1 * step + i, 1e-8);
  }
  EXPECT_GT(batchCalls, 4u);
  EXPECT_EQ(scalarCalls, 3u * batchCalls);
}
