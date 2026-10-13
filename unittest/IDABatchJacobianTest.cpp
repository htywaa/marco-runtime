#include "marco/Runtime/Multithreading/Options.h"
#include "marco/Runtime/Solvers/IDA/Instance.h"
#include "marco/Runtime/Solvers/IDA/Options.h"
#include "gtest/gtest.h"
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <utility>

using namespace marco::runtime::sundials::ida;
namespace {
std::array<double, 6> x{}, xp{};
double a = 0;
std::atomic<unsigned> scalarCalls{0};
unsigned batchCalls = 0, baselineCalls = 0;
int failBatch = 0;

struct Guard {
  Options options = getOptions();
#ifdef THREADS_ENABLE
  marco::runtime::multithreading::MultithreadingOptions threads =
      marco::runtime::multithreading::multithreadingOptions();
#endif
  std::optional<std::string> oldLimit;
  explicit Guard(unsigned workers = 1, bool noBudget = false) {
    if (auto *value = std::getenv("MARCO_IDA_BATCH_WORK_LIMIT")) oldLimit = value;
    if (noBudget) setenv("MARCO_IDA_BATCH_WORK_LIMIT", "0", 1);
    else unsetenv("MARCO_IDA_BATCH_WORK_LIMIT");
#ifdef THREADS_ENABLE
    auto &t = marco::runtime::multithreading::multithreadingOptions();
    t.enableMultithreading = true;
    t.numOfThreads = workers;
#endif
    getOptions().equidistantTimeGrid = true;
    scalarCalls = batchCalls = baselineCalls = 0;
    failBatch = 0;
  }
  ~Guard() {
    getOptions() = options;
#ifdef THREADS_ENABLE
    marco::runtime::multithreading::multithreadingOptions() = threads;
#endif
    if (oldLimit) setenv("MARCO_IDA_BATCH_WORK_LIMIT", oldLimit->c_str(), 1);
    else unsetenv("MARCO_IDA_BATCH_WORK_LIMIT");
  }
};
double getA(const uint64_t *) { return a; }
void setA(double v, const uint64_t *) { a = v; }
double getX(const uint64_t *i) { return x[i[0]]; }
void setX(double v, const uint64_t *i) { x[i[0]] = v; }
double getXP(const uint64_t *i) { return xp[i[0]]; }
void setXP(double v, const uint64_t *i) { xp[i[0]] = v; }
unsigned index(const int64_t *row) { return 3 - row[0]; }
void own(const int64_t *row, uint64_t *column) { column[0] = index(row); }
void previous(const int64_t *row, uint64_t *column) {
  column[0] = (index(row) + 5) % 6;
}
void scalarAccess(const int64_t *, uint64_t *column) { column[0] = 0; }
double f(double time, const int64_t *row) {
  auto i = index(row);
  return x[i] * x[i] + xp[i] + 2 * x[(i + 5) % 6] + 3 * a - time;
}
double scalarF(double time, const int64_t *) { return a - time; }
double j(double, const int64_t *row, const uint64_t *column, double alpha,
         uint64_t, const uint64_t *) {
  ++scalarCalls;
  return column[0] == index(row) ? 2 * x[index(row)] + alpha : 2;
}
double ja(double, const int64_t *, const uint64_t *, double, uint64_t,
          const uint64_t *) { return 3; }
double unit(double, const int64_t *, const uint64_t *, double, uint64_t,
            const uint64_t *) { return 1; }
int32_t batch(double time, double alpha, const double *direction,
              const uint64_t *offsets, double *out, uint64_t count) {
  ++batchCalls;
  // Registration order is deliberately different from IDA's scalar order.
  if (count != 6 || offsets[0] != 1 || offsets[1] != 0) return -1;
  if (failBatch == 2) {
    out[0] = std::numeric_limits<double>::quiet_NaN();
    return 0;
  }
  if (failBatch) return failBatch;
  bool zero = true;
  for (unsigned i = 0; i != 7; ++i) zero &= direction[i] == 0;
  baselineCalls += zero;
  for (unsigned row = 0; row != 6; ++row) {
    unsigned i = 5 - row;
    // A nonzero, state-dependent primal term makes dropping or retaining a
    // stale zero-direction baseline detectable independently of convergence.
    out[row] = 7 + time + x[i] +
               (2 * x[i] + alpha) * direction[offsets[0] + i] +
               2 * direction[offsets[0] + (i + 5) % 6] +
               3 * direction[offsets[1]];
  }
  return 0;
}
auto prepare(IDAInstance &instance) {
  uint64_t one[] = {1}, six[] = {6};
  int64_t scalarRange[] = {0, 1}, range[] = {-2, 4};
  auto av = instance.addAlgebraicVariable(1, one, getA, setA, "a");
  auto xv = instance.addStateVariable(1, six, getX, setX, getXP, setXP, "x");
  auto eq = instance.addEquation(range, 1, "nonlinear reverse array");
  instance.addVariableAccess(eq, xv, own);
  instance.addVariableAccess(eq, xv, previous);
  instance.addVariableAccess(eq, av, scalarAccess);
  instance.setResidualFunction(eq, f);
  instance.addJacobianFunction(eq, xv, j, 0, nullptr);
  instance.addJacobianFunction(eq, av, ja, 0, nullptr);
  auto scalarEq = instance.addEquation(scalarRange, 1, "a=time");
  instance.addVariableAccess(scalarEq, av, scalarAccess);
  instance.setResidualFunction(scalarEq, scalarF);
  instance.addJacobianFunction(scalarEq, av, unit, 0, nullptr);
  instance.setEndTime(.2);
  instance.setTimeStep(.1);
  return eq;
}

struct MatrixInput {
#if SUNDIALS_VERSION_MAJOR >= 6
  SUNContext context = nullptr;
#endif
  N_Vector y, yp, r;
  SUNMatrix matrix;
  explicit MatrixInput(long size = 7, long capacity = 19) {
#if SUNDIALS_VERSION_MAJOR >= 7
    EXPECT_EQ(SUNContext_Create(SUN_COMM_NULL, &context), 0);
#elif SUNDIALS_VERSION_MAJOR >= 6
    EXPECT_EQ(SUNContext_Create(nullptr, &context), 0);
#endif
#if SUNDIALS_VERSION_MAJOR >= 6
    y = N_VNew_Serial(size, context);
    matrix = SUNSparseMatrix(size, size, capacity, CSR_MAT, context);
#else
    y = N_VNew_Serial(size);
    matrix = SUNSparseMatrix(size, size, capacity, CSR_MAT);
#endif
    yp = N_VClone(y);
    r = N_VClone(y);
  }
  ~MatrixInput() {
    SUNMatDestroy(matrix);
    N_VDestroy(y); N_VDestroy(yp); N_VDestroy(r);
#if SUNDIALS_VERSION_MAJOR >= 6
    SUNContext_Free(&context);
#endif
  }
  int evaluate(IDAInstance &instance, double alpha, double shift = 0) {
    for (unsigned i = 0; i != 7; ++i) {
      N_VGetArrayPointer(y)[i] = .25 * i + shift;
      N_VGetArrayPointer(yp)[i] = .125 * i - shift;
    }
    return IDAInstance::jacobianMatrix(.125, alpha, y, yp, r, matrix,
                                       &instance, nullptr, nullptr, nullptr);
  }
  void check(double alpha) {
    auto *rows = SUNSparseMatrix_IndexPointers(matrix);
    auto *cols = SUNSparseMatrix_IndexValues(matrix);
    auto *data = SUNSparseMatrix_Data(matrix);
    for (unsigned row = 0; row != 7; ++row) {
      std::array<double, 7> dense{};
      for (auto p = rows[row]; p < rows[row + 1]; ++p) dense[cols[p]] = data[p];
      for (unsigned col = 0; col != 7; ++col) {
        double expected = row == 6 ? (col == 0 ? 1 : 0) :
            col == 0 ? 3 : col == 6 - row ? 2 * x[5 - row] + alpha :
            col == 1 + (10 - row) % 6 ? 2 : 0;
        EXPECT_NEAR(dense[col], expected, 2e-14) << row << "," << col;
      }
    }
  }
};
} // namespace

namespace {
int32_t tangent(double, double alpha, const double *direction,
                const uint64_t *offsets, double *out, uint64_t count) {
  ++batchCalls;
  if (count != 6 || offsets[0] != 1 || offsets[1] != 0) return -1;
  if (failBatch) return failBatch;
  bool zero = true;
  for (unsigned i = 0; i != 7; ++i) zero &= direction[i] == 0;
  baselineCalls += zero;
  for (unsigned row = 0; row != 6; ++row) {
    unsigned i = 5 - row;
    out[row] = (2 * x[i] + alpha) * direction[offsets[0] + i] +
               2 * direction[offsets[0] + (i + 5) % 6] +
               3 * direction[offsets[1]];
  }
  return 0;
}

const int preparationLayout = 0;
unsigned primalComputations = 0, preparedCalls = 0;
IDAInstance *reentrantInstance = nullptr;
MatrixInput *reentrantInput = nullptr;
int32_t preparedTangent(double time, double alpha, const double *direction,
                       const uint64_t *offsets, double *out, uint64_t count,
                       double *values, uint8_t *ready) {
  ++preparedCalls;
  if (count != 6) return -1;
  if (reentrantInstance) {
    auto *nested = reentrantInstance;
    reentrantInstance = nullptr;
    auto before = x;
    EXPECT_EQ(reentrantInput->evaluate(*nested, 99, 99), -1);
    EXPECT_EQ(before, x); // Refuse same-owner reentry before writing state.
  }
  for (unsigned i = 0; i < 6; ++i)
    if (!ready[i]) {
      values[i] = 2 * x[i];
      ready[i] = 1;
      ++primalComputations;
    }
  if (failBatch) return failBatch; // Includes a failure after preparation.
  for (unsigned row = 0; row < 6; ++row) {
    unsigned i = 5 - row;
    out[row] = (values[i] + alpha) * direction[offsets[0] + i] +
               2 * direction[offsets[0] + (i + 5) % 6] +
               3 * direction[offsets[1]];
  }
  return 0;
}

void firstAccess(const int64_t *, uint64_t *column) { column[0] = 0; }
double coupledScalarF(double time, const int64_t *) { return a * x[0] - time; }
double coupledScalarJA(double, const int64_t *, const uint64_t *, double,
                       uint64_t, const uint64_t *) { return x[0]; }
double coupledScalarJX(double, const int64_t *, const uint64_t *, double,
                       uint64_t, const uint64_t *) { return a; }
int32_t coupledScalarTangent(double, double, const double *direction,
                            const uint64_t *offsets, double *out, uint64_t n) {
  if (n != 1) return -1;
  out[0] = x[0] * direction[offsets[0]] + a * direction[offsets[1]];
  return 0;
}
unsigned sharedPrimalUses = 0;
int32_t coupledScalarPrepared(double, double, const double *direction,
                             const uint64_t *offsets, double *out, uint64_t n,
                             double *values, uint8_t *ready) {
  if (n != 1 || !ready[0]) return -1;
  ++sharedPrimalUses;
  out[0] = (values[0] / 2) * direction[offsets[0]] + a * direction[offsets[1]];
  return 0;
}
}

TEST(IDABatchJacobianTest, PreparedV3FreshScopeFailureReentryAndMultipleOwners) {
  Guard guard;
  IDAInstance first, second;
  uint64_t variables[] = {1, 0};
  for (auto *owner : {&first, &second}) {
    auto eq = prepare(*owner);
    ASSERT_TRUE(owner->setBatchTangentFunctionV2(eq, 6, tangent, 2, variables));
    ASSERT_TRUE(owner->setBatchPreparedTangentV3(
        eq, 6, preparedTangent, &preparationLayout, 6));
    ASSERT_TRUE(owner->initialize());
  }
  primalComputations = preparedCalls = 0;
  MatrixInput input;
  unsigned scopes = 0;
  for (auto *owner : {&first, &second, &first}) {
    for (double alpha : {0., 1., -3., 17.}) {
      ASSERT_EQ(input.evaluate(*owner, alpha, scopes * .25), 0);
      ++scopes;
      input.check(alpha);
      EXPECT_EQ(primalComputations, 6 * scopes);
    }
  }
  EXPECT_GT(preparedCalls, scopes);
  failBatch = 1;
  EXPECT_EQ(input.evaluate(first, 5, -10), 1);
  ++scopes;
  failBatch = 0;
  reentrantInstance = &first;
  reentrantInput = &input;
  ASSERT_EQ(input.evaluate(first, 2, 2), 0);
  ++scopes;
  input.check(2);
  EXPECT_EQ(primalComputations, 6 * scopes);
  EXPECT_EQ(reentrantInstance, nullptr);
  reentrantInput = nullptr;
}

TEST(IDABatchJacobianTest, PreparedV3BudgetFallbackAndRegistrationFailure) {
  for (bool exhausted : {false, true}) {
    Guard guard(1, exhausted);
    IDAInstance instance;
    auto eq = prepare(instance);
    uint64_t variables[] = {1, 0};
    ASSERT_TRUE(instance.setBatchTangentFunctionV2(eq, 6, tangent, 2, variables));
    ASSERT_TRUE(instance.setBatchPreparedTangentV3(
        eq, 6, preparedTangent, &preparationLayout, 6));
    ASSERT_TRUE(instance.initialize());
    MatrixInput input;
    preparedCalls = 0;
    ASSERT_EQ(input.evaluate(instance, 2), 0);
    input.check(2);
    if (exhausted) EXPECT_EQ(preparedCalls, 0u);
    // Replacing a binding invalidates the whole owner before state writes.
    ASSERT_FALSE(instance.setBatchPreparedTangentV3(
        eq, 6, preparedTangent, &preparationLayout, 6));
    auto before = x;
    EXPECT_EQ(input.evaluate(instance, 2, 80), -1);
    EXPECT_EQ(before, x);
  }
  Guard guard;
  IDAInstance instance;
  auto eq = prepare(instance);
  uint64_t variables[] = {1, 0};
  ASSERT_TRUE(instance.setBatchJacobianFunctionV1(eq, 6, batch, 2, variables));
  EXPECT_FALSE(instance.setBatchPreparedTangentV3(
      eq, 6, preparedTangent, &preparationLayout, 6));
  EXPECT_FALSE(instance.initialize()); // V1 cannot masquerade as V2.
}

TEST(IDABatchJacobianTest, PreparedV3SharesOnlyMatchingLayoutWithinOneCallback) {
  Guard guard;
  IDAInstance instance;
  auto eq = prepare(instance);
  auto scalarEquation = eq + 1;
  instance.setResidualFunction(scalarEquation, coupledScalarF);
  instance.addVariableAccess(scalarEquation, 1, firstAccess);
  instance.addJacobianFunction(scalarEquation, 0, coupledScalarJA, 0, nullptr);
  instance.addJacobianFunction(scalarEquation, 1, coupledScalarJX, 0, nullptr);
  uint64_t arrayVariables[] = {1, 0}, scalarVariables[] = {0, 1};
  ASSERT_TRUE(instance.setBatchTangentFunctionV2(eq, 6, tangent, 2, arrayVariables));
  ASSERT_TRUE(instance.setBatchTangentFunctionV2(
      scalarEquation, 1, coupledScalarTangent, 2, scalarVariables));
  ASSERT_TRUE(instance.setBatchPreparedTangentV3(
      eq, 6, preparedTangent, &preparationLayout, 6));
  ASSERT_TRUE(instance.setBatchPreparedTangentV3(
      scalarEquation, 1, coupledScalarPrepared, &preparationLayout, 6));
  ASSERT_TRUE(instance.initialize());
  MatrixInput input(7, 20);
  primalComputations = sharedPrimalUses = 0;
  unsigned scopes = 0;
  for (double shift : {0., -4., 10.}) {
    ASSERT_EQ(input.evaluate(instance, 3, shift), 0);
    ++scopes;
    EXPECT_EQ(primalComputations, scopes * 6);
    auto *rows = SUNSparseMatrix_IndexPointers(input.matrix);
    auto *cols = SUNSparseMatrix_IndexValues(input.matrix);
    auto *data = SUNSparseMatrix_Data(input.matrix);
    std::array<double, 7> dense{};
    for (auto p = rows[6]; p < rows[7]; ++p) dense[cols[p]] = data[p];
    for (unsigned col = 0; col < 7; ++col)
      EXPECT_EQ(dense[col], col == 0 ? x[0] : col == 1 ? a : 0);
  }
  EXPECT_GT(sharedPrimalUses, 3u);

  IDAInstance invalid;
  auto invalidEq = prepare(invalid);
  ASSERT_TRUE(invalid.setBatchTangentFunctionV2(invalidEq, 6, tangent, 2, arrayVariables));
  ASSERT_TRUE(invalid.setBatchTangentFunctionV2(
      invalidEq + 1, 1, coupledScalarTangent, 2, scalarVariables));
  ASSERT_TRUE(invalid.setBatchPreparedTangentV3(
      invalidEq, 6, preparedTangent, &preparationLayout, 6));
  EXPECT_FALSE(invalid.setBatchPreparedTangentV3(
      invalidEq + 1, 1, coupledScalarPrepared, &preparationLayout, 7));
  EXPECT_FALSE(invalid.initialize());
}

TEST(IDABatchJacobianTest, PureTangentV2HasNoBaselineAndFreshCompleteMatrix) {
  for (unsigned workers : {1u, 2u}) {
    Guard guard(workers);
    IDAInstance instance;
    auto eq = prepare(instance);
    uint64_t variables[] = {1, 0};
    ASSERT_TRUE(instance.setBatchTangentFunctionV2(eq, 6, tangent, 2, variables));
    ASSERT_TRUE(instance.initialize());
    MatrixInput input;
    for (double alpha : {0., -3., 1., 17.}) {
      ASSERT_EQ(input.evaluate(instance, alpha, alpha * .125), 0);
      input.check(alpha);
    }
    EXPECT_EQ(baselineCalls, 0u);
#ifdef THREADS_ENABLE
    if (workers != 1) {
      EXPECT_EQ(batchCalls, 0u);
      EXPECT_GT(scalarCalls, 0u);
      continue;
    }
#endif
    EXPECT_GT(batchCalls, 0u);
    EXPECT_EQ(scalarCalls, 0u);
    failBatch = -1;
    EXPECT_EQ(input.evaluate(instance, 3), -1);
    failBatch = 0;
    ASSERT_EQ(input.evaluate(instance, 2, -2), 0);
    input.check(2);
  }
}

TEST(IDABatchJacobianTest, PureTangentBudgetFallbackAndStaleBinding) {
  for (bool exhausted : {false, true}) {
    Guard guard(1, exhausted);
    IDAInstance instance;
    auto eq = prepare(instance);
    uint64_t variables[] = {1, 0};
    ASSERT_TRUE(instance.setBatchTangentFunctionV2(eq, 6, tangent, 2, variables));
    ASSERT_TRUE(instance.initialize());
    MatrixInput input;
    ASSERT_EQ(input.evaluate(instance, 2), 0);
    input.check(2);
    if (exhausted) {
      EXPECT_EQ(batchCalls, 0u);
      EXPECT_GT(scalarCalls, 0u);
    } else {
      instance.addJacobianFunction(eq, 1, j, 0, nullptr);
      auto state = x;
      testing::internal::CaptureStderr();
      EXPECT_LT(input.evaluate(instance, 3, 5), 0);
      testing::internal::GetCapturedStderr();
      EXPECT_EQ(state, x);
    }
  }
}

TEST(IDABatchJacobianTest, CompleteMatrixAndFreshStateDerivativeDirections) {
  for (unsigned workers : {1u, 2u}) {
    Guard guard(workers);
    IDAInstance instance;
    auto eq = prepare(instance);
    uint64_t variables[] = {1, 0};
    ASSERT_TRUE(instance.setBatchJacobianFunctionV1(eq, 6, batch, 2, variables));
    ASSERT_TRUE(instance.initialize());
    MatrixInput input;
    for (double alpha : {0., 1., -3., 17.}) {
      ASSERT_EQ(input.evaluate(instance, alpha, alpha * .125), 0);
      input.check(alpha);
    }
#ifdef THREADS_ENABLE
    if (workers != 1) {
      EXPECT_EQ(batchCalls, 0u);
      EXPECT_GT(scalarCalls, 0u);
      continue;
    }
#endif
    EXPECT_EQ(baselineCalls, 4u);
    EXPECT_GT(batchCalls, baselineCalls);
    EXPECT_EQ(scalarCalls, 0u);
  }
}

TEST(IDABatchJacobianTest, BudgetAndReplacementPreserveScalarPath) {
  for (bool noBudget : {false, true}) {
    Guard guard(1, noBudget);
    IDAInstance instance;
    auto eq = prepare(instance);
    uint64_t variables[] = {1, 0};
    ASSERT_TRUE(instance.setBatchJacobianFunctionV1(eq, 6, batch, 2, variables));
    if (!noBudget) instance.addJacobianFunction(eq, 1, j, 0, nullptr);
    ASSERT_TRUE(instance.initialize());
    MatrixInput input;
    ASSERT_EQ(input.evaluate(instance, 2), 0);
    input.check(2);
    EXPECT_EQ(batchCalls, 0u);
    EXPECT_GT(scalarCalls, 0u);
  }
}

TEST(IDABatchJacobianTest, InvalidBindingIsNotAnOptimizationMiss) {
  for (unsigned variant = 0; variant != 6; ++variant) {
    Guard guard;
    IDAInstance instance;
    auto eq = prepare(instance);
    uint64_t variables[] = {1, variant == 0 ? 1u : variant == 1 ? 2u : 0u};
    if (variant == 5)
      ASSERT_TRUE(instance.setBatchJacobianFunctionV1(eq, 6, batch, 2, variables));
    EXPECT_FALSE(instance.setBatchJacobianFunctionV1(
        eq, variant == 2 ? 5 : 6, variant == 3 ? nullptr : batch,
        2, variant == 4 ? nullptr : variables));
    EXPECT_FALSE(instance.initialize());
    EXPECT_EQ(batchCalls, 0u);
  }
}

TEST(IDABatchJacobianTest, CallbackFailureDoesNotFallThroughAndRetryIsFresh) {
  Guard guard;
  IDAInstance instance;
  auto eq = prepare(instance);
  uint64_t variables[] = {1, 0};
  ASSERT_TRUE(instance.setBatchJacobianFunctionV1(eq, 6, batch, 2, variables));
  ASSERT_TRUE(instance.initialize());
  MatrixInput input;
  for (int failure : {-3, 3, 2}) {
    failBatch = failure;
    EXPECT_EQ(input.evaluate(instance, 2), failure < 0 ? -1 : 1);
    EXPECT_EQ(scalarCalls, 0u);
    failBatch = 0;
    ASSERT_EQ(input.evaluate(instance, -2, 3), 0);
    input.check(-2);
  }
}

namespace {
int32_t reverseSupport(uint64_t row, uint64_t column) {
  if (row >= 6 || column >= 6) return -1;
  auto i = 5 - row;
  return column == i || column == (i + 5) % 6;
}
double completeSparseJ(double t, const int64_t *row, const uint64_t *column,
                      double alpha, uint64_t pool, const uint64_t *seeds) {
  if (column[0] != index(row) && column[0] != (index(row) + 5) % 6) return 0;
  return j(t, row, column, alpha, pool, seeds);
}
int32_t invalidSupport(uint64_t, uint64_t) { return -1; }
}

TEST(IDABatchJacobianTest, CertifiedSupportRetainsCSRAndZerosEveryTrial) {
  unsigned conservativeCalls = 0;
  for (bool refined : {false, true}) {
    Guard guard;
    IDAInstance instance;
    auto eq = prepare(instance);
    int64_t all[] = {0, 6};
    instance.addVariableRangeAccess(eq, 1, all, 1);
    instance.addJacobianFunction(eq, 1, completeSparseJ, 0, nullptr);
    uint64_t variables[] = {1, 0};
    ASSERT_TRUE(instance.setBatchJacobianFunctionV1(eq, 6, batch, 2, variables));
    if (refined) {
      ASSERT_TRUE(instance.setBatchJacobianSupportV1(eq, 1, reverseSupport, 5));
    }
    ASSERT_TRUE(instance.initialize());
    batchCalls = baselineCalls = 0;
    MatrixInput input(7, 43);
    for (double alpha : {0., 3., -2.}) {
      // A structural zero must not retain another trial's matrix entry.
      std::fill_n(SUNSparseMatrix_Data(input.matrix), 43, 123.);
      ASSERT_EQ(input.evaluate(instance, alpha, alpha / 4), 0);
      input.check(alpha); // Checks the complete dense matrix, including CSR outsiders.
      auto *rows = SUNSparseMatrix_IndexPointers(input.matrix);
      for (unsigned row = 0; row != 6; ++row)
        EXPECT_EQ(rows[row + 1] - rows[row], 7);
    }
    EXPECT_EQ(baselineCalls, 3u);
    if (refined) EXPECT_LT(batchCalls, conservativeCalls);
    else conservativeCalls = batchCalls;
  }
}

TEST(IDABatchJacobianTest, SupportBudgetAndReplacementKeepScalarFallback) {
  for (unsigned phase = 0; phase != 2; ++phase) {
    Guard guard(1, phase == 0);
    IDAInstance instance;
    auto eq = prepare(instance);
    uint64_t variables[] = {1, 0};
    ASSERT_TRUE(instance.setBatchJacobianFunctionV1(eq, 6, batch, 2, variables));
    ASSERT_TRUE(instance.setBatchJacobianSupportV1(eq, 1, reverseSupport, 5));
    if (phase == 1) instance.addJacobianFunction(eq, 1, j, 0, nullptr);
    ASSERT_TRUE(instance.initialize());
    MatrixInput input;
    ASSERT_EQ(input.evaluate(instance, 2), 0);
    input.check(2);
    EXPECT_EQ(batchCalls, 0u);
    EXPECT_GT(scalarCalls, 0u);
  }
}

TEST(IDABatchJacobianTest, InvalidSupportBindingOrResultIsAnError) {
  for (unsigned phase = 0; phase != 4; ++phase) {
    Guard guard;
    IDAInstance instance;
    auto eq = prepare(instance);
    uint64_t variables[] = {1, 0};
    ASSERT_TRUE(instance.setBatchJacobianFunctionV1(eq, 6, batch, 2, variables));
    if (phase == 0) {
      ASSERT_TRUE(instance.setBatchJacobianSupportV1(eq, 1, reverseSupport, 5));
      EXPECT_FALSE(instance.setBatchJacobianSupportV1(eq, 1, reverseSupport, 5));
    } else if (phase == 1) {
      EXPECT_FALSE(instance.setBatchJacobianSupportV1(eq, 99, reverseSupport, 5));
    } else if (phase == 2) {
      EXPECT_FALSE(instance.setBatchJacobianSupportV1(eq, 1, reverseSupport, 0));
    } else {
      ASSERT_TRUE(instance.setBatchJacobianSupportV1(eq, 1, invalidSupport, 5));
    }
    testing::internal::CaptureStderr();
    EXPECT_FALSE(instance.initialize());
    testing::internal::GetCapturedStderr();
  }
}

TEST(IDABatchJacobianTest, ExpensiveRefinementRetainsCertifiedBatchPlan) {
  unsigned parentCalls = 0;
  for (bool refined : {false, true}) {
    Guard guard;
    IDAInstance instance;
    auto eq = prepare(instance);
    uint64_t variables[] = {1, 0};
    ASSERT_TRUE(instance.setBatchJacobianFunctionV1(eq, 6, batch, 2, variables));
    if (refined)
      ASSERT_TRUE(instance.setBatchJacobianSupportV1(eq, 1, reverseSupport, 1000000));
    ASSERT_TRUE(instance.initialize());
    MatrixInput input;
    ASSERT_EQ(input.evaluate(instance, 3), 0);
    input.check(3);
    EXPECT_EQ(scalarCalls, 0u);
    EXPECT_GT(batchCalls, 0u);
    if (refined) EXPECT_EQ(batchCalls, parentCalls);
    else parentCalls = batchCalls;
  }
}

TEST(IDABatchJacobianTest, StaleSupportStopsBeforeCallbacksOrTrialWrites) {
  for (unsigned phase = 0; phase != 4; ++phase) {
    Guard guard;
    IDAInstance instance;
    auto eq = prepare(instance);
    uint64_t variables[] = {1, 0};
    ASSERT_TRUE(instance.setBatchJacobianFunctionV1(eq, 6, batch, 2, variables));
    ASSERT_TRUE(instance.setBatchJacobianSupportV1(eq, 1, reverseSupport, 5));
    ASSERT_TRUE(instance.initialize());
    if (phase == 0)
      EXPECT_FALSE(instance.setBatchJacobianSupportV1(eq, 1, reverseSupport, 5));
    else if (phase == 1) instance.addJacobianFunction(eq, 1, j, 0, nullptr);
    else if (phase == 2) instance.setResidualFunction(eq, f);
    else instance.addVariableAccess(eq, 1, own);
    MatrixInput input;
    auto before = x;
    testing::internal::CaptureStderr();
    EXPECT_EQ(input.evaluate(instance, 2, 123), -1);
    EXPECT_LT(IDAInstance::residualFunction(.1, input.y, input.yp, input.r,
                                           &instance), 0);
    EXPECT_FALSE(instance.calcIC());
    EXPECT_FALSE(instance.step());
    EXPECT_NE(testing::internal::GetCapturedStderr().find("stale V1"),
              std::string::npos);
    EXPECT_EQ(x, before);
    EXPECT_EQ(batchCalls, 0u);
    EXPECT_EQ(scalarCalls, 0u);
  }
}

namespace {
constexpr unsigned denseSize = 128;
std::array<double, denseSize> denseX{}, denseXP{};
bool interleavedIncidence = false;
double getDenseX(const uint64_t *i) { return denseX[i[0]]; }
void setDenseX(double v, const uint64_t *i) { denseX[i[0]] = v; }
double getDenseXP(const uint64_t *i) { return denseXP[i[0]]; }
void setDenseXP(double v, const uint64_t *i) { denseXP[i[0]] = v; }
double denseF(double t, const int64_t *row) {
  double sum = 0;
  for (unsigned i = 0; i < denseSize; ++i)
    if (!interleavedIncidence || i % 2 == row[0] % 2) sum += denseX[i];
  return 2*denseX[row[0]] + denseXP[row[0]] + .01*sum - t;
}
double denseJ(double, const int64_t *row, const uint64_t *column,
              double alpha, uint64_t, const uint64_t *) {
  ++scalarCalls;
  if (interleavedIncidence && column[0] % 2 != static_cast<uint64_t>(row[0]) % 2)
    return 0;
  return .01 + (row[0] == static_cast<int64_t>(column[0]) ? 2+alpha : 0);
}
int32_t denseBatch(double t, double alpha, const double *direction,
                   const uint64_t *offsets, double *out, uint64_t count) {
  if (count != denseSize || offsets[0] != 0) return -1;
  ++batchCalls;
  double sums[] = {0, 0};
  for (unsigned i = 0; i < denseSize; ++i) sums[i % 2] += direction[i];
  baselineCalls += sums[0] == 0 && sums[1] == 0;
  for (unsigned row = 0; row < denseSize; ++row)
    out[row] = t + denseX[row] + (2+alpha)*direction[row] +
               .01*(interleavedIncidence ? sums[row % 2] : sums[0]+sums[1]);
  return 0;
}
template <size_t Offset>
void stridedAccess(const int64_t *row, uint64_t *column) {
  column[0] = row[0] % 2 + 2 * Offset;
}
template <size_t... Offsets>
void addStridedAccesses(IDAInstance &instance, uint64_t eq, uint64_t variable,
                        std::index_sequence<Offsets...>) {
  (instance.addVariableAccess(eq, variable, stridedAccess<Offsets>), ...);
}
} // namespace

static void checkRepeatedIncidence(bool interleaved) {
  Guard guard;
  interleavedIncidence = interleaved;
  if (interleaved) setenv("MARCO_IDA_BATCH_WORK_LIMIT", "100000", 1);
  IDAInstance instance;
  uint64_t dimensions[] = {denseSize};
  int64_t ranges[] = {0, denseSize};
  auto variable = instance.addStateVariable(1, dimensions, getDenseX, setDenseX,
                                            getDenseXP, setDenseXP, "dense");
  auto eq = instance.addEquation(ranges, 1, "dense linear coupling");
  if (interleaved)
    addStridedAccesses(instance, eq, variable,
                       std::make_index_sequence<denseSize/2>{});
  else
    instance.addVariableRangeAccess(eq, variable, ranges, 1);
  instance.setResidualFunction(eq, denseF);
  instance.addJacobianFunction(eq, variable, denseJ, 0, nullptr);
  ASSERT_TRUE(instance.setBatchJacobianFunctionV1(eq, denseSize, denseBatch, 1, &variable));
  instance.setEndTime(.2);
  instance.setTimeStep(.1);
  ASSERT_TRUE(instance.initialize());
  MatrixInput input(denseSize, denseSize*denseSize);
  for (double alpha : {0., 3.}) {
    for (unsigned i = 0; i < denseSize; ++i) {
      N_VGetArrayPointer(input.y)[i] = i*.125;
      N_VGetArrayPointer(input.yp)[i] = .25;
    }
    ASSERT_EQ(IDAInstance::jacobianMatrix(.125, alpha, input.y, input.yp,
        input.r, input.matrix, &instance, nullptr, nullptr, nullptr), 0);
    auto *rows = SUNSparseMatrix_IndexPointers(input.matrix);
    auto *columns = SUNSparseMatrix_IndexValues(input.matrix);
    auto *data = SUNSparseMatrix_Data(input.matrix);
    for (unsigned row = 0; row < denseSize; ++row) {
      ASSERT_EQ(rows[row+1]-rows[row], interleaved ? denseSize/2 : denseSize);
      std::array<double, denseSize> dense{};
      for (auto p = rows[row]; p < rows[row+1]; ++p)
        dense[columns[p]] = data[p];
      for (unsigned column = 0; column < denseSize; ++column)
        EXPECT_NEAR(dense[column],
                    (interleaved && column % 2 != row % 2) ? 0 :
                        .01+(column == row ? 2+alpha : 0), 5e-13);
    }
  }
  EXPECT_EQ(scalarCalls, 0u);
  EXPECT_EQ(baselineCalls, 2u);
  EXPECT_EQ(batchCalls, 2u*((interleaved ? denseSize/2 : denseSize)+1));
}

TEST(IDABatchJacobianTest, RepeatedIncidenceHasBoundedPlanAndCompleteMatrix) {
  checkRepeatedIncidence(false);
}

TEST(IDABatchJacobianTest, InterleavedRepeatedIncidenceStaysWithinBudget) {
  checkRepeatedIncidence(true);
}
