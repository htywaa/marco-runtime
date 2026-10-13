#include "marco/Runtime/Solvers/KINSOL/Instance.h"
#include "marco/Runtime/Multithreading/Options.h"
#include "marco/Runtime/Support/MemoryManagement.h"
#include "gtest/gtest.h"
#include <array>
#include <atomic>
#include <type_traits>

#ifdef SUNDIALS_ENABLE

using marco::runtime::sundials::Equation;
using marco::runtime::sundials::Variable;
using marco::runtime::sundials::kinsol::KINSOLInstance;

// 中文：以仅 timed callback 的局部系统验证 IDA component closure 不依赖普通
// KINSOL callback，也不会在 assertions-enabled 构建中误触发断言。
// English: A timed-callback-only local system verifies that IDA component
// closure does not require ordinary KINSOL callbacks or trip assertions in
// assertions-enabled builds.
namespace {
double value = 0;

double getValue(const uint64_t *) { return value; }

void setValue(double newValue, const uint64_t *) { value = newValue; }

void singletonAccess(const int64_t *, uint64_t *result) { result[0] = 0; }

double timedResidual(double time, const int64_t *) { return value - time; }

double timedJacobian(double, const int64_t *, const uint64_t *, double,
                     uint64_t, const uint64_t *) {
  return 1;
}
} // namespace

TEST(KINSOLTimedCallbacksTest, SolvesWithTimedOnlyCallbacks) {
  value = 0;
  KINSOLInstance instance;

  uint64_t dimensions[] = {1};
  int64_t ranges[] = {0, 1};
  Variable variable = instance.addVariable(1, dimensions, getValue, setValue,
                                           "timed_singleton");
  Equation equation = instance.addEquation(ranges, 1, "x[0] - time = 0");
  instance.addVariableAccess(equation, variable, singletonAccess);
  instance.setTimedResidualFunction(equation, timedResidual);

  uint64_t unusedSeedSize = 1;
  instance.addTimedJacobianFunction(equation, variable, timedJacobian, 0,
                                    &unusedSeedSize);

  ASSERT_TRUE(instance.initialize());
  ASSERT_TRUE(instance.solve(2.5));
  EXPECT_NEAR(value, 2.5, 1e-12);
}

namespace {
double splitValues[6];

template <unsigned Offset>
double getSplitValue(const uint64_t *indices) {
  return splitValues[Offset + indices[0]];
}

template <unsigned Offset>
void setSplitValue(double value, const uint64_t *indices) {
  splitValues[Offset + indices[0]] = value;
}

template <unsigned Offset>
void splitAccess(const int64_t *indices, uint64_t *result) {
  result[0] = indices[0] - Offset;
}

double splitResidual(double time, const int64_t *indices) {
  return splitValues[indices[0]] - (time + indices[0] + 1);
}

template <unsigned Offset>
double splitJacobian(double, const int64_t *rows, const uint64_t *columns,
                     double, uint64_t, const uint64_t *) {
  return rows[0] == static_cast<int64_t>(columns[0] + Offset) ? 1 : 0;
}
} // namespace

TEST(KINSOLTimedCallbacksTest, ReusedAffineMapsRespectVariableBlockRanges) {
  for (unsigned i = 0; i < 6; ++i)
    splitValues[i] = i + 1;
  KINSOLInstance instance;
  uint64_t singleton[] = {1}, middle[] = {4};
  Variable first = instance.addVariable(1, singleton, getSplitValue<0>,
                                        setSplitValue<0>, "first");
  Variable last = instance.addVariable(1, singleton, getSplitValue<5>,
                                       setSplitValue<5>, "last");
  Variable center = instance.addVariable(1, middle, getSplitValue<1>,
                                         setSplitValue<1>, "middle");
  // One whole-array residual reads three registered column blocks. The maps
  // retain global equation coordinates, including negative translated images.
  int64_t range[] = {0, 6};
  Equation equation = instance.addEquation(range, 1, "x[i] = time+i+1");
  instance.setTimedResidualFunction(equation, splitResidual);
  instance.addVariableAccess(equation, first, splitAccess<0>);
  instance.addVariableAccess(equation, last, splitAccess<5>);
  instance.addVariableAccess(equation, center, splitAccess<1>);
  instance.addTimedJacobianFunction(equation, first, splitJacobian<0>, 0, nullptr);
  instance.addTimedJacobianFunction(equation, last, splitJacobian<5>, 0, nullptr);
  instance.addTimedJacobianFunction(equation, center, splitJacobian<1>, 0, nullptr);
  ASSERT_TRUE(instance.requestImmutableAccessLayout());
  ASSERT_TRUE(instance.configureInitializationAnchorScaling(0));
  std::vector<double> response;
  ASSERT_TRUE(instance.solveCurrentJacobian({1, 2, 3, 4, 5, 6}, response));
  ASSERT_EQ(response.size(), 6u);
  EXPECT_EQ(response, (std::vector<double>{1, 6, 2, 3, 4, 5}));
  ASSERT_TRUE(instance.solve(2.5));
  for (unsigned i = 0; i < 6; ++i)
    EXPECT_NEAR(splitValues[i], 3.5 + i, 1e-12);
}

namespace {
std::array<double, 4> matrixValues;
std::array<std::vector<uint64_t>, 4> matrixColumnVisits;

double matrixGet(const uint64_t *indices) {
  return matrixValues[2 * indices[0] + indices[1]];
}
void matrixSet(double value, const uint64_t *indices) {
  matrixValues[2 * indices[0] + indices[1]] = value;
}
template <unsigned I, unsigned J>
void matrixAccess(const int64_t *, uint64_t *indices) {
  indices[0] = I;
  indices[1] = J;
}
double matrixResidual(double time, const int64_t *indices) {
  auto row = 2 * indices[0] + indices[1];
  return matrixValues[row] - (time + row + 1);
}
double matrixJacobian(double, const int64_t *rows, const uint64_t *columns,
                      double, uint64_t, const uint64_t *) {
  auto row = 2 * rows[0] + rows[1];
  auto column = 2 * columns[0] + columns[1];
  matrixColumnVisits[row].push_back(column);
  return row == static_cast<int64_t>(column) ? 1 : 0;
}
} // namespace

TEST(KINSOLTimedCallbacksTest, MultidimensionalColumnsKeepCanonicalOrder) {
  // Crossed coordinates (0,1) and (1,0) must have one strict order. Exercise
  // actual sparse factorization and solve callbacks, including duplicate and
  // reversed access registration, rather than testing an isolated comparator.
  for (unsigned variant = 0; variant < 4; ++variant) {
    bool reverse = (variant & 1) != 0;
    bool cached = (variant & 2) != 0;
    matrixValues = {1, 2, 3, 4};
    for (auto &visits : matrixColumnVisits)
      visits.clear();
    KINSOLInstance instance;
    uint64_t dimensions[] = {2, 2};
    int64_t ranges[] = {0, 2, 0, 2};
    auto variable = instance.addVariable(2, dimensions, matrixGet, matrixSet,
                                         "matrix");
    auto equation = instance.addEquation(ranges, 2, "x[i,j]=time+2*i+j+1");
    using marco::runtime::sundials::AccessFunction;
    std::array<AccessFunction, 4> accesses = {
        matrixAccess<0, 0>, matrixAccess<0, 1>, matrixAccess<1, 0>,
        matrixAccess<1, 1>};
    for (unsigned i = 0; i < accesses.size(); ++i)
      instance.addVariableAccess(equation, variable,
                                 accesses[reverse ? 3 - i : i]);
    instance.addVariableAccess(equation, variable, matrixAccess<0, 1>);
    instance.setTimedResidualFunction(equation, matrixResidual);
    instance.addTimedJacobianFunction(equation, variable, matrixJacobian, 0,
                                      nullptr);
    if (cached) {
      ASSERT_TRUE(instance.requestImmutableAccessLayout());
    }
    ASSERT_TRUE(instance.configureInitializationAnchorScaling(0));
    std::vector<double> response;
    ASSERT_TRUE(instance.solveCurrentJacobian({4, 3, 2, 1}, response));
    EXPECT_EQ(response, (std::vector<double>{4, 3, 2, 1}));
    ASSERT_TRUE(instance.solve(2.5));
    for (unsigned row = 0; row < 4; ++row) {
      EXPECT_NEAR(matrixValues[row], 3.5 + row, 1e-12);
      ASSERT_FALSE(matrixColumnVisits[row].empty());
      EXPECT_EQ(matrixColumnVisits[row].size() % 4, 0u);
      for (size_t index = 0; index < matrixColumnVisits[row].size(); ++index)
        EXPECT_EQ(matrixColumnVisits[row][index], index % 4)
            << "row=" << row << " reversed=" << reverse;
    }
  }
}

namespace {
std::array<double, 6> reversedValues;
std::atomic<uint64_t> reverseAccessCalls{0};
double reverseGet(const uint64_t *indices) {
  return reversedValues[indices[0]];
}
void reverseSet(double x, const uint64_t *indices) {
  reversedValues[indices[0]] = x;
}
void reverseAccess(const int64_t *rows, uint64_t *columns) {
  ++reverseAccessCalls; // Passive test count, not a dependency of the map.
  columns[0] = 3 - rows[0];
}
double reverseResidual(double time, const int64_t *rows) {
  return reversedValues[3 - rows[0]] - (time + rows[0] + 3);
}
double reverseJacobian(double, const int64_t *rows, const uint64_t *columns,
                       double, uint64_t, const uint64_t *) {
  return columns[0] == static_cast<uint64_t>(3 - rows[0]) ? 1 : 0;
}
} // namespace

TEST(KINSOLTimedCallbacksTest, ImmutableGeometryPreservesTrialsAndBoundedFallback) {
#ifdef THREADS_ENABLE
  auto &options = marco::runtime::multithreading::multithreadingOptions();
  struct Restore {
    decltype(options) target;
    marco::runtime::multithreading::MultithreadingOptions saved;
    ~Restore() { target = saved; }
  } restore{options, options};
#endif
  // Negative modes retain the original path: no purity declaration, zero
  // work budget, or a storage budget that expires part-way through the plan.
  for (unsigned workers : {1u, 3u}) {
#ifdef THREADS_ENABLE
    options.numOfThreads = workers;
#endif
    for (int mode : {-1, 0, 1, 2, 3}) {
      SCOPED_TRACE(::testing::Message() << "workers=" << workers << " mode=" << mode);
      reversedValues.fill(0);
      reverseAccessCalls = 0;
      KINSOLInstance instance;
      uint64_t dimensions[] = {6};
      int64_t ranges[] = {-2, 4};
      auto variable = instance.addVariable(1, dimensions, reverseGet, reverseSet,
                                           "reverse");
      auto equation = instance.addEquation(ranges, 1, "x[3-i] = time+i+3");
      instance.addVariableAccess(equation, variable, reverseAccess);
      instance.setTimedResidualFunction(equation, reverseResidual);
      instance.addTimedJacobianFunction(equation, variable, reverseJacobian,
                                        0, nullptr);
      if (mode >= 0) {
        ASSERT_TRUE(instance.requestImmutableAccessLayout(
            mode == 0 ? 0 : 1000000, mode == 1 ? 256 : 16000000));
      }
      if (mode == 3)
        // A registration after certification must invalidate the declaration.
        instance.addVariableAccess(equation, variable, reverseAccess);
      ASSERT_TRUE(instance.initialize());
      EXPECT_FALSE(instance.requestImmutableAccessLayout());
      const auto callsAtInitialization = reverseAccessCalls.load();
      for (double time : {2.5, -3.5, 8.0}) {
        // A changed caller state must be copied again, never numerically reused.
        reversedValues.fill(time - 100);
        ASSERT_TRUE(instance.solve(time));
        for (unsigned i = 0; i < 6; ++i)
          EXPECT_NEAR(reversedValues[i], time + 6 - i, 1e-11);
        ASSERT_TRUE(instance.factorizeCurrentJacobian(time));
        std::vector<double> response;
        ASSERT_TRUE(instance.solveCurrentJacobian({1, 2, 3, 4, 5, 6}, response));
        EXPECT_EQ(response, (std::vector<double>{6, 5, 4, 3, 2, 1}));
      }
      if (mode == 2) {
        EXPECT_EQ(reverseAccessCalls.load(), callsAtInitialization);
      } else {
        EXPECT_GT(reverseAccessCalls.load(), callsAtInitialization);
      }
    }
  }
}

TEST(KINSOLTimedCallbacksTest, ImmutableGeometryDoesNotChangeEmptySystem) {
  KINSOLInstance instance;
  EXPECT_FALSE(instance.requestImmutableAccessLayout(1000001, 16000000));
  EXPECT_FALSE(instance.requestImmutableAccessLayout(1000000, 16000001));
  ASSERT_TRUE(instance.requestImmutableAccessLayout());
  ASSERT_TRUE(instance.initialize());
  ASSERT_TRUE(instance.solve(0));
}

TEST(KINSOLTimedCallbacksTest, CallbackReplacementInvalidatesFixedBindings) {
#ifdef THREADS_ENABLE
  auto &options = marco::runtime::multithreading::multithreadingOptions();
  struct Restore {
    decltype(options) target;
    marco::runtime::multithreading::MultithreadingOptions saved;
    ~Restore() { target = saved; }
  } restore{options, options};
  options.numOfThreads = 1;
#endif
  value = 0;
  KINSOLInstance instance;
  uint64_t dimensions[] = {1};
  int64_t ranges[] = {0, 1};
  auto variable = instance.addVariable(1, dimensions, getValue, setValue, "x");
  auto equation = instance.addEquation(ranges, 1, "x=time");
  instance.addVariableAccess(equation, variable, singletonAccess);
  instance.setTimedResidualFunction(equation, timedResidual);
  instance.addTimedJacobianFunction(equation, variable, timedJacobian, 0, nullptr);
  ASSERT_TRUE(instance.requestImmutableAccessLayout());
  ASSERT_TRUE(instance.initialize());
  ASSERT_TRUE(instance.factorizeCurrentJacobian(1));
  std::vector<double> response;
  ASSERT_TRUE(instance.solveCurrentJacobian({6}, response));
  EXPECT_EQ(response, (std::vector<double>{6}));

  // Replacing zero-seed callbacks is allowed by the existing registration
  // interface. A published binding must never retain the previous function.
  instance.setTimedResidualFunction(equation,
      +[](double time, const int64_t *) { return 2 * (value - time); });
  instance.addTimedJacobianFunction(equation, variable,
      +[](double, const int64_t *, const uint64_t *, double, uint64_t,
          const uint64_t *) { return 2.0; }, 0, nullptr);
  ASSERT_TRUE(instance.factorizeCurrentJacobian(2));
  ASSERT_TRUE(instance.solveCurrentJacobian({6}, response));
  EXPECT_EQ(response, (std::vector<double>{3}));
  ASSERT_TRUE(instance.solve(2));
  EXPECT_NEAR(value, 2, 1e-12);
}

namespace {
std::array<std::array<double, 3>, 2> seededValues;
std::array<uint64_t, 2> seededCalls;
std::array<uint64_t, 2> seededPools;
std::array<std::array<std::array<uint64_t, 2>, 3>, 2> seededHandles;
std::array<std::array<bool, 3>, 2> seededRowsSeen;

template <unsigned I> double seededGet(const uint64_t *indices) {
  return seededValues[I][indices[0]];
}
template <unsigned I> void seededSet(double x, const uint64_t *indices) {
  seededValues[I][indices[0]] = x;
}
void seededAccess(const int64_t *rows, uint64_t *columns) {
  columns[0] = 4 - rows[0];
}
template <unsigned I>
double seededResidual(double time, const int64_t *rows) {
  return (2 + time) * (seededValues[I][4 - rows[0]] - time - rows[0] + 2);
}
template <unsigned I>
double seededJacobian(double time, const int64_t *rows,
                      const uint64_t *columns, double, uint64_t pool,
                      const uint64_t *seeds) {
  EXPECT_NE(seeds, nullptr);
  if (!seeds)
    return 0;
  auto row = rows[0] - 2;
  if (seededCalls[I]++ != 0)
    EXPECT_EQ(pool, seededPools[I]);
  // A single worker still owns several original chunks and therefore several
  // seed sets. Bindings are stable for each row/chunk, not one set per solver.
  if (seededRowsSeen[I][row]) {
    EXPECT_EQ(seeds[0], seededHandles[I][row][0]);
    EXPECT_EQ(seeds[1], seededHandles[I][row][1]);
  }
  seededRowsSeen[I][row] = true;
  seededPools[I] = pool;
  seededHandles[I][row] = {seeds[0], seeds[1]};
  auto &memory = marco::runtime::MemoryPoolManager::getInstance().get(pool);
  double *direction = memory.get(seeds[0]);
  double *primal = memory.get(seeds[1]);
  // Overwrite every value on each call, as generated AD does. Only storage
  // identities survive a trial; no old seed or time is a numerical input.
  for (unsigned i = 0; i < 3; ++i)
    direction[i] = i == columns[0] ? 1 : 0;
  primal[0] = time;
  primal[1] = 2;
  return (primal[0] + primal[1]) * direction[4 - rows[0]];
}
} // namespace

TEST(KINSOLTimedCallbacksTest, BoundSeedsPreserveInstancesTrialsAndReverseRows) {
#ifdef THREADS_ENABLE
  auto &options = marco::runtime::multithreading::multithreadingOptions();
  struct Restore {
    decltype(options) target;
    marco::runtime::multithreading::MultithreadingOptions saved;
    ~Restore() { target = saved; }
  } restore{options, options};
  options.numOfThreads = 1;
#endif
  for (bool cached : {false, true}) {
    seededCalls.fill(0);
    seededRowsSeen = {};
    std::array<KINSOLInstance, 2> instances;
    uint64_t dimensions[] = {3}, seedSizes[] = {3, 2};
    int64_t ranges[] = {2, 5};
    for (unsigned i = 0; i < 2; ++i) {
      auto &instance = instances[i];
      auto variable = instance.addVariable(1, dimensions,
          i ? seededGet<1> : seededGet<0>,
          i ? seededSet<1> : seededSet<0>, "reverse");
      auto equation = instance.addEquation(ranges, 1, "scaled reverse");
      instance.addVariableAccess(equation, variable, seededAccess);
      instance.setTimedResidualFunction(equation,
          i ? seededResidual<1> : seededResidual<0>);
      instance.addTimedJacobianFunction(equation, variable,
          i ? seededJacobian<1> : seededJacobian<0>, 2, seedSizes);
      if (cached)
        ASSERT_TRUE(instance.requestImmutableAccessLayout());
      ASSERT_TRUE(instance.initialize());
    }
    for (double time : {0.0, 2.0, -1.0}) {
      for (unsigned i = 0; i < 2; ++i) {
        auto &instance = instances[i];
        seededValues[i].fill(time - 50);
        ASSERT_TRUE(instance.solve(time));
        for (unsigned j = 0; j < 3; ++j)
          EXPECT_NEAR(seededValues[i][j], time + 2 - j, 1e-11);
        ASSERT_TRUE(instance.factorizeCurrentJacobian(time));
        std::vector<double> response;
        ASSERT_TRUE(instance.solveCurrentJacobian({3, 2, 1}, response));
        for (unsigned j = 0; j < 3; ++j)
          EXPECT_NEAR(response[j], (j + 1) / (2 + time), 1e-12);
      }
    }
    EXPECT_GT(seededCalls[0], 3u);
    EXPECT_GT(seededCalls[1], 3u);
    EXPECT_NE(seededPools[0], seededPools[1]);
  }
}

TEST(KINSOLTimedCallbacksTest, ScalarBlocksBindActualRowsAndKeepFullFactorization) {
  using Plan = marco::runtime::sundials::kinsol::ScalarBlockPlan;
#ifdef THREADS_ENABLE
  auto &options = marco::runtime::multithreading::multithreadingOptions();
  auto saved = options;
  struct Restore {
    decltype(options) target;
    marco::runtime::multithreading::MultithreadingOptions saved;
    ~Restore() { target = saved; }
  } restore{options, saved};
  options.numOfThreads = 1;
#endif
  for (bool allowed : {false, true}) {
    Plan::Budget budget;
    KINSOLInstance instance;
    uint64_t dimensions[] = {1};
    int64_t ranges[] = {0, 1};
    auto variable = instance.addVariable(1, dimensions, getValue, setValue, "x");
    auto equation = instance.addEquation(ranges, 1, "x=time");
    instance.addVariableAccess(equation, variable, singletonAccess);
    instance.setTimedResidualFunction(equation, timedResidual);
    instance.addTimedJacobianFunction(equation, variable, timedJacobian, 0, nullptr);
    ASSERT_TRUE(instance.requestImmutableAccessLayout());
    if (!allowed) budget.work = 0;
    ASSERT_TRUE(instance.requestScalarBlockPlan({{0, 1, true, {{1, 0}}}}, budget));
    value = -5;
    ASSERT_TRUE(instance.configureInitializationAnchorScaling(0));
    for (double time : {2., -3., 0., 4.}) {
      value = time + 9;
      ASSERT_TRUE(instance.solve(time));
      EXPECT_NEAR(value, time, 1e-12);
      ASSERT_TRUE(instance.factorizeCurrentJacobian(time));
      std::vector<double> response;
      ASSERT_TRUE(instance.solveCurrentJacobian({7}, response));
      EXPECT_EQ(response, (std::vector<double>{7}));
    }
    EXPECT_EQ(instance.getScalarBlockAcceptedCalls(), allowed ? 4u : 0u);
    // Registration invalidates every optional numeric/layout object.
    instance.setTimedResidualFunction(equation, timedResidual);
    ASSERT_TRUE(instance.solve(6));
    EXPECT_NEAR(value, 6, 1e-12);
    EXPECT_EQ(instance.getScalarBlockAcceptedCalls(), allowed ? 4u : 0u);
  }
}

TEST(KINSOLTimedCallbacksTest, ScalarCertificateIsRecheckedAgainstActualAccesses) {
  using Plan = marco::runtime::sundials::kinsol::ScalarBlockPlan;
  Plan::Budget budget;
  KINSOLInstance instance;
  uint64_t dimensions[] = {1};
  int64_t ranges[] = {0, 1};
  instance.addVariable(1, dimensions, getValue, setValue, "x");
  instance.addEquation(ranges, 1, "x=time");
  EXPECT_FALSE(instance.requestScalarBlockPlan({{1, 1, true, {{1, 0}}}}, budget));
  EXPECT_FALSE(instance.requestScalarBlockPlan({{0, 1, true, {{0, 0}}}}, budget));
}

TEST(KINSOLTimedCallbacksTest, ScalarBlocksSupportOrdinaryOnlyCallbacks) {
  using Plan = marco::runtime::sundials::kinsol::ScalarBlockPlan;
#ifdef THREADS_ENABLE
  auto &options = marco::runtime::multithreading::multithreadingOptions();
  auto saved = options;
  struct Restore {
    decltype(options) target;
    marco::runtime::multithreading::MultithreadingOptions saved;
    ~Restore() { target = saved; }
  } restore{options, saved};
  options.numOfThreads = 1;
#endif
  Plan::Budget budget;
  KINSOLInstance instance;
  uint64_t dimensions[] = {1};
  int64_t ranges[] = {0, 1};
  auto variable = instance.addVariable(1, dimensions, getValue, setValue, "x");
  auto equation = instance.addEquation(ranges, 1, "x=3");
  instance.addVariableAccess(equation, variable, singletonAccess);
  instance.setResidualFunction(equation, +[](const int64_t *) { return value - 3; });
  instance.addJacobianFunction(equation, variable,
      +[](const int64_t *, const uint64_t *, uint64_t, const uint64_t *) {
        return 1.0;
      }, 0, nullptr);
  ASSERT_TRUE(instance.requestImmutableAccessLayout());
  ASSERT_TRUE(instance.requestScalarBlockPlan({{0, 1, true, {{1, 0}}}}, budget));
  value = -5;
  ASSERT_TRUE(instance.solve());
  EXPECT_NEAR(value, 3, 1e-12);
  EXPECT_EQ(instance.getScalarBlockAcceptedCalls(), 1u);
  instance.setResidualFunction(equation, +[](const int64_t *) { return value - 7; });
  ASSERT_TRUE(instance.solve());
  EXPECT_NEAR(value, 7, 1e-12);
  EXPECT_EQ(instance.getScalarBlockAcceptedCalls(), 1u);
}


namespace {
uint64_t generatedCalls=0;
bool generatedMiss=false;
void generatedResidual(double time,uint64_t rows,double *out) {
  if(rows&1) out[0]=value-time;
}
void generatedJacobian(double,uint64_t rows,uint64_t cols,double *out) {
  if((rows&1)&&(cols&1)) out[0]=1;
}
void generatedInstall(const double *x) { value=x[0]; }
void generatedRead(double *x) { x[0]=value; }
int32_t generatedHealth(const double *j,const double *,const double *,double *out) {
  if(j[0]!=1) return 0;
  out[0]=out[1]=out[2]=1;return 1;
}
int32_t generatedSolve(double time,double *x,const double *,const double *,
    double,uint64_t blocks,const uint64_t *,const uint64_t *,const uint64_t *,
    const uint64_t *,double *f,uint64_t *iterations) {
  ++generatedCalls;
  EXPECT_EQ(blocks,1u);
  // A failed generated trial can dirty its private scratch, never model state.
  x[0]=generatedMiss?123456:time;
  f[0]=0;*iterations=0;
  return generatedMiss?0:1;
}
}

TEST(KINSOLTimedCallbacksTest, VersionedKernelBindsPlanAndRestoresMissedTrials) {
  using Plan=marco::runtime::sundials::kinsol::ScalarBlockPlan;
#ifdef THREADS_ENABLE
  auto &options=marco::runtime::multithreading::multithreadingOptions();
  struct Restore {
    decltype(options) target;
    std::remove_reference_t<decltype(options)> saved;
    ~Restore() { target=saved; }
  } restore{options,options};
#endif
  for(int variant=0;variant<8;++variant) {
#ifdef THREADS_ENABLE
    options.numOfThreads=variant==7?2:1;
    options.enableMultithreading=true;
    const bool expectKernel=variant!=6 && variant!=7;
#else
    const bool expectKernel=variant!=6;
#endif
    Plan::Budget budget;
    KINSOLInstance instance;
    uint64_t dimensions[]={1};int64_t ranges[]={0,1};
    auto variable=instance.addVariable(1,dimensions,getValue,setValue,"x");
    auto equation=instance.addEquation(ranges,1,"x=time");
    instance.addVariableAccess(equation,variable,singletonAccess);
    instance.setTimedResidualFunction(equation,timedResidual);
    instance.addTimedJacobianFunction(equation,variable,timedJacobian,0,nullptr);
    ASSERT_TRUE(instance.requestImmutableAccessLayout());
    ASSERT_TRUE(instance.requestScalarBlockPlan({{0,1,true,{{1,0}}}},budget));
    const void *f[]={reinterpret_cast<const void *>(timedResidual)};
    const void *j[]={reinterpret_cast<const void *>(timedJacobian)};
    const void *g[]={reinterpret_cast<const void *>(getValue)};
    const void *s[]={reinterpret_cast<const void *>(setValue)};
    uint64_t match[]={0};
    MarcoC2KernelV1 descriptor{1,sizeof(MarcoC2KernelV1),3,0,4,1,1024,
        f,j,g,s,match,generatedResidual,generatedJacobian,generatedInstall,
        generatedRead,generatedHealth,generatedSolve};
    if(variant==1) descriptor.version=2;
    if(variant==2) descriptor.descriptorBytes=0;
    if(variant==3) j[0]=nullptr;
    if(variant==4) match[0]=1;
    if(variant==5) descriptor.scratchBytes=65537;
    if(variant==6) budget.work=0;
    if(variant>0 && variant<6) {
      EXPECT_FALSE(instance.bindC2KernelV1(descriptor));
      continue;
    }
    ASSERT_TRUE(instance.bindC2KernelV1(descriptor));
    value=-9;generatedCalls=0;
    ASSERT_TRUE(instance.configureInitializationAnchorScaling(0));
    EXPECT_EQ(instance.hasC2KernelV1(),expectKernel);
    EXPECT_FALSE(instance.bindC2KernelV1(descriptor));
    for(bool miss:{false,true,false}) {
      generatedMiss=miss;
      for(double time:{2.,-7.,4.}) {
        value=time+9;
        ASSERT_TRUE(instance.solve(time));
        EXPECT_NEAR(value,time,1e-12);
        ASSERT_TRUE(instance.factorizeCurrentJacobian(time));
        std::vector<double> response;
        ASSERT_TRUE(instance.solveCurrentJacobian({3},response));
        EXPECT_EQ(response,(std::vector<double>{3}));
      }
    }
    EXPECT_EQ(generatedCalls,expectKernel?9u:0u);
    instance.setTimedResidualFunction(equation,timedResidual);
    ASSERT_TRUE(instance.solve(13));
    EXPECT_NEAR(value,13,1e-12);
    EXPECT_EQ(generatedCalls,expectKernel?9u:0u);
  }
}

#endif // SUNDIALS_ENABLE
