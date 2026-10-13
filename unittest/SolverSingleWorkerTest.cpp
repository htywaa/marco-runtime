#include "marco/Runtime/Multithreading/Options.h"
#include "marco/Runtime/Multithreading/ThreadPool.h"
#include "marco/Runtime/Solvers/IDA/Instance.h"
#include "marco/Runtime/Solvers/IDA/Options.h"
#include "marco/Runtime/Solvers/KINSOL/Instance.h"
#include "gtest/gtest.h"
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <type_traits>
#include "marco/Runtime/Simulation/Options.h"

using marco::runtime::sundials::ida::IDAInstance;
using marco::runtime::sundials::kinsol::KINSOLInstance;

namespace {
struct Slot {
  double value = 0;
  std::atomic<unsigned> residuals{0}, jacobians{0}, callerCalls{0};
};
std::array<Slot, 3> slots;
thread_local unsigned callerToken = 0;
bool throwResidual = false;
bool nest = false;
bool nestedPassed = false;
KINSOLInstance *inner = nullptr;

struct OptionsGuard {
#ifdef THREADS_ENABLE
  decltype(marco::runtime::multithreading::multithreadingOptions()) savedRef =
      marco::runtime::multithreading::multithreadingOptions();
  std::remove_reference_t<decltype(savedRef)> saved = savedRef;
#endif
  marco::runtime::sundials::ida::Options idaSaved =
      marco::runtime::sundials::ida::getOptions();
  explicit OptionsGuard(unsigned workers = 1, bool enabled = true) {
#ifdef THREADS_ENABLE
    savedRef.numOfThreads = workers;
    savedRef.enableMultithreading = enabled;
#endif
    marco::runtime::sundials::ida::getOptions().equidistantTimeGrid = true;
    callerToken = 17;
    throwResidual = nest = nestedPassed = false;
    for (auto &slot : slots) {
      slot.value = -1;
      slot.residuals = slot.jacobians = slot.callerCalls = 0;
    }
  }
  ~OptionsGuard() {
#ifdef THREADS_ENABLE
    savedRef = saved;
#endif
    marco::runtime::sundials::ida::getOptions() = idaSaved;
    callerToken = 0;
    throwResidual = nest = false;
    inner = nullptr;
  }
};

template <unsigned I> double get(const uint64_t *) { return slots[I].value; }
template <unsigned I> void set(double value, const uint64_t *) {
  slots[I].value = value;
}
void access(const int64_t *, uint64_t *indices) { indices[0] = 0; }
template <unsigned I> double residual(double time, const int64_t *) {
  ++slots[I].residuals;
  if (callerToken == 17)
    ++slots[I].callerCalls;
  if (throwResidual)
    throw std::runtime_error("single worker callback exception");
  if (I == 0 && nest && !nestedPassed)
    nestedPassed = inner->solve(4.5) && std::abs(slots[2].value - 4.5) < 1e-10;
  return slots[I].value - time;
}
template <unsigned I>
double jacobian(double, const int64_t *, const uint64_t *, double, uint64_t,
                const uint64_t *) {
  ++slots[I].jacobians;
  if (callerToken == 17)
    ++slots[I].callerCalls;
  return 1;
}
template <unsigned I> void prepare(KINSOLInstance &instance) {
  uint64_t dimensions[] = {1};
  int64_t ranges[] = {0, 1};
  auto variable = instance.addVariable(1, dimensions, get<I>, set<I>, "x");
  auto equation = instance.addEquation(ranges, 1, "x = time");
  instance.addVariableAccess(equation, variable, access);
  instance.setTimedResidualFunction(equation, residual<I>);
  instance.addTimedJacobianFunction(equation, variable, jacobian<I>, 0, nullptr);
}
template <unsigned I> void prepare(IDAInstance &instance) {
  uint64_t dimensions[] = {1};
  int64_t ranges[] = {0, 1};
  auto variable = instance.addAlgebraicVariable(1, dimensions, get<I>, set<I>, "x");
  auto equation = instance.addEquation(ranges, 1, "x = time");
  instance.addVariableAccess(equation, variable, access);
  instance.setResidualFunction(equation, residual<I>);
  instance.addJacobianFunction(equation, variable, jacobian<I>, 0, nullptr);
  instance.setEndTime(0.25);
  instance.setTimeStep(0.25);
}
void expectCaller(unsigned i) {
  EXPECT_GT(slots[i].residuals, 0u);
  EXPECT_GT(slots[i].jacobians, 0u);
  EXPECT_EQ(slots[i].callerCalls,
            slots[i].residuals.load() + slots[i].jacobians.load());
}
} // namespace

TEST(SolverSingleWorkerTest, KINSOLPreservesChunksAndCallerTLS) {
  for (unsigned requested : {0u, 1u, 4u}) {
    OptionsGuard options(requested, requested != 4);
    KINSOLInstance instance;
    prepare<0>(instance);
    ASSERT_TRUE(instance.solve(2.5));
    EXPECT_NEAR(slots[0].value, 2.5, 1e-10);
    expectCaller(0);
  }
}

TEST(SolverSingleWorkerTest, IDAResidualAndJacobianUseCallerTLS) {
  OptionsGuard options;
  IDAInstance instance;
  prepare<0>(instance);
  ASSERT_TRUE(instance.calcIC());
  ASSERT_TRUE(instance.step());
  EXPECT_NEAR(slots[0].value, 0.25, 1e-10);
  expectCaller(0);
}

TEST(SolverSingleWorkerTest, PassiveProfileCountsActualCallsWithoutAddingWork) {
  auto &simulation = marco::runtime::simulation::getOptions();
  struct Restore {
    bool &value;
    bool saved;
    ~Restore() { value = saved; }
  } restore{simulation.profiling, simulation.profiling};
  for (unsigned workers : {1u, 2u}) {
    std::array<unsigned, 4> reference{};
    for (bool profile : {false, true}) {
      OptionsGuard options(workers);
      simulation.profiling = profile;
      testing::internal::CaptureStderr();
      bool solved = false;
      {
        IDAInstance ida;
        KINSOLInstance kinsol;
        prepare<0>(ida);
        prepare<1>(kinsol);
        solved = ida.calcIC() && ida.step() && kinsol.solve(2.5);
      }
      std::string output = testing::internal::GetCapturedStderr();
      ASSERT_TRUE(solved);
      EXPECT_NEAR(slots[0].value, 0.25, 1e-10);
      EXPECT_NEAR(slots[1].value, 2.5, 1e-10);
      std::array<unsigned, 4> calls{slots[0].residuals.load(),
          slots[0].jacobians.load(), slots[1].residuals.load(),
          slots[1].jacobians.load()};
      if (!profile) {
        reference = calls;
        EXPECT_EQ(output.find("-call-statistics]"), std::string::npos);
        continue;
      }
      EXPECT_EQ(calls, reference);
      for (unsigned solver : {0u, 1u}) {
        std::string prefix = solver ? "[kinsol-call-statistics]"
                                    : "[ida-call-statistics]";
        size_t begin = output.find(prefix);
        ASSERT_NE(begin, std::string::npos);
        std::string row = output.substr(begin, output.find('\n', begin)-begin);
        auto count = [&](const char *label) {
          size_t offset = row.find(label);
          EXPECT_NE(offset, std::string::npos);
          return offset == std::string::npos ? 0ul
              : std::stoul(row.substr(offset+std::string(label).size()));
        };
        EXPECT_EQ(count("scalar_residual_calls="), calls[solver*2]);
        EXPECT_EQ(count("scalar_jacobian_calls="), calls[solver*2+1]);
        if (!solver) {
          EXPECT_EQ(count("IDASolve_calls="), 1ul);
          EXPECT_EQ(count("schur_scalar_jacobian_calls="), 0ul);
          EXPECT_EQ(count("candidate_scalar_residual_calls="), 0ul);
          EXPECT_EQ(count("probe_extra_scalar_residual_calls="), 0ul);
          EXPECT_NE(row.find("scalar_scope=registered_equation_iteration"),
                    std::string::npos);
        }
      }
    }
  }
}

TEST(SolverSingleWorkerTest, DistinctInstancesCanRunOnDistinctCallers) {
  OptionsGuard options;
  KINSOLInstance a, b;
  prepare<0>(a);
  prepare<1>(b);
  ASSERT_TRUE(a.initialize());
  ASSERT_TRUE(b.initialize());
  bool okA = false, okB = false;
  std::thread first([&] { callerToken = 17; okA = a.solve(2.5); });
  std::thread second([&] { callerToken = 17; okB = b.solve(3.5); });
  first.join();
  second.join();
  ASSERT_TRUE(okA && okB);
  EXPECT_NEAR(slots[0].value, 2.5, 1e-10);
  EXPECT_NEAR(slots[1].value, 3.5, 1e-10);
  expectCaller(0);
  expectCaller(1);
}

TEST(SolverSingleWorkerTest, CallbackCanSolveAnotherInstance) {
  OptionsGuard options;
  KINSOLInstance outer, nested;
  prepare<0>(outer);
  prepare<2>(nested);
  ASSERT_TRUE(outer.initialize());
  ASSERT_TRUE(nested.initialize());
  inner = &nested;
  nest = true;
  ASSERT_TRUE(outer.solve(2.5));
  EXPECT_TRUE(nestedPassed);
  expectCaller(0);
  expectCaller(2);
}

TEST(SolverSingleWorkerTest, MultiWorkerCallbacksKeepWorkerThreads) {
  OptionsGuard options(2);
  KINSOLInstance instance;
  prepare<0>(instance);
  ASSERT_TRUE(instance.solve(2.5));
  EXPECT_NEAR(slots[0].value, 2.5, 1e-10);
#ifdef THREADS_ENABLE
  EXPECT_EQ(slots[0].callerCalls, 0u);
#else
  expectCaller(0);
#endif
}

TEST(SolverSingleWorkerTest, EmptyTasksAndDestruction) {
  OptionsGuard options;
  { marco::runtime::ThreadPool empty; empty.wait(); }
  { KINSOLInstance empty; }
  { IDAInstance empty; }
  EXPECT_EQ(slots[0].residuals, 0u);
}

TEST(SolverSingleWorkerTest, StatisticsReportTheActualCallerThread) {
  OptionsGuard options;
  auto &simulation = marco::runtime::simulation::getOptions();
  const bool savedDebug = simulation.debug;
  simulation.debug = true;
  testing::internal::CaptureStderr();
  bool solved = false;
  {
    KINSOLInstance instance;
    prepare<0>(instance);
    solved = instance.solve(2.5);
  }
  std::string output = testing::internal::GetCapturedStderr();
  simulation.debug = savedDebug;
  ASSERT_TRUE(solved);
  std::ostringstream thread;
  thread << std::this_thread::get_id();
  EXPECT_NE(output.find("worker 0 (thread " + thread.str() + ")"), std::string::npos);
  EXPECT_NE(output.find("scalar equations=1"), std::string::npos);
  expectCaller(0);
}

TEST(SolverSingleWorkerTest, ExceptionKeepsBuildSpecificBehavior) {
#ifdef THREADS_ENABLE
  EXPECT_EXIT({
    std::set_terminate([] { std::_Exit(23); });
    OptionsGuard options;
    KINSOLInstance instance;
    prepare<0>(instance);
    throwResidual = true;
    instance.solve(2.5);
  }, testing::ExitedWithCode(23), "");
#else
  OptionsGuard options;
  KINSOLInstance instance;
  prepare<0>(instance);
  throwResidual = true;
  EXPECT_THROW(instance.solve(2.5), std::runtime_error);
#endif
}

TEST(SolverSingleWorkerTest, IDAExceptionKeepsBuildSpecificBehavior) {
#ifdef THREADS_ENABLE
  EXPECT_EXIT({
    std::set_terminate([] { std::_Exit(23); });
    OptionsGuard options;
    IDAInstance instance;
    prepare<0>(instance);
    throwResidual = true;
    instance.calcIC();
  }, testing::ExitedWithCode(23), "");
#else
  OptionsGuard options;
  IDAInstance instance;
  prepare<0>(instance);
  throwResidual = true;
  EXPECT_THROW(instance.calcIC(), std::runtime_error);
#endif
}
