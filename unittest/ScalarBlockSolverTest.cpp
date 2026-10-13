#include "marco/Runtime/Solvers/KINSOL/ScalarBlockSolver.h"
#include "gtest/gtest.h"
#include <cmath>
#include <limits>

using Plan = marco::runtime::sundials::kinsol::ScalarBlockPlan;
using Solver = marco::runtime::sundials::kinsol::ScalarBlockSolver;
using Sparse = marco::runtime::sundials::kinsol::SparseNonlinearSystem;

namespace {
std::unique_ptr<const Plan> makePlan(Plan::Budget &budget) {
  std::vector<Plan::Row> rows{
      {0, 1, true, {{0, 1}, {0, 0}}},
      {1, 7, true, {{3, 0}, {4, 1}, {0, 0}}},
      {2, 6, true, {{2, 0}, {4, 0}, {0, 0}}}};
  auto [status, plan] = Plan::build(rows, budget);
  EXPECT_EQ(status, Plan::Status::Ready);
  return std::move(plan);
}

Sparse::Configuration options() { return {3, 6, 1e-12, 1e-14, 100, true, true}; }
double derivative(uint32_t row, uint32_t column, const std::vector<double> &x) {
  if (row == 0) return column == 0 ? 2 * x[0] : 0;
  if (row == 1) {
    if (column == 0) return x[1] + 2 * x[0] * x[2];
    return column == 1 ? x[0] : x[0] * x[0];
  }
  return column == 0 ? 0 : 1;
}
} // namespace

TEST(ScalarBlockSolverTest, BothRootBranchesAndCompleteAffineCoupling) {
  Plan::Budget budget;
  auto plan = makePlan(budget);
  ASSERT_NE(plan, nullptr);
  for (double root : {2.0, -2.0}) {
    std::vector<double> prepared;
    auto solver = Solver::create(*plan, options(),
        [root, &prepared](uint32_t row, const std::vector<double> &x) {
          EXPECT_EQ(prepared, x);
          if (row == 0) return x[0] * x[0] - 4;
          if (row == 1) return x[0] * x[1] + x[0] * x[0] * x[2] -
                               (root * 3 + 4 * 5);
          return x[1] + x[2] - 8;
        }, [&prepared](uint32_t row, uint32_t column,
                       const std::vector<double> &x) {
          EXPECT_EQ(prepared, x);
          return derivative(row, column, x);
        }, budget, [&prepared](const auto &state) { prepared = state; });
    ASSERT_NE(solver, nullptr);
    ASSERT_TRUE(solver->setScaling({1, 0.5, 0.1}, {1, 0.1, 0.2}));
    std::vector<double> x{root * 0.75, 4, 4};
    ASSERT_EQ(solver->solve(0, x), Solver::Result::Solved);
    EXPECT_NEAR(x[0], root, 1e-12);
    EXPECT_NEAR(x[1], 3, 1e-12);
    EXPECT_NEAR(x[2], 5, 1e-12);
  }
}

TEST(ScalarBlockSolverTest, EveryInvocationUsesNewExternalAndLocalState) {
  Plan::Budget budget;
  auto plan = makePlan(budget);
  ASSERT_NE(plan, nullptr);
  double root = 2;
  auto solver = Solver::create(*plan, options(),
      [&](uint32_t row, const std::vector<double> &x) {
        if (row == 0) return x[0] * x[0] - root * root;
        if (row == 1) return x[0] * x[1] + x[0] * x[0] * x[2] -
                             (root * 3 + root * root * 5);
        return x[1] + x[2] - 8;
      }, derivative, budget);
  ASSERT_NE(solver, nullptr);
  for (double value : {2.0, 3.0, -4.0, 2.0}) {
    root = value;
    solver->setLineSearchEnabled(value == 2.0);
    std::vector<double> x{value * 0.9, 0, 0};
    ASSERT_EQ(solver->solve(1, x), Solver::Result::Solved);
    EXPECT_NEAR(x[0], value, 1e-11);
    EXPECT_NEAR(x[1], 3, 1e-11);
    EXPECT_NEAR(x[2], 5, 1e-11);
  }
}

TEST(ScalarBlockSolverTest, SingularBlockOrInvalidTrialRestoresOriginalWarmStart) {
  Plan::Budget budget;
  auto plan = makePlan(budget);
  ASSERT_NE(plan, nullptr);
  double squaredRoot = 1;
  bool fail = false;
  auto solver = Solver::create(*plan, options(),
      [&](uint32_t row, const std::vector<double> &x) {
        if (fail) return std::numeric_limits<double>::quiet_NaN();
        if (row == 0) return x[0] * x[0] - squaredRoot;
        if (row == 1) return x[0] * x[1] + x[0] * x[0] * x[2] - 3;
        return x[1] + x[2] - 2;
      }, derivative, budget);
  ASSERT_NE(solver, nullptr);
  for (unsigned which = 0; which < 3; ++which) {
    fail = which == 1;
    squaredRoot = which == 0 ? 1 : -1;
    std::vector<double> x{1, 5, 6};
    EXPECT_EQ(solver->solve(0, x), Solver::Result::RetainOriginal);
    EXPECT_EQ(x, (std::vector<double>{1, 5, 6}));
  }
  fail = false;
  squaredRoot = 4;
  std::vector<double> x{1.5, 0, 0};
  EXPECT_EQ(solver->solve(0, x), Solver::Result::Solved);
  EXPECT_NEAR(x[0], 2, 1e-12);
}

TEST(ScalarBlockSolverTest, StationaryWarmStartAndBadScalingKeepOriginal) {
  Plan::Budget budget;
  auto plan = makePlan(budget);
  auto solver = Solver::create(*plan, options(),
      [](uint32_t row, const std::vector<double> &x) {
        return row == 0 ? x[0] * x[0] - 4 : x[1] + x[2];
      }, derivative, budget);
  ASSERT_NE(solver, nullptr);
  EXPECT_FALSE(solver->setScaling({1, 0, 1}, {1, 1, 1}));
  EXPECT_FALSE(solver->setScaling({1, 1}, {1, 1}));
  std::vector<double> x{0, 1, 2};
  EXPECT_EQ(solver->solve(0, x), Solver::Result::RetainOriginal);
  EXPECT_EQ(x, (std::vector<double>{0, 1, 2}));
  budget.storage = 0;
  EXPECT_EQ(Solver::create(*plan, options(),
      [](uint32_t, const std::vector<double> &) { return 0; },
      derivative, budget), nullptr);
}

TEST(ScalarBlockSolverTest, ExplicitChainChecksCurrentCoefficientAndWholeResidual) {
  Plan::Budget budget;
  std::vector<Plan::Row> rows{{0, 1, true, {{1, 0}}, true},
                             {1, 3, true, {{1, 0}, {2, 0}}, true}};
  auto [status, plan] = Plan::build(rows, budget);
  ASSERT_EQ(status, Plan::Status::Ready);
  double parameter = 2, coefficient = 3;
  bool badValue = false;
  std::vector<double> prepared;
  auto solver = Solver::create(*plan, {2, 3, 1e-12, 1e-14, 100, true, true},
      [&](uint32_t row, const auto &x) {
        EXPECT_EQ(prepared, x);
        return row == 0 ? x[0] - parameter : coefficient * x[1] - x[0];
      }, [&](uint32_t row, uint32_t column, const auto &x) {
        EXPECT_EQ(prepared, x);
        return row == 0 ? (column == 0 ? 1. : 0.)
                        : (column == 0 ? -1. : coefficient);
      }, budget, [&](const auto &x) { prepared = x; });
  ASSERT_NE(solver, nullptr);
  ASSERT_TRUE(solver->setScaling({1, 2}, {1, 3}));
  auto original = std::vector<double>{11, 12};
  auto x = original;
  EXPECT_EQ(solver->solve(0, x), Solver::Result::RetainOriginal);
  EXPECT_EQ(x, original); // Missing optional callback is never executed.
  solver->setExplicitValues([&](uint32_t row, const auto &state) {
    EXPECT_EQ(prepared, state);
    if (badValue) return 123.; // Finite but inconsistent: final F must reject it.
    return row == 0 ? parameter : state[0] / coefficient;
  });
  for (double p : {2., -7., 1.}) {
    parameter = p;
    x = original;
    ASSERT_EQ(solver->solve(0, x), Solver::Result::Solved);
    EXPECT_DOUBLE_EQ(x[0], p);
    EXPECT_NEAR(x[1], p / coefficient, 1e-14);
  }
  for (int which = 0; which != 4; ++which) {
    coefficient = which == 0 ? 0 : which == 1 ?
        std::numeric_limits<double>::quiet_NaN() : 3;
    badValue = which == 2;
    parameter = which == 3 ? std::numeric_limits<double>::infinity() : 2;
    x = original;
    EXPECT_EQ(solver->solve(0, x), Solver::Result::RetainOriginal);
    EXPECT_EQ(x, original);
  }
}

TEST(ScalarBlockSolverTest, CoalescedAffineChainRetainsWholeResidualAndRecovery) {
  Plan::Budget budget;
  std::vector<Plan::Row> rows{
      {0, 1, true, {{0, 1}}},
      {1, 3, true, {{1, 0}, {2, 0}}},
      {2, 6, true, {{2, 0}, {4, 0}}}};
  auto [status, plan] = Plan::build(rows, budget);
  ASSERT_EQ(status, Plan::Status::Ready);
  ASSERT_EQ(plan->getExecutionBlocks().size(), 2u);
  double coefficient = 2;
  bool invalid = false;
  auto solver = Solver::create(*plan, options(),
      [&](uint32_t row, const std::vector<double> &x) {
        if (invalid) return std::numeric_limits<double>::quiet_NaN();
        if (row == 0) return x[0] * x[0] - 4;
        if (row == 1) return x[1] - 2 * x[0];
        return coefficient * x[2] - 3 * x[1];
      }, [&](uint32_t row, uint32_t column, const auto &x) -> double {
        if (row == 0) return column == 0 ? 2 * x[0] : 0;
        if (row == 1) return column == 0 ? -2 : column == 1 ? 1 : 0;
        return column == 1 ? -3 : column == 2 ? coefficient : 0;
      }, budget);
  ASSERT_NE(solver, nullptr);
  for (double branch : {1.0, -1.0}) {
    std::vector<double> x{1.5 * branch, 0, 0};
    ASSERT_EQ(solver->solve(0, x), Solver::Result::Solved);
    EXPECT_NEAR(x[0], 2 * branch, 1e-12);
    EXPECT_NEAR(x[1], 4 * branch, 1e-12);
    EXPECT_NEAR(x[2], 6 * branch, 1e-12);
  }
  const std::vector<double> original{1.5, 5, 6};
  for (int phase = 0; phase < 3; ++phase) {
    invalid = phase == 0;
    coefficient = phase == 1 ? 0 : 2;
    auto x = original;
    if (phase < 2) {
      EXPECT_EQ(solver->solve(0, x), Solver::Result::RetainOriginal);
      EXPECT_EQ(x, original);
    } else {
      EXPECT_EQ(solver->solve(0, x), Solver::Result::Solved);
      EXPECT_NEAR(x[2], 6, 1e-12);
    }
  }
}
