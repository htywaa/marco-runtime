#include "marco/Runtime/Solvers/KINSOL/ScalarBlockPlan.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <random>

using Plan = marco::runtime::sundials::kinsol::ScalarBlockPlan;

TEST(ScalarBlockPlanTest, PreservesPairingAndSourceQuadraticBranchObligation) {
  std::vector<Plan::Row> rows{
      {2, 4, true, {{0, 4}}},
      {0, 5, true, {{1, 0}, {4, 0}}},
      {1, 3, true, {{1, 0}, {2, 0}}}};
  Plan::Budget budget;
  auto [status, plan] = Plan::build(rows, budget);
  ASSERT_EQ(status, Plan::Status::Ready);
  ASSERT_NE(plan, nullptr);
  ASSERT_EQ(plan->getBlocks().size(), 3u);
  EXPECT_EQ(plan->getBlocks()[0].kind, Plan::Kind::SourceQuadratic);
  for (size_t i = 0; i < rows.size(); ++i) {
    EXPECT_EQ(plan->getBlocks()[i].rows, std::vector<uint32_t>({uint32_t(i)}));
    EXPECT_EQ(plan->getBlocks()[i].columns,
              std::vector<uint32_t>({rows[i].matchedColumn}));
  }
}

TEST(ScalarBlockPlanTest, AffineBlockMayDependNonlinearlyOnPredecessors) {
  std::vector<Plan::Row> rows{
      {0, 1, true, {{0, 1}}},
      {1, 7, true, {{3, 0}, {4, 1}, {0, 1}}},
      {2, 6, true, {{2, 0}, {4, 0}}}};
  Plan::Budget budget;
  auto [status, plan] = Plan::build(rows, budget);
  ASSERT_EQ(status, Plan::Status::Ready);
  ASSERT_EQ(plan->getBlocks().size(), 2u);
  EXPECT_EQ(plan->getBlocks()[1].kind, Plan::Kind::Affine);
  EXPECT_EQ(plan->getBlocks()[1].rows, (std::vector<uint32_t>{1, 2}));
}

TEST(ScalarBlockPlanTest, ExplicitProofOnlySelectsUniqueAffineSingletons) {
  Plan::Budget budget;
  std::vector<Plan::Row> rows{{0, 1, true, {{1, 0}}, true},
                             {1, 3, true, {{1, 0}, {2, 0}}, true}};
  auto first = Plan::build(rows, budget);
  ASSERT_EQ(first.first, Plan::Status::Ready);
  ASSERT_EQ(first.second->getExecutionBlocks().size(), 2u);
  for (const auto &block : first.second->getExecutionBlocks())
    EXPECT_EQ(block.kind, Plan::Kind::Explicit);
  // A cycle requires the existing joint linear solve, even if each isolated
  // expression has an individually unique value for fixed predecessors.
  rows[0].reads = 3;
  rows[0].terms.push_back({2, 0});
  auto cycle = Plan::build(rows, budget);
  ASSERT_EQ(cycle.first, Plan::Status::Ready);
  ASSERT_EQ(cycle.second->getExecutionBlocks().size(), 1u);
  EXPECT_EQ(cycle.second->getExecutionBlocks()[0].kind, Plan::Kind::Affine);
  rows[0].terms[0] = {0, 1};
  EXPECT_EQ(Plan::build(rows, budget).first, Plan::Status::InvalidInput);
}

TEST(ScalarBlockPlanTest, CoalescingKeepsPairingAndProvesUnionAffine) {
  Plan::Budget budget;
  std::vector<Plan::Row> rows{
      {2, 4, true, {{0, 4}}},
      {0, 5, true, {{1, 0}, {4, 0}}},
      {1, 3, true, {{1, 0}, {2, 0}}}};
  auto [status, plan] = Plan::build(rows, budget);
  ASSERT_EQ(status, Plan::Status::Ready);
  ASSERT_EQ(plan->getBlocks().size(), 3u);
  ASSERT_EQ(plan->getExecutionBlocks().size(), 2u);
  EXPECT_EQ(plan->getExecutionBlocks()[0].kind, Plan::Kind::SourceQuadratic);
  EXPECT_EQ(plan->getExecutionBlocks()[1].rows, (std::vector<uint32_t>{1, 2}));
  EXPECT_EQ(plan->getExecutionBlocks()[1].columns, (std::vector<uint32_t>{0, 1}));

  // The final equation is still affine in its own unknown, but squaring the
  // preceding unknown forbids combining the two into a linear solve.
  rows[2].terms = {{0, 1}, {2, 0}};
  auto nonlinearUnion = Plan::build(rows, budget);
  ASSERT_EQ(nonlinearUnion.first, Plan::Status::Ready);
  EXPECT_EQ(nonlinearUnion.second->getExecutionBlocks().size(), 3u);
  rows[2].terms = {{3, 0}};
  auto variableCoefficient = Plan::build(rows, budget);
  ASSERT_EQ(variableCoefficient.first, Plan::Status::Ready);
  EXPECT_EQ(variableCoefficient.second->getExecutionBlocks().size(), 3u);
}

TEST(ScalarBlockPlanTest, NonlinearCouplingOrUpstreamLocalKeepsWholeSolver) {
  for (auto rows : {std::vector<Plan::Row>{{0, 3, true, {{3, 0}}},
                                         {1, 3, true, {{1, 0}, {2, 0}}}},
                    std::vector<Plan::Row>{{0, 1, true, {{1, 0}}},
                                         {1, 3, true, {{1, 0}, {0, 2}}}}}) {
    Plan::Budget budget;
    auto [status, plan] = Plan::build(rows, budget);
    EXPECT_EQ(status, Plan::Status::Unavailable);
    EXPECT_EQ(plan, nullptr);
  }
}

TEST(ScalarBlockPlanTest, InvalidCertificatesNeverPublishPartialBlocks) {
  const std::vector<Plan::Row> valid{{0, 1, true, {{1, 0}}},
                                     {1, 3, true, {{1, 0}, {2, 0}}}};
  for (unsigned which = 0; which < 5; ++which) {
    auto rows = valid;
    if (which == 0) rows[1].matchedColumn = 0;
    if (which == 1) rows[1].reads |= 4;
    if (which == 2) rows[1].terms.pop_back();
    if (which == 3) rows[1].terms.push_back({1, 1});
    if (which == 4) rows[1].matchedColumn = 2;
    Plan::Budget budget;
    auto [status, plan] = Plan::build(rows, budget);
    EXPECT_EQ(status, Plan::Status::InvalidInput) << which;
    EXPECT_EQ(plan, nullptr);
  }
  auto rows = valid;
  rows[1].polynomialCertified = false;
  Plan::Budget budget;
  EXPECT_EQ(Plan::build(rows, budget).first, Plan::Status::Unavailable);
}

TEST(ScalarBlockPlanTest, CumulativeBudgetAndEmptyOrLargeInputs) {
  const std::vector<Plan::Row> rows{{0, 1, true, {{1, 0}}}};
  for (unsigned which = 0; which < 3; ++which) {
    Plan::Budget budget;
    if (which == 0) budget.work = 0;
    if (which == 1) budget.storage = 0;
    if (which == 2) budget.partitions = 0;
    auto [status, plan] = Plan::build(rows, budget);
    EXPECT_EQ(status, Plan::Status::BudgetExhausted);
    EXPECT_EQ(plan, nullptr);
  }
  Plan::Budget budget;
  auto first = Plan::build(rows, budget);
  ASSERT_EQ(first.first, Plan::Status::Ready);
  auto remaining = budget;
  auto second = Plan::build(rows, budget);
  ASSERT_EQ(second.first, Plan::Status::Ready);
  EXPECT_LT(budget.work, remaining.work);
  EXPECT_LT(budget.storage, remaining.storage);
  EXPECT_LT(budget.partitions, remaining.partitions);
  EXPECT_EQ(Plan::build({}, budget).first, Plan::Status::Unavailable);
  EXPECT_EQ(Plan::build(std::vector<Plan::Row>(65), budget).first,
            Plan::Status::Unavailable);
}

TEST(ScalarBlockPlanTest, SmallGraphsMatchIndependentReachabilityReference) {
  std::mt19937 rng(5417);
  for (unsigned sample = 0; sample < 256; ++sample) {
    const size_t n = 1 + rng() % 6;
    std::vector<Plan::Row> rows(n);
    std::vector<uint32_t> matching(n);
    for (size_t i = 0; i < n; ++i) matching[i] = i;
    std::shuffle(matching.begin(), matching.end(), rng);
    std::vector<std::vector<size_t>> edges(n);
    for (size_t i = 0; i < n; ++i) {
      rows[i] = {matching[i], uint64_t(1) << matching[i], true, {}};
      for (size_t writer = 0; writer < n; ++writer)
        if (i == writer || rng() % 2) {
          rows[i].reads |= uint64_t(1) << matching[writer];
          edges[i].push_back(writer);
        }
      for (size_t column = 0; column < n; ++column)
        if (rows[i].reads & (uint64_t(1) << column))
          rows[i].terms.push_back({uint64_t(1) << column, 0});
    }
    std::vector<std::vector<bool>> reach(n, std::vector<bool>(n));
    for (size_t start = 0; start < n; ++start) {
      std::vector<size_t> queue{start};
      reach[start][start] = true;
      for (size_t at = 0; at < queue.size(); ++at)
        for (auto next : edges[queue[at]])
          if (!reach[start][next]) {
            reach[start][next] = true;
            queue.push_back(next);
          }
    }
    Plan::Budget budget;
    auto [status, plan] = Plan::build(rows, budget);
    ASSERT_EQ(status, Plan::Status::Ready) << sample;
    std::vector<size_t> owner(n, n);
    for (size_t b = 0; b < plan->getBlocks().size(); ++b) {
      const auto &block = plan->getBlocks()[b];
      ASSERT_EQ(block.kind, Plan::Kind::Affine);
      for (size_t i = 0; i < block.rows.size(); ++i) {
        ASSERT_EQ(owner[block.rows[i]], n);
        owner[block.rows[i]] = b;
        EXPECT_EQ(block.columns[i], matching[block.rows[i]]);
      }
    }
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j) {
        EXPECT_EQ(owner[i] == owner[j], reach[i][j] && reach[j][i]);
        if (reach[i][j]) {
          EXPECT_GE(owner[i], owner[j]);
        }
      }
  }
}
