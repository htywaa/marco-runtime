#include "marco/Runtime/Solvers/IDA/Instance.h"
#include "gtest/gtest.h"
#include <cmath>
#include <limits>

using namespace marco::runtime::sundials::ida;

// 中文：覆盖局部 chart condition 的秩阈值、硬切换边界及重启恢复策略。
// English: Cover local-chart rank thresholds, the hard switch boundary, and
// IDA restart-recovery policy.
TEST(IDAConstraintConditionTest, WellConditionedMatrix) {
  ConstraintConditionEstimate estimate =
      estimateConstraintCondition({2.0, 0.0, 0.0, 0.5}, 2);

  EXPECT_EQ(estimate.rank, 2);
  EXPECT_FALSE(estimate.rankDeficient);
  EXPECT_DOUBLE_EQ(estimate.minimumPivot, 0.5);
  EXPECT_DOUBLE_EQ(estimate.maximumPivot, 2.0);
  EXPECT_DOUBLE_EQ(estimate.oneNormCondition, 4.0);
}

TEST(IDAConstraintConditionTest, RankDeficientMatrix) {
  ConstraintConditionEstimate estimate =
      estimateConstraintCondition({1.0, 2.0, 2.0, 4.0}, 2);

  EXPECT_EQ(estimate.rank, 1);
  EXPECT_TRUE(estimate.rankDeficient);
  EXPECT_TRUE(std::isinf(estimate.oneNormCondition));
}

TEST(IDAConstraintConditionTest, RejectsPivotBelowScaledThreshold) {
  double epsilon = std::numeric_limits<double>::epsilon();
  ConstraintConditionEstimate estimate =
      estimateConstraintCondition({1.0, 0.0, 0.0, 50.0 * epsilon}, 2);

  EXPECT_EQ(estimate.rank, 1);
  EXPECT_TRUE(estimate.rankDeficient);
}

TEST(IDAConstraintConditionTest, RejectsHardConditionBoundary) {
  ConstraintConditionEstimate below =
      estimateConstraintCondition({1.0, 0.0, 0.0, 2e-8}, 2);
  ConstraintConditionEstimate boundary =
      estimateConstraintCondition({1.0, 0.0, 0.0, 1e-8}, 2);

  EXPECT_FALSE(requiresDynamicStateSelection(below, false));
  EXPECT_TRUE(requiresDynamicStateSelection(boundary, false));
}

TEST(IDAConstraintConditionTest, EscalatesIllConditionedFailedTrial) {
  ConstraintConditionEstimate estimate =
      estimateConstraintCondition({1.0, 0.0, 0.0, 2e-8}, 2);

  EXPECT_FALSE(requiresDynamicStateSelection(estimate, false));
  EXPECT_TRUE(requiresDynamicStateSelection(estimate, true));
}

TEST(IDAConstraintConditionTest, RequestsSwitchAtSafeChartHysteresis) {
  ConstraintConditionEstimate below =
      estimateConstraintCondition({1.0, 0.0, 0.0, 1.0001e-3}, 2);
  ConstraintConditionEstimate boundary =
      estimateConstraintCondition({1.0, 0.0, 0.0, 1e-3}, 2);

  EXPECT_FALSE(shouldRequestConstraintStateSetSwitch(below, false));
  EXPECT_TRUE(shouldRequestConstraintStateSetSwitch(boundary, false));
}

TEST(IDAConstraintConditionTest, BasisHealthIncludesLocalClosureCondition) {
  ConstraintBasisHealthCertificate certificate;
  certificate.chart.oneNormCondition = 2;
  certificate.localClosure.oneNormCondition = 1e4;

  EXPECT_EQ(classifyConstraintBasisHealth(certificate, false),
            ConstraintBasisHealthAction::RequestSwitch);
}

TEST(IDAConstraintConditionTest, BasisHealthIncludesDomainResponse) {
  ConstraintBasisHealthCertificate certificate;
  certificate.chart.oneNormCondition = 2;
  certificate.localClosure.oneNormCondition = 3;
  certificate.responseAvailable = true;
  certificate.responseAmplification = 1e4;

  EXPECT_EQ(classifyConstraintBasisHealth(certificate, false),
            ConstraintBasisHealthAction::RequestSwitch);
  certificate.responseAmplification = 1e8;
  EXPECT_EQ(classifyConstraintBasisHealth(certificate, false),
            ConstraintBasisHealthAction::Unsafe);
}

TEST(IDAConstraintConditionTest, MissingResponseDoesNotInvalidateChart) {
  ConstraintBasisHealthCertificate certificate;
  certificate.chart.oneNormCondition = 2;
  certificate.localClosure.oneNormCondition = 3;

  EXPECT_EQ(classifyConstraintBasisHealth(certificate, false),
            ConstraintBasisHealthAction::Accept);
}

TEST(IDAConstraintConditionTest, AcceptsSafeStateSetBoundary) {
  EXPECT_TRUE(isSafeConstraintStateSetCandidate(true, 1e4, 1e-10, 1e-9));
}

TEST(IDAConstraintConditionTest, RejectsUnsafeStateSetCandidates) {
  EXPECT_FALSE(isSafeConstraintStateSetCandidate(false, 1.0, 0.0, 0.0));
  EXPECT_FALSE(isSafeConstraintStateSetCandidate(true, 1e4 + 1.0, 0.0, 0.0));
  EXPECT_FALSE(isSafeConstraintStateSetCandidate(true, 1.0, 1.1e-10, 0.0));
  EXPECT_FALSE(isSafeConstraintStateSetCandidate(true, 1.0, 0.0, 1.1e-9));
  EXPECT_FALSE(isSafeConstraintStateSetCandidate(
      true, std::numeric_limits<double>::infinity(), 0.0, 0.0));
}

TEST(IDAConstraintConditionTest,
     RequiresCompleteMergedDomainResidualCertificate) {
  ConstraintResidualRoleCounts domainExpected{/*position=*/2,
                                               /*tangent=*/2,
                                               /*highest=*/2,
                                               /*support=*/4};
  ConstraintResidualRoleCounts complete{/*position=*/2,
                                        /*tangent=*/2,
                                        /*highest=*/2,
                                        /*support=*/4};
  ConstraintResidualRoleCounts oneSemanticComponent{/*position=*/1,
                                                     /*tangent=*/1,
                                                     /*highest=*/1,
                                                     /*support=*/2};
  ConstraintResidualRoleCounts roleMismatch{/*position=*/2,
                                            /*tangent=*/2,
                                            /*highest=*/1,
                                            /*support=*/5};

  EXPECT_TRUE(hasCompleteConstraintResidualCertificate(domainExpected,
                                                       complete));
  EXPECT_FALSE(hasCompleteConstraintResidualCertificate(
      domainExpected, oneSemanticComponent));
  EXPECT_FALSE(hasCompleteConstraintResidualCertificate(domainExpected,
                                                        roleMismatch));
}

TEST(IDAConstraintConditionTest,
     KeepsExecutionDomainAndSemanticConditionsDisjoint) {
  ConstraintSwitchScope domain{ConstraintSwitchScopeKind::ExecutionDomain, 4};
  ConstraintSwitchScope semantic{
      ConstraintSwitchScopeKind::SemanticComponent, 4};
  std::vector<ConstraintSwitchHealthSample> samples = {
      {domain, 3.0}, {semantic, 100.0}, {domain, 7.0}};

  EXPECT_DOUBLE_EQ(
      getMaximumConstraintSwitchScopeCondition(samples, domain).value(), 7.0);
  EXPECT_DOUBLE_EQ(
      getMaximumConstraintSwitchScopeCondition(samples, semantic).value(),
      100.0);
  EXPECT_FALSE(getMaximumConstraintSwitchScopeCondition(
                   samples,
                   {ConstraintSwitchScopeKind::ExecutionDomain, 5})
                   .has_value());
}

TEST(IDAConstraintConditionTest, SelectsSafeEpochByStateSelectCost) {
  std::vector<ConstraintExecutionEpochCandidate> candidates = {
      {4, 1, 0, 9, 1.0, true},
      {3, 0, 2, 0, 4.0, true},
      {2, 0, 1, 0, 9.0, true},
      {1, 0, 0, 0, 20.0, false},
  };

  EXPECT_EQ(selectConstraintExecutionEpoch(candidates), 2);
}

TEST(IDAConstraintConditionTest, UsesAvoidThenConditionThenStableId) {
  std::vector<ConstraintExecutionEpochCandidate> candidates = {
      {5, 0, 0, 1, 2.0, true},
      {4, 0, 0, 2, 8.0, true},
      {3, 0, 0, 2, 1.0, true},
      {2, 0, 0, 2, 1.0, true},
  };

  EXPECT_EQ(selectConstraintExecutionEpoch(candidates), 2);
}

TEST(IDAConstraintConditionTest, RejectsUnsafeEpochSet) {
  std::vector<ConstraintExecutionEpochCandidate> candidates = {
      {0, 0, 0, 0, 1.0, false},
      {1, 0, 0, 0, 2.0, false},
  };

  EXPECT_EQ(selectConstraintExecutionEpoch(candidates), std::nullopt);
}

TEST(IDAConstraintConditionTest, SuppressesOnlyExpectedResidualFailures) {
  EXPECT_TRUE(shouldSuppressIDAError(IDA_RES_FAIL, true, false));
  EXPECT_TRUE(shouldSuppressIDAError(IDA_REP_RES_ERR, false, true));
  EXPECT_FALSE(shouldSuppressIDAError(IDA_RES_FAIL, false, false));
  EXPECT_FALSE(shouldSuppressIDAError(IDA_REP_RES_ERR, false, false));
  EXPECT_FALSE(shouldSuppressIDAError(IDA_CONV_FAIL, true, true));
  EXPECT_FALSE(shouldSuppressIDAError(IDA_LSETUP_FAIL, true, true));
}

TEST(IDAConstraintConditionTest, RestartRecoveryUsesAcceptedStepAndRamps) {
  IDARestartRecoveryState state = createIDARestartRecoveryState(
      /*acceptedStep=*/0.08, /*outputStep=*/0.01,
      /*configuredInitialStep=*/0, /*configuredMinimumStep=*/0,
      /*configuredMaximumStep=*/0);
  ASSERT_TRUE(state.active);
  EXPECT_DOUBLE_EQ(state.initialStep, 0.02);
  EXPECT_DOUBLE_EQ(state.currentStepLimit, 0.04);
  EXPECT_EQ(state.currentMaximumOrder, 1);

  EXPECT_FALSE(advanceIDARestartRecoveryState(state));
  EXPECT_DOUBLE_EQ(state.currentStepLimit, 0.08);
  EXPECT_EQ(state.currentMaximumOrder, 2);
  EXPECT_FALSE(advanceIDARestartRecoveryState(state));
  EXPECT_FALSE(advanceIDARestartRecoveryState(state));
  EXPECT_TRUE(advanceIDARestartRecoveryState(state));
  EXPECT_FALSE(state.active);
  EXPECT_EQ(state.currentMaximumOrder, 5);
}

TEST(IDAConstraintConditionTest, RestartRecoveryHonorsConfiguredBounds) {
  IDARestartRecoveryState state = createIDARestartRecoveryState(
      /*acceptedStep=*/1.0, /*outputStep=*/0.1,
      /*configuredInitialStep=*/0, /*configuredMinimumStep=*/0.05,
      /*configuredMaximumStep=*/0.1);
  ASSERT_TRUE(state.active);
  EXPECT_DOUBLE_EQ(state.initialStep, 0.05);
  EXPECT_DOUBLE_EQ(state.targetStep, 0.1);
  EXPECT_DOUBLE_EQ(state.currentStepLimit, 0.1);
}
