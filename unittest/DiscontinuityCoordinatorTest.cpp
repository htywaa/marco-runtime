#include "marco/Runtime/Solvers/SUNDIALS/DiscontinuityCoordinator.h"
#include "gtest/gtest.h"
#include <string>

using namespace marco::runtime::sundials;

// 中文：验证 superdense 事务按 configuration cycle 终止，而不是依赖固定次数
// 上限，并确认拒绝记录只在当前物理时刻有效。
// English: Verify that superdense transactions terminate on configuration
// cycles rather than a fixed count, and that rejection records are scoped to
// the current physical time.
TEST(DiscontinuityCoordinatorTest, AllowsMoreThanSixteenAcyclicTransitions) {
  SolverDiscontinuityCoordinator coordinator;
  coordinator.beginPhysicalTime(1.0, "state=0");
  for (uint64_t state = 0; state < 32; ++state) {
    SolverDiscontinuityTransaction transaction{
        SolverDiscontinuityKind::ConstraintBasisSwitch, 0, state, state + 1,
        "condition"};
    EXPECT_EQ(coordinator.commit(transaction,
                                 "state=" + std::to_string(state + 1)),
              SolverDiscontinuityCoordinator::CommitResult::Committed);
  }
  EXPECT_EQ(coordinator.getSuperdenseIndex(), 32u);
}

TEST(DiscontinuityCoordinatorTest, DetectsSameTimeConfigurationCycle) {
  SolverDiscontinuityCoordinator coordinator;
  coordinator.beginPhysicalTime(1.0, "state=A");
  EXPECT_EQ(coordinator.commit(
                {SolverDiscontinuityKind::ConstraintBasisSwitch, 0, 0, 1,
                 "condition"},
                "state=B"),
            SolverDiscontinuityCoordinator::CommitResult::Committed);
  EXPECT_EQ(coordinator.commit(
                {SolverDiscontinuityKind::ModelEvent, 7, 1, 0, "when"},
                "state=A"),
            SolverDiscontinuityCoordinator::CommitResult::ConfigurationCycle);
}

TEST(DiscontinuityCoordinatorTest, RejectionsAreScopedToPhysicalTime) {
  SolverDiscontinuityCoordinator coordinator;
  SolverDiscontinuityTransaction transaction{
      SolverDiscontinuityKind::ConstraintBasisSwitch, 0, 0, 1, "condition"};
  coordinator.beginPhysicalTime(1.0, "state=A");
  coordinator.reject(transaction);
  EXPECT_TRUE(coordinator.isRejected(transaction));

  coordinator.beginPhysicalTime(2.0, "state=A");
  EXPECT_FALSE(coordinator.isRejected(transaction));
}

TEST(DiscontinuityCoordinatorTest,
     BasisAndEventTransactionsShareSuperdenseSequence) {
  SolverDiscontinuityCoordinator coordinator;
  coordinator.beginPhysicalTime(3.0, "basis=0,event=0");
  EXPECT_EQ(coordinator.commit(
                {SolverDiscontinuityKind::ModelEvent, 2, 0, 1, "when"},
                "basis=0,event=1"),
            SolverDiscontinuityCoordinator::CommitResult::Committed);
  EXPECT_EQ(coordinator.commit(
                {SolverDiscontinuityKind::ConstraintBasisSwitch, 4, 0, 1,
                 "condition"},
                "basis=1,event=1"),
            SolverDiscontinuityCoordinator::CommitResult::Committed);
  EXPECT_EQ(coordinator.getSuperdenseIndex(), 2u);
}
