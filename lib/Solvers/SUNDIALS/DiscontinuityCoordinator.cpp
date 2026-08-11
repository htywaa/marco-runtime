#include "marco/Runtime/Solvers/SUNDIALS/DiscontinuityCoordinator.h"
#include <tuple>

namespace marco::runtime::sundials {

// 中文：transaction 的稳定全序使同一物理时刻的拒绝/提交集合具有确定表示。
// English: A stable total order gives rejected and committed transaction sets
// a deterministic representation at one physical time.
bool SolverDiscontinuityTransaction::operator<(
    const SolverDiscontinuityTransaction &other) const {
  return std::tie(kind, domain, fromState, toState, cause) <
         std::tie(other.kind, other.domain, other.fromState, other.toState,
                  other.cause);
}

void SolverDiscontinuityCoordinator::beginPhysicalTime(
    double time, const std::string &acceptedConfiguration) {
  // 中文：物理时间推进时重置 superdense 事务历史；同一时间内保留所有已访问
  // configuration，以检测事件/状态基切换循环，而不是依赖固定切换次数。
  // English: Advancing physical time resets superdense transaction history.
  // Within one time instant all visited configurations are retained to detect
  // event/state-basis cycles without relying on a fixed switch count.
  if (!active || time != physicalTime) {
    active = true;
    physicalTime = time;
    superdenseIndex = 0;
    visitedConfigurations.clear();
    rejectedTransactions.clear();
    committedTransactions.clear();
  }
  if (visitedConfigurations.empty()) {
    visitedConfigurations.insert(acceptedConfiguration);
  }
}

bool SolverDiscontinuityCoordinator::isRejected(
    const SolverDiscontinuityTransaction &transaction) const {
  // 中文：拒绝集合按完整事务身份查询；同一 domain 的另一条候选边仍可尝试。
  // English: Rejection is keyed by the complete transaction identity, so a
  // different candidate edge in the same domain remains eligible.
  return rejectedTransactions.count(transaction) != 0;
}

void SolverDiscontinuityCoordinator::reject(
    const SolverDiscontinuityTransaction &transaction) {
  rejectedTransactions.insert(transaction);
}

bool SolverDiscontinuityCoordinator::wouldRepeatConfiguration(
    const std::string &configuration) const {
  return visitedConfigurations.count(configuration) != 0;
}

SolverDiscontinuityCoordinator::CommitResult
SolverDiscontinuityCoordinator::commit(
    const SolverDiscontinuityTransaction &transaction,
    const std::string &configuration) {
  // 中文：只有新 configuration 才能提交并推进 superdense index；循环候选由
  // 调用方回滚到 accepted checkpoint。
  // English: Only a new configuration may commit and advance the superdense
  // index. Cyclic candidates are rolled back to the accepted checkpoint by the
  // caller.
  if (wouldRepeatConfiguration(configuration)) {
    return CommitResult::ConfigurationCycle;
  }
  committedTransactions.insert(transaction);
  visitedConfigurations.insert(configuration);
  ++superdenseIndex;
  return CommitResult::Committed;
}

} // namespace marco::runtime::sundials
