#ifndef MARCO_RUNTIME_SOLVERS_SUNDIALS_DISCONTINUITYCOORDINATOR_H
#define MARCO_RUNTIME_SOLVERS_SUNDIALS_DISCONTINUITYCOORDINATOR_H

#include <cstdint>
#include <set>
#include <string>

namespace marco::runtime::sundials {

/// 中文：Modelica 事件与 constraint-basis switch 共用同一 superdense-time
/// transaction 类型，但各自保留原因与目标配置。
/// English: Modelica events and constraint-basis switches share one
/// superdense-time transaction type while retaining distinct causes and target
/// configurations.
enum class SolverDiscontinuityKind {
  ConstraintBasisSwitch,
  ModelEvent
};

struct SolverDiscontinuityTransaction {
  /// 中文：domain 标识可独立切换的 component/event 域；from/to 与 cause 一起
  /// 构成同一物理时刻内稳定、可去重的事务身份。
  /// English: domain identifies an independently switchable component/event
  /// domain. Together with from/to and cause it forms a stable, deduplicated
  /// transaction identity at one physical time.
  SolverDiscontinuityKind kind{SolverDiscontinuityKind::ConstraintBasisSwitch};
  uint64_t domain{0};
  uint64_t fromState{0};
  uint64_t toState{0};
  std::string cause;

  bool operator<(const SolverDiscontinuityTransaction &other) const;
};

/// 中文：协调同一 physical time 上的配置访问、拒绝与提交，使用 visited
/// configurations 检测循环而不是固定限制切换次数。
/// English: Coordinate configuration visits, rejections, and commits at one
/// physical time, detecting cycles through visited configurations instead of
/// a fixed switch-count limit.
class SolverDiscontinuityCoordinator {
public:
  enum class CommitResult { Committed, ConfigurationCycle };

  void beginPhysicalTime(double time,
                         const std::string &acceptedConfiguration);

  /// 中文：reject 记录失败边；commit 只接受未访问过的完整配置并推进 superdense
  /// index，因此循环检测不依赖任意固定切换次数上限。
  /// English: reject records failed edges, while commit accepts only a new
  /// complete configuration and advances the superdense index, making cycle
  /// detection independent of an arbitrary switch-count limit.
  bool isRejected(const SolverDiscontinuityTransaction &transaction) const;
  void reject(const SolverDiscontinuityTransaction &transaction);

  bool wouldRepeatConfiguration(const std::string &configuration) const;
  CommitResult commit(const SolverDiscontinuityTransaction &transaction,
                      const std::string &configuration);

  uint64_t getSuperdenseIndex() const { return superdenseIndex; }
  double getPhysicalTime() const { return physicalTime; }
  uint64_t getCommittedTransactions() const {
    return committedTransactions.size();
  }

private:
  /// 中文：这些集合只在一个 physical time 内有效；时间推进时清空，事件迭代或
  /// basis switch 留在同一时刻时则持续累积。
  /// English: These sets are scoped to one physical time. They reset when time
  /// advances and accumulate across event iterations or basis switches at the
  /// same instant.
  bool active{false};
  double physicalTime{0};
  uint64_t superdenseIndex{0};
  std::set<std::string> visitedConfigurations;
  std::set<SolverDiscontinuityTransaction> rejectedTransactions;
  std::set<SolverDiscontinuityTransaction> committedTransactions;
};

} // namespace marco::runtime::sundials

#endif // MARCO_RUNTIME_SOLVERS_SUNDIALS_DISCONTINUITYCOORDINATOR_H
