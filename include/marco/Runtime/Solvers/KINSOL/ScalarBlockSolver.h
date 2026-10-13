#ifndef MARCO_RUNTIME_SOLVERS_KINSOL_SCALARBLOCKSOLVER_H
#define MARCO_RUNTIME_SOLVERS_KINSOL_SCALARBLOCKSOLVER_H
#ifdef SUNDIALS_ENABLE

#include "marco/Runtime/Solvers/KINSOL/ScalarBlockPlan.h"
#include "marco/Runtime/Solvers/KINSOL/SparseNonlinearSystem.h"
#include <functional>
#include <memory>
#include <vector>

namespace marco::runtime::sundials::kinsol {

/// Optional execution of a certified small scalar closure. The callbacks must
/// evaluate the certified pure residual at the complete supplied local state;
/// all external inputs are fixed for this invocation. No numeric value survives
/// into the next invocation. Existing full-closure health checks remain owned
/// by IDA. This object is neither reentrant nor concurrently callable.
// 中文：每次 solve 重新建立当前状态；仅工作区容量跨调用保留，数值结果不跨 trial 缓存。
// English: Each solve establishes a fresh current state; only workspace capacity persists, not
// numerical results across trials.
class ScalarBlockSolver {
public:
  using Residual = std::function<double(uint32_t, const std::vector<double> &)>;
  using Jacobian = std::function<double(uint32_t, uint32_t,
                                        const std::vector<double> &)>;
  using Preparation = std::function<void(const std::vector<double> &)>;
  enum class Result { Solved, RetainOriginal };

  static std::unique_ptr<ScalarBlockSolver> create(
      const ScalarBlockPlan &plan,
      SparseNonlinearSystem::Configuration configuration,
      Residual residual, Jacobian jacobian, ScalarBlockPlan::Budget &budget,
      Preparation preparation = {});
  bool setScaling(const std::vector<double> &variables,
                  const std::vector<double> &residuals);
  void setLineSearchEnabled(bool enabled);
  void setExplicitValues(Residual callback) { explicitValues = std::move(callback); }

  /// 中文：数值证明未命中时恢复输入向量，并请求调用方从相同 warm start 执行原整体闭包。
  ///       调用方须先恢复模型存储；成功输出已通过完整缩放残差及根分支守卫。
  /// On a missed numeric proof, restores the input vector and asks the caller
  /// to execute the original whole closure from that same warm start. The
  /// caller must restore its model storage before doing so. Successful output
  /// has passed the full scaled residual test as well as the root-branch guard.
  Result solve(double time, std::vector<double> &values);
  uint64_t getLastNonlinearIterations() const { return lastIterations; }

private:
  friend class KINSOLInstance;
  ScalarBlockSolver(const ScalarBlockPlan &plan,
                    SparseNonlinearSystem::Configuration configuration,
                    Residual residual, Jacobian jacobian,
                    Preparation preparation);
  struct Block {
    const ScalarBlockPlan::Block *plan;
    std::unique_ptr<SparseNonlinearSystem> solver;
    std::vector<double> residual, rhs, solution;
  };
  void install(Block &block, N_Vector variables);
  const ScalarBlockPlan &plan;
  SparseNonlinearSystem::Configuration configuration;
  Residual residual;
  Residual explicitValues;
  Jacobian jacobian;
  Preparation preparation;
  std::vector<Block> blocks;
  // 中文：只保留容量；结果仅能由所属 facade 在紧接着的成功点检查中读取。
  //       下一次求解先替换每个条目，不能继承旧读取权威。
  // Retained capacity only. A result is readable by the owning facade solely
  // during its immediate successful-point inspection; the next solve replaces
  // every entry before it can acquire that authority again.
  std::vector<double> state, initial, residualScale, finalResidual;
  uint64_t lastIterations{0};
};
} // namespace marco::runtime::sundials::kinsol
#endif
#endif
