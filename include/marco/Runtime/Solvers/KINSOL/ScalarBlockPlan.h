#ifndef MARCO_RUNTIME_SOLVERS_KINSOL_SCALARBLOCKPLAN_H
#define MARCO_RUNTIME_SOLVERS_KINSOL_SCALARBLOCKPLAN_H

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace marco::runtime::sundials::kinsol {

/// Optional execution plan for an already scalar, small C2 closure. Input
/// matching and reads belong to the active execution layout, not a new match.
/// Polynomial supports must come from a compiler proof of the actual pure
/// residual. Samples of F/J, or numeric zeros, cannot provide that proof.
// 中文：小型且原本已标量化的组件才可分块；结构多项式证明来自编译器，数值采样不能签发证明。
// English: Partition only small, already-scalar components; the compiler supplies polynomial
// structure proof, which numeric samples cannot establish.
class ScalarBlockPlan {
public:
  struct Term {
    // 中文：各变量指数为零、一或二，总次数可超过二；前驱平方乘当前未知量，对当前块仍属仿射。
    // Each variable has exponent 0, 1, or 2. Total degree may exceed two:
    // a predecessor's square times a current unknown is affine in this block.
    uint64_t linear = 0;
    uint64_t squared = 0;
  };

  struct Row {
    uint32_t matchedColumn = 0;
    uint64_t reads = 0;
    bool polynomialCertified = false;
    std::vector<Term> terms;
    // 中文：编译器为该行原配对签发唯一隔离证明，执行适配器还绑定实际可选回调。
    // A compiler-issued unique isolation of this row's original pairing.
    // The execution adapter also binds its actual optional callback.
    bool explicitValueCertified = false;
  };

  struct Budget {
    uint64_t work = 1000000;
    uint64_t storage = 16000000;
    uint64_t partitions = 256;
  };

  enum class Status { Ready, Unavailable, InvalidInput, BudgetExhausted };
  enum class Kind { Affine, SourceQuadratic, Explicit };

  struct Block {
    Kind kind;
    std::vector<uint32_t> rows;
    // 中文：与 rows 一一对应，严格使用调用方原选配对。
    // Parallel to rows: exactly the caller's selected pairing.
    std::vector<uint32_t> columns;
  };

  /// 中文：仿射指前驱固定时相对于当前块未知量为仿射，不证明每个数值点 Jacobian 非奇异。
  ///       二次块只接受单未知量且无局部前驱的情况；求解器仍须认证根分支、完整残差及健康状态。
  /// Affine means affine in this block's unknowns, with predecessor values
  /// fixed. It does not certify a nonsingular numeric Jacobian. A quadratic
  /// block is accepted only for one unknown with no local predecessors. The
  /// solver must still certify the root branch, full residual, and health.
  static std::pair<Status, std::unique_ptr<const ScalarBlockPlan>>
  build(const std::vector<Row> &rows, Budget &budget);

  const std::vector<Block> &getBlocks() const { return blocks; }

  /// 中文：相邻仿射组件仅在每个原多项式对完整并集仍仿射时共用现有线性求解。
  ///       上面的最小 SCC 证书与这里的执行分组合并保持独立。
  /// Adjacent affine components may share one existing linear solve only if
  /// every original polynomial is affine in the complete union. The minimal
  /// SCC certificate above is kept distinct from this execution grouping.
  const std::vector<Block> &getExecutionBlocks() const { return executionBlocks; }

private:
  ScalarBlockPlan() = default;
  std::vector<Block> blocks;
  std::vector<Block> executionBlocks;
};

} // namespace marco::runtime::sundials::kinsol

#endif
