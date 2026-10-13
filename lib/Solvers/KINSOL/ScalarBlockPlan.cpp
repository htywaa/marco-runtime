#include "marco/Runtime/Solvers/KINSOL/ScalarBlockPlan.h"
#include <algorithm>
#include <array>

namespace marco::runtime::sundials::kinsol {
namespace {
unsigned countBits(uint64_t value) {
  unsigned count = 0;
  while (value) {
    value &= value - 1;
    ++count;
  }
  return count;
}

bool charge(ScalarBlockPlan::Budget &budget, uint64_t work,
            uint64_t storage = 0, uint64_t partitions = 0) {
  if (work > budget.work || storage > budget.storage ||
      partitions > budget.partitions)
    return false;
  budget.work -= work;
  budget.storage -= storage;
  budget.partitions -= partitions;
  return true;
}
} // namespace

std::pair<ScalarBlockPlan::Status, std::unique_ptr<const ScalarBlockPlan>>
// 中文：先验证原配对唯一性和完整多项式 support，再按真实依赖求块；不重新匹配或数组展开。
// English: Validate original pairing uniqueness and complete polynomial support before
// deriving blocks from dependencies; do not rematch or scalarize arrays.
ScalarBlockPlan::build(const std::vector<Row> &rows, Budget &budget) {
  const auto fail = [](Status status) {
    return std::make_pair(status, std::unique_ptr<const ScalarBlockPlan>());
  };
  // 中文：这里明确不是数组标量化路线。
  // This is deliberately not an array scalarization route.
  const size_t size = rows.size();
  if (!size || size > 64)
    return fail(Status::Unavailable);
  if (!charge(budget, size, sizeof(ScalarBlockPlan) +
                            5 * 64 * sizeof(uint64_t)))
    return fail(Status::BudgetExhausted);
  const uint64_t whole = size == 64 ? ~uint64_t(0)
                                   : (uint64_t(1) << size) - 1;
  uint64_t matched = 0;
  std::array<uint32_t, 64> writers{};
  for (size_t row = 0; row < size; ++row) {
    const Row &entry = rows[row];
    if (entry.matchedColumn >= size || (entry.reads & ~whole))
      return fail(Status::InvalidInput);
    const uint64_t column = uint64_t(1) << entry.matchedColumn;
    if ((matched & column) || !(entry.reads & column))
      return fail(Status::InvalidInput);
    matched |= column;
    writers[entry.matchedColumn] = row;
    if (!entry.polynomialCertified)
      return fail(Status::Unavailable);
    uint64_t polynomialReads = 0;
    for (const Term &term : entry.terms) {
      if (!charge(budget, 1 + 2 * 64))
        return fail(Status::BudgetExhausted);
      if ((term.linear & term.squared) ||
          ((term.linear | term.squared) & ~entry.reads) ||
          (entry.explicitValueCertified && (term.squared & column)))
        return fail(Status::InvalidInput);
      polynomialReads |= term.linear | term.squared;
    }
    // 中文：不能由观测到零系数推断省略的局部读取。
    // No omitted local read is inferred from an observed zero coefficient.
    if (polynomialReads != entry.reads)
      return fail(Status::InvalidInput);
  }

  // 中文：固定大小可达闭包只作用于已存在的标量行，用来证明 SCC 和 predecessor 顺序。
  // English: Fixed-size reachability operates only on existing scalar rows to prove SCCs and
  // predecessor order.
  std::array<uint64_t, 64> reaches{};
  for (size_t reader = 0; reader < size; ++reader) {
    reaches[reader] = uint64_t(1) << reader;
    if (!charge(budget, size))
      return fail(Status::BudgetExhausted);
    for (size_t column = 0; column < size; ++column)
      if (rows[reader].reads & (uint64_t(1) << column))
        reaches[reader] |= uint64_t(1) << writers[column];
  }
  // 中文：在原标量行上计算固定大小精确可达性；不扫描外围域，也不按元素展开取得这些行。
  // Fixed-size exact reachability on original scalar rows. No ambient domain
  // scan or per-element expansion is used to obtain these rows.
  for (size_t pivot = 0; pivot < size; ++pivot) {
    if (!charge(budget, size))
      return fail(Status::BudgetExhausted);
    for (size_t row = 0; row < size; ++row)
      if (reaches[row] & (uint64_t(1) << pivot))
        reaches[row] |= reaches[pivot];
  }

  std::array<uint64_t, 64> members{};
  std::array<uint64_t, 64> predecessors{};
  uint64_t assigned = 0;
  size_t components = 0;
  for (size_t first = 0; first < size; ++first) {
    if (assigned & (uint64_t(1) << first))
      continue;
    if (!charge(budget, size, sizeof(Block) + size * 2 * sizeof(uint32_t), 1))
      return fail(Status::BudgetExhausted);
    uint64_t component = 0;
    for (size_t row = 0; row < size; ++row)
      if ((reaches[first] & (uint64_t(1) << row)) &&
          (reaches[row] & (uint64_t(1) << first)))
        component |= uint64_t(1) << row;
    members[components] = component;
    predecessors[components] = reaches[first] & ~component;
    assigned |= component;
    ++components;
  }
  if (assigned != whole)
    return fail(Status::InvalidInput);

  auto result = std::unique_ptr<ScalarBlockPlan>(new ScalarBlockPlan);
  result->blocks.reserve(components);
  uint64_t emitted = 0;
  for (size_t index = 0; index < components; ++index) {
    size_t selected = components;
    if (!charge(budget, components))
      return fail(Status::BudgetExhausted);
    for (size_t component = 0; component < components; ++component)
      if (!(members[component] & emitted) &&
          !(predecessors[component] & ~emitted)) {
        selected = component;
        break;
      }
    if (selected == components)
      return fail(Status::InvalidInput);
    Block block{Kind::Affine, {}, {}};
    const size_t memberCount = countBits(members[selected]);
    block.rows.reserve(memberCount);
    block.columns.reserve(memberCount);
    uint64_t columns = 0;
    for (size_t row = 0; row < size; ++row)
      if (members[selected] & (uint64_t(1) << row)) {
        block.rows.push_back(row);
        block.columns.push_back(rows[row].matchedColumn);
        columns |= uint64_t(1) << rows[row].matchedColumn;
      }
    unsigned degree = 0;
    for (auto row : block.rows)
      for (const Term &term : rows[row].terms) {
        if (!charge(budget, 1 + 2 * 64))
          return fail(Status::BudgetExhausted);
        degree = std::max(degree, countBits(term.linear & columns) +
                                     2 * countBits(term.squared & columns));
      }
    if (degree > 1) {
      if (block.rows.size() != 1 || predecessors[selected])
        return fail(Status::Unavailable);
      block.kind = Kind::SourceQuadratic;
    } else if (block.rows.size() == 1 &&
               rows[block.rows.front()].explicitValueCertified) {
      block.kind = Kind::Explicit;
    }
    result->blocks.push_back(std::move(block));
    emitted |= members[selected];
  }

  // 中文：合并改变执行粒度而不改变原证书；必须在联合 unknown 集上重新检查仿射性。
  // English: Coalescing changes execution granularity, not the original certificate; recheck
  // affinity over the combined unknown set.
  // Coalesce adjacent linear components, never merely their endpoint sets.
  // A predecessor may be a coefficient of a later block: bringing it into
  // the same solve can make that union nonlinear. Recheck all original terms
  // against the union before changing the execution granularity.
  if (!charge(budget, size, components * sizeof(Block) +
                           2 * size * sizeof(uint32_t)))
    return fail(Status::BudgetExhausted);
  result->executionBlocks.reserve(components);
  for (const Block &block : result->blocks) {
    bool combine = block.kind == Kind::Affine &&
                   !result->executionBlocks.empty() &&
                   result->executionBlocks.back().kind == Kind::Affine;
    if (combine) {
      const Block &previous = result->executionBlocks.back();
      uint64_t columns = 0;
      for (auto column : previous.columns) columns |= uint64_t(1) << column;
      for (auto column : block.columns) columns |= uint64_t(1) << column;
      for (const Block *part : {&previous, &block}) {
        for (auto row : part->rows) {
          for (const Term &term : rows[row].terms) {
            if (!charge(budget, 1 + 2 * 64))
              return fail(Status::BudgetExhausted);
            if (countBits(term.linear & columns) +
                    2 * countBits(term.squared & columns) > 1)
              combine = false;
          }
        }
      }
    }
    if (combine) {
      auto &previous = result->executionBlocks.back();
      // 中文：有界向量的增长/复制单独计费，与原证书存储分开；这不是逐 trial 工作。
      // Account for the bounded vectors' possible growth/copy, separately
      // from the original certificate's storage. This is not per-trial work.
      if (!charge(budget, previous.rows.size() + block.rows.size(),
                  2 * (previous.rows.size() + block.rows.size()) *
                      sizeof(uint32_t)))
        return fail(Status::BudgetExhausted);
      previous.rows.insert(previous.rows.end(), block.rows.begin(),
                           block.rows.end());
      previous.columns.insert(previous.columns.end(), block.columns.begin(),
                              block.columns.end());
    } else {
      result->executionBlocks.push_back(block);
    }
  }
  return {Status::Ready, std::move(result)};
}
} // namespace marco::runtime::sundials::kinsol
