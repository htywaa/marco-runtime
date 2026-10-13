#ifdef SUNDIALS_ENABLE
#include "marco/Runtime/Solvers/KINSOL/ScalarBlockSolver.h"
#include <algorithm>
#include <cmath>

namespace marco::runtime::sundials::kinsol {
ScalarBlockSolver::ScalarBlockSolver(
    const ScalarBlockPlan &plan,
    SparseNonlinearSystem::Configuration configuration,
    Residual residual, Jacobian jacobian, Preparation preparation)
    : plan(plan), configuration(configuration), residual(std::move(residual)),
      jacobian(std::move(jacobian)), preparation(std::move(preparation)),
      state(configuration.size),
      initial(configuration.size), residualScale(configuration.size, 1),
      finalResidual(configuration.size) {}

std::unique_ptr<ScalarBlockSolver> ScalarBlockSolver::create(
    const ScalarBlockPlan &plan,
    SparseNonlinearSystem::Configuration configuration,
    Residual residual, Jacobian jacobian, ScalarBlockPlan::Budget &budget,
    Preparation preparation) {
  if (!configuration.size || configuration.size > 64 || !residual || !jacobian)
    return {};
  uint64_t work = configuration.size, storage = sizeof(ScalarBlockSolver) +
                               4 * configuration.size * sizeof(double);
  uint64_t rows = 0;
  for (const auto &block : plan.getExecutionBlocks()) {
    rows += block.rows.size();
    if (block.kind == ScalarBlockPlan::Kind::Explicit) {
      work += 1;
      storage += sizeof(Block);
      continue;
    }
    work += block.rows.size() * block.rows.size() + 1;
    storage += sizeof(Block) + sizeof(SparseNonlinearSystem) +
        (block.rows.size() * block.rows.size() + 16 * block.rows.size()) *
            (sizeof(double) + sizeof(uint64_t));
  }
  if (rows != configuration.size || work > budget.work || storage > budget.storage)
    return {};
  budget.work -= work;
  budget.storage -= storage;
  auto result = std::unique_ptr<ScalarBlockSolver>(new ScalarBlockSolver(
      plan, configuration, std::move(residual), std::move(jacobian),
      std::move(preparation)));
  result->blocks.reserve(plan.getExecutionBlocks().size());
  for (const auto &planBlock : plan.getExecutionBlocks()) {
    result->blocks.push_back({&planBlock, {}, {}, {}, {}});
    Block &block = result->blocks.back();
    if (planBlock.kind == ScalarBlockPlan::Kind::Explicit)
      continue;
    const auto count = planBlock.rows.size();
    block.residual.resize(count);
    block.rhs.resize(count);
    block.solution.reserve(count);
    auto options = configuration;
    options.size = count;
    options.nonZeros = count * count;
    auto *owner = result.get();
    auto *entry = &block;
    block.solver = std::make_unique<SparseNonlinearSystem>(
        options,
        [owner, entry](N_Vector variables, N_Vector values) {
          owner->install(*entry, variables);
          auto *out = N_VGetArrayPointer(values);
          for (size_t i = 0; i < entry->plan->rows.size(); ++i) {
            out[i] = owner->residual(entry->plan->rows[i], owner->state);
            entry->residual[i] = out[i];
            if (!std::isfinite(out[i])) return KIN_SYSFUNC_FAIL;
          }
          return KIN_SUCCESS;
        },
        [owner, entry](N_Vector variables, N_Vector, SUNMatrix matrix) {
          owner->install(*entry, variables);
          const auto count = entry->plan->rows.size();
          auto *pointers = SUNSparseMatrix_IndexPointers(matrix);
          auto *indices = SUNSparseMatrix_IndexValues(matrix);
          auto *data = SUNSparseMatrix_Data(matrix);
          for (size_t row = 0; row < count; ++row) {
            pointers[row] = row * count;
            for (size_t column = 0; column < count; ++column) {
              const auto offset = row * count + column;
              indices[offset] = column;
              data[offset] = owner->jacobian(entry->plan->rows[row],
                  entry->plan->columns[column], owner->state);
              if (!std::isfinite(data[offset])) return KIN_SYSFUNC_FAIL;
            }
          }
          pointers[count] = count * count;
          return KIN_SUCCESS;
        });
    if (!block.solver->initialize()) return {};
  }
  return result;
}

// 中文：一次向量回调先安装完整当前块，再准备依赖；不能为不同标量项混用不同 trial 的值。
// English: Install the complete block state before callback preparation; scalar entries cannot
// mix values from different trials.
void ScalarBlockSolver::install(Block &block, N_Vector variables) {
  const auto *values = N_VGetArrayPointer(variables);
  for (size_t i = 0; i < block.plan->columns.size(); ++i)
    state[block.plan->columns[i]] = values[i];
  // 中文：每次向量回调只准备一个完整不可变状态，不逐标量条目准备；即使值重复，新回调/trial 也重新准备。
  // One complete, immutable state per vector callback, never per scalar entry.
  // A new callback/trial always prepares afresh, including repeated values.
  if (preparation) preparation(state);
}

bool ScalarBlockSolver::setScaling(const std::vector<double> &variables,
                                   const std::vector<double> &residuals) {
  if (variables.size() != configuration.size ||
      residuals.size() != configuration.size) return false;
  for (size_t i = 0; i < variables.size(); ++i)
    if (!(variables[i] > 0) || !std::isfinite(variables[i]) ||
        !(residuals[i] > 0) || !std::isfinite(residuals[i])) return false;
  for (auto &block : blocks) {
    if (!block.solver) continue;
    std::vector<double> x, f;
    for (auto column : block.plan->columns) x.push_back(variables[column]);
    for (auto row : block.plan->rows) f.push_back(residuals[row]);
    if (!block.solver->setScaling(x, f)) return false;
  }
  residualScale = residuals;
  return true;
}

void ScalarBlockSolver::setLineSearchEnabled(bool enabled) {
  for (auto &block : blocks)
    if (block.solver) block.solver->setLineSearchEnabled(enabled);
}

// 中文：数值条件不成立时还原输入向量并交还整体闭包；最终仍检查所有行残差和二次根分支。
// English: A failed numerical condition restores the input vector and returns to the whole
// closure; all residual rows and the quadratic branch remain checked.
ScalarBlockSolver::Result ScalarBlockSolver::solve(
    double time, std::vector<double> &values) {
  lastIterations = 0;
  if (values.size() != configuration.size) return Result::RetainOriginal;
  state = values;
  initial = values;
  if (preparation) preparation(state);
  auto decline = [&] {
    values = initial;
    return Result::RetainOriginal;
  };
  for (auto &block : blocks) {
    if (block.plan->kind == ScalarBlockPlan::Kind::Explicit) {
      if (!explicitValues) return decline();
      auto row = block.plan->rows.front();
      auto column = block.plan->columns.front();
      // 中文：在当前前驱状态重新求值；结构上仿射的行并不保证每个 trial 的数值系数非零。
      // Re-evaluate at the current predecessor state. A structurally affine
      // row need not have a nonzero numeric coefficient at every trial.
      double diagonal = jacobian(row, column, state);
      if (!std::isfinite(diagonal) || diagonal == 0) return decline();
      double value = explicitValues(row, state);
      if (!std::isfinite(value)) return decline();
      state[column] = value;
      if (preparation) preparation(state);
      continue;
    }
    auto *x = N_VGetArrayPointer(block.solver->getVariablesVector());
    for (size_t i = 0; i < block.plan->columns.size(); ++i)
      x[i] = state[block.plan->columns[i]];
    if (block.plan->kind == ScalarBlockPlan::Kind::Affine) {
      if (!block.solver->factorize(time)) return decline();
      for (size_t i = 0; i < block.rhs.size(); ++i)
        block.rhs[i] = -block.residual[i];
      if (!block.solver->solveFactorized(block.rhs, block.solution))
        return decline();
      for (size_t i = 0; i < block.solution.size(); ++i) {
        x[i] += block.solution[i];
        if (!std::isfinite(x[i])) return decline();
      }
    } else {
      // 中文：源二次表达式的导数符号标识相对驻点的根分支。
      //       保持原 KINSOL warm start 和线搜索，不能用无符号平方根代替。
      // For a source quadratic, the derivative's sign identifies the root
      // branch relative to its stationary point. Keep the original KINSOL
      // warm start and line search; never replace it by an unsigned sqrt.
      const auto row = block.plan->rows.front();
      const auto column = block.plan->columns.front();
      const double before = jacobian(row, column, state);
      if (!std::isfinite(before) || before == 0) return decline();
      auto status = block.solver->solve(time);
      lastIterations += block.solver->getLastNonlinearIterations();
      if (status != SparseNonlinearSystem::SolveStatus::Success)
        return decline();
      install(block, block.solver->getVariablesVector());
      const double after = jacobian(row, column, state);
      if (!std::isfinite(after) || after == 0 ||
          std::signbit(before) != std::signbit(after)) return decline();
    }
    install(block, block.solver->getVariablesVector());
  }
  // 中文：各块均已求解不能替代完整闭包残差，也不能替代 owner 独立的 pre/post/failed-trial 健康检查。
  // A collection of solved blocks does not replace the complete closure's
  // residual or the owner's independent pre/post/failed-trial health checks.
  for (uint32_t row = 0; row < configuration.size; ++row) {
    double value = residual(row, state);
    if (!std::isfinite(value) ||
        std::abs(value) * residualScale[row] > configuration.functionNormTolerance)
      return decline();
    finalResidual[row] = value;
  }
  values = state;
  return Result::Solved;
}
} // namespace marco::runtime::sundials::kinsol
#endif
