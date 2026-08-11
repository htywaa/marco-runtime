#include "marco/Runtime/Solvers/KINSOL/Instance.h"
#include "gtest/gtest.h"

#ifdef SUNDIALS_ENABLE

using marco::runtime::sundials::Equation;
using marco::runtime::sundials::Variable;
using marco::runtime::sundials::kinsol::KINSOLInstance;

// 中文：以仅 timed callback 的局部系统验证 IDA component closure 不依赖普通
// KINSOL callback，也不会在 assertions-enabled 构建中误触发断言。
// English: A timed-callback-only local system verifies that IDA component
// closure does not require ordinary KINSOL callbacks or trip assertions in
// assertions-enabled builds.
namespace {
double value = 0;

double getValue(const uint64_t *) { return value; }

void setValue(double newValue, const uint64_t *) { value = newValue; }

void singletonAccess(const int64_t *, uint64_t *result) { result[0] = 0; }

double timedResidual(double time, const int64_t *) { return value - time; }

double timedJacobian(double, const int64_t *, const uint64_t *, double,
                     uint64_t, const uint64_t *) {
  return 1;
}
} // namespace

TEST(KINSOLTimedCallbacksTest, SolvesWithTimedOnlyCallbacks) {
  value = 0;
  KINSOLInstance instance;

  uint64_t dimensions[] = {1};
  int64_t ranges[] = {0, 1};
  Variable variable = instance.addVariable(1, dimensions, getValue, setValue,
                                           "timed_singleton");
  Equation equation = instance.addEquation(ranges, 1, "x[0] - time = 0");
  instance.addVariableAccess(equation, variable, singletonAccess);
  instance.setTimedResidualFunction(equation, timedResidual);

  uint64_t unusedSeedSize = 1;
  instance.addTimedJacobianFunction(equation, variable, timedJacobian, 0,
                                    &unusedSeedSize);

  ASSERT_TRUE(instance.initialize());
  ASSERT_TRUE(instance.solve(2.5));
  EXPECT_NEAR(value, 2.5, 1e-12);
}

#endif // SUNDIALS_ENABLE
