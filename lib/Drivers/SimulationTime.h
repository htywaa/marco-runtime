#ifndef MARCO_RUNTIME_DRIVERS_SIMULATIONTIME_H
#define MARCO_RUNTIME_DRIVERS_SIMULATIONTIME_H

#include <algorithm>
#include <cmath>

namespace marco::runtime::internal {
// 中文：补偿求和避免反复加步长产生漂移；最终步仍实际积分到终点。
// English: Compensated addition prevents accumulated clock drift. The final
// interval is still integrated, rather than merely relabeling an earlier state.
class SimulationTime {
public:
  SimulationTime(double start, double end) : current(start), end(end) {}
  bool hasNext() const { return current < end; }
  double clipStep(double requested) const {
    return std::min(requested, end - current);
  }
  bool canAdvance(double step) const {
    const double next = step == end - current ? end :
        std::min(current + (step - correction), end);
    return std::isfinite(step) && step > 0 && next > current;
  }
  double acceptStep(double step) {
    const double increment = step - correction;
    const double next = current + increment;
    if (step == end - current || next >= end) {
      current = end;
      correction = 0;
    } else {
      correction = (next - current) - increment;
      current = next;
    }
    return current;
  }

private:
  double current, end, correction{0};
};
} // namespace marco::runtime::internal
#endif
