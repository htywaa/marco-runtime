#ifndef MARCO_RUNTIME_PROFILING_PROFILER_H
#define MARCO_RUNTIME_PROFILING_PROFILER_H

#include <cstdint>
#include <string>

namespace marco::runtime::profiling {
// 中文：记录一次并行方程遍历中每个 worker 实际处理的 chunk 和标量方程数。
// English: Records how many chunks and scalar equations each worker processed
// during one parallel equation traversal.
struct ParallelThreadWorkStats {
  uint64_t chunks{0};
  uint64_t scalarEquations{0};
};

class Profiler {
public:
  Profiler(const std::string &name);

  Profiler(const Profiler &other) = delete;

  virtual ~Profiler();

  const std::string &getName() const;

  virtual void reset() = 0;
  virtual void print() const = 0;

private:
  std::string name;
};
} // namespace marco::runtime::profiling

#endif // MARCO_RUNTIME_PROFILING_PROFILER_H
