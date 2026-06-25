#include "marco/Runtime/Solvers/KINSOL/Profiler.h"
#include "marco/Runtime/Profiling/Profiling.h"
#include <iostream>

#ifdef MARCO_PROFILING

namespace marco::runtime::profiling {
static void recordParallelWork(
    std::vector<ParallelThreadWorkStats> &target,
    const std::vector<ParallelThreadWorkStats> &threadWork) {
  if (target.size() < threadWork.size()) {
    target.resize(threadWork.size());
  }

  for (size_t i = 0, e = threadWork.size(); i < e; ++i) {
    target[i].chunks += threadWork[i].chunks;
    target[i].scalarEquations += threadWork[i].scalarEquations;
  }
}

static void printParallelWork(
    const char *label, const std::vector<ParallelThreadWorkStats> &threadWork) {
  uint64_t totalChunks = 0;
  uint64_t totalScalarEquations = 0;
  uint64_t activeWorkers = 0;

  for (const ParallelThreadWorkStats &work : threadWork) {
    totalChunks += work.chunks;
    totalScalarEquations += work.scalarEquations;

    if (work.chunks != 0 || work.scalarEquations != 0) {
      ++activeWorkers;
    }
  }

  std::cerr << label << " parallel worker slots used: " << activeWorkers
            << "/" << threadWork.size() << "\n";
  std::cerr << label << " parallel chunks processed: " << totalChunks << "\n";
  std::cerr << label << " parallel scalar equations processed: "
            << totalScalarEquations << "\n";

  for (size_t i = 0, e = threadWork.size(); i < e; ++i) {
    std::cerr << "  worker " << i << ": chunks=" << threadWork[i].chunks
              << ", scalar equations=" << threadWork[i].scalarEquations
              << "\n";
  }
}

KINSOLProfiler::KINSOLProfiler() : Profiler("KINSOL") {
  registerProfiler(*this);
}

void KINSOLProfiler::reset() {
  std::lock_guard<std::mutex> lockGuard(mutex);
  residualsCallCounter = 0;
  residualsTimer.reset();
  residualsParallelWork.clear();
  partialDerivativesCallCounter = 0;
  partialDerivativesTimer.reset();
  partialDerivativesParallelWork.clear();
  copyVarsFromMARCOTimer.reset();
  copyVarsIntoMARCOTimer.reset();
}

void KINSOLProfiler::print() const {
  std::lock_guard<std::mutex> lockGuard(mutex);

  std::cerr << "Number of computations of the residuals: "
            << residualsCallCounter << "\n";
  std::cerr << "Time spent on computing the residuals: "
            << residualsTimer.totalElapsedTime<std::milli>() << " ms\n";
  printParallelWork("Residual", residualsParallelWork);
  std::cerr << "Number of computations of the partial derivatives: "
            << partialDerivativesCallCounter << "\n";
  std::cerr << "Time spent on computing the partial derivatives: "
            << partialDerivativesTimer.totalElapsedTime<std::milli>()
            << " ms\n";
  printParallelWork("Jacobian", partialDerivativesParallelWork);
  std::cerr << "Time spent on copying the variables from MARCO: "
            << copyVarsFromMARCOTimer.totalElapsedTime<std::milli>()
            << " ms\n";
  std::cerr << "Time spent on copying the variables into MARCO: "
            << copyVarsIntoMARCOTimer.totalElapsedTime<std::milli>()
            << " ms\n";
}

void KINSOLProfiler::incrementResidualsCallCounter() {
  std::lock_guard<std::mutex> lockGuard(mutex);
  ++residualsCallCounter;
}

void KINSOLProfiler::incrementPartialDerivativesCallCounter() {
  std::lock_guard<std::mutex> lockGuard(mutex);
  ++partialDerivativesCallCounter;
}

void KINSOLProfiler::recordResidualsParallelWork(
    const std::vector<ParallelThreadWorkStats> &threadWork) {
  std::lock_guard<std::mutex> lockGuard(mutex);
  recordParallelWork(residualsParallelWork, threadWork);
}

void KINSOLProfiler::recordPartialDerivativesParallelWork(
    const std::vector<ParallelThreadWorkStats> &threadWork) {
  std::lock_guard<std::mutex> lockGuard(mutex);
  recordParallelWork(partialDerivativesParallelWork, threadWork);
}

KINSOLProfiler &kinsolProfiler() {
  static KINSOLProfiler obj;
  return obj;
}
} // namespace marco::runtime::profiling

#endif // MARCO_PROFILING
