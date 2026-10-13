#include "marco/Runtime/Drivers/RungeKutta/Driver.h"
#include "marco/Runtime/Drivers/RungeKutta/CLI.h"
#include "marco/Runtime/Simulation/Options.h"
#include "marco/Runtime/Simulation/Profiler.h"
#include "marco/Runtime/Simulation/Runtime.h"
#include "marco/Runtime/Solvers/RungeKutta/Options.h"
#include "marco/Runtime/Solvers/RungeKutta/Profiler.h"
#include "../SimulationTime.h"
#include <cmath>
#include <iostream>

namespace marco::runtime {
RungeKutta::RungeKutta(Simulation *simulation) : Driver(simulation) {}

#ifdef CLI_ENABLE
std::unique_ptr<cli::Category> RungeKutta::getCLIOptions() {
  return std::make_unique<rungekutta::CommandLineOptions>();
}
#endif // CLI_ENABLE

int RungeKutta::run() {
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[Runge-Kutta] Starting simulation" << std::endl;
  }

  const double startTime = getTime();
  const double endTime = simulation::getOptions().endTime;
  double timeStep = rungekutta::getOptions().timeStep;
  if (!std::isfinite(startTime) || !std::isfinite(endTime) || endTime < startTime ||
      !std::isfinite(timeStep) || timeStep <= 0) {
    std::cerr << "[Runge-Kutta] Invalid simulation interval or time step.\n";
    return EXIT_FAILURE;
  }

  internal::SimulationTime clock(startTime, endTime);
  while (clock.hasNext()) {
    // 中文：最终试步必须受剩余时域限制，接受后以新时间判断终止。
    // English: Bound the final trial by the remaining interval and decide
    // termination from the accepted time, not the previous step's time.
    timeStep = clock.clipStep(timeStep);
    // Compute the next values of the state variables.
    if (marco::runtime::simulation::getOptions().debug) {
      std::cerr << "[Runge-Kutta] Updating state variables" << std::endl;
    }

    uint64_t iteration = 0;
    double error;

    do {
      if (!clock.canAdvance(timeStep)) {
        std::cerr << "[Runge-Kutta] Time step cannot advance simulation time.\n";
        return EXIT_FAILURE;
      }
      RUNGEKUTTA_PROFILER_STATEVAR_START
      tryStep(timeStep);
      RUNGEKUTTA_PROFILER_STATEVAR_STOP

      RUNGEKUTTA_PROFILER_ERROR_START
      error = estimateError();
      RUNGEKUTTA_PROFILER_ERROR_STOP
      if (!std::isfinite(error)) {
        std::cerr << "[Runge-Kutta] Nonfinite error estimate.\n";
        return EXIT_FAILURE;
      }

      if (error >= rungekutta::getOptions().tolerance) {
        timeStep /= 2;
      }
    } while (error >= rungekutta::getOptions().tolerance &&
             iteration++ < rungekutta::getOptions().maxIterations);

    if (error >= rungekutta::getOptions().tolerance &&
        iteration >= rungekutta::getOptions().maxIterations) {
      std::cerr << "Could not find an appropriate time step for adaptive "
                   "Runge-Kutta. Increase the maximum number of iterations or "
                   "the maximum tolerance.";

      return EXIT_FAILURE;
    }

    // Update the time.
    setTime(clock.acceptStep(timeStep));

    RUNGEKUTTA_PROFILER_VARCOPY_START
    acceptStep();
    RUNGEKUTTA_PROFILER_VARCOPY_STOP

    if (iteration == 1) {
      // The time step was big enough to converge in only one iteration.
      // Try a bigger one in the next step.
      timeStep *= 2;
    }

    // Move to the next step.
    if (marco::runtime::simulation::getOptions().debug) {
      std::cerr << "[Runge-Kutta] Updating time and non-state variables"
                << std::endl;
    }

    RUNGEKUTTA_PROFILER_NONSTATEVAR_START
    updateNonStateVariables();
    RUNGEKUTTA_PROFILER_NONSTATEVAR_STOP

    if (marco::runtime::simulation::getOptions().debug) {
      std::cerr << "[Runge-Kutta] Printing values" << std::endl;
    }

    // Print the values.
    getSimulation()->getPrinter()->printValues();
  }

  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[Runge-Kutta] Simulation finished" << std::endl;
  }

  return EXIT_SUCCESS;
}
} // namespace marco::runtime

namespace marco::runtime {
std::unique_ptr<Driver> getDriver(Simulation *simulation) {
  return std::make_unique<RungeKutta>(simulation);
}
} // namespace marco::runtime
