#include "marco/Runtime/Drivers/EulerForward/Driver.h"
#include "marco/Runtime/Drivers/EulerForward/CLI.h"
#include "marco/Runtime/Simulation/Options.h"
#include "marco/Runtime/Simulation/Profiler.h"
#include "marco/Runtime/Simulation/Runtime.h"
#include "marco/Runtime/Solvers/EulerForward/Options.h"
#include "marco/Runtime/Solvers/EulerForward/Profiler.h"
#include "../SimulationTime.h"
#include <cmath>
#include <iostream>

namespace marco::runtime {
EulerForward::EulerForward(Simulation *simulation) : Driver(simulation) {}

#ifdef CLI_ENABLE
std::unique_ptr<cli::Category> EulerForward::getCLIOptions() {
  return std::make_unique<eulerforward::CommandLineOptions>();
}
#endif // CLI_ENABLE

int EulerForward::run() {
  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[Euler Forward] Starting simulation" << std::endl;
  }

  const double startTime = getTime();
  const double endTime = simulation::getOptions().endTime;
  const double requestedStep = eulerforward::getOptions().timeStep;
  if (!std::isfinite(startTime) || !std::isfinite(endTime) || endTime < startTime ||
      !std::isfinite(requestedStep) || requestedStep <= 0) {
    std::cerr << "[Euler Forward] Invalid simulation interval or time step.\n";
    return EXIT_FAILURE;
  }

  internal::SimulationTime clock(startTime, endTime);
  while (clock.hasNext()) {
    // 中文：非整步终点需要执行最后一个较短步，不能提前退出或越界。
    // English: Execute the shorter final step instead of stopping early or
    // advancing beyond the requested endpoint.
    const double timeStep = clock.clipStep(requestedStep);
    if (!clock.canAdvance(timeStep)) {
      std::cerr << "[Euler Forward] Time step cannot advance simulation time.\n";
      return EXIT_FAILURE;
    }
    // Compute the next values of the state variables.
    if (marco::runtime::simulation::getOptions().debug) {
      std::cerr << "[Euler Forward] Updating state variables" << std::endl;
    }

    EULER_FORWARD_PROFILER_STATEVAR_START
    updateStateVariables(timeStep);
    EULER_FORWARD_PROFILER_STATEVAR_STOP

    // Move to the next step.
    if (marco::runtime::simulation::getOptions().debug) {
      std::cerr << "[Euler Forward] Updating time and non-state variables"
                << std::endl;
    }

    EULER_FORWARD_PROFILER_NONSTATEVAR_START
    setTime(clock.acceptStep(timeStep));

    updateNonStateVariables();
    EULER_FORWARD_PROFILER_NONSTATEVAR_STOP

    if (marco::runtime::simulation::getOptions().debug) {
      std::cerr << "[Euler Forward] Printing values" << std::endl;
    }

    // Print the values.
    getSimulation()->getPrinter()->printValues();
  }

  if (marco::runtime::simulation::getOptions().debug) {
    std::cerr << "[Euler Forward] Simulation finished" << std::endl;
  }

  return EXIT_SUCCESS;
}
} // namespace marco::runtime

namespace marco::runtime {
std::unique_ptr<Driver> getDriver(Simulation *simulation) {
  return std::make_unique<EulerForward>(simulation);
}
} // namespace marco::runtime
