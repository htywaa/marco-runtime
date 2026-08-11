#ifdef CLI_ENABLE
#ifdef SUNDIALS_ENABLE

#include "marco/Runtime/Drivers/IDA/CLI.h"
#include "marco/Runtime/Solvers/IDA/Options.h"
#include <cstdlib>
#include <iostream>

namespace marco::runtime::sundials::ida {
std::string CommandLineOptions::getTitle() const { return "IDA"; }

void CommandLineOptions::printCommandLineOptions(std::ostream &os) const {
  // clang-format off
  os << "  --time-step=<value>                    Set the time step (in seconds)." << std::endl;

  os << "  --ida-equations-chunks-factor=<value>  Set the factor which, once multiplied by the threads count, determines the number of equation chunks. Defaults to " << getOptions().equationsChunksFactor << "." << std::endl;

  os << "  --ida-relative-tolerance=<value>       Set the relative tolerance. Defaults to " << getOptions().relativeTolerance << "." << std::endl;
  os << "  --ida-absolute-tolerance=<value>       Set the absolute tolerance. Defaults to " << getOptions().absoluteTolerance << "." << std::endl;
  os << "  --ida-max-algebraic-abs-tol=<value>    Set the maximum absolute tolerance allowed for algebraic variables. Defaults to " << getOptions().maxAlgebraicAbsoluteTolerance << "." << std::endl;
  os << "  --ida-time-scaling-factor-ic=<value>   Set the dividing factor for the initial step size guess (the higher the value, the smaller the step). Defaults to " << getOptions().timeScalingFactorInit << "." << std::endl;

  os << "  --ida-max-steps=<value>                Set the maximum number of steps to be taken by the solver in its attempt to reach the next output time. Defaults to " << getOptions().maxSteps << "." << std::endl;
  os << "  --ida-initial-step-size=<value>        Set the initial step size. Defaults to " << getOptions().initialStepSize << "." << std::endl;
  os << "  --ida-min-step-size=<value>            Set the minimum absolute value of the step size. Defaults to " << getOptions().minStepSize << "." << std::endl;
  os << "  --ida-max-step-size=<value>            Set the maximum absolute value of the step size. Defaults to " << getOptions().maxStepSize << "." << std::endl;
  os << "  --ida-max-err-test-fails=<value>       Set the maximum number of error test failures in attempting one step. Defaults to " << getOptions().maxErrTestFails << "." << std::endl;
  // 中文：成对展示正反 override，明确用户可覆盖 dummy-state 模型默认策略。
  // English: Show both override directions so users can explicitly replace a
  // dummy-state model's default policy.
  os << "  --ida-suppress-alg-vars                Suppress algebraic variables in the local error test." << std::endl;
  os << "  --ida-include-alg-vars                 Include algebraic variables in the local error test, overriding a model default." << std::endl;
  os << "  --ida-max-nonlin-iters=<value>         Maximum number of nonlinear solver iterations in one solve attempt. Defaults to " << getOptions().maxNonlinIters << "." << std::endl;
  os << "  --ida-max-conv-fails=<value>           Maximum number of nonlinear solver convergence failures in one step. Defaults to " << getOptions().maxConvFails << "." << std::endl;
  os << "  --ida-nonlin-conv-coef=<value>         Safety factor in the nonlinear convergence test. Defaults to " << getOptions().nonlinConvCoef << "." << std::endl;
  os << "  --ida-nonlin-conv-coef-ic=<value>      Positive constant in the Newton iteration convergence test within the initial condition calculation. Defaults to " << getOptions().nonlinConvCoefIC << "." << std::endl;
  os << "  --ida-max-steps-ic=<value>             Maximum number of steps allowed for IC. Defaults to " << getOptions().maxStepsIC << "." << std::endl;
  os << "  --ida-max-jacs-ic=<value>              Maximum number of the approximate Jacobian or preconditioner evaluations allowed when the Newton iteration appears to be slowly converging. Defaults to " << getOptions().maxNumJacsIC << "." << std::endl;
  os << "  --ida-max-iters-ic=<value>             Maximum number of Newton iterations allowed in any one attempt to solve the initial conditions calculation problem. Defaults to " << getOptions().maxNumItersIC << "." << std::endl;
  os << "  --ida-line-search-off                  Disable the linesearch algorithm." << std::endl;

  os << "  --ida-print-jacobian                   Whether to print the Jacobian matrices while debugging." << std::endl;
  // clang-format on
}

void CommandLineOptions::parseCommandLineOptions(
    const argh::parser &options) const {
  // clang-format off
  getOptions().equidistantTimeGrid = static_cast<bool>(options("time-step") >> getOptions().timeStep);
  options("ida-relative-tolerance", getOptions().relativeTolerance) >> getOptions().relativeTolerance;
  options("ida-absolute-tolerance", getOptions().absoluteTolerance) >> getOptions().absoluteTolerance;
  options("ida-max-algebraic-abs-tol", getOptions().maxAlgebraicAbsoluteTolerance) >> getOptions().maxAlgebraicAbsoluteTolerance;
  options("ida-time-scaling-factor-ic", getOptions().timeScalingFactorInit) >> getOptions().timeScalingFactorInit;
  options("ida-equations-chunks-factor", getOptions().equationsChunksFactor) >> getOptions().equationsChunksFactor;
  options("ida-max-steps", getOptions().maxSteps) >> getOptions().maxSteps;
  options("ida-initial-step-size", getOptions().initialStepSize) >> getOptions().initialStepSize;
  options("ida-min-step-size", getOptions().minStepSize) >> getOptions().minStepSize;
  options("ida-max-step-size", getOptions().maxStepSize) >> getOptions().maxStepSize;
  options("ida-max-err-test-fails", getOptions().maxErrTestFails) >> getOptions().maxErrTestFails;
  // 中文：CLI 显式值优先于模型默认；正反开关互斥，避免无法追踪的顺序覆盖。
  // English: An explicit CLI choice overrides the model default. The positive
  // and negative switches are mutually exclusive to avoid order-dependent
  // behavior.
  bool suppressAlgebraic = options["ida-suppress-alg-vars"];
  bool includeAlgebraic = options["ida-include-alg-vars"];
  if (suppressAlgebraic && includeAlgebraic) {
    std::cerr << "Conflicting IDA options: --ida-suppress-alg-vars and "
                 "--ida-include-alg-vars cannot be used together."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (suppressAlgebraic) {
    getOptions().suppressAlgOverride = SUNTRUE;
  } else if (includeAlgebraic) {
    getOptions().suppressAlgOverride = SUNFALSE;
  } else {
    getOptions().suppressAlgOverride.reset();
  }
  options("ida-max-nonlin-iters", getOptions().maxNonlinIters) >> getOptions().maxNonlinIters;
  options("ida-max-conv-fails", getOptions().maxConvFails) >> getOptions().maxConvFails;
  options("ida-nonlin-conv-coef", getOptions().nonlinConvCoef) >> getOptions().nonlinConvCoef;
  options("ida-nonlin-conv-coef-ic", getOptions().nonlinConvCoefIC) >> getOptions().nonlinConvCoefIC;
  options("ida-max-steps-ic", getOptions().maxStepsIC) >> getOptions().maxStepsIC;
  options("ida-max-jacs-ic", getOptions().maxNumJacsIC) >> getOptions().maxNumJacsIC;
  options("ida-max-iters-ic", getOptions().maxNumItersIC) >> getOptions().maxNumItersIC;
  getOptions().lineSearchOff = options["ida-line-search-off"] ? SUNTRUE : SUNFALSE;
  getOptions().printJacobian = options["ida-print-jacobian"];
  // clang-format on
}
} // namespace marco::runtime::sundials::ida

#endif // SUNDIALS_ENABLE
#endif // CLI_ENABLE
