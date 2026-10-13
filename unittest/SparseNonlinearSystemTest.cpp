#ifdef SUNDIALS_ENABLE

#include "marco/Runtime/Solvers/IDA/Instance.h"
#include "marco/Runtime/Solvers/KINSOL/SparseNonlinearSystem.h"
#include "marco/Runtime/Simulation/Options.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cmath>

using marco::runtime::sundials::kinsol::SparseNonlinearSystem;
using marco::runtime::sundials::ida::composeExecutionGroupDerivative;

// 中文：验证共享稀疏非线性内核、缩放 condition 估计和跨 execution-group
// sensitivity 组合，不构造全模型 dense Jacobian。
// English: Verify the shared sparse nonlinear kernel, scaled condition
// estimation, and cross-execution-group sensitivity composition without
// constructing a full-model dense Jacobian.
namespace {

void setDiagonal(SUNMatrix matrix, const std::vector<double> &diagonal) {
  sunindextype *rows = SUNSparseMatrix_IndexPointers(matrix);
  sunindextype *columns = SUNSparseMatrix_IndexValues(matrix);
  realtype *values = SUNSparseMatrix_Data(matrix);
  rows[0] = 0;
  for (uint64_t i = 0; i < diagonal.size(); ++i) {
    rows[i + 1] = static_cast<sunindextype>(i + 1);
    columns[i] = static_cast<sunindextype>(i);
    values[i] = diagonal[i];
  }
}

TEST(SparseNonlinearSystemTest, NonlinearIterationsBelongToEachSolve) {
  double target = 1;
  SparseNonlinearSystem system(
      {1, 1, 1e-12, 1e-14, 100, false, true},
      [&](N_Vector variables, N_Vector residuals) {
        N_VGetArrayPointer(residuals)[0] =
            N_VGetArrayPointer(variables)[0] - target;
        return KIN_SUCCESS;
      },
      [](N_Vector, N_Vector, SUNMatrix jacobian) {
        setDiagonal(jacobian, {1});
        return KIN_SUCCESS;
      });
  ASSERT_TRUE(system.initialize());
  for (double root : {1.0, 2.0, 3.0}) {
    target = root;
    ASSERT_EQ(system.solve(root), SparseNonlinearSystem::SolveStatus::Success);
    EXPECT_DOUBLE_EQ(N_VGetArrayPointer(system.getVariablesVector())[0], root);
    EXPECT_EQ(system.getLastNonlinearIterations(), 1u);
  }
  ASSERT_EQ(system.solve(4), SparseNonlinearSystem::SolveStatus::Success);
  EXPECT_EQ(system.getLastSolveFlag(), KIN_INITIAL_GUESS_OK);
  EXPECT_EQ(system.getLastNonlinearIterations(), 0u);
}

TEST(SparseNonlinearSystemTest, SmallStepWithoutResidualConvergenceIsRecoverable) {
  SparseNonlinearSystem system(
      {1, 1, 1e-12, 1e-4, 100, false, true},
      [](N_Vector variables, N_Vector residuals) {
        // The actual derivative is 1e12. The initial Newton step is smaller
        // than the selected step tolerance but has not solved the equation.
        double x = N_VGetArrayPointer(variables)[0];
        N_VGetArrayPointer(residuals)[0] = std::exp(1e12 * x);
        return KIN_SUCCESS;
      },
      [](N_Vector variables, N_Vector, SUNMatrix jacobian) {
        setDiagonal(jacobian,
                    {1e12 * std::exp(1e12 *
                                    N_VGetArrayPointer(variables)[0])});
        return KIN_SUCCESS;
      });
  ASSERT_TRUE(system.initialize());
  EXPECT_EQ(system.solve(0),
            SparseNonlinearSystem::SolveStatus::RecoverableFailure);
  EXPECT_EQ(system.getLastSolveFlag(), KIN_STEP_LT_STPTOL);
  EXPECT_GT(std::exp(1e12 *
                     N_VGetArrayPointer(system.getVariablesVector())[0]),
            1e-12);
}

TEST(SparseNonlinearSystemTest, PassiveStatisticsPreserveCallsAndFactorReuse) {
  auto &options = marco::runtime::simulation::getOptions();
  struct Restore {
    bool &value;
    bool saved;
    ~Restore() { value = saved; }
  } restore{options.profiling, options.profiling};
  std::vector<char> reference;
  for (bool profile : {false, true}) {
    options.profiling = profile;
    std::vector<char> calls;
    double coefficient = 2;
    SparseNonlinearSystem system(
        {1, 1, 1e-12, 1e-14, 100, true, true},
        [&](N_Vector variables, N_Vector residuals) {
          calls.push_back('r');
          N_VGetArrayPointer(residuals)[0] =
              coefficient * N_VGetArrayPointer(variables)[0] - 4;
          return KIN_SUCCESS;
        },
        [&](N_Vector, N_Vector, SUNMatrix jacobian) {
          calls.push_back('j');
          setDiagonal(jacobian, {coefficient});
          return KIN_SUCCESS;
        });
    ASSERT_TRUE(system.initialize());
    N_VGetArrayPointer(system.getVariablesVector())[0] = 3;
    ASSERT_EQ(system.solve(0), SparseNonlinearSystem::SolveStatus::Success);
    EXPECT_NEAR(N_VGetArrayPointer(system.getVariablesVector())[0], 2, 1e-12);
    ASSERT_TRUE(system.factorize(0));
    size_t afterFactor = calls.size();
    std::vector<double> response;
    ASSERT_TRUE(system.solveFactorized({6}, response));
    EXPECT_DOUBLE_EQ(response[0], 3);
    ASSERT_TRUE(system.solveFactorized({8}, response));
    EXPECT_DOUBLE_EQ(response[0], 4);
    EXPECT_EQ(calls.size(), afterFactor); // Multiple RHS do not reevaluate J.
    SparseNonlinearSystem::ConditionEstimate condition;
    ASSERT_TRUE(system.estimateScaledSubmatrixCondition({0}, {0}, condition));
    EXPECT_EQ(calls.size(), afterFactor); // A statistics read is passive too.
    const auto &stats = system.getStatistics();
    if (profile) {
      EXPECT_EQ(stats.solves, 1u);
      EXPECT_EQ(stats.factorizations, 1u);
      EXPECT_EQ(stats.rightHandSides, 2u);
      EXPECT_EQ(stats.conditions, 1u);
      EXPECT_EQ(stats.residuals, std::count(calls.begin(), calls.end(), 'r'));
      EXPECT_EQ(stats.jacobians, std::count(calls.begin(), calls.end(), 'j'));
      EXPECT_EQ(stats.nonlinearIterations, system.getLastNonlinearIterations());
      ASSERT_EQ(stats.solveFlags.size(), 1u);
      EXPECT_EQ(stats.solveFlags.at(KIN_SUCCESS), 1u);
      EXPECT_TRUE(system.getLastResidualObservation().available);
      EXPECT_TRUE(system.getLastResidualObservation().finite);
      EXPECT_NEAR(system.getLastResidualObservation().maximumNorm, 0, 1e-12);
    } else {
      EXPECT_EQ(stats.solves + stats.residuals + stats.jacobians +
                    stats.factorizations + stats.rightHandSides +
                    stats.conditions + stats.solveNanoseconds +
                    stats.factorNanoseconds + stats.rhsNanoseconds +
                    stats.conditionNanoseconds, 0u);
      EXPECT_TRUE(stats.solveFlags.empty());
      EXPECT_FALSE(system.getLastResidualObservation().available);
    }
    // A changed numerical matrix requires a fresh factorization. This is a
    // test-only comparison, not a new production reuse policy.
    coefficient = 4;
    ASSERT_TRUE(system.factorize(1));
    ASSERT_TRUE(system.solveFactorized({8}, response));
    EXPECT_DOUBLE_EQ(response[0], 2);
    if (profile) {
      EXPECT_DOUBLE_EQ(system.getLastResidualObservation().maximumNorm, 4);
      EXPECT_DOUBLE_EQ(system.getLastResidualObservation().scaledMaximumNorm, 4);
    }
    if (!profile)
      reference = calls;
    else
      EXPECT_EQ(calls, reference);
  }
}

TEST(SparseNonlinearSystemTest, LocalToleranceQualificationDependsOnEquationScale) {
  for (double variableUnit : {1e-3, 1.0, 1e3}) {
    for (double equationUnit : {1e-12, 1.0, 1e12}) {
      SCOPED_TRACE(::testing::Message() << variableUnit << ", " << equationUnit);
      SparseNonlinearSystem system(
          {1, 1, 1e-9, 1e-12, 100, true, true},
          [&](N_Vector variables, N_Vector residuals) {
            double x = N_VGetArrayPointer(variables)[0] / variableUnit;
            N_VGetArrayPointer(residuals)[0] = equationUnit * (x*x-4);
            return KIN_SUCCESS;
          },
          [&](N_Vector variables, N_Vector, SUNMatrix jacobian) {
            double x = N_VGetArrayPointer(variables)[0] / variableUnit;
            setDiagonal(jacobian, {2*x*equationUnit/variableUnit});
            return KIN_SUCCESS;
          });
      ASSERT_TRUE(system.initialize());
      N_VGetArrayPointer(system.getVariablesVector())[0] = 3*variableUnit;
      ASSERT_TRUE(system.factorize(0));
      ASSERT_TRUE(system.setScalingFromVariableNominals({3*variableUnit}));
      for (double initial : {3.0, 4.0, 5.0}) {
        N_VGetArrayPointer(system.getVariablesVector())[0] = initial*variableUnit;
        ASSERT_EQ(system.solve(0), SparseNonlinearSystem::SolveStatus::Success);
        double root = N_VGetArrayPointer(system.getVariablesVector())[0]/variableUnit;
        double scale = system.getResidualScale()[0];
        EXPECT_LE(std::abs(equationUnit*(root*root-4)*scale), 1e-9);
        if (equationUnit >= 1) {
          EXPECT_NEAR(root, 2.0, 1e-8);
          EXPECT_GT(system.getLastNonlinearIterations(), 0u);
        } else {
          // The existing max(1,row norm) anchor intentionally does not magnify
          // a tiny residual. Solver success alone cannot qualify physical
          // accuracy or justify a universal relaxed-tolerance default.
          EXPECT_DOUBLE_EQ(scale, 1.0);
          EXPECT_GT(std::abs(root-2.0), 0.01);
          if (initial == 3) {
            EXPECT_EQ(system.getLastSolveFlag(), KIN_INITIAL_GUESS_OK);
            EXPECT_EQ(system.getLastNonlinearIterations(), 0u);
          }
        }
      }
    }
  }
}

TEST(SparseNonlinearSystemTest, SolvesWithSparseKLUAndAnchorScaling) {
  SparseNonlinearSystem system(
      {/*size=*/1, /*nonZeros=*/1, /*functionNormTolerance=*/1e-12,
       /*scaledStepTolerance=*/1e-14, /*maximumNewtonStep=*/100,
       /*lineSearch=*/true, /*quietErrors=*/true},
      [](N_Vector variables, N_Vector residuals) {
        double x = N_VGetArrayPointer(variables)[0];
        N_VGetArrayPointer(residuals)[0] = x * x - 4;
        return KIN_SUCCESS;
      },
      [](N_Vector variables, N_Vector residuals, SUNMatrix jacobian) {
        setDiagonal(jacobian, {2 * N_VGetArrayPointer(variables)[0]});
        return KIN_SUCCESS;
      });

  ASSERT_TRUE(system.initialize());
  N_VGetArrayPointer(system.getVariablesVector())[0] = 3;
  ASSERT_TRUE(system.factorize(0));
  ASSERT_TRUE(system.setScalingFromVariableNominals({3}));
  EXPECT_EQ(system.solve(0), SparseNonlinearSystem::SolveStatus::Success);
  EXPECT_NEAR(N_VGetArrayPointer(system.getVariablesVector())[0], 2.0, 1e-12);
}

TEST(SparseNonlinearSystemTest, SparseSolveProducesAnalyticSchurCorrection) {
  constexpr double state = 3;
  constexpr double cj = 5;
  SparseNonlinearSystem system(
      {/*size=*/1, /*nonZeros=*/1, /*functionNormTolerance=*/1e-12,
       /*scaledStepTolerance=*/1e-14, /*maximumNewtonStep=*/100,
       /*lineSearch=*/true, /*quietErrors=*/true},
      [](N_Vector variables, N_Vector residuals) {
        N_VGetArrayPointer(residuals)[0] = N_VGetArrayPointer(variables)[0];
        return KIN_SUCCESS;
      },
      [](N_Vector variables, N_Vector residuals, SUNMatrix jacobian) {
        setDiagonal(jacobian, {1});
        return KIN_SUCCESS;
      });

  ASSERT_TRUE(system.initialize());
  ASSERT_TRUE(system.factorize(0));
  std::vector<double> response;
  ASSERT_TRUE(system.solveFactorized({-2 * state}, response));
  ASSERT_EQ(response.size(), 1u);
  double reduced = cj - response[0];
  EXPECT_DOUBLE_EQ(reduced, cj + 2 * state);
}

TEST(SparseNonlinearSystemTest,
     PropagatesSensitivityAcrossExecutionGroupDAG) {
  SparseNonlinearSystem upstream(
      {/*size=*/1, /*nonZeros=*/1, /*functionNormTolerance=*/1e-12,
       /*scaledStepTolerance=*/1e-14, /*maximumNewtonStep=*/100,
       /*lineSearch=*/true, /*quietErrors=*/true},
      [](N_Vector variables, N_Vector residuals) {
        N_VGetArrayPointer(residuals)[0] =
            2 * N_VGetArrayPointer(variables)[0];
        return KIN_SUCCESS;
      },
      [](N_Vector variables, N_Vector residuals, SUNMatrix jacobian) {
        setDiagonal(jacobian, {2});
        return KIN_SUCCESS;
      });
  SparseNonlinearSystem downstream(
      {/*size=*/1, /*nonZeros=*/1, /*functionNormTolerance=*/1e-12,
       /*scaledStepTolerance=*/1e-14, /*maximumNewtonStep=*/100,
       /*lineSearch=*/true, /*quietErrors=*/true},
      [](N_Vector variables, N_Vector residuals) {
        N_VGetArrayPointer(residuals)[0] =
            3 * N_VGetArrayPointer(variables)[0];
        return KIN_SUCCESS;
      },
      [](N_Vector variables, N_Vector residuals, SUNMatrix jacobian) {
        setDiagonal(jacobian, {3});
        return KIN_SUCCESS;
      });

  ASSERT_TRUE(upstream.initialize());
  ASSERT_TRUE(downstream.initialize());
  ASSERT_TRUE(upstream.factorize(0));
  ASSERT_TRUE(downstream.factorize(0));

  // G1 = 2*z1 - s, hence response1 = J1^-1*G1_s = -1/2.
  std::vector<double> response1;
  ASSERT_TRUE(upstream.solveFactorized({-1}, response1));
  ASSERT_EQ(response1.size(), 1u);

  // G2 = 3*z2 + 4*z1 - 5*s. The total derivative is
  // G2_s - G2_z1*response1 = -3, hence response2 = -1.
  double downstreamDerivative = 0;
  ASSERT_TRUE(composeExecutionGroupDerivative(
      -5, {4}, {response1[0]}, downstreamDerivative));
  std::vector<double> response2;
  ASSERT_TRUE(
      downstream.solveFactorized({downstreamDerivative}, response2));
  ASSERT_EQ(response2.size(), 1u);

  // R = s' + 7*z1 + 11*z2. The complete Schur derivative is
  // cj - 7*response1 - 11*response2.
  constexpr double cj = 6;
  double reduced = 0;
  ASSERT_TRUE(composeExecutionGroupDerivative(
      cj, {7, 11}, {response1[0], response2[0]}, reduced));
  EXPECT_DOUBLE_EQ(response1[0], -0.5);
  EXPECT_DOUBLE_EQ(response2[0], -1);
  EXPECT_DOUBLE_EQ(reduced, cj + 14.5);
}

TEST(SparseNonlinearSystemTest,
     LeavesDisconnectedExecutionGroupsOutOfSensitivity) {
  double result = 0;
  EXPECT_TRUE(composeExecutionGroupDerivative(3.5, {}, {}, result));
  EXPECT_DOUBLE_EQ(result, 3.5);
  EXPECT_FALSE(composeExecutionGroupDerivative(3.5, {1}, {}, result));
}

TEST(SparseNonlinearSystemTest, ExtractsOnlyRequestedSparseEntries) {
  SparseNonlinearSystem system(
      {/*size=*/2, /*nonZeros=*/2, /*functionNormTolerance=*/1e-12,
       /*scaledStepTolerance=*/1e-14, /*maximumNewtonStep=*/100,
       /*lineSearch=*/true, /*quietErrors=*/true},
      [](N_Vector variables, N_Vector residuals) {
        double *values = N_VGetArrayPointer(variables);
        double *result = N_VGetArrayPointer(residuals);
        result[0] = 2 * values[0];
        result[1] = 3 * values[1];
        return KIN_SUCCESS;
      },
      [](N_Vector variables, N_Vector residuals, SUNMatrix jacobian) {
        setDiagonal(jacobian, {2, 3});
        return KIN_SUCCESS;
      });

  ASSERT_TRUE(system.initialize());
  ASSERT_TRUE(system.factorize(0));
  std::vector<double> entries;
  ASSERT_TRUE(system.extractEntries({0, 0, 1}, {0, 1, 1}, entries));
  EXPECT_EQ(entries, (std::vector<double>{2, 0, 3}));
}

TEST(SparseNonlinearSystemTest,
     EstimatesDenseConditionInNominalScaledCoordinates) {
  SparseNonlinearSystem system(
      {/*size=*/2, /*nonZeros=*/2, /*functionNormTolerance=*/1e-12,
       /*scaledStepTolerance=*/1e-14, /*maximumNewtonStep=*/100,
       /*lineSearch=*/true, /*quietErrors=*/true},
      [](N_Vector variables, N_Vector residuals) { return KIN_SUCCESS; },
      [](N_Vector variables, N_Vector residuals, SUNMatrix jacobian) {
        setDiagonal(jacobian, {1e-12, 1e12});
        return KIN_SUCCESS;
      });

  ASSERT_TRUE(system.initialize());
  ASSERT_TRUE(system.factorize(0));
  ASSERT_TRUE(system.setScalingFromVariableNominals({1e12, 1e-12}));
  SparseNonlinearSystem::ConditionEstimate estimate;
  ASSERT_TRUE(system.estimateScaledSubmatrixCondition({0, 1}, {0, 1},
                                                       estimate));
  EXPECT_FALSE(estimate.sparseEstimator);
  EXPECT_FALSE(estimate.rankDeficient);
  EXPECT_EQ(estimate.rank, 2u);
  EXPECT_NEAR(estimate.oneNormCondition, 1.0, 1e-12);
}

TEST(SparseNonlinearSystemTest,
     UsesSparseKLUConditionEstimatorAboveDenseLimit) {
  constexpr uint64_t size = 65;
  SparseNonlinearSystem system(
      {/*size=*/size, /*nonZeros=*/size, /*functionNormTolerance=*/1e-12,
       /*scaledStepTolerance=*/1e-14, /*maximumNewtonStep=*/100,
       /*lineSearch=*/true, /*quietErrors=*/true},
      [](N_Vector variables, N_Vector residuals) { return KIN_SUCCESS; },
      [](N_Vector variables, N_Vector residuals, SUNMatrix jacobian) {
        std::vector<double> diagonal(size, 1);
        for (uint64_t i = 0; i < size; ++i) {
          diagonal[i] = i % 2 == 0 ? 1e-9 : 1e9;
        }
        setDiagonal(jacobian, diagonal);
        return KIN_SUCCESS;
      });

  ASSERT_TRUE(system.initialize());
  ASSERT_TRUE(system.factorize(0));
  std::vector<double> nominals(size, 1);
  std::vector<uint64_t> indices(size, 0);
  for (uint64_t i = 0; i < size; ++i) {
    nominals[i] = i % 2 == 0 ? 1e9 : 1e-9;
    indices[i] = i;
  }
  ASSERT_TRUE(system.setScalingFromVariableNominals(nominals));
  SparseNonlinearSystem::ConditionEstimate estimate;
  ASSERT_TRUE(
      system.estimateScaledSubmatrixCondition(indices, indices, estimate));
  EXPECT_TRUE(estimate.sparseEstimator);
  EXPECT_FALSE(estimate.rankDeficient);
  EXPECT_EQ(estimate.rank, size);
  EXPECT_NEAR(estimate.oneNormCondition, 1.0, 1e-8);
}

TEST(SparseNonlinearSystemTest, CallbackFailureIsFatal) {
  SparseNonlinearSystem system(
      {/*size=*/1, /*nonZeros=*/1, /*functionNormTolerance=*/1e-12,
       /*scaledStepTolerance=*/1e-14, /*maximumNewtonStep=*/100,
       /*lineSearch=*/true, /*quietErrors=*/true},
      [](N_Vector variables, N_Vector residuals) {
        return KIN_SYSFUNC_FAIL;
      },
      [](N_Vector variables, N_Vector residuals, SUNMatrix jacobian) {
        setDiagonal(jacobian, {1});
        return KIN_SUCCESS;
      });

  EXPECT_EQ(system.solve(0),
            SparseNonlinearSystem::SolveStatus::FatalFailure);
}

TEST(SparseNonlinearSystemTest, ScratchReusePreservesTrialsAndNestedInstances) {
  int nestedCalls = 0;
  SparseNonlinearSystem nested(
      {1, 1, 1e-12, 1e-14, 100, true, true},
      [&](N_Vector variables, N_Vector residuals) {
        ++nestedCalls;
        N_VGetArrayPointer(residuals)[0] =
            5 * N_VGetArrayPointer(variables)[0];
        return KIN_SUCCESS;
      },
      [](N_Vector, N_Vector, SUNMatrix jacobian) {
        setDiagonal(jacobian, {5});
        return KIN_SUCCESS;
      });
  ASSERT_TRUE(nested.initialize());
  bool fail = false;
  int residualCalls = 0;
  int jacobianCalls = 0;
  SparseNonlinearSystem system(
      {2, 2, 1e-12, 1e-14, 100, true, true},
      [&](N_Vector variables, N_Vector residuals) {
        ++residualCalls;
        const double *x = N_VGetArrayPointer(variables);
        double *f = N_VGetArrayPointer(residuals);
        f[0] = x[0] * x[0];
        f[1] = 3 * x[1];
        return fail ? KIN_SYSFUNC_FAIL : KIN_SUCCESS;
      },
      [&](N_Vector variables, N_Vector residuals, SUNMatrix jacobian) {
        ++jacobianCalls;
        const double *x = N_VGetArrayPointer(variables);
        EXPECT_DOUBLE_EQ(N_VGetArrayPointer(residuals)[0], x[0] * x[0]);
        EXPECT_DOUBLE_EQ(N_VGetArrayPointer(residuals)[1], 3 * x[1]);
        // A nested, distinct solver owns disjoint scratch and numerical state.
        if (!nested.factorize(0))
          return KIN_SYSFUNC_FAIL;
        std::vector<double> response;
        if (!nested.solveFactorized({15}, response))
          return KIN_SYSFUNC_FAIL;
        EXPECT_DOUBLE_EQ(response[0], 3);
        setDiagonal(jacobian, {2 * x[0], 3});
        return KIN_SUCCESS;
      });
  ASSERT_TRUE(system.initialize());
  using Status = SparseNonlinearSystem::FactorizationStatus;
  for (int trial = 1; trial <= 4; ++trial) {
    N_VGetArrayPointer(system.getVariablesVector())[0] = trial;
    N_VGetArrayPointer(system.getVariablesVector())[1] = -trial;
    fail = true;
    EXPECT_EQ(system.factorizeWithStatus(0), Status::EvaluationFailure);
    fail = false;
    ASSERT_EQ(system.factorizeWithStatus(0), Status::Success);
    std::vector<double> response;
    ASSERT_TRUE(system.solveFactorized({2.0 * trial, -6}, response));
    EXPECT_EQ(response, (std::vector<double>{1, -2}));
    ASSERT_TRUE(system.solveFactorized({0, 0}, response));
    EXPECT_EQ(response, (std::vector<double>{0, 0}));
    EXPECT_FALSE(system.solveFactorized({1}, response));
    ASSERT_TRUE(system.solveFactorized({-4.0 * trial, 9}, response));
    EXPECT_EQ(response, (std::vector<double>{-2, 3}));
  }
  EXPECT_EQ(residualCalls, 8);
  EXPECT_EQ(jacobianCalls, 4);
  EXPECT_EQ(nestedCalls, 4);
}

TEST(SparseNonlinearSystemTest,
     ConditionScratchRecomputesChangedMatricesScalesAndSubsets) {
  std::vector<double> diagonal{2, 3, 4};
  SparseNonlinearSystem system(
      {3, 3, 1e-12, 1e-14, 100, true, true},
      [](N_Vector, N_Vector residuals) {
        N_VConst(0, residuals);
        return KIN_SUCCESS;
      },
      [&](N_Vector, N_Vector, SUNMatrix jacobian) {
        setDiagonal(jacobian, diagonal);
        return KIN_SUCCESS;
      });
  ASSERT_TRUE(system.factorize(0));
  SparseNonlinearSystem::ConditionEstimate estimate;
  auto check = [&](const std::vector<uint64_t> &indices, double expected) {
    ASSERT_TRUE(system.estimateScaledSubmatrixCondition(indices, indices,
                                                        estimate));
    EXPECT_FALSE(estimate.rankDeficient);
    EXPECT_EQ(estimate.rank, indices.size());
    EXPECT_NEAR(estimate.oneNormCondition, expected, 1e-12);
  };
  check({0, 1, 2}, 2);
  check({2, 0}, 2);
  check({1}, 1);
  check({}, 1);
  EXPECT_FALSE(system.estimateScaledSubmatrixCondition({0, 0}, {0, 1},
                                                       estimate));
  check({0, 1, 2}, 2);
  // Same time and storage do not authorize any numerical reuse.
  diagonal = {8, 3, 4};
  ASSERT_TRUE(system.factorize(0));
  check({0, 1, 2}, 8.0 / 3.0);
  ASSERT_TRUE(system.setScaling({1, 1, 1}, {1.0 / 8, 1.0 / 3, 1.0 / 4}));
  check({0, 1, 2}, 1);
  diagonal = {0, 3, 4};
  EXPECT_EQ(system.factorizeWithStatus(0),
            SparseNonlinearSystem::FactorizationStatus::Singular);
  ASSERT_TRUE(system.estimateScaledSubmatrixCondition({0}, {0}, estimate));
  EXPECT_TRUE(estimate.rankDeficient);
  diagonal = {8, 3, 4};
  ASSERT_TRUE(system.factorize(0));
  check({0, 1, 2}, 1);
}

TEST(SparseNonlinearSystemTest, FactorizationSeparatesSingularAndCallbackFailure) {
  bool failCallback = false;
  bool singular = true;
  SparseNonlinearSystem system(
      {/*size=*/2, /*nonZeros=*/2, /*functionNormTolerance=*/1e-12,
       /*scaledStepTolerance=*/1e-14, /*maximumNewtonStep=*/100,
       /*lineSearch=*/true, /*quietErrors=*/true},
      [&](N_Vector variables, N_Vector residuals) {
        N_VConst(0, residuals);
        return failCallback ? KIN_SYSFUNC_FAIL : KIN_SUCCESS;
      },
      [&](N_Vector variables, N_Vector residuals, SUNMatrix jacobian) {
        setDiagonal(jacobian, {1, singular ? 0.0 : 1.0});
        return KIN_SUCCESS;
      });
  using Status = SparseNonlinearSystem::FactorizationStatus;
  EXPECT_EQ(system.factorizeWithStatus(0), Status::Singular);
  failCallback = true;
  EXPECT_EQ(system.factorizeWithStatus(0), Status::EvaluationFailure);
  failCallback = false;
  singular = false;
  EXPECT_EQ(system.factorizeWithStatus(0), Status::Success);
  std::vector<double> response;
  ASSERT_TRUE(system.solveFactorized({2, -3}, response));
  EXPECT_EQ(response, (std::vector<double>{2, -3}));
  // Initial singularity and a failed refactor after a successful setup have
  // different KLU lifetimes. Neither may expose stale factors to a RHS solve.
  for (int trial = 0; trial < 2; ++trial) {
    singular = true;
    EXPECT_EQ(system.factorizeWithStatus(0), Status::Singular);
    EXPECT_FALSE(system.solveFactorized({2, -3}, response));
    singular = false;
    ASSERT_EQ(system.factorizeWithStatus(0), Status::Success);
    ASSERT_TRUE(system.solveFactorized({2, -3}, response));
    EXPECT_EQ(response, (std::vector<double>{2, -3}));
    failCallback = true;
    EXPECT_EQ(system.factorizeWithStatus(0), Status::EvaluationFailure);
    EXPECT_FALSE(system.solveFactorized({2, -3}, response));
    failCallback = false;
  }
}

TEST(SparseNonlinearSystemTest, ConditionMatchesIndependentRankOneInverse) {
  // Sherman-Morrison gives an independent closed-form inverse, including
  // non-symmetric matrices and row/column permutations that exercise pivoting.
  // Dimensions cover both the interleaved and original larger-block paths.
  for (uint64_t size : {1, 2, 3, 4, 7, 8, 16, 17, 64}) {
    SCOPED_TRACE(size);
    std::vector<double> diagonal(size), u(size), v(size);
    double denominator = 1;
    for (uint64_t i = 0; i < size; ++i) {
      diagonal[i] = 2.0 + double(i) / (size + 1);
      u[i] = (int(i % 3) - 1) / 8.0;
      v[i] = (int(i % 5) - 2) / 16.0;
      denominator += u[i] * v[i] / diagonal[i];
    }
    double norm = 0, inverseNorm = 0;
    for (uint64_t column = 0; column < size; ++column) {
      double sum = 0, inverseSum = 0;
      for (uint64_t row = 0; row < size; ++row) {
        sum += std::abs((row == column ? diagonal[row] : 0) +
                        u[row] * v[column]);
        inverseSum += std::abs((row == column ? 1 / diagonal[row] : 0) -
            (u[row] / diagonal[row]) * (v[column] / diagonal[column]) /
                denominator);
      }
      norm = std::max(norm, sum);
      inverseNorm = std::max(inverseNorm, inverseSum);
    }
    SparseNonlinearSystem system(
        {size, size * size, 1e-12, 1e-14, 100, true, true},
        [](N_Vector, N_Vector residuals) {
          N_VConst(0, residuals);
          return KIN_SUCCESS;
        },
        [&](N_Vector, N_Vector, SUNMatrix matrix) {
          auto *rows = SUNSparseMatrix_IndexPointers(matrix);
          auto *columns = SUNSparseMatrix_IndexValues(matrix);
          auto *values = SUNSparseMatrix_Data(matrix);
          for (uint64_t row = 0; row < size; ++row) {
            rows[row] = row * size;
            const auto sourceRow = (row + 1) % size;
            for (uint64_t column = 0; column < size; ++column) {
              const auto sourceColumn = size - column - 1;
              columns[row * size + column] = column;
              values[row * size + column] =
                  (sourceRow == sourceColumn ? diagonal[sourceRow] : 0) +
                  u[sourceRow] * v[sourceColumn];
            }
          }
          rows[size] = size * size;
          return KIN_SUCCESS;
        });
    ASSERT_TRUE(system.factorize(0));
    std::vector<uint64_t> indices(size);
    for (uint64_t i = 0; i < size; ++i) indices[i] = i;
    SparseNonlinearSystem::ConditionEstimate estimate;
    ASSERT_TRUE(system.estimateScaledSubmatrixCondition(indices, indices,
                                                        estimate));
    EXPECT_FALSE(estimate.rankDeficient);
    EXPECT_EQ(estimate.rank, size);
    EXPECT_NEAR(estimate.oneNormCondition, norm * inverseNorm, 1e-11);
    // Re-querying a smaller geometry must not expose stale matrix columns.
    ASSERT_TRUE(system.estimateScaledSubmatrixCondition({0}, {size - 1},
                                                        estimate));
    if (!estimate.rankDeficient) {
      EXPECT_DOUBLE_EQ(estimate.oneNormCondition, 1);
    }
    ASSERT_TRUE(system.estimateScaledSubmatrixCondition(indices, indices,
                                                        estimate));
    EXPECT_NEAR(estimate.oneNormCondition, norm * inverseNorm, 1e-11);
  }
}

} // namespace

#endif // SUNDIALS_ENABLE
