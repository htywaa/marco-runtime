#ifdef SUNDIALS_ENABLE

#include "marco/Runtime/Solvers/IDA/Instance.h"
#include "marco/Runtime/Solvers/KINSOL/SparseNonlinearSystem.h"
#include "gtest/gtest.h"
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

} // namespace

#endif // SUNDIALS_ENABLE
