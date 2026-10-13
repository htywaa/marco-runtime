add_custom_target(MARCORuntimeUnitTests)

set_target_properties(MARCORuntimeUnitTests
    PROPERTIES FOLDER "MARCO runtime unit tests")

include(GoogleTest)

marco_add_unittest(MARCORuntimeTest
    BuiltInFunctionsTest.cpp
    MathTest.cpp
    UtilityFunctionsTest.cpp)

if (MARCO_ENABLE_SUNDIALS)
    # 中文：这些测试直接覆盖 IDA/KINSOL 共享稀疏内核与切换事务。
    # English: These tests cover the shared IDA/KINSOL sparse core and switch transactions.
    target_sources(MARCORuntimeTest PRIVATE
        DiscontinuityCoordinatorTest.cpp
        IDAConstraintConditionTest.cpp
        IDALocalOnlyTest.cpp
        IDABatchResidualTest.cpp
        IDABatchJacobianTest.cpp
        KINSOLTimedCallbacksTest.cpp
        SolverSingleWorkerTest.cpp
        ScalarBlockPlanTest.cpp
    ScalarBlockSolverTest.cpp
        SparseNonlinearSystemTest.cpp)
    target_compile_definitions(MARCORuntimeTest PRIVATE SUNDIALS_ENABLE)
endif()

target_link_libraries(MARCORuntimeTest
    PUBLIC
    Support
    OptionsSimulation)

if (MARCO_CLI)
    target_sources(MARCORuntimeTest PRIVATE
        SimulationOutputTest.cpp
        ../lib/Drivers/Driver.cpp)
    target_compile_definitions(MARCORuntimeTest PRIVATE CLI_ENABLE)
    target_link_libraries(MARCORuntimeTest PUBLIC Simulation_static PrinterCSV)

    foreach(driver IN ITEMS EulerForward RungeKutta)
        marco_add_unittest(MARCORuntime${driver}Test DriverEndTimeTest.cpp)
        target_compile_definitions(MARCORuntime${driver}Test PRIVATE CLI_ENABLE)
        if (driver STREQUAL "RungeKutta")
            target_compile_definitions(MARCORuntime${driver}Test PRIVATE TEST_RUNGE_KUTTA)
        endif()
        target_link_libraries(MARCORuntime${driver}Test PUBLIC
            Support OptionsSimulation Simulation_static PrinterCSV
            Driver${driver} Solver${driver})
        if (MARCO_ENABLE_SUNDIALS)
            target_link_libraries(MARCORuntime${driver}Test PUBLIC DriverKINSOL)
        endif()
    endforeach()
endif()

if (MARCO_ENABLE_SUNDIALS)
    target_link_libraries(MARCORuntimeTest PUBLIC
        SolverIDA_static
        SolverKINSOL_static)
    if (MARCO_CLI)
        target_sources(MARCORuntimeTest PRIVATE
            KINSOLCommandLineTest.cpp IDACommandLineTest.cpp)
        target_compile_definitions(MARCORuntimeTest PRIVATE CLI_ENABLE)
        target_link_libraries(MARCORuntimeTest PUBLIC DriverKINSOL DriverIDA)
    endif()
endif()
