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
        KINSOLTimedCallbacksTest.cpp
        SparseNonlinearSystemTest.cpp)
    target_compile_definitions(MARCORuntimeTest PRIVATE SUNDIALS_ENABLE)
endif()

target_link_libraries(MARCORuntimeTest
    PUBLIC
    Support
    OptionsSimulation)

if (MARCO_ENABLE_SUNDIALS)
    target_link_libraries(MARCORuntimeTest PUBLIC
        SolverIDA_static
        SolverKINSOL_static)
endif()
