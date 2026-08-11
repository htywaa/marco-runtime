marco_add_runtime_shared_library(SolverSUNDIALS
    # 中文：统一 superdense discontinuity 事务协调状态基与未来 Modelica 事件。
    # English: One superdense transaction coordinates basis switches and future Modelica events.
    DiscontinuityCoordinator.cpp
    Instance.cpp
    Profiler.cpp)

marco_add_runtime_static_library(SolverSUNDIALS_static
    # 中文：静态 runtime 使用同一事务协调语义。
    # English: The static runtime uses the same transaction-coordination semantics.
    DiscontinuityCoordinator.cpp
    Instance.cpp
    Profiler.cpp)

set(LINKED_LIBS
    Modeling)

target_link_libraries(SolverSUNDIALS
    PUBLIC
    ${LINKED_LIBS})

target_link_libraries(SolverSUNDIALS_static
    PUBLIC
    ${LINKED_LIBS})

if (MARCO_ENABLE_SUNDIALS)
    set(LINKED_LIBS
        SUNDIALS::KLU
        SUNDIALS::nvecserial
        SUNDIALS::sunmatrixsparse)

    target_link_libraries(SolverSUNDIALS
        PUBLIC
        ${LINKED_LIBS})

    target_link_libraries(SolverSUNDIALS_static
        PUBLIC
        ${LINKED_LIBS})
endif()

if (MARCO_PROFILING)
    target_link_libraries(SolverSUNDIALS
        PUBLIC
        Profiling)

    target_link_libraries(SolverSUNDIALS_static
            PUBLIC
            Profiling)
endif()
