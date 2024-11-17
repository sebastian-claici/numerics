# Add Catch2 as a dependency
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.0.1 # Replace with the latest version if necessary
)
FetchContent_MakeAvailable(Catch2)

add_executable(tests 
    main.cpp 
    test_core.cpp 
    test_ops.cpp 
    test_linalg.cpp
)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

include(CTest)
include(Catch)

catch_discover_tests(tests)

