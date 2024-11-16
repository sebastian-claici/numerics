cmake_minimum_required(VERSION 3.10)
project(Numerical_Recipes)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wno-reorder")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

# Check if the compiler is clang++ and set it if available
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Using Clang: ${CMAKE_CXX_COMPILER}")
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
endif()

# Check and enable OpenMP
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found and enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "OpenMP not found")
endif()

# Add Catch2 as a dependency
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.0.1 # Replace with the latest version if necessary
)
FetchContent_MakeAvailable(Catch2)

add_executable(tests test/test_ops.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

include(CTest)
include(Catch)

catch_discover_tests(tests)
