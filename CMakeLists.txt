cmake_minimum_required(VERSION 3.10)
project(Numerical_Recipes)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wno-reorder")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(src/include)

add_executable(main src/main.cpp)
