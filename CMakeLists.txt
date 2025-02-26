cmake_minimum_required(VERSION 3.10)
project(circle_algorithm_benchmark)

# Enforce out-of-source builds (prevent source tree pollution)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR 
    "In-source builds are not allowed. Please create a 'build' directory and run CMake from there.\n"
    "Example:\n"
    "  mkdir -p build\n"
    "  cd build\n"
    "  cmake ..\n"
    "  make\n"
    "Or use CMake's '-B' option to specify the build directory:\n"
    "  cmake -B build && cmake --build build\n"
    "Removing created CMakeCache.txt file and CMakeFiles directory..."
  )
endif()

# Enable C++23 features
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required packages
find_package(benchmark REQUIRED)
find_package(GTest REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Set optimization flags for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -flto")

# List all header files
set(HEADER_FILES
  src/placement_algorithms.h
  src/utils.hpp
)

# Add the main algorithm library
add_library(placement_algorithms
  src/placement_algorithms.cpp
  src/utils.cpp
  ${HEADER_FILES}
)

# Add compiler flags
target_compile_options(placement_algorithms PRIVATE 
  -Wall -Wextra -Wpedantic -Werror=return-type
)

# Add the benchmark executable
add_executable(benchmark_algorithms
  src/benchmark_algorithms.cpp
)

# Add the test executable
add_executable(test_algorithms
  src/test_algorithms.cpp
)

# Apply same compiler flags to executables
target_compile_options(benchmark_algorithms PRIVATE 
  -Wall -Wextra -Wpedantic -Werror=return-type
)
target_compile_options(test_algorithms PRIVATE 
  -Wall -Wextra -Wpedantic -Werror=return-type
)

# Link-time optimization for all targets
set_property(TARGET placement_algorithms PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
set_property(TARGET benchmark_algorithms PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
set_property(TARGET test_algorithms PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

# Link dependencies
target_link_libraries(benchmark_algorithms
  placement_algorithms
  benchmark::benchmark
  benchmark::benchmark_main
)

target_link_libraries(test_algorithms
  placement_algorithms
  GTest::GTest
  GTest::Main
)
