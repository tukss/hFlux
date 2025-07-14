cmake_minimum_required(VERSION 3.10)

# Set the project name
project(hflux
  LANGUAGES C CXX
  VERSION 0.1)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(HDF5 COMPONENTS C CXX)
find_package(Kokkos REQUIRED)

option(BUILD_SHARED_LIBS "Build shared library" ON)

add_subdirectory(src)
# We leave unit tests on by default in the early development stages.
option(ENABLE_UNIT_TESTS "Enable unit tests" ON)
mark_as_advanced(ENABLE_UNIT_TESTS)
# tests
if(ENABLE_UNIT_TESTS)
  message("\nConfiguring tests")
  enable_testing()
  add_subdirectory(test)
endif()

