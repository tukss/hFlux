cmake_minimum_required(VERSION 3.10)

# Set the project name
project(hflux
  LANGUAGES CXX
  VERSION 0.1)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS "Build shared library" ON)

add_subdirectory(src)
