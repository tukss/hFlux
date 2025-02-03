cmake_minimum_required(VERSION 3.10)

# Set the project name
project(hFlux)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the source files (all .cpp files in the src folder)
file(GLOB SOURCES "src/*.cpp")

# Add the library (shared library)
add_library(hFlux SHARED ${SOURCES})

# Include the headers directory for the library
target_include_directories(hFlux PRIVATE "external/mdspan/include/mdspan")
target_include_directories(hFlux PUBLIC "src")

# Optionally set the output directory for the shared library
set_target_properties(hFlux PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
