# ------------------------------------------------------------
# Define a helper to add a CTest target
# ------------------------------------------------------------
# Usage:
#   add_hFlux(<test-name>
#     SRCS    <list of source files...>
#     INPUT   <input-file-basename>   # looked for under ../inputs/<name>
#   )
#
# The INPUT argument should be just the filename (e.g. "conservation_test.input");
# the function will prefix it with ${CMAKE_CURRENT_SOURCE_DIR}/../inputs/
#
function(add_hFlux_test test_name)
  cmake_parse_arguments(
    PT   # prefix for options/args
    ""   # no boolean options
    "INPUT"   # one-value args
    "SRCS"    # multi-value args
    ${ARGN}
  )

  if(NOT PT_SRCS)
    message(FATAL_ERROR "add_hFlux_test(${test_name}): missing SRCS")
  endif()
  if(NOT PT_INPUT)
    message(FATAL_ERROR "add_hFlux_test(${test_name}): missing INPUT")
  endif()

  # 1) Add the executable
  add_executable(${test_name} ${PT_SRCS})

  # 2) Add include directories and link hflux library
  target_link_libraries(${test_name} PRIVATE hflux::hflux)

  # 3) C++20
  target_compile_features(${test_name} PRIVATE cxx_std_20)

  # 5) Register the test with CTest, passing the input file
  add_test(
    NAME ${test_name}
    COMMAND ${test_name}
            -i  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../inputs/${PT_INPUT}>
  )
endfunction()
