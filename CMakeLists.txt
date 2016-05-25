cmake_minimum_required(VERSION 2.8)

project(Motion-Deblurring)

# Set compiler flag for C++ 2011 standard
if (CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_CXX_FLAGS "-std=c++0x")
endif()

# some options
option(IMWRITE "write intermediate images" OFF)

# were does the binaries and libs are delivered?
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include_directories(motion-deblurring/include
                    two-phase-kernel/include
                    external/argtable
                    external/match/src
                    utils/
                    tools/)

add_subdirectory(motion-deblurring/)
add_subdirectory(two-phase-kernel/)
add_subdirectory(external/argtable/)
add_subdirectory(external/match/)
add_subdirectory(utils/)
add_subdirectory(tools/)
