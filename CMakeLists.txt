cmake_minimum_required(VERSION 2.8)

project(Motion-Deblurring)

# Set compiler flag for C++ 2011 standard
if (CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_CXX_FLAGS "-std=c++0x")
endif()

include_directories(motion-deblurring/include
                    two-phase-kernel/include
                    external/)

add_subdirectory(motion-deblurring/)
add_subdirectory(two-phase-kernel/)
add_subdirectory(external/)
