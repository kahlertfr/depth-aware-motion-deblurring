cmake_minimum_required(VERSION 2.8)

project(Two-Phase-Kernel)

# Set compiler flag for C++ 2011 standard
if (CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_CXX_FLAGS "-std=c++0x")
endif()

find_package(OpenCV 3 REQUIRED)

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# were does the binaries and libs are delivered?
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Header files are separated from the source to increase
# interface abstraction
include_directories(include/)

# -------------------------
# two phase kernel estimation library
# -------------------------

# Add paths to linker search and installed rpath
# @see: https://cmake.org/Wiki/CMake_RPATH_handling#Common_questions
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(LIB_SOURCES src/two_phase_psf_estimation.cpp
                src/kernel_initialization.cpp)
add_library(lib2psfest SHARED ${LIB_SOURCES})
set_target_properties(lib2psfest PROPERTIES
                                 OUTPUT_NAME "2psfest")
target_link_libraries(lib2psfest ${OpenCV_LIBS})

# The main.cc is just the command line interface for the library. Using this
# separation you can use the library functions independently of the 
# executable
add_executable(two-phase-kernel src/main.cpp)
target_link_libraries(two-phase-kernel lib2psfest libargtable)

# ------------
# Installation
# ------------
install(TARGETS lib2psfest two-phase-kernel
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY include/
        DESTINATION include/two_phase_kernel
        FILES_MATCHING PATTERN "*.hpp")
