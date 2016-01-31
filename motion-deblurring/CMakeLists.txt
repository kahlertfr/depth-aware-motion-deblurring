cmake_minimum_required(VERSION 2.8)

project(Motion-Deblurring)

# Set compiler flag for C++ 2011 standard
if (CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_CXX_FLAGS "-std=c++0x")
endif()

find_package(OpenCV 3 REQUIRED)

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# Header files are separated from the source to increase
# interface abstraction
include_directories(include/)

# -------------------------
# Motion deblurring library
# -------------------------

# Add paths to linker search and installed rpath
# @see: https://cmake.org/Wiki/CMake_RPATH_handling#Common_questions
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(LIB_SOURCES src/depth_aware_deblurring.cpp
                src/disparity_estimation.cpp
                src/region_tree.cpp)
add_library(libmdeblur SHARED ${LIB_SOURCES} $<TARGET_OBJECTS:utils>)
set_target_properties(libmdeblur PROPERTIES
                                 OUTPUT_NAME "mdeblur")
target_link_libraries(libmdeblur ${OpenCV_LIBS})

# The main.cc is just the command line interface for the library. Using this
# separation you can use the library functions independently of the 
# executable
add_executable(motion-deblurring src/main.cpp)
target_link_libraries(motion-deblurring libmdeblur libargtable)

# ------------
# Installation
# ------------
install(TARGETS libmdeblur motion-deblurring 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY include/
        DESTINATION include/motion_deblurring
        FILES_MATCHING PATTERN "*.hpp")
