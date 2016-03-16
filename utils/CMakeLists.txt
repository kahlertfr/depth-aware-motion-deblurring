# Small utility library containing some neat
# and convinient functions to make programming
# less redundant
project(utils)

set(SOURCES utils.cpp
            coherence_filter.cpp
            deconvolution.cpp)

add_library(utils OBJECT ${SOURCES})
set_target_properties(utils PROPERTIES POSITION_INDEPENDENT_CODE ON)