project(match)

set(LIB_SOURCES src/match.cpp
                src/data.cpp
                src/statistics.cpp
                src/kz1.cpp
                src/kz2.cpp
                src/bvz.cpp
                src/corr.cpp
                src/image.cpp
                maxflow/graph.cpp
                maxflow/maxflow.cpp)

# Not the nicest way but header and source files
# are not separated and include each other without
# their parent directory as prefix
include_directories(src energy maxflow)

add_library(libmatch SHARED ${LIB_SOURCES})
set_target_properties(libmatch PROPERTIES
                               OUTPUT_NAME "match")

add_executable(match src/main.cpp)
target_link_libraries(match libmatch)