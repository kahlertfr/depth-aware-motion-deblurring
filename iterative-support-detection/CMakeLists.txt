project(iterative-support-detection)

find_package(OpenCV 3 REQUIRED)

add_executable(isd main.cpp isd.cpp $<TARGET_OBJECTS:utils>)
target_link_libraries(isd ${OpenCV_LIBS})