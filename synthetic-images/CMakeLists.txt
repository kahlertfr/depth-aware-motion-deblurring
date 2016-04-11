project(synthetic-images)

find_package(OpenCV 3 REQUIRED)

add_executable(conv convolve.cpp)
target_link_libraries(conv ${OpenCV_LIBS})