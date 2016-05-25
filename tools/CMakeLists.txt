project(synthetic-images)

find_package(OpenCV 3 REQUIRED)

add_executable(conv2 convolve.cpp)
target_link_libraries(conv2 libmdeblur)

add_executable(deconv deconvolve.cpp)
target_link_libraries(deconv libmdeblur)