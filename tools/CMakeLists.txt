project(synthetic-images)

find_package(OpenCV 3 REQUIRED)

add_executable(conv2 convolve.cpp)
target_link_libraries(conv2 libmdeblur)

add_executable(deconv deconvolve.cpp)
target_link_libraries(deconv libmdeblur)

add_executable(top-level-deconv top_level_deconv.cpp)
target_link_libraries(top-level-deconv libmdeblur)

add_executable(shock-filter shock_filter.cpp)
target_link_libraries(shock-filter libmdeblur)