add_executable(motion-deblurring main.cpp argtable3.c)
target_link_libraries(motion-deblurring ${OpenCV_LIBS})