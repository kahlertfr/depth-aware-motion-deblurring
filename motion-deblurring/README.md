# Reference Implementation - Depth-Aware Motion Deblurring

["Depth-Aware Motion Deblurring paper from Xu and Jia"][Xu12]


## Requirements

- [OpenCV 3.0](http://opencv.org/). [Installation guide][OpenCV-install]



## Building

This project uses [CMake](http://cmake.org/) as build tool chain. Use an additional build folder to have clean source folders.

```bash
# create a build directory (all CMake Files will be saved there)
mkdir build
cd build

# Create a build configuration
# CMAKE_BUILD_TYPE= Release for disabling output (doesn't show/save images ...)
cmake -D CMAKE_BUILD_TYPE=Release ..
make motion-deblurring

# Executable can be found in build/bin
bin/motion-deblurring ../images/mouse-left.jpg ../images/mouse-right.jpg
```


[OpenCV-install]: http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html#table-of-content-introduction
[Xu12]: http://www.cse.cuhk.edu.hk/leojia/papers/depth_deblur_iccp12.pdf