# Reference Implementation - Depth-Aware Motion Deblurring

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
make

# Executable can be found in build/bin
```


[OpenCV-install]: http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html#table-of-content-introduction