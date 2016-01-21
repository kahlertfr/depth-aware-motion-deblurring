# Reference Implementation - Two-Phase Kernel Estimation for Robust Motion Deblurring

["Two-Phase Kernel Estimation paper from Xu and Jia"][Xu10]

This Implementation concentrates on the first part of the algorithm which is PSF estimation for single images. It can be used independently.


## Requirements

- [OpenCV 3.0](http://opencv.org/). [Installation guide][OpenCV-install]
- [argtable3](http://www.argtable.org/) for a commandline tool


## Building

This project uses [CMake](http://cmake.org/) as build tool chain. Use an additional build folder to have clean source folders.

The easiest way is way of building this project is from the toplevel because all toplevel will be resolved correctly. Otherwise you need to ensure CMake can find a libargtable or you modify the CMakeLists to not build a command line tool. If you done this follow the next steps:

```bash
# create a build directory (all CMake Files will be saved there)
mkdir build
cd build

# Create a build configuration
# CMAKE_BUILD_TYPE= Release for disabling output (doesn't show/save images ...)
cmake -D CMAKE_BUILD_TYPE=Release ..
make two-phase-kernel

# Executable can be found in build/bin
bin/two-phase-kernel ../images/mouse-left.jpg
```


[OpenCV-install]: http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html#table-of-content-introduction
[Xu10]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.170.6990&rep=rep1&type=pdf