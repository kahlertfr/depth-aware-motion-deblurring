/***********************************************************************
 * Author:       Franziska Krüger
 *
 * Description:
 * ------------
 * Convolves an image with a kernel using matlabs conv2 method
 * 
 ************************************************************************
*/

#include <iostream>                     // cout, cerr, endl
#include <stdexcept>                    // throw exception

#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite
#include <opencv2/imgproc/imgproc.hpp>  // convert

using namespace std;
using namespace cv;


/**
 * Works like matlab conv2 with "valid"-flag
 * (partial copy of the method that could be found in utils folder)
 */
void conv2(const Mat& src, Mat& dst, const Mat& kernel) {
    int padSizeX = kernel.cols - 1;
    int padSizeY = kernel.rows - 1;

    Mat zeroPadded;
    copyMakeBorder(src, zeroPadded, padSizeY, padSizeY, padSizeX, padSizeX,
                   BORDER_CONSTANT, Scalar::all(0));
    
    Point anchor(0, 0);

    // // openCV is doing a correlation in their filter2D function ...
    // Mat fkernel;
    // flip(kernel, fkernel, -1);

    Mat tmp;
    filter2D(zeroPadded, tmp, -1, kernel, anchor);

    // src =
    //     1 2 3 4
    //     1 2 3 4
    //     1 2 3 4
    // 
    // zeroPadded =
    //     0 0 1 2 3 4 0 0
    //     0 0 1 2 3 4 0 0
    //     0 0 1 2 3 4 0 0
    // 
    // kernel =
    //     0.5 0 0.5
    // 
    // tmp =
    //     0.5 1 2 3 1.5 2 0 2
    //     0.5 1 2 3 1.5 2 0 2
    //     0.5 1 2 3 1.5 2 0 2
    //     |<----------->|      full
    //         |<---->|         same
    //           |-|            valid
    // 
    // the last column is complete rubbish, because openCV's
    // filter2D uses reflected borders (101) by default.
    
    // crop padding
    Mat cropped;

    // variables cannot be declared in case statements
    int width  = -1;
    int height = -1;

    width  = src.cols - kernel.cols + 1;
            height = src.rows - kernel.rows + 1;
            cropped = tmp(Rect((tmp.cols - padSizeX - width) / 2,
                               (tmp.rows - padSizeY - height) / 2,
                               width,
                               height));

    cropped.copyTo(dst);
}


int main(int argc, char** argv) {
    Mat src, kernel, mask, conv;

    if (argc < 3) {
        cerr << "usage: conv <image> <kernel>" << endl;
        return 1;
    }

    string image = argv[1];
    string kernelName = argv[2];

    // mouse with mask
    src = imread(image, CV_LOAD_IMAGE_GRAYSCALE);
    kernel = imread(kernelName, CV_LOAD_IMAGE_GRAYSCALE);

    if (!src.data || !kernel.data) {
        throw runtime_error("Can not load images!");
    }

    kernel.convertTo(kernel, CV_32F);
    kernel /= sum(kernel)[0];  // mouse kernel is not energy preserving

    // // create mask
    // int border = 50;
    // Mat tmpmask = Mat::ones(src.rows - border * 2, src.cols - border * 2, CV_8U);
    // copyMakeBorder(tmpmask, mask, border, border, border, border,
    //                BORDER_CONSTANT, Scalar::all(0));
    // mask *= 255;

    conv2(src, conv, kernel);
    imwrite("conv-" + image, conv);

    return 0;
}