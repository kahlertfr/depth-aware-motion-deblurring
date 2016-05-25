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

#include "utils.hpp"

using namespace std;
using namespace cv;


int main(int argc, char** argv) {
    Mat src, kernel, mask, conv;

    if (argc < 3) {
        cerr << "usage: conv2 <image> <kernel>" << endl;
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

    deblur::conv2(src, conv, kernel, deblur::VALID);
    imwrite("conv.png", conv);

    return 0;
}