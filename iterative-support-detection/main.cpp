/***********************************************************************
 * Author:       Franziska Krüger
 *
 * Description:
 * ------------
 * Iterative support detection to create sparse PSFs while obtaining
 * the blur kernel structure.
 * 
 ************************************************************************
*/

#include <iostream>                     // cout, cerr, endl
#include <stdexcept>                    // throw exception

#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite
#include <opencv2/imgproc/imgproc.hpp>  // convert

#include "deconvolution.hpp"

using namespace std;
using namespace cv;


int main(int argc, char** argv) {
    Mat src, kernel;

    if (argc < 3) {
        cerr << "usage: conv <kernel> <image>" << endl;
        return 1;
    }

    string image = argv[1];
    string kernelName = argv[2];

    // mouse with mask
    src = imread(image, CV_LOAD_IMAGE_GRAYSCALE);
    src.convertTo(src, CV_32F);
    src /= 255;

    kernel = imread(kernelName, CV_LOAD_IMAGE_GRAYSCALE);

    if (!src.data || !kernel.data) {
        throw runtime_error("Can not load images!");
    }

    kernel.convertTo(kernel, CV_32F);
    kernel /= sum(kernel)[0];  // mouse kernel is not energy preserving

    // show deconv with unrefined kernel
    Mat dst;
    deblur::deconvolveFFT(src, dst, kernel);
    threshold(dst, dst, 0.0, -1, THRESH_TOZERO);
    threshold(dst, dst, 1.0, -1, THRESH_TRUNC);
    dst.convertTo(dst, CV_8U, 255);
    imwrite("deconv-unrefined-kernel.png", dst);


    // refine kernel
    
    imwrite("kernel-refined.png", kernel);


    // show deconv with refined kernel
    deblur::deconvolveFFT(src, dst, kernel);
    threshold(dst, dst, 0.0, -1, THRESH_TOZERO);
    threshold(dst, dst, 1.0, -1, THRESH_TRUNC);
    dst.convertTo(dst, CV_8U, 255);
    imwrite("deconv-refined-kernel.png", dst);

    return 0;
}