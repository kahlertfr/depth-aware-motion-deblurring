/***********************************************************************
 * Author:       Franziska Krüger
 *
 * Description:
 * ------------
 * Deconvolves an image with a kernel using the deconvolveFFT and the
 * deconvolveIRLS method.
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
    Mat src, kernel, mask, deconv;

    if (argc < 3) {
        cerr << "usage: deconv <image> <kernel> [<mask>]" << endl;
        return 1;
    }

    string image = argv[1];
    string kernelName = argv[2];

    src = imread(image, CV_LOAD_IMAGE_GRAYSCALE);
    src.convertTo(src, CV_32F);
    src /= 255;

    kernel = imread(kernelName, CV_LOAD_IMAGE_GRAYSCALE);

    if (!src.data || !kernel.data) {
        throw runtime_error("Can not load images!");
    }

    kernel.convertTo(kernel, CV_32F);
    kernel /= sum(kernel)[0];  // kernel is not energy preserving

    // load mask
    if (argc > 3) {
        string maskName = argv[3];
        mask = imread(maskName, CV_LOAD_IMAGE_GRAYSCALE);
        mask /= 255;
    } else {
        // mask for whole image
        mask = Mat::ones(src.size(), CV_8U);
    }

    deblur::deconvolveFFT(src, deconv, kernel, mask);
    // save like matlab imshow([deconv])
    threshold(deconv, deconv, 0.0, -1, THRESH_TOZERO);
    threshold(deconv, deconv, 1.0, -1, THRESH_TRUNC);
    deconv.convertTo(deconv, CV_8U, 255);
    imwrite("deconvFFT.png", deconv);

    deblur::deconvolveIRLS(src, deconv, kernel, mask);
    // save like matlab imshow([deconv])
    threshold(deconv, deconv, 0.0, -1, THRESH_TOZERO);
    threshold(deconv, deconv, 1.0, -1, THRESH_TRUNC);
    deconv.convertTo(deconv, CV_8U, 255);
    imwrite("deconvIRLS.png", deconv);

    return 0;
}