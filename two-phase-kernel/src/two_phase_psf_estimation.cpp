#include <iostream>                     // cout, cerr, endl
#include <complex>                      // complex numbers
#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite

#include "utils.hpp"
#include "kernel_initialization.hpp"

#include "two_phase_psf_estimation.hpp"

using namespace cv;
using namespace std;


namespace TwoPhaseKernelEstimation {

    void estimateKernel(Mat& psf, const Mat& blurredImage, const int psfWidth, const Mat& mask) {
        // set expected kernel witdh to odd number
        int width = (psfWidth % 2 == 0) ? psfWidth + 1 : psfWidth;

        // phase one: initialize kernel
        // 
        Mat kernel;

        // convert blurred image to gray
        Mat blurredGray;
        cvtColor(blurredImage, blurredGray, CV_BGR2GRAY);

        // TODO: change number of pyrLevel and iterations
        initKernel(kernel, blurredGray, width, mask, 1, 1);

        #ifndef NDEBUG
            imshow("kernel", kernel);
        #endif
    }


    void estimateKernel(Mat& psf, const Mat& image, const int psfWidth) {
        Mat mask = Mat(image.rows, image.cols, CV_8U, 1);
        estimateKernel(psf, image, psfWidth, mask);
    }
}