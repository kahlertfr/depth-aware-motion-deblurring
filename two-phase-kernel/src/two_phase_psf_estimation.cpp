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
        

        // // delta function as image with one white pixel
        // Mat delta = Mat::zeros(50, 50, CV_32F);
        // delta.at<float>(25, 25) = 1;
        // Mat deltaUchar;
        // convertFloatToUchar(deltaUchar, delta);
        // imshow("spatial domain", deltaUchar);

        // Mat fourier;
        // FFT(delta, fourier);

        // for (int x = 0; x < fourier.cols; x++) {
        //     for (int y = 0; y < fourier.rows; y++) {
        //         // complex entries at the current position
        //         complex<float> k(fourier.at<Vec2f>(y, x)[0], fourier.at<Vec2f>(y, x)[1]);
        //         cout << k << endl;
        //     }
        // }

        // Mat uchar;
        // showComplexImage("fourier domain", fourier);


        #ifndef NDEBUG
            // imshow("kernel", kernel);
        #endif
    }


    void estimateKernel(Mat& psf, const Mat& image, const int psfWidth) {
        Mat mask = Mat(image.rows, image.cols, CV_8U, 1);
        estimateKernel(psf, image, psfWidth, mask);
    }
}