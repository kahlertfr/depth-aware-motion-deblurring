#include <iostream>                     // cout, cerr, endl

#include "two_phase_psf_estimation.hpp"

using namespace cv;
using namespace std;


namespace TwoPhasePSFEstimation {

    void estimateKernel(cv::Mat& psf, const cv::Mat& image, const int psfWidth) {
        Mat kernel = Mat::zeros(psfWidth, psfWidth, CV_8U);

        psf = kernel;
    }
}