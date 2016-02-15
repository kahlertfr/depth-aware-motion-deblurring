/***********************************************************************
 * Author:       Franziska Krüger
 * Requirements: OpenCV 3
 *
 * Description:
 * ------------
 * Non-blind deconvolution in Fourier Domain.
 * 
 ***********************************************************************
 */

#ifndef DECONVOLUTION_H
#define DECONVOLUTION_H

#include <opencv2/opencv.hpp>


namespace DepthAwareDeblurring {

    /**
     * Non-blind deconvolution.
     * 
     * @param src    blurred grayvalue image
     * @param dst    latent image
     * @param kernel energy preserving kernel
     * @param we     weight
     */
    void deconvolve(cv::Mat src, cv::Mat& dst, cv::Mat& kernel, float we = 0.001);

}

#endif
