/***********************************************************************
 * Author:       Franziska Krüger
 * Requirements: OpenCV 3
 *
 * Description:
 * ------------
 * Phase One: Kernel initialization 
 * 
 ***********************************************************************
 */

#ifndef KERNEL_INIT_H
#define KERNEL_INIT_H

#include <opencv2/opencv.hpp>


namespace TwoPhaseKernelEstimation {

    /**
     * Initializes the kernel through an iterative process
     * 
     * @param kernel      resulting kernel
     * @param blurredGray gray value image
     * @param width       approximate width of kernel
     * @param mask        mask which region should be computed
     * @param pyrLevel    number of image pyramid levels
     * @param iterations  number of iterations
     * @param thresholdR  threshold for confidence value
     * @param thresholdS  threshold for selected edges
     */
    void initKernel(cv::Mat& kernel, const cv::Mat& blurredGray, const int width, const cv::Mat& mask,
                    const int pyrLevel = 3, const int iterations = 3,
                    float thresholdR = 0.25, float thresholdS = 50);
}

#endif
