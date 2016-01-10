/***********************************************************************
 * Author:       Franziska Krüger
 * Requirements: OpenCV 3
 *
 * Description:
 * ------------
 * Two-phase kernel estimation for robust motion deblurring by Xu and Jia.
 * 
 ***********************************************************************
 */

#ifndef TWO_PHASE_PSF_ESTIMATION_H
#define TWO_PHASE_PSF_ESTIMATION_H

#include <opencv2/opencv.hpp>


namespace TwoPhaseKernelEstimation {

    /**
     * Start two-phase kernel estimation algortihm
     * 
     * @param psf       Mat to store the estimated PSF kernel
     * @param image     blurred image
     * @param psfWidth  expected width of PSF kernel before computing
     * @param mask      binary mask of region that should be computed
     */
    void estimateKernel(cv::Mat& psf, const cv::Mat& image, const int psfWidth, const cv::Mat& mask);

}

#endif
