/******************************************************************************
 * Author:       Franziska Krüger
 * Requirements: OpenCV 3
 *
 * Description:
 * ------------
 * Computes an salient edge map as for the P map in "Fast motion deblurring"
 * from Cho and Lee.
 * 
 ******************************************************************************
 */

#ifndef EDGE_MAP_H
#define EDGE_MAP_H

#include <array>
#include <opencv2/opencv.hpp>


namespace DepthAwareDeblurring {

    /**
     * Creates a more robust gradient. First, the image will be filtered
     * with a bilateral and shock filter to reduce blur. After that, the
     * gradients in x- and y-direction will be calculated.
     */
    void gradientMaps(const cv::Mat& image, std::array<cv::Mat,2>& gradients);

    /**
     * Selectes salient edges from a gradient image.
     * 
     * @param gradients   gradients of x and y direction
     * @param thresholded output
     * @param psfWidth    approximate psf width (m)
     * @param r           r * m pixel of largest magnitude will be used
     * @param mask        region mask
     */
    void thresholdGradients(const std::array<cv::Mat,2>& gradients,
                            std::array<cv::Mat,2>& thresholded, const int psfWidth,
                            const cv::InputArray& mask = cv::noArray(), const int r = 5);


}

#endif