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


namespace deblur {

    /**
     * Creates a more robust gradient. First, the image will be filtered
     * with a bilateral and shock filter to reduce blur. After that, the
     * gradients in x- and y-direction will be calculated.
     * 
     * @param image     blurred grayvalue image
     * @param gradients resulting gradients for x and y-direction
     */
    void gradientMaps(const cv::Mat& image, std::array<cv::Mat,2>& gradients);

    /**
     * Selectes salient edges from a gradient image.
     * 
     * @param gradients   gradients of x and y direction
     * @param thresholded resulting thresholded gradients
     * @param psfWidth    approximate psf width (m)
     * @param mask        region mask
     * @param r           r * m pixel of largest magnitude will be used
     */
    void thresholdGradients(const std::array<cv::Mat,2>& gradients,
                            std::array<cv::Mat,2>& thresholded, const int psfWidth,
                            const cv::InputArray& mask = cv::noArray(), const int r = 2);

    /**
     * Creates robust gradients and threshold them afterwards to get an salient edge map
     * (this is a combination of the two methods above)
     * 
     * @param image    blurred grayvalue image
     * @param egdeMaps resulting thresholded gradients
     * @param psfWidth approximate psf width (m)
     * @param mask     region mask
     * @param r        r * m pixel of largest magnitude will be used
     */
    void computeSalientEdgeMap(const cv::Mat& image, std::array<cv::Mat,2>& edgeMaps,
                               const int psfWidth, const cv::InputArray& mask = cv::noArray(),
                               const int r = 2);

}

#endif