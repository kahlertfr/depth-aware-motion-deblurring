/***********************************************************************
 * Author:       Franziska Krüger
 * Requirements: OpenCV 3
 *
 * Description:
 * ------------
 * Iterative support detection for kernel refinement
 * 
 ***********************************************************************
 */

#ifndef ISD_H
#define ISD_H

#include <opencv2/opencv.hpp>


namespace deblur {

    /**
     * Iterative support detection for kernel refinement
     * 
     * @param src    kernel
     * @param dst    refined kernel
     */
    void isd(const cv::Mat& src, cv::Mat& dst);
}

#endif
