/***********************************************************************
 * Author:       WinCoder@qq.com & Franziska Krüger
 * Reference:    http://blog.csdn.net/bluecol/article/details/49924739
 * Requirements: OpenCV 3
 *
 * Description:
 * ------------
 * inspired by
 *  Joachim Weickert "Coherence-Enhancing Shock Filters"
 *  http://www.mia.uni-saarland.de/Publications/weickert-dagm03.pdf
 *
 * Example:
 *    Mat dst = CoherenceFilter(I,11,11,0.5,4);
 *   imshow("shock filter",dst);
 * 
 ***********************************************************************
 */

#ifndef COHERENCE_FILTER_H
#define COHERENCE_FILTER_H

#include <opencv2/opencv.hpp>


namespace deblur {

    /**
    * Shock filters an image.
    * 
    * @param img        input image ranging value from 0 to 255.
    * @param sigma      sobel kernel size.
    * @param str_sigma  neighborhood size,see detail in reference[2]
    * @param blend      blending coefficient.default value 0.5.
    * @param iter       number of iteration.
    * @param shockImage filtered image
    */
    void coherenceFilter(const cv::Mat& img, const int sigma, const int str_sigma,
                         const float blend, const int iter, cv::Mat& shockImage);

}

#endif