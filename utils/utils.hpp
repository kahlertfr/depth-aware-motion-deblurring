/***********************************************************************
 * Author:       Franziska Krüger
 * Requirements: OpenCV 3
 *
 * Description:
 * ------------
 * Collection of some useful functions. 
 * 
 ***********************************************************************
 */

#ifndef UTILS_COLLECTION_H
#define UTILS_COLLECTION_H

#include <vector>
#include <algorithm>           // std::sort
#include <array>
#include <cmath>               // sqrt
#include <opencv2/opencv.hpp>


namespace deblur {

    inline float norm(float a, float b) {
        return sqrt(a * a + b * b);
    }

    /**
     * Applies DFT after expanding input image to optimal size for Fourier transformation
     * 
     * @param image   input image with 1 channel
     * @param complex result as 2 channel matrix with complex numbers
     */
    void FFT(const cv::Mat& image, cv::Mat& complex);

    /**
     * Converts a matrix containing floats to a matrix
     * conatining uchars
     * 
     * @param ucharMat resulting matrix
     * @param floatMat input matrix
     */
    void convertFloatToUchar(cv::Mat& ucharMat, const cv::Mat& floatMat);

    /**
     * Rearrange quadrants of an image so that the origin is at the image center.
     * This is useful for fourier images. 
     */
    void swapQuadrants(cv::Mat& image);

    /**
     * Displays a matrix with complex numbers stored as 2 channels
     * Copied from: http://docs.opencv.org/2.4/doc/tutorials/core/
     * discrete_fourier_transform/discrete_fourier_transform.html
     * 
     * @param windowName name of window
     * @param complex    matrix that should be displayed
     */
    void showComplexImage(const std::string windowName, const cv::Mat& complex);

    /**
     * Normalizes an input array into range [-1, 1] by conserving
     * zero.
     *
     * Example:
     *
     *      [-415, 471]   =>  [-0.88, 1]
     *
     * Works inplace.
     * 
     * @param src  Input matrix
     * @param dst  Normalized matrix
     */
    void normalizeOne(cv::Mat& src, cv::Mat& dst);

    /**
     * Convinient shorthand for inplace-normalization into range
     * [-1, 1]
     * 
     * @param  input source and destination
     */
    inline void normalizeOne(cv::Mat& input) {
        normalizeOne(input, input);
    }

    template<typename T>
    void normalizeOne(T& src, T& dst) {
        std::array<double,4> extrema;
        std::array<double,4> maxima;

        cv::minMaxLoc(src[0], &(extrema[0]), &(extrema[1]));
        cv::minMaxLoc(src[1], &(extrema[2]), &(extrema[3]));

        for (int i = 0; i < 4; ++i) {
            maxima[i] = std::abs(extrema[i]);
        }

        std::sort(maxima.begin(), maxima.end());
        const double scale = maxima[3];

        cv::normalize(src[0], dst[0], extrema[0] / scale, extrema[1] / scale, cv::NORM_MINMAX);
        cv::normalize(src[1], dst[1], extrema[2] / scale, extrema[3] / scale, cv::NORM_MINMAX);
    }

    inline void normalizeOne(std::array<cv::Mat,2>& src, std::array<cv::Mat,2>& dst) {
        normalizeOne<std::array<cv::Mat,2>>(src, dst);
    }

    inline void normalizeOne(std::array<cv::Mat,2>& input) {
        normalizeOne<std::array<cv::Mat,2>>(input, input);
    }

    inline void normalizeOne(std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst) {
        assert(src.size() == 2 && "Vector must contain 2 channels");
        normalizeOne<std::vector<cv::Mat>>(src, dst);
    }

    inline void normalizeOne(std::vector<cv::Mat>& input) {
        normalizeOne<std::vector<cv::Mat>>(input, input);
    }

}

#endif