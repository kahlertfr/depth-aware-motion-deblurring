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

    /**
     * Enumeration for conviniend call of two views
     */
    enum view { LEFT, RIGHT };

    /**
     * Flags for matlabs conv2 method
     */
    enum ConvShape {
        FULL,
        SAME,
        VALID,
    };
    

    inline float norm(float a, float b) {
        return sqrt(a * a + b * b);
    }

    /**
     * Works like matlab conv2
     *
     * The shape parameter controls the result matrix size:
     * 
     *  - FULL  Returns the full two-dimensional convolution
     *  - SAME  Returns the central part of the convolution of the same size as A
     *  - VALID Returns only those parts of the convolution that are computed without
     *          the zero-padded edges
     */
    void conv2(const cv::Mat& src, cv::Mat& dst, const cv::Mat& kernel, ConvShape shape = FULL);

    /**
     * Applies DFT after expanding input image to optimal size for Fourier transformation
     * 
     * @param src input image with 1 channel
     * @param dst result as 2 channel matrix with complex numbers
     */
    void fft(const cv::Mat& src, cv::Mat& dst);

    /**
     * Calls OpenCv dft function with adding the input matrix
     * as real channel of a complex matrix (2-channel matrix).
     * Without any padding!
     * 
     * @param src input image with 1 channel
     * @param dst result as 2 channel matrix with complex numbers
     */
    void dft(const cv::Mat& src, cv::Mat& dst);

    /**
     * Converts a matrix containing floats to a matrix
     * conatining uchars
     * 
     * @param src input matrix
     * @param dst resulting matrix
     */
    void convertFloatToUchar(const cv::Mat& src, cv::Mat& dst);

    /**
     * Displays a float matrix
     * which entries are outside range [0,1]
     * 
     * @param name window name
     * @param src  float matrix
     */
    void showFloat(const std::string name, const cv::Mat& src, const bool write = false);

    /**
     * Displays a float matrix as for gradients in range [-1, 1]
     * where the zero becomes 128 as grayvalue.
     * 
     * @param name window name
     * @param src  float matrix
     */
    void showGradients(const std::string name, const cv::Mat& src, const bool write = false);

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
     * Returns the real part of a complex matrix. Same as the "real()"
     * function in MatLab
     * 
     * @param  src  Complex input matrix of floats 
     * @return      Real part of the complex matrix
     */
    cv::Mat realMat(const cv::Mat& src);

    /**
     * Computes one channel gradients:
     *     sqrt(x² + y²)
     * 
     * @param gradients gradients of x- and y-direction
     * @param gradient  resulting combined gradients
     */
    void normedGradients(std::array<cv::Mat, 2>& gradients, cv::Mat& gradient);

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

    /**
     * Fills pixel in a given range with a given uchar.
     * 
     * @param image image to work on
     * @param start starting point
     * @param end   end point
     * @param color color for filling
     */
    void fillPixel(cv::Mat& image, const cv::Point start, const cv::Point end, const uchar color);

    /**
     * fill the black regions with the neighboring pixel colors (half way the left one
     * and half way the right one) and blur the resulting image. Copy the original region
     * over it.
     * 
     * @param taperedRegion resulting image
     * @param region        region image
     * @param mask          mask of region
     */
    void edgeTaper(cv::Mat& taperedRegion, cv::Mat& region, cv::Mat& mask, cv::Mat& image);
}

#endif