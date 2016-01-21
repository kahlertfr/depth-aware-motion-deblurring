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

#include <opencv2/opencv.hpp>
#include <cmath>                        // sqrt


namespace TwoPhaseKernelEstimation {

    inline float norm(float a, float b) {
        return sqrt(a * a + b * b);
    }


    /**
     * Applies DFT after expanding input image to optimal size for Fourier transformation
     * 
     * @param image   input image with 1 channel
     * @param complex result as 2 channel matrix with complex numbers
     */
    inline void FFT(const cv::Mat& image, cv::Mat& complex) {
        assert(image.type() == CV_32F && "fft works on 32FC1-images");

        // for fast DFT expand image to optimal size
        cv::Mat padded;
        int m = cv::getOptimalDFTSize( image.rows );
        int n = cv::getOptimalDFTSize( image.cols );

        // on the border add zero pixels
        cv::copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols,
                           cv::BORDER_CONSTANT, cv::Scalar::all(0));

        // Add to the expanded another plane with zeros
        cv::Mat planes[] = {padded, cv::Mat::zeros(padded.size(), CV_32F)};
        cv::merge(planes, 2, complex);

        // this way the result may fit in the source matrix
        cv::dft(complex, complex, cv::DFT_COMPLEX_OUTPUT);

        assert(padded.size() == complex.size() && "Resulting complex matrix must be of same size");
    }


    /**
     * Converts a matrix containing floats to a matrix
     * conatining uchars
     * 
     * @param ucharMat resulting matrix
     * @param floatMat input matrix
     */
    inline void convertFloatToUchar(cv::Mat& ucharMat, const cv::Mat& floatMat) {
        // find min and max value
        double min; double max;
        cv::minMaxLoc(floatMat, &min, &max);

        // if the matrix is in the range [0, 1] just scale with 255
        if (min >= 0 && max < 1) {
            floatMat.convertTo(ucharMat, CV_8U, 255.0);
        } else {
            cv::Mat copy;
            floatMat.copyTo(copy);

            // handling that floats could be negative
            copy -= min;

            // convert and show
            copy.convertTo(ucharMat, CV_8U, 255.0/(max-min));
        }
    }


    /**
     * Rearrange quadrants of an image so that the origin is at the image center.
     * This is useful for fourier images. 
     */
    inline void swapQuadrants(cv::Mat& image) {
        // rearrange the quadrants of Fourier image  so that the origin is at the image center
        int cx = image.cols/2;
        int cy = image.rows/2;

        cv::Mat q0(image, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
        cv::Mat q1(image, cv::Rect(cx, 0, cx, cy));  // Top-Right
        cv::Mat q2(image, cv::Rect(0, cy, cx, cy));  // Bottom-Left
        cv::Mat q3(image, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

        cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
        q2.copyTo(q1);
        tmp.copyTo(q2);
    }


    /**
     * Displays a matrix with complex numbers stored as 2 channels
     * Copied from: http://docs.opencv.org/2.4/doc/tutorials/core/
     * discrete_fourier_transform/discrete_fourier_transform.html
     * 
     * @param windowName name of window
     * @param complex    matrix that should be displayed
     */
    inline void showComplexImage(const std::string windowName, const cv::Mat& complex) {
        // compute the magnitude and switch to logarithmic scale
        // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
        cv::Mat planes[] = {cv::Mat::zeros(complex.size(), CV_32F), cv::Mat::zeros(complex.size(), CV_32F)};
        cv::split(complex, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
        cv::magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
        cv::Mat magI = planes[0];

        magI += cv::Scalar::all(1);                    // switch to logarithmic scale
        cv::log(magI, magI);

        // crop the spectrum, if it has an odd number of rows or columns
        magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

        swapQuadrants(magI);

        cv::normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
                                                // viewable image form (float between values 0 and 1).

        cv::imshow(windowName, magI);
    }
}

#endif