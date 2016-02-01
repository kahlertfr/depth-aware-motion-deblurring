/******************************************************************************
 * Author:       Franziska Krüger
 * Requirements: OpenCV 3
 *
 * Description:
 * ------------
 * Computes PSFs for a stereo image with a given disparity map.
 * It uses a region tree for this.
 * 
 ******************************************************************************
 */

#ifndef ITERATIVE_PSF_H
#define ITERATIVE_PSF_H

#include <opencv2/opencv.hpp>

#include "region_tree.hpp"


namespace DepthAwareDeblurring {

    class IterativePSF {

      public:

        IterativePSF(const cv::Mat& disparityMapM, const cv::Mat& disparityMapR,
                     const int layers, cv::Mat* imageLeft, cv::Mat* imageRight,
                     const int maxTopLevelNodes, const int width);

        /**
         * Estimates the PSFs of the top-level regions.
         *
         * Because the algorithm for estimation isn't working yet it loads
         * kernel images as initial kernel estimation.
         * 
         * @param filePrefix for loading kernel images
         */
        void toplevelKernelEstimation(const std::string filePrefix);

        /**
         * Estimates the kernel of all middle and leaf level nodes.
         * Uses candidate selection for minimizing the error of the estimated PSF.
         * 
         */
        void midLevelKernelEstimation();


      private:
        /**
         * Approximate psf kernel width
         */
        int psfWidth;

        /**
         * region tree build form disparity map that stores PSFs for 
         * different depth layers
         */
        RegionTree regionTree;

        /**
         * Enhanced gradients (using bilateral and shock filtering) of left image
         * in x and y direction
         */
        std::array<cv::Mat,2> enhancedGradsLeft;

        /**
         * Enhanced gradients (using bilateral and shock filtering) of right image
         * in x and y direction
         */
        std::array<cv::Mat,2> enhancedGradsRight;

        /**
         * Enhanced gradients (using bilateral and shock filtering) of left image
         * in x and y direction
         */
        std::array<cv::Mat,2> gradsLeft;

        /**
         * Enhanced gradients (using bilateral and shock filtering) of right image
         * in x and y direction
         */
        std::array<cv::Mat,2> gradsRight;

        /**
         * Estimates the PSF of a region jointly on the reference and matching view.
         * 
         * @param maskLeft       mask for region of matching view
         * @param maskRight      mask for region of reference view
         * @param psf            resulting psf
         */
        void jointPSFEstimation(const cv::Mat& maskLeft, const cv::Mat& maskRight, cv::Mat& psf);

        /**
         * Computes the enhanced and simple gradients
         */
        void gradientComputation();
    };
}

#endif