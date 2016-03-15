/******************************************************************************
 * Author:       Franziska Krüger
 * Requirements: OpenCV 3
 *
 * Description:
 * ------------
 * One pass of the depth-aware deblurring algorithm:
 *     1. disparity estimation
 *     2. region tree reconstruction
 *     3. PSF estimation for top level nodes
 *     4. PSF estimation for mid and leaf level nodes
 *     5. deconvolution of input images
 * 
 ******************************************************************************
 */

#ifndef DEPTH_DEBLUR_H
#define DEPTH_DEBLUR_H

#include <opencv2/opencv.hpp>

#include "region_tree.hpp"


namespace deblur {

    class DepthDeblur {

      public:

        /**
         * Constructor for depth-deblurring of stereo images
         * 
         * @param imageLeft  blurred left view
         * @param imageRight blurred right view
         * @param width      approximate PSF width
         */
        DepthDeblur(cv::Mat& imageLeft, cv::Mat& imageRight, const int width);

        /**
         * Disparity estimation of two blurred images
         * where occluded regions are filled and where the disparity map is 
         * quantized to l regions.
         * 
         * @param inverse  determine if the disparity is calculated from right to left
         */
        void disparityEstimation();

        /**
         * Creates a region tree from disparity maps
         * 
         * @param maxTopLevelNodes  maximal number of nodes at the top level
         */
        void regionTreeReconstruction(const int maxTopLevelNodes);

        /**
         * Estimates the PSFs of the top-level regions.
         *
         * There is a possibility to load kernel-images because
         * the used algorithm doesn't work very well.
         * 
         */
        void toplevelKernelEstimation();

        /**
         * Estimates the kernel of all middle and leaf level nodes.
         * Uses candidate selection for minimizing the error of the estimated PSF.
         * 
         */
        void midLevelKernelEstimation();

        /**
         * Deconvolves the two views for each depth layer.
         * 
         * @param dst   deconvolved image
         * @param view  determine which view is deconvolved
         * @param color use color image
         */
        void deconvolve(cv::Mat& dst, view view, bool color = false);


      private:

        /**
         * both views
         */
        const std::array<cv::Mat, 2> images;

        /**
         * both gray views
         */
        std::array<cv::Mat, 2> grayImages;

        /**
         * Approximate psf kernel width
         *
         * this is an odd number.
         */
        const int psfWidth;

        /**
         * number of disparity layers and region tree leaf nodes
         *
         * this is an even number.
         */
        const int layers;

        /**
         * quantized disparity maps for left-right and right-left disparity
         */
        std::array<cv::Mat, 2> disparityMaps;

        /**
         * region tree build form disparity map that stores PSFs for 
         * different depth layers
         */
        RegionTree regionTree;

        /**
         * Gradients of left image in x and y direction
         */
        std::array<cv::Mat,2> gradsLeft;

        /**
         * Gradients of right image in x and y direction
         */
        std::array<cv::Mat,2> gradsRight;

        int current;

        /**
         * Estimates the PSF of a region jointly on the reference and matching view.
         * 
         * @param maskLeft       mask for region of matching view
         * @param maskRight      mask for region of reference view
         * @param psf            resulting psf
         */
        void jointPSFEstimation(const cv::Mat& maskLeft, const cv::Mat& maskRight,
                                const std::array<cv::Mat,2>& salientEdgesLeft,
                                const std::array<cv::Mat,2>& salientEdgesRight, cv::Mat& psf);

        /**
         * Computes gradients of the two blurred images
         */
        void computeBlurredGradients();

        /**
         * Executes a joint psf estimation after computing the salient edge map of
         * this region node and saves the psf in the region tree.
         * 
         * @param id     current node
         */
        void estimateChildPSF(int id);

        /**
         * Calculates the entropy of the kernel
         *
         * H(k) = -1 * sum_x(x*log(x))
         * 
         * @param  kernel [description]
         * @return        [description]
         */
        float computeEntropy(cv::Mat& kernel);

        /**
         * Selects candiates for psf selection
         * 
         * The following psfs are candidates:
         *      - own psf (also it may be unreliable)
         *      - parent psf
         *      - reliable sibbling psf
         *      
         * @param candiates resulting vector of candidates
         * @param id        current node id
         * @param sId       sibbling node id
         */
        void candidateSelection(std::vector<cv::Mat>& candiates, int id, int sId);

        /**
         * Selects a suitable PSF for the given Node
         *
         * @param candiates possible PSFs
         * @param id        node ID
         */
        void psfSelection(std::vector<cv::Mat>& candiates, int id);

        /**
         * Computed the correlation of gradient magnitudes inside the same region
         * of two images.
         *
         * X and Y are normed gradients ∥∇I∥_2
         *
         *              E(X - μx) * E(Y - μy)
         * corr(X,Y) = ----------------------
         *                     σx * σy
         *
         * where E is the expectation operator
         *       σ is signal standard deviation
         *       μ is signal standard mean
         * 
         * @param  image1 first image
         * @param  image2 second image
         * @param  mask   mask of the region
         * @return        correlation value
         */
        float gradientCorrelation(cv::Mat& image1, cv::Mat& image2, cv::Mat& mask);

    };
}

#endif