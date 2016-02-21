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
         * There is a possibility to load kernel-images because
         * the used algorithm doesn't work very well.
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
         * Gradients of left image in x and y direction
         */
        std::array<cv::Mat,2> gradsLeft;

        /**
         * Gradients of right image in x and y direction
         */
        std::array<cv::Mat,2> gradsRight;

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