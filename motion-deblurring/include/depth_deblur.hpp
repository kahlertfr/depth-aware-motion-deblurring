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

#include <stack>
#include <queue>                        // FIFO queue
#include <mutex>
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
         * @param threads number of threads for parallel deconvolution
         */
        void midLevelKernelEstimation(int nThreads);

        /**
         * Deconvolves the two views for each depth layer.
         * 
         * @param dst     deconvolved image
         * @param view    determine which view is deconvolved
         * @param threads number of threads for parallel deconvolution
         * @param color   use color image
         */
        void deconvolve(cv::Mat& dst, view view, int nThreads = 1, bool color = false);


      private:

        /**
         * both views as CV_8UC3
         */
        std::array<cv::Mat, 2> images;

        /**
         * both gray views CV_8U
         */
        std::array<cv::Mat, 2> grayImages;

        /**
         * both views as float images CV_32F for computation with better precision
         */
        std::array<cv::Mat, 2> floatImages;

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
         * The images are equal if the value is nearly 1.
         * 
         * @param  image1 first image
         * @param  image2 second image
         * @param  mask   mask of the region
         * @return        correlation value
         */
        float gradientCorrelation(cv::Mat& image1, cv::Mat& image2, cv::Mat& mask);


    // methods and variables used for parallel computation ++++++++++++++++++++++++++++++++++++++++++
        
        /**
         * mutex for queue or stack acces
         */
        std::mutex m;

        /**
         * mutex for counting visited leafs
         */
        std::mutex mCounter;

        /**
         * stack for parallel computation of region deconvolution
         */
        std::stack<int> regionStack;

        /**
         * results of region deconvolution
         */
        std::vector<cv::Mat> regionDeconv;

        /**
         * queue for parallel mid psf estimation
         */
        std::queue<int> remainingNodes;

        /**
         * count visited leafs and use it as break condition
         */
        int visitedLeafs;

        /*
         * This method is used by threads for parallel deconvolution of the regions. It 
         * uses a thread safe access to the regionStack.
         * Results are stored in regionDeonv.
         * 
         */
        void deconvolveRegion(const view view, const bool color);

        /**
         * Provides a mutex lock to safely get and pop the top item
         * of the shared stack. Returns false if the stack is empty.
         * 
         */
        bool safeStackAccess(std::stack<int>* sharedStack, int& item);

        /*
         * This method is used by threads for parallel mid level psf estimation. It 
         * uses a thread safe access to the remainingNodes queue.
         * 
         */
        void midLevelKernelEstimationNode();

        /**
         * Provides a mutex lock to safely get and pop the top item
         * of the shared queue. Returns false if the queue is empty.
         * 
         */
        bool safeQueueAccess(std::queue<int>* sharedQueue, int& item);

    };
}

#endif