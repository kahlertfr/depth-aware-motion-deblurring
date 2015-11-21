/***********************************************************************
 * Author:       Franziska Krüger
 * Requirements: OpenCV 3
 *
 * Description:
 * ------------
 * Reference Implemenatation of the Depth-Aware Motion Deblurring
 * Algorithm by Xu and Jia (2012).
 *
 * Algorithm:
 * First-Pass Estimation
 * 1. Stereo-matching-based disparity estimation
 * 2. Region-tree construction
 * 3. PSF estimation for top-level regions in trees
 *     3.1 PSF refinement
 *     3.2 PSF candidate generation and selection
 * 4. If not leaf-level propagate PSF estimation to lower-level regions
 *   and go to step 3.1
 * 5. Blur removal given the PSF estimation
 *
 * Second-Pass Estimation
 * 6. Update disparities based on the deblurred images
 * 7. PSF estimation by repeating STeps 2-5
 * 
 ************************************************************************
*/

#include <iostream>   // cout, cerr, endl
#include <stdexcept>  // throw exception
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

namespace DepthAwareDeblurring {

    /**
     * Uses OpenCV semi global block matching algorithmus to obtain
     * a disparity map with occlusion as black regions
     *
     */
    void semiGlobalBlockMatching(const Mat &left, const Mat &right, Mat &disparityMap) {
        // set up stereo block match algorithm
        // (found nice parameter values for a good result on many images)
        Ptr<StereoSGBM> sgbm = StereoSGBM::create(-64,   // minimum disparity
                                                  16*12, // Range of disparity
                                                  5,     // Size of the block window. Must be odd
                                                  800,   // P1 disparity smoothness (default: 0)
                                                         // penalty for disparity changes +/- 1
                                                  2400,  // P2 disparity smoothness (default: 0)
                                                         // penalty for disparity changes more than 1
                                                  10,    // disp12MaxDiff (default: 0)
                                                  4,     // preFilterCap (default: 0)
                                                  1,     // uniquenessRatio (default: 0)
                                                  150,   // speckleWindowSize (default: 0, 50-200)
                                                  2);    // speckleRange (default: 0)
                                                  
        sgbm->compute(left, right, disparityMap);

        // get its extreme values
        double minVal; double maxVal;
        minMaxLoc(disparityMap, &minVal, &maxVal);
        cout << minVal << " " << maxVal << endl;

        // convert disparity map to values between 0 and 255
        // scale factor for convertion is: 255 / (max - min)
        disparityMap.convertTo(disparityMap, CV_8UC1, 255/(maxVal - minVal));
    }


    /**
     * Disparity estimation of two blurred images
     * from both direction left to right and right to left
     *  
     */
    void disparityEstimation(const Mat &blurredLeft, const Mat &blurredRight,
                             Mat &disparityMap) {
        // down sample images to roughly reduce blur for disparity estimation
        // (down sampling ratio is 2)
        Mat blurredLeftSmall, blurredRightSmall;

        // because we checked that both images are of the same size
        // the new size is the same for both too
        Size downsampledSize = Size(blurredLeftSmall.cols / 2, blurredLeftSmall.rows / 2);

        // downsample with Gaussian pyramid
        pyrDown(blurredLeft, blurredLeftSmall, downsampledSize);
        pyrDown(blurredRight, blurredRightSmall, downsampledSize);

        imshow("blurred left image", blurredLeftSmall);
        imshow("blurred right image", blurredRightSmall);

        // convert color images to gray images
        cvtColor(blurredLeftSmall, blurredLeftSmall, CV_BGR2GRAY);
        cvtColor(blurredRightSmall, blurredRightSmall, CV_BGR2GRAY);

        // disparity map with occlusions as black regions
        // 
        // here a different algorithm as the paper approach is used
        // because it is more convinient to use a OpenCV implementation.
        Mat disparityMapSmall;
        semiGlobalBlockMatching(blurredLeftSmall, blurredRightSmall, disparityMapSmall);

        // TODO: fill occlusion with smallest neighborhood disparity
        // (because just relatively small disparities can be occluded)
        
        // upsample disparity map to original resolution
        pyrUp(disparityMapSmall, disparityMap, Size(blurredLeft.cols, blurredLeft.rows));
        
        imshow("disparity left to right", disparityMap);
    }


    /**
     * Starts alogrithm with given blurred images as
     * OpenCV matrices
     * 
     */
    void runAlgorithm(const Mat &blurredLeft, const Mat &blurredRight) {
        // check if images have the same size
        if (blurredLeft.cols != blurredRight.cols || blurredLeft.rows != blurredRight.rows) {
            throw runtime_error("ParallelTRDiff::runAlgorithm():Images aren't of same size!");
        }

        // initial disparity estimation of blurred images
        Mat disparityMap;
        disparityEstimation(blurredLeft, blurredRight, disparityMap);
        

        // Wait for a key stroke
        waitKey(0);
    }


    /**
     * Loads images from given filenames and starts the algorithm
     * 
     */
    void runAlgorithm(string filenameLeft, string filenameRight) {
        cout << "loads images..." << endl;

        Mat blurredLeft, blurredRight;
        blurredLeft = imread(filenameLeft, 1);
        blurredRight = imread(filenameRight, 1);

        if (!blurredLeft.data || !blurredRight.data) {
            throw runtime_error("ParallelTRDiff::runAlgorithm():Can not load images!");
        }

        runAlgorithm(blurredLeft, blurredRight);
    }
}