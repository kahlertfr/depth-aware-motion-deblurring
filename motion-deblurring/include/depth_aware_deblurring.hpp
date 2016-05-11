/***********************************************************************
 * Author:       Franziska Krüger
 * Requirements: OpenCV 3
 *
 * Description:
 * ------------
 * Reference Implementation of the Depth-Aware Motion Deblurring
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
 * 7. PSF estimation by repeating Steps 2-5
 * 
 ************************************************************************
*/

#ifndef DEPTH_AWARE_DEBLURRING_H
#define DEPTH_AWARE_DEBLURRING_H

#include <string>
#include <opencv2/opencv.hpp> // cv::Mat


namespace deblur {

    /**
     * Starts depth-aware motion deblurring algorithm with given blurred images (matrices)
     * 
     * @param blurredLeft       OpenCV matrix of blurred left image
     * @param blurredRight      OpenCV matrix of blurred right image
     * @param deblurredLeft     result left
     * @param deblurredRight    result right
     * @param psfWidth          approximate PSF width
     * @param layers            number of regions / disparity layers
     * @param maxTopLevelNodes  maximum of top level nodes in region tree construction
     * @param maxDisparity      maximum disparity between left and right view
     */
    void runDepthDeblur(const cv::Mat &blurredLeft, const cv::Mat &blurredRight,
                        cv::Mat& deblurredLeft, cv::Mat& deblurredRight, const int threads = 1,
                        int psfWidth = 35, const int layers = 12, const int maxTopLevelNodes = 3,
                        const int maxDisparity = 80);

    /**
     * Loads images from given filenames and then starts the depth-aware motion 
     * deblurring algorithm
     * 
     * @param filenameLeft        relative or absolute path to blurred left image
     * @param filenameRight       relative or absolute path to blurred right image
     * @param psfWidth            approximate PSF width
     * @param layers            number of regions / disparity layers
     * @param maxTopLevelNodes    maximum of top level nodes in region tree construction
     * @param maxDisparity        maximum disparity between left and right view
     * @param filenameDeblurLeft  filename for result left
     * @param filenameDeblurRight filename for result right
     */
    void runDepthDeblur(const std::string filenameLeft, const std::string filenameRight,
                        const int threads = 1, int psfWidth = 35, const int layers = 12,
                        const int maxTopLevelNodes = 3, const int maxDisparity = 80, 
                        const std::string filenameDeblurLeft = "deblur-left.png",
                        const std::string filenameDeblurRight = "deblur-right.png");

}

#endif