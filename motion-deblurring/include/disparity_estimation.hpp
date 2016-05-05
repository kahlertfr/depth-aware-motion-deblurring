/***********************************************************************
 * Author:       Franziska Krüger
 * Requirements: OpenCV 3
 *
 * Description:
 * ------------
 * Some utils for disparity estimation.
 * 
 ***********************************************************************
 */

#ifndef DISPARITY_ESTIMATION_H
#define DISPARITY_ESTIMATION_H

#include <opencv2/opencv.hpp>


namespace deblur {

    /**
     * Fills occlusion regions (where the value is smaller than a given threshold) 
     * with smallest neighborhood disparity (in a row) because just relatively small 
     * disparities can be occluded.
     * 
     * @param disparityMap disparity map where occlusions will be filled
     * @param threshold    threshold for detecting occluded regions
     */
    void fillOcclusionRegions(cv::Mat& disparityMap, const uchar threshold = 0);

    /**
     * Uses OpenCVs semi global block matching algorithm to obtain
     * a disparity map with occlusion as black regions
     * 
     * @param left         left image
     * @param right        right image
     * @param disparityMap disparity with occlusions
     */
    void semiGlobalBlockMatching(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparityMap);

    /**
     * Quantizes two images with kmeans algorithm and sorts the
     * clustering depending on the color value of the cluster centers
     * such that the clustering represents depth graduation.
     *
     * Using two images at once results in the same clusters for the same color
     * in both images (otherwise the clusters may differ depending on the color range).
     * 
     * @param images          input images
     * @param k               cluster number
     * @param quantizedImages clustered images
     */
    void quantizeImage(const std::array<cv::Mat,2>& images, const int k, std::array<cv::Mat,2>& quantizedImages);
}

#endif
