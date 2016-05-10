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
     * Enumeration for conviniend call of different disparity algorithms
     */
    enum disparityAlgo { SGBM, MATCH };

    /**
     * Disparity estimation algorithm using "Computing Visual Correspondence with Occlusions
     * using Graph Cuts" algorithm from Vladimir Kolmogorov and Ramin Zabih
     * 
     * @param images       left and right image gray value
     * @param dMaps        resulting maps for left-right and right-left disparity
     * @param maxDisparity estimated maximum disparity
     */
    void disparityFilledMatch(const std::array<cv::Mat, 2>& images, std::array<cv::Mat, 2>& dMaps,
                              int maxDisparity);

    /**
     * Disparity estimation using the SGBM algorithm and filling the occlusions 
     * with the smallest disparity value of the neighborhood (on the line)
     * 
     * @param images left and right image gray value
     * @param dMaps  resulting maps for left-right and right-left disparity
     */
    void disparityFilledSGBM(const std::array<cv::Mat, 2>& images, std::array<cv::Mat, 2>& dMaps);

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
