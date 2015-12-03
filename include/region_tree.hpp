/******************************************************************************
 * Author:       Franziska Krüger
 * Requirements: OpenCV 3
 *
 * Description:
 * ------------
 * This class provides a region tree (binary tree) on a disparity map that
 * saves bianry masks of each disparity layer S(i) with depth i on the 
 * leaf nodes.
 *
 * Each other layer is caculated the following way:
 *     - merge S(i) and S(j) if i and j are neighboring numbers and
 *       i = ⌊j/2⌋ · 2
 *
 * example:
 *
 *      S({0, 1, 2, 3})            S({4, 5, 6, 7})
 *          /      \                   /      \
 *   S({0, 1})   S({2, 3})     S({4, 5})   S({6, 7})
 *      /    \      /    \       /    \      /    \
 *    S(0)  S(1)  S(2)  S(3)   S(4)  S(5)  S(6)  S(7)   --> store binary
 *                                                          mask of this layers
 * 
 ******************************************************************************
 */

#ifndef REGION_TREE_H
#define REGION_TREE_H

#include <opencv2/opencv.hpp>

namespace DepthAwareDeblurring {

    typedef std::pair<std::vector<int>, std::vector<int>> node;

    class RegionTree {

      public:
        RegionTree();

        /**
         * Stores nodes as pairs with their containing layers and the node ids
         * of the children.
         */
        std::vector<node> tree;

        /**
         * Walk through the tree in a top to bottom manner
         */
        std::vector<int> topLevelNodeIds;

        /**
         * Each node gets a Point Spread Function
         */
        std::vector<cv::Mat> pointSpreadFunctions;


        /**
         * Creates the binary masks of each disparity layer and sets up the tree
         * @param quantizedDisparityMap  disparity map with values from [0, layers - 1]
         * @param layers                 number of different disparity values
         * @param image                  image to which the disparity map belongs
         */
        void create(const cv::Mat &quantizedDisparityMap, const int layers, const cv::Mat *image);


        /**
         * Creates an image where everything is black but the region of the image
         * that is determined by the disparity layers contained in the node
         * 
         * @param nodeId       Id of the node in the region tree
         * @param regionImage  image with the resulting region
         */
        void getImage(const int nodeId, cv::Mat &regionImage);


      private:
        /**
         * Binary masks of each disparity layer
         */
        std::vector<cv::Mat> _masks;

        const cv::Mat* _originalImage;
    };
}

#endif