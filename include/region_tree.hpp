/******************************************************************************
 * Author:       Franziska Krüger
 * Requirements: OpenCV 3
 *
 * Description:
 * ------------
 * This class provides a region tree on a disparity map that
 * saves binary masks of each disparity layer S(i) with depth i on the 
 * leaf nodes.
 *
 * A region tree contains a specific number of binary trees where each node
 * describes a region which contains multiple disparity layers. The region
 * of the original image can be computed by adding the masks of the contained
 * disparity layers.
 *
 * Each middle or top node is calculated the following way:
 *     - merge S(i) and S(j) if i and j are neighboring numbers and
 *       i = ⌊j/2⌋ · 2
 *
 * example:
 *     - 8 disparity layers and a maximum of 2 nodes at top level
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

    /**
     * A Node stores both children and parent because of walking
     * easily from top to bottom and bottom to top through the tree
     */
    struct node {
        std::vector<int>     layers;    // contained disparity layers
        int                  parent;    // index of parent node
        std::pair<int, int>  children;  // indices of child nodes
        std::vector<cv::Mat> psf;
    };


    class RegionTree {

      public:
        RegionTree();

        /**
         * Store the top level nodes to walk through the tree in a top to bottom manner
         */
        std::vector<int> topLevelNodeIds;

        /**
         * Access the nodes stored in the tree with their index.
         * 
         * @param  i index
         * @return   corresponding node
         */
        inline node &operator[](int i) {
            return tree[i];
        }

        /**
         * Creates the binary masks of each disparity layer and sets up the tree
         * 
         * @param quantizedDisparityMap  disparity map with values from [0, layers - 1]
         * @param layers                 number of different disparity values
         * @param image                  image to which the disparity map belongs
         * @param maxTopLevelNodes       maximum number of nodes at top level
         */
        void create(const cv::Mat &quantizedDisparityMap, const int layers, const cv::Mat *image,
                    const int maxTopLevelNodes=3);


        /**
         * Creates an image where everything is black but the region of the image
         * that is determined by the disparity layers contained in the node
         * 
         * @param nodeId       Id of the node in the region tree
         * @param regionImage  image with the resulting region
         */
        void getRegionImage(const int nodeId, cv::Mat &regionImage);


      private:
        /**
         * Stores nodes with their containing layers and the node ids
         * of the children and the parent node.
         */
        std::vector<node> tree;

        /**
         * Binary masks of each disparity layer
         */
        std::vector<cv::Mat> _masks;

        /**
         * Pointer to color image (for generating region image)
         */
        const cv::Mat* _originalImage;
    };
}

#endif