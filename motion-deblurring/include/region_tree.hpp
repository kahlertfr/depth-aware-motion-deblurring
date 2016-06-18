/******************************************************************************
 * Author:       Franziska Krüger
 * Requirements: OpenCV 3
 *
 * Description:
 * ------------
 * This class provides a region tree on a disparity maps of a stereo view.
 * It saves binary masks of each disparity layer S(i) with depth i on the 
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

#include "utils.hpp"

namespace deblur {

    /**
     * A Node stores both children and parent because of walking
     * easily from top to bottom and bottom to top through the tree
     */
    struct node {
        std::vector<int>     layers;    // contained disparity layers
        int                  parent;    // index of parent node
        std::pair<int, int>  children;  // indices of child nodes
        cv::Mat              psf;       // kernel
        float                entropy;   // entropy of kernel
    };


    class RegionTree {

      public:
        RegionTree();

        /**
         * Store the top level nodes to walk through the tree in a top to bottom manner
         */
        std::vector<int> topLevelNodeIds;

        /**
         * Pointer to original images of left and right view (for generating region image)
         */
        std::array<cv::Mat*, 2> images;

        /**
         * Stores nodes with their containing layers and the node ids
         * of the children and the parent node.
         */
        std::vector<node> _tree;

        /**
         * Binary masks of each disparity layer for each view
         */
        std::array<std::vector<cv::Mat>, 2> _masks;

        /**
         * Access the nodes stored in the tree with their index.
         * 
         * @param  i index
         * @return   corresponding node
         */
        inline node &operator[](int i) {
            return _tree[i];
        }

        /**
         * Get all nodes of the same level of the current node
         * 
         * @param id  current node
         */
        std::vector<int> getLevelPeers(int id);

        /**
         * Creates the binary masks of each disparity layer for each view and sets up the tree
         * 
         * @param quantizedDisparityMapL left-right disparity map with values from [0, layers - 1]
         * @param quantizedDisparityMapR right-left disparity map with values from [0, layers - 1]
         * @param layers                 number of different disparity values
         * @param imageLeft              image of left view
         * @param imageRight             image of right view
         * @param maxTopLevelNodes       maximum number of nodes at top level
         */
        void create(const cv::Mat& quantizedDisparityMapL, const cv::Mat& quantizedDisparityMapR,
                    const int layers, cv::Mat* imageLeft, cv::Mat* imageRight,
                    const int maxTopLevelNodes = 3);

        /**
         * Returns the depth mask of both view for a specific node by adding all mask of all layers.
         * 
         * @param nodeId  Id of the node in the region tree
         * @param maskL   mask of both regions
         */
        inline void getMasks(const int nodeId, std::array<cv::Mat, 2>& masks) const {
            getMask(nodeId, masks[deblur::LEFT], deblur::LEFT);
            getMask(nodeId, masks[deblur::RIGHT], deblur::RIGHT);
        };

        /**
         * Returns the depth mask for a specific node in one view by adding all mask of all layers.
         * 
         * @param nodeId  Id of the node in the region tree
         * @param maskL   mask of left region
         * @param view    LEFT or RIGHT view
         */
        void getMask(const int nodeId, cv::Mat& mask, const deblur::view view) const;

        /**
         * Creates an image where everything is black but the region of the image
         * that is determined by the disparity layers contained in the node
         * 
         * @param nodeId       Id of the node in the region tree
         * @param regionImage  image with the resulting region
         * @param mask         mask of this region
         */
        void getRegionImage(const int nodeId, cv::Mat &regionImage, cv::Mat &mask,
                            const deblur::view view) const;

        /**
         * Returns total number of nodes in this region tree
         */
        inline int size() { return _tree.size(); }
    };
}

#endif