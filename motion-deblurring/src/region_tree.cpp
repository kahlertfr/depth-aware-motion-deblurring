#include <iostream>                     // cout, cerr, endl
#include <math.h>                       // pow
#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite
#include <opencv2/imgproc/imgproc.hpp>  // convert

#include "region_tree.hpp"


using namespace cv;
using namespace std;


namespace deblur {

    RegionTree::RegionTree(){}

    void RegionTree::create(const Mat& quantizedDisparityMapL, const Mat& quantizedDisparityMapR,
                            const int layers, Mat* imageLeft, Mat* imageRight,
                            const int maxTopLevelNodes){

        // save a pointer to the original image
        images[LEFT] = imageLeft;
        images[RIGHT] = imageRight;

        // the size of masks is determinded by the number of disparity layers
        _masks[LEFT].reserve(layers);
        _masks[RIGHT].reserve(layers);

        // save each disparity layer as binary mask
        for (int l = 0; l < layers; l++) {
            Mat maskLeft, maskRight;

            // find all pixels that have the color l
            //      1 - pixel has color
            //      0 - pixel has other color
            inRange(quantizedDisparityMapL, l, l, maskLeft);
            inRange(quantizedDisparityMapR, l, l, maskRight);

            _masks[LEFT].push_back(maskLeft);
            _masks[RIGHT].push_back(maskRight);

            // store leaf node in tree
            // which doesn't have any child nodes
            node n;
            n.layers = {l};
            n.children = {-1,-1};
            tree.push_back(n);
        }


        // the tree contains all leaf nodes
        // now find the parent nodes
        int level = 0;
        int startId = 0;
        int endId = layers;

        while (true) {
            // reached level with top level nodes
            // the number of the nodes of the previous level in the binary tree
            // can be caculated by: layers / (2^level)
            // where the leafs are at level 0
            if ((layers / pow(2, level)) <= maxTopLevelNodes) {
                for (int i = startId; i < tree.size(); i++) {
                    topLevelNodeIds.push_back(i);
                    tree[i].parent = -1;
                }
                break;
            } 

            // go through all nodes of the previous level
            for (int i = startId; i < endId; i++) {
                // there is no neighbor so stop building this subtree
                if (i + 1 >= endId) {
                    // found a top level node
                    topLevelNodeIds.push_back(i);
                } else {
                    // get two child nodes
                    node child1 = tree[i];
                    node child2 = tree[i + 1];

                    node n;

                    // save contained disparity layers of the new node
                    n.layers.reserve(child1.layers.size() + child2.layers.size());
                    n.layers = child1.layers;
                    n.layers.insert(n.layers.end(), child2.layers.begin(), child2.layers.end());

                    // save child node ids
                    n.children = {i, i + 1};

                    tree.push_back(n);

                    // save parent id of child nodes
                    int parentId = tree.size() - 1;
                    tree[i].parent = parentId;
                    tree[i + 1].parent = parentId;

                    // jump over child2
                    i++;
                }
            }

            // update indices
            level ++;
            startId = endId;
            endId = tree.size();
        };


        #ifdef IMWRITE
            // print tree
            for(int i = 0; i < tree.size(); i++) {
                node n = tree[i];
                cout << "    n" << i << ": ";
                for (int b = 0; b < n.layers.size(); b++) {
                    cout << n.layers[b] << " ";
                }

                if (n.parent != -1)
                    cout << " p(n" << n.parent << ")";
                if (n.children.first != -1)
                    cout << " c(n" << n.children.first << ", n" << n.children.second << ")";
                cout << endl;
            }

            cout << "    top level nodes: ";
            for(int i = 0; i < topLevelNodeIds.size(); i++) {
                cout << topLevelNodeIds[i] << " ";
            }
            cout << endl;
        #endif
    }


    void RegionTree::getMask(const int nodeId, Mat& mask, const view view) const {
        // a region contains multiple layers
        vector<int> region = tree[nodeId].layers;

        mask = Mat::zeros(images[view]->rows, images[view]->cols, CV_8U);

        // adding all masks contained by this node
        for (int i = 0; i < region.size(); i++) {
            add(mask, _masks[view][region[i]], mask);
        }
    }


    void RegionTree::getRegionImage(const int nodeId, Mat &regionImage, Mat &mask,
                                    const view view) const {
        getMask(nodeId, mask, view);

        // create an image with this mask
        images[view]->copyTo(regionImage, mask);
    }


    /**
     * Fills pixel in a given range with a given uchar.
     * 
     * @param image image to work on
     * @param start starting point
     * @param end   end point
     * @param color color for filling
     */
    static void fillPixel(Mat &image, const Point start, const Point end, const uchar color) {
        for (int row = start.y; row <= end.y; row++) {
            for (int col = start.x; col <= end.x; col++) {
                image.at<uchar>(row, col) = color;
            }
        }
    }


    void RegionTree::edgeTaper(Mat& taperedRegion, Mat& region, Mat& mask, Mat& image) const {
        assert(region.type() == CV_8U && "gray values needed");

        Mat taperedHorizontal;
        region.copyTo(taperedHorizontal);

        // search for black regions
        uchar threshold = 0;

        uchar left = 0;
        Point start(-1,-1);

        // fill the black regions with the gray values
        // go through each pixel
        for (int row = 0; row < region.rows; row++) {
            for (int col = 0; col < region.cols; col++) {
                uchar value = region.at<uchar>(row, col);

                // check if in black region
                if (start != Point(-1, -1)) {
                    // found next colored pixel or reached end of the row
                    if (value > threshold || col == region.cols - 1) {
                        // fill first half of the detected region
                        Point end(col - ((col - start.x) / 2),
                                  row - ((row - start.y) / 2));

                        // if at the start of the row set the color for the 
                        // first half to the same color as for the second one
                        if (start.x == 0) {
                            left = value;
                        }

                        fillPixel(taperedHorizontal, start, end, left);

                        // if at the end of the row set the color for the 
                        // second half to the same color as for the first one
                        if (col == region.cols - 1) {
                            value = left;
                        }

                        // fill second half of the detected region
                        fillPixel(taperedHorizontal, end, Point(col, row), value);
                        
                        // reset the start point of the region
                        start = Point(-1,-1);
                    }
                } else {
                    // found new occluded pixel
                    if (value <= threshold) {
                        // there is no left neighbor at column 0 so check it
                        left = (col > 0) ? region.at<uchar>(row, col - 1) : 0;
                        start = Point(col, row);
                    }
                }
            }
        }

        Mat taperedVertical;
        region.copyTo(taperedVertical);

        left = 0;
        start = Point(-1,-1);

        // second run for vertical filling
        for (int col = 0; col < region.cols; col++) {
            for (int row = 0; row < region.rows; row++) {
                uchar value = region.at<uchar>(row, col);

                // check if in black region
                if (start != Point(-1, -1)) {
                    // found next colored pixel or reached end of the row
                    if (value > threshold || row == region.rows - 1) {
                        // fill first half of the detected region
                        Point end(col - ((col - start.x) / 2),
                                  row - ((row - start.y) / 2));

                        // if at the start of the row set the color for the 
                        // first half to the same color as for the second one
                        if (start.y == 0) {
                            left = value;
                        }

                        fillPixel(taperedVertical, start, end, left);

                        // if at the end of the row set the color for the 
                        // second half to the same color as for the first one
                        if (row == region.rows - 1) {
                            value = left;
                        }

                        // fill second half of the detected region
                        fillPixel(taperedVertical, end, Point(col, row), value);
                        
                        // reset the start point of the region
                        start = Point(-1,-1);
                    }
                } else {
                    // found new occluded pixel
                    if (value <= threshold) {
                        // there is no left neighbor at column 0 so check it
                        left = (row > 0) ? region.at<uchar>(row - 1, col) : 0;
                        start = Point(col, row);
                    }
                }
            }
        }

        // add the horizontal and vertical filled images
        addWeighted(taperedHorizontal, 0.5, taperedVertical, 0.5, 0.0, taperedRegion);

        // fill the region within the mask to avoid blurring the inside of the region
        // over its borders (this will reduce the frequency at the border of the region)
        left = 0;
        start = Point(-1,-1);

        for (int row = 0; row < region.rows; row++) {
            for (int col = 0; col < region.cols; col++) {
                uchar value = taperedRegion.at<uchar>(row, col);

                // check if inside mask
                if (start != Point(-1, -1)) {
                    // found pixel next to region inside mask or reached end of the row
                    if (mask.at<uchar>(row, col) == 0 || col == region.cols - 1) {
                        // cout << (int)value << endl;
                        // fill first half of the detected region
                        Point end(col - ((col - start.x) / 2),
                                  row - ((row - start.y) / 2));

                        // if at the start of the row set the color for the 
                        // first half to the same color as for the second one
                        if (start.x == 0) {
                            left = value;
                        }

                        fillPixel(taperedRegion, start, end, left);

                        // if at the end of the row set the color for the 
                        // second half to the same color as for the first one
                        if (col == region.cols - 1) {
                            value = left;
                        }

                        // fill second half of the detected region
                        fillPixel(taperedRegion, end, Point(col, row), value);
                        
                        // reset the start point of the region
                        start = Point(-1,-1);
                    }
                } else {
                    // region inside mask begins
                    if (mask.at<uchar>(row, col) > 0) {
                        // there is no left neighbor at column 0 so check it
                        left = (col > 0) ? taperedRegion.at<uchar>(row, col - 1) : 0;
                        start = Point(col, row);
                    }
                }
            }
        }
     
        // add the original image 
        Mat imageGauss;
        GaussianBlur(image, imageGauss, Size(19, 19), 0, 0, BORDER_DEFAULT);
        addWeighted(taperedRegion, 0.7, imageGauss, 0.3, 0.0, taperedRegion);
        GaussianBlur(taperedRegion, taperedRegion, Size(51, 51), 0, 0, BORDER_DEFAULT);
        
        region.copyTo(taperedRegion, mask);
    }
}