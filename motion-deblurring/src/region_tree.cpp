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
            //      255 - pixel has color
            //      0   - pixel has other color
            inRange(quantizedDisparityMapL, l, l, maskLeft);
            inRange(quantizedDisparityMapR, l, l, maskRight);

            // set range of the masks to 0 and 1
            maskRight /= 255;
            maskLeft /= 255;

            _masks[LEFT].push_back(maskLeft);
            _masks[RIGHT].push_back(maskRight);

            // #ifdef IMWRITE
            //     // save masks
            //     string filename = "mask-left-" + to_string(l) + ".png";
            //     imwrite(filename, maskLeft * 255);
            //     filename = "mask-right-" + to_string(l) + ".png";
            //     imwrite(filename, maskRight * 255);
            // #endif

            // store leaf node in tree
            // which doesn't have any child nodes
            node n;
            n.layers = {l};
            n.children = {-1, -1};
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
                    tree[i].parent = -1;
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
}