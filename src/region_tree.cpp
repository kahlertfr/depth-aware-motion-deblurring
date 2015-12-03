#include <iostream>                     // cout, cerr, endl
#include <math.h>                       // pow
#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite
#include <opencv2/imgproc/imgproc.hpp>  // convert

#include "region_tree.hpp"

using namespace cv;
using namespace std;


namespace DepthAwareDeblurring {

    RegionTree::RegionTree(){}

    void RegionTree::create(const Mat &quantizedDisparityMap, const int layers, const Mat *image){
        // save the original image
        _originalImage = image;

        _masks.reserve(layers);

        // save each disparity layer as binary mask
        for (int l = 0; l < layers; l++) {
            Mat mask;
            inRange(quantizedDisparityMap, l, l, mask);
            _masks.push_back(mask);

            // store leaf nodes
            node n;
            n.first = {l};
            tree.push_back(n);
        }


        // the tree contains all leaf nodes
        // now find the parent nodes
        int level = 0;
        int startId = 0;
        int endId = layers;

        int numberOfTopLevelNodes = 3;

        while (true) {
            // next level starts at current tree size
            int nextStart = tree.size();
            int counter = 0;

            // reached level with top level nodes
            if ((layers / pow(2, level)) < numberOfTopLevelNodes) {
                for (int i = startId; i < tree.size(); i++) {
                    topLevelNodeIds.push_back(i);
                }
                break;
            } 

            // go through all nodes of the previous level
            for (int i = startId; i < endId; i++) {
                // there is no neighbor (just one child)
                if (i + 1 >= endId) {
                    // found a top level node
                    topLevelNodeIds.push_back(i);
                } else {
                    // get two child nodes
                    node child1 = tree[i];
                    node child2 = tree[i + 1];

                    node n;

                    // save contained disparity layers of the new node
                    n.first.reserve(child1.first.size() + child2.first.size());
                    n.first = child1.first;
                    n.first.insert(n.first.end(), child2.first.begin(), child2.first.end());

                    // save child node ids
                    n.second = {i, i + 1};

                    tree.push_back(n);

                    // jump over child2
                    i++;
                    counter++;
                }
            }

            // update indices
            level ++;
            startId = nextStart;
            endId = startId + counter;
        };

        cout << tree.size() << endl;
        for(int i = 0; i < tree.size(); i++) {
            node n = tree[i];
            cout << i << ": ";
            for (int b = 0; b < n.first.size(); b++) {
                cout << n.first[b] << " ";
            }
            cout << endl;
        }

        cout << "   top level nodes: ";
        for(int i = 0; i < topLevelNodeIds.size(); i++) {
            cout << topLevelNodeIds[i] << " ";
        }
        cout << endl;
    }


    void RegionTree::getImage(const int nodeId, Mat &regionImage) {
        vector<int> layers = tree[nodeId].first;

        Mat mask = Mat::zeros(_originalImage->rows, _originalImage->cols, CV_8U);

        // adding all masks contained by this node
        for (int i = 0; i < layers.size(); i++) {
            add(mask, _masks[layers[i]], mask);
        }

        // create an image with this mask
        _originalImage->copyTo(regionImage, mask);
    }

}