#include <iostream>                     // cout, cerr, endl
#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite

#include "region_tree.hpp"

using namespace cv;
using namespace std;


namespace DepthAwareDeblurring {

    RegionTree::RegionTree(){}

    void RegionTree::create(const Mat &quantizedDisparityMap, const int layers, const Mat &image){
        cout << "create region tree " << endl; 

        masks.reserve(layers);

        // save each disparity layer as binary mask
        for (int l = 0; l < layers; l++) {
            Mat mask;
            inRange(quantizedDisparityMap, l, l, mask);
            masks.push_back(mask);
        }
    }
}