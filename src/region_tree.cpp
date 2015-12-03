#include <iostream>                     // cout, cerr, endl
#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite

#include "region_tree.hpp"

using namespace cv;
using namespace std;


namespace DepthAwareDeblurring {

    RegionTree::RegionTree(){}

    void RegionTree::create(const Mat &disparityMap, const Mat &image){
        cout << "create region tree " << endl;
    }
}