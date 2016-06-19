/***********************************************************************
 * Author:       Franziska Krüger
 *
 * Description:
 * ------------
 * Runs stereo matching algorithm on a stereo image pair.
 * 
 ************************************************************************
*/

#include <iostream>                     // cout, cerr, endl
#include <stdexcept>                    // throw exception

#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite
#include <opencv2/imgproc/imgproc.hpp>  // convert

#include "disparity_estimation.hpp"

using namespace std;
using namespace cv;


int main(int argc, char** argv) {
    int layers = 12;


    Mat left, right, dmap;

    if (argc < 3) {
        cerr << "usage: disparity <left-view> <right-view>" << endl;
        return 1;
    }

    // mouse with mask
    left = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    right = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

    if (!left.data || !right.data) {
        throw runtime_error("Can not load images!");
    }

    array<Mat, 2> views, quants;
    array<Mat, 2> dmaps = { Mat::zeros(views[0].size(), CV_8U),
                            Mat::zeros(views[1].size(), CV_8U)};

    // downsample images
    Size downsampledSize = Size(views[0].cols / 2, views[0].rows / 2);
    pyrDown(left, views[0], downsampledSize);
    pyrDown(right, views[1], downsampledSize);

    // // semi global block matching
    deblur::disparityFilledSGBM(views, dmaps);

    double min1; double max1;
    minMaxLoc(dmaps[0], &min1, &max1);
    dmaps[0].convertTo(dmap, CV_8U, 255.0/(max1-min1));
    imwrite("dmap-sgbm-left.png", dmap);
    minMaxLoc(dmaps[1], &min1, &max1);
    dmaps[1].convertTo(dmap, CV_8U, 255.0/(max1-min1));
    imwrite("dmap-sgbm-right.png", dmap);

    // quantize
    deblur::quantizeImage(dmaps, layers, quants);

    minMaxLoc(quants[0], &min1, &max1);
    quants[0].convertTo(dmap, CV_8U, 255.0/(max1-min1));
    imwrite("dmap-quant-sgbm-left.png", dmap);
    minMaxLoc(quants[1], &min1, &max1);
    quants[1].convertTo(dmap, CV_8U, 255.0/(max1-min1));
    imwrite("dmap-quant-sgbm-right.png", dmap);


    // disparity using graph-cut
    deblur::disparityFilledMatch(views, dmaps, views[0].cols / 4);

    minMaxLoc(dmaps[0], &min1, &max1);
    dmaps[0].convertTo(dmap, CV_8U, 255.0/(max1-min1));
    imwrite("dmap-graph-cut-left.png", dmap);
    minMaxLoc(dmaps[1], &min1, &max1);
    dmaps[1].convertTo(dmap, CV_8U, 255.0/(max1-min1));
    imwrite("dmap-graph-cut-right.png", dmap);

    // quantize
    deblur::quantizeImage(dmaps, layers, quants);

    minMaxLoc(quants[0], &min1, &max1);
    quants[0].convertTo(dmap, CV_8U, 255.0/(max1-min1));
    imwrite("dmap-quant-graph-cut-left.png", dmap);
    minMaxLoc(quants[1], &min1, &max1);
    quants[1].convertTo(dmap, CV_8U, 255.0/(max1-min1));
    imwrite("dmap-quant-graph-cut-right.png", dmap);

    return 0;
}