/***********************************************************************
 * Author:       Franziska Krüger
 *
 * Description:
 * ------------
 * Shock filters an image.
 * 
 ************************************************************************
*/

#include <iostream>                     // cout, cerr, endl
#include <stdexcept>                    // throw exception

#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite
#include <opencv2/imgproc/imgproc.hpp>  // convert

#include "coherence_filter.hpp"

using namespace std;
using namespace cv;


int main(int argc, char** argv) {
    Mat src, dst;

    if (argc < 2) {
        cerr << "usage: shock-filter <image>" << endl;
        return 1;
    }

    string image = argv[1];

    // mouse with mask
    src = imread(image, CV_LOAD_IMAGE_GRAYSCALE);
    src.convertTo(src, CV_32F);

    if (!src.data) {
        throw runtime_error("Can not load image!");
    }

    deblur::coherenceFilter(src, dst);

    dst.convertTo(dst, CV_8U);
    imwrite("shock-filter.png", dst);

    return 0;
}