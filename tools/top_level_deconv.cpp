/***********************************************************************
 * Author:       Franziska Krüger
 *
 * Description:
 * ------------
 * Runs the depth-aware motion deblurring algorithm only on the top-level
 * regions (without psf-refinement in the mid-level regions).
 * 
 ************************************************************************
*/

#include <iostream>                     // cout, cerr, endl
#include <stdexcept>                    // throw exception

#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite
#include <opencv2/imgproc/imgproc.hpp>  // convert

#include "depth_deblur.hpp"
#include "utils.hpp"
#include "disparity_estimation.hpp"

using namespace std;
using namespace cv;
using namespace deblur;


int main(int argc, char** argv) {
    // parameter
    const int threads = 4;
    const int psfWidth = 35;
    const int layers = 12;
    const int maxTopLevelNodes = 3;
    const int maxDisparity = 80;

    Mat srcLeft, srcRight, dstLeft, dstRight;

    if (argc < 3) {
        cerr << "usage: top-level-deconv <left> <right>" << endl;
        return 1;
    }

    string leftName = argv[1];
    string rightName = argv[2];

    // load images
    srcLeft = imread(leftName, 1);
    srcRight = imread(rightName, 1);

    if (!srcLeft.data || !srcRight.data) {
        throw runtime_error("Can not load images!");
    }

    cout << "Start Depth-Aware Motion Deblurring with" << endl;
    cout << "   image left:          " << leftName << endl;
    cout << "   image right:         " << rightName << endl;
    cout << "   max disparity:       " << maxDisparity << endl;
    cout << "   approx. PSF width:   " << psfWidth << endl;
    cout << "   layers/regions:      " << layers << endl;
    cout << "   max top level nodes: " << maxTopLevelNodes << endl;
    cout << "   threads:             " << threads << endl;
    cout << endl;

    // 
    // slighty changed code of runDepthDeblur follows
    // 

    // check if images have the same size
    if (srcLeft.cols != srcRight.cols || srcLeft.rows != srcRight.rows) {
        throw runtime_error("Images aren't of same size!");
    }

    // approximate PSF width has to be greater than 0
    if (psfWidth < 1) {
        throw runtime_error("PSF width has to be greater zero!");
    }


    #ifdef IMWRITE
        imwrite("input-left.png", srcLeft);
        imwrite("input-right.png", srcRight);
    #endif

    // views for disparity estimation
    // they will be updated after the first pass
    array<Mat, 2> deblurViews;
    srcLeft.copyTo(deblurViews[LEFT]);
    srcRight.copyTo(deblurViews[RIGHT]);

    // two passes through algorithm
    for (int i = 0; i < 2; i++) {
        cout << i + 1 << ". Pass Estimation" << endl;

        // this class holds everything needed for one step of the depth-aware deblurring
        DepthDeblur depthDeblur(srcLeft, srcRight, psfWidth, layers);

        // initial disparity estimation of blurred images
        // here: left image is matching image and right image is reference image
        //       I_m(x) = I_r(x + d_m(x))
        cout << " Step 1: disparity estimation" << endl;
        // FIXME: parameter for max disparity
        depthDeblur.disparityEstimation(deblurViews, MATCH, maxDisparity);
        

        cout << " Step 2: region tree reconstruction" << endl;
        depthDeblur.regionTreeReconstruction(maxTopLevelNodes);


        cout << " Step 3: PSF estimation for top-level regions in trees" << endl;
        depthDeblur.toplevelKernelEstimation();


        // cout << " Step 3.1: Iterative PSF estimation" << endl;
        // cout << "   ... jointly compute PSF for middle & leaf level-regions of both views" << endl;
        // depthDeblur.midLevelKernelEstimation(threads);


        cout << " Step 4: Blur removal given top-level PSF" << endl;
        // set new left and right view for second pass
        if ((i + 1) < 2) {
            Mat deconvLeft, deconvRight;
            // use threads
            depthDeblur.deconvolveTopLevel(deconvLeft, LEFT, threads);
            depthDeblur.deconvolveTopLevel(deconvRight, RIGHT, threads);
            
            // this deconvolved images will be used for a disparity update
            deconvLeft.copyTo(deblurViews[LEFT]);
            deconvRight.copyTo(deblurViews[RIGHT]);
        } else {
            // deblur final images
            depthDeblur.deconvolveTopLevel(deblurViews[LEFT], LEFT, threads);
            depthDeblur.deconvolveTopLevel(deblurViews[RIGHT], RIGHT, threads);

            // FIXME: deblur color images
        }

        #ifdef IMWRITE
            string filename = "deconv-" + to_string(i + 1) + "-left.png";
            imwrite(filename, deblurViews[LEFT]);
            filename = "deconv-" + to_string(i + 1) + "-right.png";
            imwrite(filename, deblurViews[LEFT]);
        #endif

        // FIXME: skip second pass because the result is of the first is too bad :(
        // i++;
    }

    deblurViews[LEFT].copyTo(dstLeft);
    deblurViews[RIGHT].copyTo(dstRight);
    
    cout << "finished Algorithm" << endl;




    imwrite("deconv-left.png", dstLeft);
    imwrite("deconv-right.png", dstRight);

    return 0;
}