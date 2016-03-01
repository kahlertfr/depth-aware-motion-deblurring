#include <iostream>                     // cout, cerr, endl
#include <stdexcept>                    // throw exception
#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite
#include <opencv2/imgproc/imgproc.hpp>  // convert

#include "depth_deblur.hpp"             // for one step of the depth-aware deblurring
#include "utils.hpp"                    // convertFloatToUchar

#include "depth_aware_deblurring.hpp"


using namespace std;
using namespace cv;


namespace deblur {

    void depthDeblur(const Mat& blurredLeft, const Mat& blurredRight,
                     Mat& deblurredLeft, Mat& deblurredRight,
                     int psfWidth, const int maxTopLevelNodes) {
        // check if images have the same size
        if (blurredLeft.cols != blurredRight.cols || blurredLeft.rows != blurredRight.rows) {
            throw runtime_error("Images aren't of same size!");
        }

        // approximate PSF width has to be greater than 0
        if (psfWidth < 1) {
            throw runtime_error("PSF width has to be greater zero!");
        }


        #ifdef IMWRITE
            imwrite("input-left.png", blurredLeft);
            imwrite("input-right.png", blurredRight);
        #endif


        // input images at each iteration step
        Mat left, right;
        blurredLeft.copyTo(left);
        blurredRight.copyTo(right);

        // two passes through algorithm
        for (int i = 0; i < 2; i++) {
            cout << i + 1 << ". Pass Estimation" << endl;

            // this class holds everything needed for one step of the depth-aware deblurring
            DepthDeblur depthDeblur(left, right, psfWidth);

            // initial disparity estimation of blurred images
            // here: left image is matching image and right image is reference image
            //       I_m(x) = I_r(x + d_m(x))
            cout << " Step 1: disparity estimation" << endl;
            depthDeblur.disparityEstimation();
            

            cout << " Step 2: region tree reconstruction" << endl;
            depthDeblur.regionTreeReconstruction(maxTopLevelNodes);


            cout << " Step 3: PSF estimation for top-level regions in trees" << endl;
            depthDeblur.toplevelKernelEstimation("left");


            cout << " Step 3.1: Iterative PSF estimation" << endl;
            cout << "   ... jointly compute PSF for middle & leaf level-regions of both views" << endl;
            depthDeblur.midLevelKernelEstimation();


            cout << " Step 4: Blur removal given PSF estimate" << endl;
            // set new left and right view for next pass
            if (i==1) {
                Mat deconvLeft, deconvRight;
                depthDeblur.deconvolve(deconvLeft, LEFT);
                depthDeblur.deconvolve(deconvRight, RIGHT);
                
                deconvLeft.copyTo(left);
                deconvRight.copyTo(right);
            } else {
                // deblur color images
                depthDeblur.deconvolve(deblurredLeft, LEFT, true);
                depthDeblur.deconvolve(deblurredRight, RIGHT, true);
            }

            // TODO: update parameters
            i++; // for debugging without second pass
        }
        
        cout << "finished Algorithm" << endl;
    }


    void depthDeblur(const string filenameLeft, const string filenameRight,
                     const int psfWidth, const int maxTopLevelNodes,
                     const string filenameResultLeft, const string filenameResultRight) {

        // load images
        Mat blurredLeft, blurredRight;
        blurredLeft = imread(filenameLeft, 1);
        blurredRight = imread(filenameRight, 1);

        if (!blurredLeft.data || !blurredRight.data) {
            throw runtime_error("Can not load images!");
        }

        Mat left, right;
        depthDeblur(blurredLeft, blurredRight, left, right, psfWidth, maxTopLevelNodes);

        imwrite(filenameResultLeft, left);
        imwrite(filenameResultRight, right);
    }
}