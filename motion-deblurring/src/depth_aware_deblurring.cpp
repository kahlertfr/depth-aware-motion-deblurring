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

    void depthDeblur(const Mat &blurredLeft, const Mat &blurredRight,
                     const int psfWidth, const int maxTopLevelNodes) {
        // check if images have the same size
        if (blurredLeft.cols != blurredRight.cols || blurredLeft.rows != blurredRight.rows) {
            throw runtime_error("Images aren't of same size!");
        }

        // approximate PSF width has to be greater than 0
        if (psfWidth < 1) {
            throw runtime_error("PSF width has to be greater zero!");
        }

        // compute gray value images
        Mat grayLeft, grayRight;
        if (blurredLeft.type() == CV_8UC3) {
            cvtColor(blurredLeft, grayLeft, CV_BGR2GRAY);
        }
        else {
            grayLeft = blurredLeft;
        }

        if (blurredRight.type() == CV_8UC3) {
            cvtColor(blurredRight, grayRight, CV_BGR2GRAY);
        }
        else {
            grayRight = blurredRight;
        }


        #ifndef NDEBUG
            imshow("blurred left image", blurredLeft);
            imshow("blurred right image", blurredRight);
        #endif


        // input images at each iteration step
        Mat left, right;
        grayLeft.copyTo(left);
        grayRight.copyTo(right);

        // two passes through algorithm
        for (int i = 0; i < 2; i++) {
            cout << i + 1 << ". Pass Estimation" << endl;

            // this class holds everything needed for one step of the depth-aware deblurring
            DepthDeblur depthDeblur(&left, &right, psfWidth);

            // // initial disparity estimation of blurred images
            // // here: left image is matching image and right image is reference image
            // //       I_m(x) = I_r(x + d_m(x))
            // cout << " Step 1: disparity estimation" << endl;
            // depthDeblur.disparityEstimation();
            

            // cout << " Step 2: region tree reconstruction" << endl;
            // depthDeblur.regionTreeReconstruction(maxTopLevelNodes);


            // // compute PSFs for toplevels of the region trees
            // cout << " Step 3: PSF estimation for top-level regions in trees" << endl;
            // depthDeblur.toplevelKernelEstimation("left");


            // cout << " Step 3.1: Iterative PSF estimation" << endl;
            // cout << "   ... jointly compute PSF for middle & leaf level-regions of both views" << endl;
            // depthDeblur.midLevelKernelEstimation();

            cout << " Step 4: Blur removal given PSF estimate" << endl;
            depthDeblur.deconvolve();
            
            // TODO: deconvolve images
            // TODO: set new left and right input image 
            // TODO: update parameters
            i++; // for debugging without second pass
        }
        
        cout << "finished Algorithm" << endl;

        #ifndef NDEBUG
            // Wait for a key stroke
            waitKey(0);
        #endif
    }


    void depthDeblur(const string filenameLeft, const string filenameRight,
                     const int psfWidth, const int maxTopLevelNodes) {

        // load images
        Mat blurredLeft, blurredRight;
        blurredLeft = imread(filenameLeft, 1);
        blurredRight = imread(filenameRight, 1);

        if (!blurredLeft.data || !blurredRight.data) {
            throw runtime_error("Can not load images!");
        }

        depthDeblur(blurredLeft, blurredRight, psfWidth, maxTopLevelNodes);
    }
}