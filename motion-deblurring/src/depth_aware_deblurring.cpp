#include <iostream>                     // cout, cerr, endl
#include <stdexcept>                    // throw exception
#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite
#include <opencv2/imgproc/imgproc.hpp>  // convert

#include "depth_deblur.hpp"             // for one step of the depth-aware deblurring
#include "utils.hpp"                    // convertFloatToUchar
#include "disparity_estimation.hpp"     // SGBM MATCH

#include "depth_aware_deblurring.hpp"


using namespace std;
using namespace cv;


namespace deblur {

    void runDepthDeblur(const Mat& blurredLeft, const Mat& blurredRight,
                        Mat& deblurredLeft, Mat& deblurredRight, const int threads,
                        int psfWidth, const int layers, const int maxTopLevelNodes,
                        const int maxDisparity) {
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

        // views for disparity estimation
        // they will be updated after the first pass
        array<Mat, 2> deblurViews;
        blurredLeft.copyTo(deblurViews[LEFT]);
        blurredRight.copyTo(deblurViews[RIGHT]);

        // two passes through algorithm
        for (int i = 0; i < 2; i++) {
            cout << i + 1 << ". Pass Estimation" << endl;

            // this class holds everything needed for one step of the depth-aware deblurring
            DepthDeblur depthDeblur(blurredLeft, blurredRight, psfWidth, layers, DepthDeblur::IRLS);

            // initial disparity estimation of blurred images
            // here: left image is matching image and right image is reference image
            //       I_m(x) = I_r(x + d_m(x))
            cout << " Step 1: disparity estimation" << endl;
            depthDeblur.disparityEstimation(deblurViews, MATCH, maxDisparity);
            

            cout << " Step 2: region tree reconstruction" << endl;
            depthDeblur.regionTreeReconstruction(maxTopLevelNodes);


            cout << " Step 3: PSF estimation for top-level regions in trees" << endl;
            depthDeblur.toplevelKernelEstimation();


            cout << " Step 3.1: Iterative PSF estimation" << endl;
            cout << "   ... jointly compute PSF for middle & leaf level-regions of both views" << endl;
            depthDeblur.midLevelKernelEstimation(threads);


            cout << " Step 4: Blur removal given PSF estimate" << endl;
            // set new left and right view for second pass
            if ((i + 1) < 2) {
                Mat deconvLeft, deconvRight;
                // use threads
                depthDeblur.deconvolve(deconvLeft, LEFT, threads);
                depthDeblur.deconvolve(deconvRight, RIGHT, threads);
                
                // this deconvolved images will be used for a disparity update
                deconvLeft.copyTo(deblurViews[LEFT]);
                deconvRight.copyTo(deblurViews[RIGHT]);
            } else {
                // deblur final images
                depthDeblur.deconvolve(deblurViews[LEFT], LEFT, threads);
                depthDeblur.deconvolve(deblurViews[RIGHT], RIGHT, threads);

                // FIXME: deblur color images
            }

            #ifdef IMWRITE
                imwrite("deconv-" + to_string(i + 1) + "-left.png", deblurViews[LEFT]);
                imwrite("deconv-" + to_string(i + 1) + "-right.png", deblurViews[RIGHT]);
            #endif

            // FIXME: skip second pass because the result is of the first is too bad :(
            i++;
        }

        deblurViews[LEFT].copyTo(deblurredLeft);
        deblurViews[RIGHT].copyTo(deblurredRight);
        
        cout << "finished Algorithm" << endl;
    }


    void runDepthDeblur(const string filenameLeft, const string filenameRight,
                        const int threads, const int psfWidth, const int layers,
                        const int maxTopLevelNodes, const int maxDisparity,
                        const string filenameResultLeft, const string filenameResultRight) {

        // load images
        Mat blurredLeft, blurredRight;
        blurredLeft = imread(filenameLeft, 1);
        blurredRight = imread(filenameRight, 1);

        if (!blurredLeft.data || !blurredRight.data) {
            throw runtime_error("Can not load images!");
        }

        Mat left, right;
        runDepthDeblur(blurredLeft, blurredRight, left, right, threads, psfWidth, layers, maxTopLevelNodes,
                       maxDisparity);

        imwrite(filenameResultLeft, left);
        imwrite(filenameResultRight, right);
    }
}