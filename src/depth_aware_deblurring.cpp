#include <iostream>                     // cout, cerr, endl
#include <stdexcept>                    // throw exception
#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite
#include <opencv2/imgproc/imgproc.hpp>  // convert

#include "disparity_estimation.hpp"     // SGBM, fillOcclusions, quantize

using namespace std;
using namespace cv;

namespace DepthAwareDeblurring {

    /**
     * Step 1: Disparity estimation of two blurred images
     * where occluded regions are filled and where the disparity map is 
     * quantized to l regions.
     * 
     * @param blurredLeft  left blurred input image
     * @param blurredRight right blurred input image
     * @param l            quantizes disparity values until l regions remains
     * @param disparityMap quantized disparity map
     * @param inverse      determine if the disparity is calculated from right to left
     */
    void quantizedDisparityEstimation(const Mat &blurredLeft, const Mat &blurredRight,
                                      const int l, Mat &disparityMap, bool inverse=false) {

        // down sample images to roughly reduce blur for disparity estimation
        Mat blurredLeftSmall, blurredRightSmall;

        // because we checked that both images are of the same size
        // the new size is the same for both too
        // (down sampling ratio is 2)
        Size downsampledSize = Size(blurredLeftSmall.cols / 2, blurredLeftSmall.rows / 2);

        // down sample with Gaussian pyramid
        pyrDown(blurredLeft, blurredLeftSmall, downsampledSize);
        pyrDown(blurredRight, blurredRightSmall, downsampledSize);

        #ifndef NDEBUG
            imshow("blurred left image", blurredLeftSmall);
            imshow("blurred right image", blurredRightSmall);
        #endif

        // convert color images to gray images
        cvtColor(blurredLeftSmall, blurredLeftSmall, CV_BGR2GRAY);
        cvtColor(blurredRightSmall, blurredRightSmall, CV_BGR2GRAY);

        // disparity map with occlusions as black regions
        // 
        // here a different algorithm as the paper approach is used
        // because it is more convenient to use a OpenCV implementation.
        // TODO: functions pointer
        Mat disparityMapSmall;

        // if the disparity is caculated from right to left flip the images
        // because otherwise SGBM will not work
        if (inverse) {
            Mat blurredLeftFlipped, blurredRightFlipped;
            flip(blurredLeftSmall, blurredLeftFlipped, 1);
            flip(blurredRightSmall, blurredRightFlipped, 1);
            blurredLeftFlipped.copyTo(blurredLeftSmall);
            blurredRightFlipped.copyTo(blurredRightSmall);
        }

        DisparityEstimation::semiGlobalBlockMatching(blurredLeftSmall, blurredRightSmall, disparityMapSmall);

        // flip back the disparity map
        if (inverse) {
            Mat disparityFlipped;
            flip(disparityMapSmall, disparityFlipped, 1);
            disparityFlipped.copyTo(disparityMapSmall);
        }

        #ifndef NDEBUG
            string prefix = (inverse) ? "_inverse" : "";
            imshow("original disparity map " + prefix, disparityMapSmall);
            imwrite("dmap_small" + prefix + ".jpg", disparityMapSmall);
        #endif

        // fill occlusion regions (= value < 10)
        DisparityEstimation::fillOcclusionRegions(disparityMapSmall, 10);

        #ifndef NDEBUG
            imshow("disparity map with filled occlusion " + prefix, disparityMapSmall);
            imwrite("dmap_small_filled" + prefix + ".jpg", disparityMapSmall);
        #endif

        // quantize the image
        Mat quantizedDisparity;
        DisparityEstimation::quantizeImage(disparityMapSmall, l, quantizedDisparity);

        #ifndef NDEBUG
            // convert quantized image to be displayable
            Mat disparityViewable;
            double min; double max;
            minMaxLoc(quantizedDisparity, &min, &max);
            quantizedDisparity.convertTo(disparityViewable, CV_8U, 255.0/(max-min));

            imshow("quantized disparity map " + prefix, disparityViewable);
            imwrite("dmap_final" + prefix + ".jpg", disparityViewable);
        #endif

        // up sample disparity map to original resolution
        pyrUp(quantizedDisparity, disparityMap, Size(blurredLeft.cols, blurredLeft.rows));
    }


    void runAlgorithm(const Mat &blurredLeft, const Mat &blurredRight) {
        // check if images have the same size
        if (blurredLeft.cols != blurredRight.cols || blurredLeft.rows != blurredRight.rows) {
            throw runtime_error("ParallelTRDiff::runAlgorithm():Images aren't of same size!");
        }

        // initial disparity estimation of blurred images
        cout << "Step 1: disparity estimation ..." << endl;
        Mat disparityMap1;  // left to rigth disparity
        Mat disparityMap2;  // right to left disparity

        // quantization factor is approximated PSF width/height
        int l = 25;
        cout << "... left to right" << endl;
        quantizedDisparityEstimation(blurredLeft, blurredRight, l, disparityMap1);
        cout << "... right to left" << endl;
        quantizedDisparityEstimation(blurredRight, blurredLeft, l, disparityMap2, true);
        
        cout << "Step 2: region tree reconstruction ..." << endl;
        // to be continued ...

        // Wait for a key stroke
        waitKey(0);
    }


    void runAlgorithm(const string filenameLeft, const string filenameRight) {
        cout << "loads images..." << endl;

        Mat blurredLeft, blurredRight;
        blurredLeft = imread(filenameLeft, 1);
        blurredRight = imread(filenameRight, 1);

        if (!blurredLeft.data || !blurredRight.data) {
            throw runtime_error("ParallelTRDiff::runAlgorithm():Can not load images!");
        }

        runAlgorithm(blurredLeft, blurredRight);
    }
}