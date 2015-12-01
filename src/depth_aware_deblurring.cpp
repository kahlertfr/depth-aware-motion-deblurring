/***********************************************************************
 * Author:       Franziska Krüger
 * Requirements: OpenCV 3
 *
 * Description:
 * ------------
 * Reference Implementation of the Depth-Aware Motion Deblurring
 * Algorithm by Xu and Jia (2012).
 *
 * Algorithm:
 * First-Pass Estimation
 * 1. Stereo-matching-based disparity estimation
 * 2. Region-tree construction
 * 3. PSF estimation for top-level regions in trees
 *     3.1 PSF refinement
 *     3.2 PSF candidate generation and selection
 * 4. If not leaf-level propagate PSF estimation to lower-level regions
 *   and go to step 3.1
 * 5. Blur removal given the PSF estimation
 *
 * Second-Pass Estimation
 * 6. Update disparities based on the deblurred images
 * 7. PSF estimation by repeating Steps 2-5
 * 
 ************************************************************************
*/

#include <iostream>                     // cout, cerr, endl
#include <stdexcept>                    // throw exception
#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite
#include <opencv2/imgproc/imgproc.hpp>  // convert
#include <opencv2/calib3d/calib3d.hpp>  // sgbm

using namespace std;
using namespace cv;

namespace DepthAwareDeblurring {

    /**
     * Fills pixel in a given range with a given uchar.
     * 
     * @param image image to work on
     * @param start starting point
     * @param end   end point
     * @param color color for filling
     */
    void fillPixel(Mat &image, const Point start, const Point end, const uchar color) {
        for (int row = start.y; row <= end.y; row++) {
            for (int col = start.x; col <= end.x; col++) {
                image.at<uchar>(row, col) = color;
            }
        }
    }


    /**
     * Fills occlusion regions (where the value is smaller than a given threshold) 
     * with smallest neighborhood disparity (in a row) because just relatively small 
     * disparities can be occluded.
     * 
     * @param disparityMap disparity map where occlusions will be filled
     * @param threshold    threshold for detecting occluded regions
     */
    void fillOcclusionRegions(Mat &disparityMap, const int threshold = 0) {
        uchar minDisparity = 255;
        Point start(-1,-1);

        // go through each pixel
        for (int row = 0; row < disparityMap.rows; row++) {
            for (int col = 0; col < disparityMap.cols; col++) {
                uchar value = disparityMap.at<uchar>(row, col);

                // check if in occluded region
                if (start != Point(-1, -1)) {
                    // found next disparity or reached end of the row
                    if (value > threshold || col == disparityMap.cols - 1) {
                        // compare current disparity - find smallest
                        minDisparity = (minDisparity < value || col == disparityMap.cols - 1)
                                       ? minDisparity
                                       : value;

                        // fill whole region and reset the start point of the region
                        fillPixel(disparityMap, start, Point(col, row), minDisparity);
                        start = Point(-1,-1);
                    }
                } else {
                    // found new occluded pixel
                    if (value <= threshold) {
                        // there is no left neighbor at column 0 so check it
                        minDisparity = (col > 0) ? disparityMap.at<uchar>(row, col - 1) : 255;
                        start = Point(col, row);
                    }
                }
            }
        }
    }


    /**
     * Uses OpenCVs semi global block matching algorithm to obtain
     * a disparity map with occlusion as black regions
     * 
     * @param left         left image
     * @param right        right image
     * @param disparityMap disparity with occlusions
     */
    void semiGlobalBlockMatching(const Mat &left, const Mat &right, Mat &disparityMap) {
        // set up stereo block match algorithm
        // (found nice parameter values for a good result on many images)
        int minDis = -64;             // minimum disparity
        int disRange = 16 * 10;       // range: Maximum disparity minus minimum disparity
        int blockSize = 9;            // Matched block size (have to be odd: 3-11)
        int p1 = 900;                 // P1 disparity smoothness (default: 0)
                                      // penalty for disparity changes +/- 1
        int p2 = 3000;                // P2 disparity smoothness (default: 0)
                                      // penalty for disparity changes more than 1
        int disp12MaxDiff = 2;        // Maximum allowed difference in the left-right disparity check
        int preFilterCap = 20;        // Truncation value for the prefiltered image pixels
        int uniquenessRatio = 1;      // Margin in percentage by which the best (minimum) computed cost
                                      // function value should “win” the second best value to consider 
                                      // the found match correct (5-15)
        int speckleWindowSize = 150;  // Maximum size of smooth disparity regions (50-200)
        int speckleRange = 1;         // Maximum disparity variation within each connected component (1-2)
        bool fullDP = true;          // Set it to true to run the full-scale two-pass dynamic programming algorithm

        // #ifndef NDEBUG
        //     cout << "  parameter of SGBM" << endl;
        //     cout << "    min disparity     " << minDis << endl;
        //     cout << "    disparity range   " << disRange << endl;
        //     cout << "    block size        " << blockSize << endl;
        //     cout << "    P1                " << p1 << endl;
        //     cout << "    P2                " << p2 << endl;
        //     cout << "    disp12MaxDiff     " << disp12MaxDiff << endl;
        //     cout << "    preFilterCap      " << preFilterCap << endl;
        //     cout << "    uniquenessRatio   " << uniquenessRatio << endl;
        //     cout << "    speckleWindowSize " << speckleWindowSize << endl;
        //     cout << "    speckleRange      " << speckleRange << endl;
        //     cout << "    fullDP            " << fullDP << endl;
        // #endif

        Ptr<StereoSGBM> sgbm = StereoSGBM::create(minDis, disRange, blockSize // );
                                                , p1, p2, disp12MaxDiff, preFilterCap, uniquenessRatio,
                                                speckleWindowSize, speckleRange, fullDP);
                                                  
        sgbm->compute(left, right, disparityMap);

        // get its extreme values
        double minVal; double maxVal;
        minMaxLoc(disparityMap, &minVal, &maxVal);

        // convert disparity map to values between 0 and 255
        // scale factor for conversion is: 255 / (max - min)
        disparityMap.convertTo(disparityMap, CV_8UC1, 255/(maxVal - minVal));
    }


    /**
     * Quantizes a given picture with kmeans algorithm.
     * 
     * @param image          input image
     * @param k              cluster number
     * @param quantizedImage clustered image
     */
    void quantizeImage(const Mat& image, const int k, Mat& quantizedImage) {
        // map the image to the samples
        Mat samples(image.total(), 1, CV_32F);

        for (int row = 0; row < image.rows; row++) {
            for (int col = 0; col < image.cols; col++) {
                    samples.at<float>(row + col * image.rows, 0) = image.at<uchar>(row, col);
            }    
        }

        // kmeans clustering
        Mat labels;
        Mat centers;

        kmeans(samples,                // input
               k,                      // number of cluster
               labels,                 // found labeling (1D array)
               TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), 
                                       // termination criterion: here number of iterations
               5,                      // attempts - execution with different inital labelings
               KMEANS_RANDOM_CENTERS,  // flags: random initial centers
               centers);               // output of cluster center

        // map the clustering to an image
        Mat newImage(image.size(), CV_32F);
        for (int row = 0; row < image.rows; row++) {
            for (int col = 0; col < image.cols; col++) {
                // use color of cluster center for clustered image
                int cluster_idx = labels.at<int>(row + col * image.rows, 0);

                newImage.at<float>(row,col) = centers.at<float>(cluster_idx, 0);
            }
        }

        newImage.copyTo(quantizedImage);
    }


    /**
     * Step 1: Disparity estimation of two blurred images
     * where occluded regions are filled and the disparity map quantized.
     * 
     * @param blurredLeft  left blurred input image
     * @param blurredRight right blurred input image
     * @param l            quantizes disparity values until l regions remains
     * @param disparityMap quantized disparity map
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

        semiGlobalBlockMatching(blurredLeftSmall, blurredRightSmall, disparityMapSmall);

        // flip back the disparity map
        if (inverse) {
            Mat disparityFlipped;
            flip(disparityMapSmall, disparityFlipped, 1);
            disparityFlipped.copyTo(disparityMapSmall);
        }

        #ifndef NDEBUG
            string prefix = (inverse) ? "inverse" : "";
            imshow("original disparity map " + prefix, disparityMapSmall);
            imwrite("dmap_small_" + prefix + ".jpg", disparityMapSmall);
        #endif

        // fill occlusion regions (= value < 10)
        fillOcclusionRegions(disparityMapSmall, 10);


        #ifndef NDEBUG
            imshow("disparity map with filled occlusion " + prefix, disparityMapSmall);
            imwrite("dmap_small_filled_" + prefix + ".jpg", disparityMapSmall);
        #endif

        // // // additional step deviate from paper: smooth the disparity with a median filter
        // // // to eliminate noise
        // // Mat smooth(disparityMapSmall.size(), CV_8U);
        // // medianBlur(disparityMapSmall, smooth, 7);

        // // quantize the image
        // Mat quantizedDisparity;
        // quantizeImage(disparityMapSmall, l, quantizedDisparity);

        // // convert quantized image to be displayable
        // double min; double max;
        // minMaxLoc(quantizedDisparity, &min, &max);
        // quantizedDisparity.convertTo(quantizedDisparity, CV_8U, 255.0/(max-min));

        // #ifndef NDEBUG
        //     imshow("quantized disparity map " + prefix, quantizedDisparity);
        //     imwrite("dmap_final_" + prefix + ".jpg", quantizedDisparity);
        // #endif
        

        // up sample disparity map to original resolution
        // pyrUp(quantizedDisparity, disparityMap, Size(blurredLeft.cols, blurredLeft.rows));
        pyrUp(disparityMapSmall, disparityMap, Size(blurredLeft.cols, blurredLeft.rows));
    }


    /**
     * Starts depth-aware motion deblurring algorithm with given blurred images (matrices)
     * 
     * @param blurredLeft  OpenCV matrix of blurred left image
     * @param blurredRight OpenCV matrix of blurred right image
     */
    void runAlgorithm(const Mat &blurredLeft, const Mat &blurredRight) {
        // check if images have the same size
        if (blurredLeft.cols != blurredRight.cols || blurredLeft.rows != blurredRight.rows) {
            throw runtime_error("ParallelTRDiff::runAlgorithm():Images aren't of same size!");
        }

        // initial disparity estimation of blurred images
        cout << "Step 1: disparity estimation ..." << endl;
        Mat disparityMap1;
        Mat disparityMap2;

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


    /**
     * Loads images from given filenames and then starts the depth-aware motion 
     * deblurring algorithm
     * 
     * @param filenameLeft  relative or absolute path to blurred left image
     * @param filenameRight relative or absolute path to blurred right image
     */
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