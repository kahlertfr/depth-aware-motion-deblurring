#include <iostream>                     // cout, cerr, endl
#include <opencv2/imgproc/imgproc.hpp>  // convert

#include <opencv2/calib3d/calib3d.hpp>  // sgbm
#include "match.h"                      // match algorithm

#include "disparity_estimation.hpp"
#include "utils.hpp"                    // LEFT RIGHT fillPixel

using namespace cv;
using namespace std;


namespace deblur {

    void disparityFilledMatch(const array<Mat, 2>& images, array<Mat, 2>& dMaps,
                              int maxDisparity) {

        bool color = (images[LEFT].type() == CV_8UC3);

        Mat left, right;

        if (color) {
            // OpenCV stores colorimages as BGR so we have to convert them
            cvtColor(images[LEFT], left, COLOR_BGR2RGB);
            cvtColor(images[RIGHT], right, COLOR_BGR2RGB);
        } else {
            left = images[LEFT];
            right = images[RIGHT];
        }

        // create new Match object
        Match match(left.data, right.data, Coord(left.cols, left.rows), color);

        // parameters
        int lambda = 20;

        // disparity range
        Coord disp_base(-1 * maxDisparity, 0); // min disparity
        Coord disp_max(0, 0);                  // max disparity

        Match::Parameters kz2_params = {
            true,                  // subpixel
            Match::Parameters::L2, // data_cost
            1,                     // denominator

            // Smoothness
            // ----------
            5,          // I_threshold
            8,          // I_threshold2
            20,         // interaction_radius
            3 * lambda, // lambda1
            lambda,     // lambda2
            5 * lambda, // K

            MATCH_INFINITY,  // occlusion_penalty
            1,               // iter_max // FIXME: for edbug
            false,           // randomize_every_iteration
            5                // w_size
        };

        match.SetParameters(&kz2_params);
        match.SetDispRange(disp_base, disp_max);

        // using the "Computing Visual Correspondence with Occlusions using Graph Cuts" algorithm from
        // Vladimir Kolmogorov and Ramin Zabih
        match.KZ2();

        // doing cross-checking afterwards
        match.CROSS_CHECK();

        // fill occlusions
        match.FILL_OCCLUSIONS();

        // transfer the results to OpenCV (left-rigth)
        Mat disparity_left(left.size(), CV_8U);
        match.SaveScaledXLeft(disparity_left.data, false);
        disparity_left.copyTo(dMaps[LEFT]);

        // to get the results for right-left disparity
        // we have to swap the images
        match.SWAP_IMAGES();
        // save the image with inverted colors
        Mat disparity_right(right.size(), CV_8U);
        match.SaveScaledXLeft(disparity_right.data, true);
        disparity_right.copyTo(dMaps[RIGHT]);

        imshow("Disparity filled in method", dMaps[RIGHT]);
        waitKey(0);
    }


    void disparityFilledSGBM(const array<Mat, 2>& images, array<Mat, 2>& dMaps) {
        // disparity map for left-right
        semiGlobalBlockMatching(images[LEFT], images[RIGHT], dMaps[LEFT]);

        // disparity map from right to left
        // therfore flip the images because otherwise SGBM will not work
        Mat smallLeftFlipped, smallRightFlipped;
        flip(images[LEFT], smallLeftFlipped, 1);
        flip(images[RIGHT], smallRightFlipped, 1);
        smallLeftFlipped.copyTo(images[LEFT]);
        smallRightFlipped.copyTo(images[RIGHT]);

        // disparity map for left-right
        semiGlobalBlockMatching(images[RIGHT], images[LEFT], dMaps[RIGHT]);

        // flip disparity map back
        Mat disparityFlipped;
        flip(dMaps[RIGHT], disparityFlipped, 1);
        disparityFlipped.copyTo(dMaps[RIGHT]);


        // fill occlusion regions (= value < 10)
        fillOcclusionRegions(dMaps[LEFT], 10);
        fillOcclusionRegions(dMaps[RIGHT], 10);

        // median filter
        Mat median;
        medianBlur(dMaps[LEFT], median, 9);
        median.copyTo(dMaps[LEFT]);
        medianBlur(dMaps[RIGHT], median, 9);
        median.copyTo(dMaps[RIGHT]);
    }


    void fillOcclusionRegions(Mat& disparityMap, const uchar threshold) {
        assert(disparityMap.type() == CV_8U && "gray values needed");

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


    void semiGlobalBlockMatching(const Mat& left, const Mat& right, Mat& disparityMap) {
        // set up stereo block match algorithm
        // (found nice parameter values for a good result on many images)
        int minDis = -64;             // minimum disparity
        int disRange = 16 * 10;       // range: Maximum disparity minus minimum disparity
        int blockSize = 9;            // Matched block size (have to be odd: 3-11)
        int p1 = 600;                 // P1 disparity smoothness (default: 0)
                                      // penalty for disparity changes +/- 1
        int p2 = 3000;                // P2 disparity smoothness (default: 0)
                                      // penalty for disparity changes more than 1
        int disp12MaxDiff = 2;        // Maximum allowed difference in the left-right disparity check
        int preFilterCap = 0;        // Truncation value for the prefiltered image pixels
        int uniquenessRatio = 1;      // Margin in percentage by which the best (minimum) computed cost
                                      // function value should “win” the second best value to consider 
                                      // the found match correct (5-15)
        int speckleWindowSize = 50;    // Maximum size of smooth disparity regions (50-200)
        int speckleRange = 1;         // Maximum disparity variation within each connected component (1-2)
        bool fullDP = true;           // Set it to true to run the full-scale two-pass dynamic programming algorithm

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

        Ptr<StereoSGBM> sgbm = StereoSGBM::create(minDis, disRange, blockSize,
                                                  p1, p2, disp12MaxDiff, preFilterCap, uniquenessRatio,
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
     * Compares a pair depending on its second value.
     */
    bool comparePairOnSecond (const pair<int, float> i, const pair<int, float>  j) { 
        return (i.second < j.second);
    }


    void quantizeImage(const array<Mat,2>& images, const int k, array<Mat,2>& quantizedImages) {
        assert(images[0].size() == images[1].size() && "Both images have to be of the same size");

        int totalPixels = images[0].total() + images[1].total();
        int rows = images[0].rows;
        int cols = images[0].cols;
        int offset = images[0].total();
        
        // map the image to the samples
        Mat samples(totalPixels, 1, CV_32F);

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                    samples.at<float>(row + col * rows, 0)          = images[0].at<uchar>(row, col);
                    samples.at<float>(row + col * rows + offset, 0) = images[1].at<uchar>(row, col);
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

        // sort clusters such that they represent the ordered disparities
        // store pairs of (cluster, color) in a vector and sort it depending on the color
        vector<pair<int, float>> clusters;
        clusters.reserve(k);

        for (int i = 0; i < k; i++) {
            pair<int, float> cluster(i, centers.at<float>(i, 0));
            clusters.push_back(cluster);
        }

        sort(clusters.begin(), clusters.end(), comparePairOnSecond);

        // create mapping of old cluster to new (ordered) cluster
        vector<int> mapping;
        mapping.resize(k);

        for (int i = 0; i < k; i++) {
            mapping[clusters[i].first] = i;
        }

        // map the clustering to an image
        Mat newImage1(images[0].size(), CV_8U);
        Mat newImage2(images[1].size(), CV_8U);

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                // store new cluster
                int cluster_idx = labels.at<int>(row + col * rows, 0);
                newImage1.at<uchar>(row,col) = mapping[cluster_idx];

                cluster_idx = labels.at<int>(row + col * rows + offset, 0);
                newImage2.at<uchar>(row,col) = mapping[cluster_idx];
            }
        }

        newImage1.copyTo(quantizedImages[0]);
        newImage2.copyTo(quantizedImages[1]);
    }
}
