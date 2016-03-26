#include <iostream>                     // cout, cerr, endl
#include <opencv2/imgproc/imgproc.hpp>  // convert
#include <opencv2/calib3d/calib3d.hpp>  // sgbm

#include "disparity_estimation.hpp"


using namespace cv;
using namespace std;


namespace deblur {

    void quantizedDisparityEstimation(const Mat& blurredLeft, const Mat& blurredRight,
                                          int l, Mat& disparityMap, bool inverse) {

        assert(blurredLeft.type() == CV_8U && "gray values needed");

        // down sample images to roughly reduce blur for disparity estimation
        Mat blurredLeftSmall, blurredRightSmall;

        // because we checked that both images are of the same size
        // the new size is the same for both too
        // (down sampling ratio is 2)
        Size downsampledSize = Size(blurredLeftSmall.cols / 2, blurredLeftSmall.rows / 2);

        // down sample with Gaussian pyramid
        pyrDown(blurredLeft, blurredLeftSmall, downsampledSize);
        pyrDown(blurredRight, blurredRightSmall, downsampledSize);

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

        // fill occlusion regions (= value < 10)
        fillOcclusionRegions(disparityMapSmall, 10);

        // median filter
        Mat median;
        medianBlur(disparityMapSmall, median, 9);
        median.copyTo(disparityMapSmall);

        // quantize the image
        Mat quantizedDisparity;
        quantizeImage(disparityMapSmall, l, quantizedDisparity);

        #ifdef IMWRITE
            // convert quantized image to be displayable
            Mat disparityViewable;
            double min; double max;
            minMaxLoc(quantizedDisparity, &min, &max);
            quantizedDisparity.convertTo(disparityViewable, CV_8U, 255.0/(max-min));

            // imshow("quantized disparity map " + prefix, disparityViewable);
            string filename = "dmap-" + to_string(inverse) + ".png";
            imwrite(filename, disparityViewable);
        #endif

        // up sample disparity map to original resolution without interpolation
        resize(quantizedDisparity, disparityMap, Size(blurredLeft.cols, blurredLeft.rows), 0, 0, INTER_NEAREST);
    }


    /**
     * Fills pixel in a given range with a given uchar.
     * 
     * @param image image to work on
     * @param start starting point
     * @param end   end point
     * @param color color for filling
     */
    static void fillPixel(Mat &image, const Point start, const Point end, const uchar color) {
        for (int row = start.y; row <= end.y; row++) {
            for (int col = start.x; col <= end.x; col++) {
                image.at<uchar>(row, col) = color;
            }
        }
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
        int preFilterCap = 40;        // Truncation value for the prefiltered image pixels
        int uniquenessRatio = 1;      // Margin in percentage by which the best (minimum) computed cost
                                      // function value should “win” the second best value to consider 
                                      // the found match correct (5-15)
        int speckleWindowSize = 150;  // Maximum size of smooth disparity regions (50-200)
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
        Mat newImage(image.size(), CV_8U);
        for (int row = 0; row < image.rows; row++) {
            for (int col = 0; col < image.cols; col++) {
                // store new cluster
                int cluster_idx = labels.at<int>(row + col * image.rows, 0);
                newImage.at<uchar>(row,col) = mapping[cluster_idx];
            }
        }

        newImage.copyTo(quantizedImage);
    }
}
