#include <iostream>
#include <opencv2/opencv.hpp>
#include "match.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: disparity <left> <right>" << endl;
        return 1;
    }

    Mat left  = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat right = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    if (left.size() != right.size()) {
        cerr << "Error: Images must have same dimensions" << endl;
        return 1;
    }

    Mat left_rgb, right_rgb;
    cvtColor(left, left_rgb, COLOR_BGR2RGB);
    cvtColor(right, right_rgb, COLOR_BGR2RGB);

    Match match(left_rgb.data, right_rgb.data, Coord(left.cols, left.rows), true);

    int lambda = 20;

    // disparity range
    Coord disp_base(-70, 0); // min disparity
    Coord disp_max(0, 0);    // max disparity

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
        2,               // iter_max
        false,           // randomize_every_iteration
        5                // w_size
    };

    match.SetParameters(&kz2_params);
    match.SetDispRange(disp_base, disp_max);

    cerr << "Run KZ2 ..." << endl;
    match.KZ2();
    match.CROSS_CHECK();

    // Transfer result to OpenCV
    Mat disparity_left(left.size(), CV_8U);
    match.SaveXLeft(disparity_left.data, false);

    // Mark occlusions red
    Mat occluded;
    cvtColor(disparity_left, occluded, COLOR_GRAY2BGR);

    for (int row = 0; row < occluded.rows; ++row) {
        for (int col = 0; col < occluded.cols; ++col) {
            if (disparity_left.at<uchar>(row, col) == OCCLUDED) {
                occluded.at<Vec3b>(row, col) = { 0, 0, 255 };
            }
        }
    }

    imshow("Disparity occluded", occluded);

    match.FILL_OCCLUSIONS();
    match.SaveScaledXLeft(disparity_left.data, false);
    imshow("Disparity filled", disparity_left);

    waitKey(0);
    return 0;
}