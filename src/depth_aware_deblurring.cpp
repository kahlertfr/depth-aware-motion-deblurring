/***********************************************************************
 * Author:       Franziska Krüger
 *
 * Description:
 * ------------
 * Reference Implemenatation of the Depth-Aware Motion Deblurring
 * Algorithm by Xu and Jia.
 * 
 ************************************************************************
*/

#include <iostream>  // cout, cerr, endl
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

namespace DepthAwareDeblurring {

    /**
     * Starts the algorithm
     */
    void runAlgorithm(string filenameLeft, string filenameRight) {
        cout << "loads images..." << endl;

        Mat blurredLeft, blurredRight;
        blurredLeft = imread(filenameLeft, 1);
        blurredRight = imread(filenameRight, 1);

        imshow("Blurred left image", blurredLeft);
        imshow("Blurred right image", blurredRight);

        // Wait for a key stroke
        waitKey(0);
    }
}