/***********************************************************************
 * Author:       Franziska Krüger
 *
 * Description:
 * ------------
 * Runs the psf selection process with an image and given psf candidates.
 * 
 ************************************************************************
*/

#include <iostream>                     // cout, cerr, endl
#include <stdexcept>                    // throw exception

#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite
#include <opencv2/imgproc/imgproc.hpp>  // convert

#include "depth_deblur.hpp"
#include "deconvolution.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;
using namespace deblur;


class PSFSelection : public DepthDeblur {

  public:

    PSFSelection(const cv::Mat& imageLeft, const cv::Mat& imageRight) 
      : DepthDeblur(imageLeft, imageRight, 0, 0)
    {
        // do nothing constructor
    }

    void run(Mat& image, vector<Mat>& candidates, Mat& mask) {
        // mocking region tree
        regionTree.images[LEFT] = &image;
        regionTree.images[RIGHT] = &image;

        // compute entropy
        for (int i = 0; i < candidates.size(); i++) {
            // region tree mocking
            node n;
            n.layers = {0};
            regionTree._tree.push_back(n);
            regionTree._masks[LEFT].push_back(mask);
            regionTree._masks[RIGHT].push_back(mask);

            
            regionTree[i].entropy = computeEntropy(candidates[i]);

            cout << "entropy for " << i << ": " << regionTree[i].entropy << endl;
        }

        Mat winner;
        psfSelection(candidates, winner, 0);
    }
};


int main(int argc, char** argv) {
    // parameter
    const int threads = 4;
    const int psfWidth = 35;
    const int layers = 12;
    const int maxTopLevelNodes = 3;
    const int maxDisparity = 80;

    Mat src, psf1, psf2, psf3;

    if (argc < 5) {
        cerr << "usage: psf-selection <image> <psf1> <psf2> <psf3> [<mask>]" << endl;
        return 1;
    }

    string imageName = argv[1];
    string psfName1 = argv[2];
    string psfName2 = argv[3];
    string psfName3 = argv[4];

    // load images
    src = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
    psf1 = imread(psfName1, CV_LOAD_IMAGE_GRAYSCALE);
    psf2 = imread(psfName2, CV_LOAD_IMAGE_GRAYSCALE);
    psf3 = imread(psfName3, CV_LOAD_IMAGE_GRAYSCALE);

    if (!src.data || !psf1.data || !psf2.data || !psf3.data) {
        throw runtime_error("Can not load images!");
    }

    // load mask
    Mat mask;
    if (argc > 5) {
        string maskName = argv[5];
        mask = imread(maskName, CV_LOAD_IMAGE_GRAYSCALE);
    } else {
        // mask for whole image
        mask = Mat::ones(src.size(), CV_8U);
    }

    // energy preserving kernels
    psf1.convertTo(psf1, CV_32F);
    psf1 /= sum(psf1)[0];
    psf2.convertTo(psf2, CV_32F);
    psf2 /= sum(psf2)[0];
    psf3.convertTo(psf3, CV_32F);
    psf3 /= sum(psf3)[0];

    // psf selection with given candidates
    vector<Mat> candidates;
    candidates.push_back(psf1);
    candidates.push_back(psf2);
    candidates.push_back(psf3);

    PSFSelection selection(src, src);
    selection.run(src, candidates, mask);

    // deconvolve image with all kernels
    Mat floatSrc;
    src.convertTo(floatSrc, CV_32F);
    floatSrc /= 255;

    Mat deconv;
    for (int i = 0; i < candidates.size(); i++) {
        deblur::deconvolveFFT(floatSrc, deconv, candidates[i]);

        // save like matlab imshow([deconv])
        threshold(deconv, deconv, 0.0, -1, THRESH_TOZERO);
        threshold(deconv, deconv, 1.0, -1, THRESH_TRUNC);
        deconv.convertTo(deconv, CV_8U, 255);
        imwrite("deconv-psf" + to_string(i) + ".png", deconv);        
    }

    return 0;
}