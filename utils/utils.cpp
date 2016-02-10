#include "utils.hpp"

using namespace cv;
using namespace std;

namespace deblur {

    void fft(const Mat& src, Mat& dst) {
        if (src.type() == CV_32F) {
            // for fast DFT expand image to optimal size
            Mat padded;
            int m = getOptimalDFTSize( src.rows );
            int n = getOptimalDFTSize( src.cols );

            // on the border add zero pixels
            copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols,
                               BORDER_CONSTANT, Scalar::all(0));

            // add to the expanded real plane another imagniary plane with zeros
            Mat planes[] = {padded,
                                Mat::zeros(padded.size(), CV_32F)};
            merge(planes, 2, dst);
        } else if (src.type() == CV_32FC2) {
            src.copyTo(dst);
        } else {
            assert(false && "fft works on 32FC1- and 32FC1-images");
        }

        // this way the result may fit in the source matrix
        // 
        // DFT_COMPLEX_OUTPUT suppress to creation of a dense CCS matrix
        // but we want a simple complex matrix
        dft(dst, dst, DFT_COMPLEX_OUTPUT);

        // assert(padded.size() == dst.size() && "Resulting complex matrix must be of same size");
    }


    void dft(Mat& src, Mat& dst){
        // create complex matrix
        Mat planes[] = {src,
                        Mat::zeros(src.size(), CV_32F)};
        merge(planes, 2, dst);

        // DFT_COMPLEX_OUTPUT suppress to creation of a dense CCS matrix
        // but we want a simple complex matrix
        dft(dst, dst, DFT_COMPLEX_OUTPUT);
    }


    void convertFloatToUchar(const Mat& src, Mat& dst) {
        // find min and max value
        double min; double max;
        minMaxLoc(src, &min, &max);

        // if the matrix is in the range [0, 1] just scale with 255
        if (min >= 0 && max < 1) {
            src.convertTo(dst, CV_8U, 255.0/(max-min));
        } else {
            Mat copy;
            src.copyTo(copy);

            // handling that floats could be negative
            copy -= min;

            // convert and show
            copy.convertTo(dst, CV_8U, 255.0/(max-min));
        }
    }


    void showFloat(const string name, const Mat& src, const bool write){
        assert(src.type() == CV_32F && "works for single channel ");

        Mat srcUchar;
        convertFloatToUchar(src, srcUchar);
        imshow(name, srcUchar);

        if (write){
            string filename = name + ".png";
            imwrite(filename, srcUchar);
        }
    }


    void swapQuadrants(Mat& image) {
        // rearrange the quadrants of Fourier image  so that the origin is at the image center
        int cx = image.cols/2;
        int cy = image.rows/2;

        Mat q0(image, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
        Mat q1(image, Rect(cx, 0, cx, cy));  // Top-Right
        Mat q2(image, Rect(0, cy, cx, cy));  // Bottom-Left
        Mat q3(image, Rect(cx, cy, cx, cy)); // Bottom-Right

        Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
        q2.copyTo(q1);
        tmp.copyTo(q2);
    }


    void showComplexImage(const string windowName, const Mat& complex) {
        // compute the magnitude and switch to logarithmic scale
        // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
        Mat planes[] = {Mat::zeros(complex.size(), CV_32F), Mat::zeros(complex.size(), CV_32F)};
        split(complex, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
        magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
        Mat magI = planes[0];

        magI += Scalar::all(1);                    // switch to logarithmic scale
        log(magI, magI);

        // crop the spectrum, if it has an odd number of rows or columns
        magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

        swapQuadrants(magI);

        normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
                                                // viewable image form (float between values 0 and 1).

        imshow(windowName, magI);
    }


    void normalizeOne(Mat& src, Mat& dst) {
        if (src.channels() == 1) {
            double min, max;
            minMaxLoc(src, &min, &max);
            const double scale = std::max(abs(min), abs(max));
            normalize(src, dst, min / scale, max / scale, NORM_MINMAX);
        } else if (src.channels() == 2) {
            vector<Mat> channels;
            vector<Mat> tmp(2);

            split(src, channels);
            normalizeOne(channels, tmp);
            merge(tmp, dst);

        } else {
            assert(false && "Input must have 1- or 2-channels");
        }
    }
}
