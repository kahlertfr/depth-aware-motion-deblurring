#include "edge_map.hpp"
#include "coherence_filter.hpp"
#include "utils.hpp" // convertFloatToUchar

using namespace cv;
using namespace std;
using namespace deblur;


namespace DepthAwareDeblurring {

    void gradientMaps(const Mat& image, array<Mat, 2>& gradients) {
        assert(image.type() == CV_8U && "Input image must be grayscaled");

        Mat bilateral;
        bilateralFilter(image, bilateral,
                        5,     // diamter / support size in pxiel
                        0.5,   // range (color) sigma
                        2.0);  // spatial sigma

        // #ifndef NDEBUG
        //     imshow("bilateral", bilateral);
        // #endif

        Mat shock;
        coherenceFilter(bilateral, shock);

        // #ifndef NDEBUG
        //     imshow("shock", shock);
        // #endif
        
        const int delta = 0;
        const int ddepth = CV_32F;
        const int ksize = 3;
        const int scale = 1;

        Sobel(shock, gradients[0], ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
        Sobel(shock, gradients[1], ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

        // #ifndef NDEBUG
        //     Mat normalized;
        //     convertFloatToUchar(normalized, gradients[0]);
        //     imshow("x-gradient", normalized);
        // #endif
    }


    void thresholdGradients(const std::array<cv::Mat,2>& gradients, std::array<cv::Mat,2>& thresholded,
                            const int psfWidth, const InputArray& mask, const int r) {

        assert(gradients[0].size() == gradients[1].size() && "Gradients must be of same size");
            
        // Apply mask
        array<Mat,2> masked = { Mat::zeros(gradients[0].size(), gradients[0].type()),
                                Mat::zeros(gradients[1].size(), gradients[1].type()) };

        gradients[0].copyTo(masked[0], mask);
        gradients[1].copyTo(masked[1], mask);

        #ifndef NDEBUG
            Mat ucharGrads;
            convertFloatToUchar(ucharGrads, masked[0]);
            imshow("input gradients", ucharGrads);
        #endif

        Mat magnitude, angle;
        cartToPolar(masked[0], masked[1], magnitude, angle, true);

        // quantizies magnitude to 255 bins
        Mat discreteMag;
        convertFloatToUchar(discreteMag, magnitude);

        // histograms for 4 bins of angles (45 degrees)
        uchar histo[4][255] = {{0}};
        Mat histoMags[4] = { Mat::zeros(gradients[0].size(), CV_8U),
                             Mat::zeros(gradients[0].size(), CV_8U),
                             Mat::zeros(gradients[0].size(), CV_8U),
                             Mat::zeros(gradients[1].size(), CV_8U)};

        // create the four histograms
        for (int row = 0; row < discreteMag.rows; ++row) {
            for (int col = 0; col < discreteMag.cols; ++col) {
                uchar color = discreteMag.at<uchar>(row, col);
                if (color != 0) {
                    // split gradient into bins depending on its angle
                    // this is the quantization of the angles by 45 degrees
                    int index = (int)(angle.at<float>(row, col) / 45) % 4;

                    histo[index][color]++;
                    histoMags[index].at<uchar>(row, col) = color;
                }
            }
        }
        
        // get range (threshold) of colors that keep at least r*psfWidth pixel of
        // the largest magnitude of each quantized angle
        Mat thresholdedGrads = Mat::zeros(gradients[0].size(), gradients[0].type());
        // TODO: parameter for r
        // int quantity = r * psfWidth;
        int quantity = r * psfWidth * psfWidth;

        // for each histogram
        for (int i = 0; i < 4; i++) {
            int minValue = 255;
            int reachedQuantity = 0;

            // find color value that keeps the claimed number of pixel
            while(reachedQuantity < quantity) {
                reachedQuantity += histo[i][minValue];
                minValue--;
            }

            // get mask of this values
            Mat mask;
            inRange(histoMags[i], minValue, 255, mask);

            // copy the values of the original magnitude to the thresholdedGrads one
            magnitude.copyTo(thresholdedGrads, mask);
        }

        polarToCart(thresholdedGrads, angle, thresholded[0], thresholded[1]);
    }

}