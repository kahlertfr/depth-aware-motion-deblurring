#include <iostream>                     // cout, cerr, endl
#include <math.h>                       // sqrt
#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite

#include "two_phase_psf_estimation.hpp"

using namespace cv;
using namespace std;


namespace TwoPhaseKernelEstimation {

    float norm(float a, float b) {
        return sqrt(a * a + b * b);
    }

    /**
     * Compute usefulness of gradients:
     * 
     *           ||sum_y∈Nh(x) ∇B(y)||
     *  r(x) = ----------------------------
     *          sum_y∈Nh(x) ||∇B(y)|| + 0.5
     *          
     * @param confidence result
     * @param gradients  matrix with x and y gradients
     * @param width      width of window for Nh
     * @param mask       binary mask of region that should be computed
     */
    void computeGradientConfidence(Mat& confidence, const Mat& gradients, const int width,
                                   const Mat& mask) {
        confidence = Mat::zeros(gradients.rows, gradients.cols, CV_32F);

        // get range for Nh window
        int range = width / 2;

        // go through all pixels
        for (int x = width; x < (gradients.cols - width); x++) {
            for (int y = width; y < (gradients.rows - width); y++) {
                // check if inside region
                if (mask.at<uchar>(y, x) != 0) {
                    pair<float, float> sum = {0, 0};  // sum of the part: ||sum_y∈Nh(x) ∇B(y)||
                    float innerSum = 0;               // sum of the part: sum_y∈Nh(x) ||∇B(y)||

                    // sum all gradient values inside the window (width x width) around pixel
                    for (int xOffset = range * -1; xOffset <= range; xOffset++) {
                        for (int yOffset = range * -1; yOffset <= range; yOffset++) {
                            Vec2f gradient = gradients.at<Vec2f>(y + yOffset, x + xOffset);

                            sum.first += gradient[0];
                            sum.second += gradient[1];

                            // norm of gradient
                            innerSum += norm(gradient[0], gradient[1]);
                        }
                    }

                    confidence.at<float>(y, x) = norm(sum.first, sum.second) / (innerSum + 0.5);
                }
            }
        }
    }


    /**
     * Converts a matrix containing floats to a matrix
     * conatining uchars
     * 
     * @param ucharMat resulting matrix
     * @param floatMat input matrix
     */
    void convertFloatToUchar(Mat& ucharMat, const Mat& floatMat) {
        // find min and max value
        double min; double max;
        minMaxLoc(floatMat, &min, &max);

        // if the matrix is in the range [0, 1] just scale with 255
        if (min >= 0 && max < 1) {
            floatMat.convertTo(ucharMat, CV_8U, 255.0);
        } else {
            // handling that floats could be negative
            floatMat -= min;

            // convert and show
            floatMat.convertTo(ucharMat, CV_8U, 255.0/(max-min));
        }
    }


    /* 
    *  reference: http://blog.csdn.net/bluecol/article/details/49924739
    *  
    *  Coherence-Enhancing Shock Filters
    *  Author:WinCoder@qq.com
    *  inspired by
    *  Joachim Weickert "Coherence-Enhancing Shock Filters"
    *  http://www.mia.uni-saarland.de/Publications/weickert-dagm03.pdf
    *  
    *   Paras:
    *   @img        : input image ranging value from 0 to 255.
    *   @sigma      : sobel kernel size.
    *   @str_sigma  : neighborhood size,see detail in reference[2]
    *   @belnd      : blending coefficient.default value 0.5.
    *   @iter       : number of iteration.
    *    
    *   Example:
    *   Mat dst = CoherenceFilter(I,11,11,0.5,4);
    *   imshow("shock filter",dst);
    */
    Mat coherenceFilter(Mat img,int sigma, int str_sigma, float blend, int iter)
    {
        Mat I = img.clone();
        int height = I.rows;
        int width  = I.cols;

        for(int i = 0;i <iter; i++)
        {
            Mat gray;
            cvtColor(I,gray,COLOR_BGR2GRAY);
            Mat eigen;
            cornerEigenValsAndVecs(gray,eigen,str_sigma,3);

            vector<Mat> vec;
            split(eigen,vec);

            Mat x,y;
            x = vec[2];
            y = vec[3];

            Mat gxx,gxy,gyy;
            Sobel(gray,gxx,CV_32F,2,0,sigma);
            Sobel(gray,gxy,CV_32F,1,1,sigma);
            Sobel(gray,gyy,CV_32F,0,2,sigma);

            Mat ero;
            Mat dil;
            erode(I,ero,Mat());
            dilate(I,dil,Mat());

            Mat img1 = ero;
            for(int nY = 0;nY<height;nY++)
            {
                for(int nX = 0;nX<width;nX++)
                {
                    if(x.at<float>(nY,nX)* x.at<float>(nY,nX)* gxx.at<float>(nY,nX)
                        + 2*x.at<float>(nY,nX)* y.at<float>(nY,nX)* gxy.at<float>(nY,nX)
                        + y.at<float>(nY,nX)* y.at<float>(nY,nX)* gyy.at<float>(nY,nX)<0)
                    {
                            img1.at<Vec3b>(nY,nX) = dil.at<Vec3b>(nY,nX);
                    }
                }
            }
            I = I*(1.0-blend)+img1*blend;
        }
        return I;
    }


    /**
     * The final selected edges for kernel estimation are determined as:
     * ∇I^s = ∇I · H (M ||∇I||_2 − τ_s )
     * where H is the Heaviside step function.
     * 
     * @param image      input image which will be shockfiltered (I)
     * @param mask       mask for ruling out some pixel (M)
     * @param selection  result (∇I^s)
     */
    void selectEdges(const Mat& image, const Mat& mask, const float threshold, Mat& selection) {
        // shock filter the input image
        Mat shockImage = coherenceFilter(image, 11, 11, 0.5, 4);
        imshow("shock filter", shockImage);

        // gradients of shock filtered image
        int delta = 0;
        int ddepth = CV_32F;
        int ksize = 3;
        int scale = 1;

        // TODO: convert to gray or not?
        Mat gray, xGradients, yGradients;
        cvtColor(shockImage, gray, CV_BGR2GRAY);
        Sobel(gray, xGradients, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
        Sobel(gray, yGradients, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

        // merge the gradients of x- and y-direction to one matrix
        Mat gradients;
        vector<Mat> grads = {xGradients, yGradients};
        merge(grads, gradients);

        #ifndef NDEBUG
            // display gradients
            Mat xGradientsViewable, yGradientsViewable;
            convertFloatToUchar(xGradientsViewable, xGradients);
            convertFloatToUchar(yGradientsViewable, yGradients);
            imshow("x gradient shock", xGradientsViewable);
            imshow("y gradient shock", yGradientsViewable);
        #endif

        // compute final selected edges
        selection = Mat::zeros(image.rows, image.cols, CV_32FC2);
        // cout << gradients.cols << " " << gradients.rows << endl;
        for (int x = 0; x < gradients.cols; x++) {
            for (int y = 0; y < gradients.rows; y++) {
                // if the mask is zero at the current coordinate the result
                // of the equation (see method description) is zero too.
                // So nothing has to be computed for this case
                if (mask.at<uchar>(y, x) != 0) {
                    Vec2f gradient = gradients.at<Vec2f>(y, x);

                    // if the following equation doesn't hold the value
                    // is also zero and nothing has to be computed
                    if ((norm(gradient[0], gradient[1]) - threshold) > 0) {
                        selection.at<Vec2f>(y,x) = {gradient[0], gradient[1]};
                    }
                }
            }
        }

        #ifndef NDEBUG
            // display gradients
            int from_tox[] = {0, 0};
            mixChannels(selection, xGradients, from_tox, 1);
            int from_toy[] = {1, 0};
            mixChannels(selection, yGradients, from_toy, 1);
            convertFloatToUchar(xGradientsViewable, xGradients);
            convertFloatToUchar(yGradientsViewable, yGradients);
            imshow("x gradient selection", xGradientsViewable);
            imshow("y gradient selection", yGradientsViewable);
        #endif
    }


    void estimateKernel(Mat& psf, const Mat& image, const int psfWidth, const Mat& mask) {
        // set expected kernel witdh to odd number
        int width = (psfWidth % 2 == 0) ? psfWidth + 1 : psfWidth;

        // phase one: initialize kernel
        // 
        // all-zer kernel
        Mat kernel = Mat::zeros(psfWidth, psfWidth, CV_8U);

        // build an image pyramid
        int level = 1;  // TODO: add parameter for this

        vector<Mat> pyramid;
        pyramid.push_back(image);

        for (int i = 0; i < (level - 1); i++) {
            Mat downImage;
            pyrDown(pyramid[i], downImage, Size(pyramid[i].cols/2, pyramid[i].rows/2));
            pyramid.push_back(downImage);
        }

        // go through image image pyramid
        for (int i = 0; i < pyramid.size(); i++) {
            imshow("pyr " + i, pyramid[i]);
            // compute image gradient for x and y direction
            // 
            // gaussian blur
            GaussianBlur(pyramid[i], pyramid[i], Size(3,3), 0, 0, BORDER_DEFAULT);

            // convert it to gray
            Mat gray;
            cvtColor(pyramid[i], gray, CV_BGR2GRAY);

            Mat xGradients, yGradients;
            int delta = 0;
            int ddepth = CV_32F;
            int ksize = 3;
            int scale = 1;

            // gradient x
            Sobel(gray, xGradients, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);

            // gradient y
            Sobel(gray, yGradients, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

            // #ifndef NDEBUG
            //     // display gradients
            //     Mat xGradientsViewable, yGradientsViewable;
            //     convertFloatToUchar(xGradientsViewable, xGradients);
            //     convertFloatToUchar(yGradientsViewable, yGradients);
            //     imshow("x gradient", xGradientsViewable);
            //     imshow("y gradient", yGradientsViewable);
            // #endif

            // merge gradients to one matrix with x and y gradients
            Mat gradients;
            vector<Mat> grads = {xGradients, yGradients};
            merge(grads, gradients);

            // TODO: remove borders of region - how?

            // compute gradient confidence for al pixels
            Mat gradientConfidence;
            computeGradientConfidence(gradientConfidence, gradients, width, mask);

            #ifndef NDEBUG
                // print confidence matrix
                Mat confidenceUchar;
                convertFloatToUchar(confidenceUchar, gradientConfidence);
                imshow("confidence", confidenceUchar);
            #endif

            // create mask for ruling out pixel belonging to small confidence-values
            // M = H(r - τ_r) where H is Heaviside step function
            Mat edgeMask;
            float threshold = 0.25;  // TODO: value? confidence is between 0 and 1
            inRange(gradientConfidence, threshold, 1, edgeMask);
            imshow("edge mask", edgeMask);

            int iterations = 1;  // TODO: add parameter for this
            for (int i = 0; i < iterations; i++) {
                // select edges for kernel estimation
                Mat selectedEdges;
                // TODO: threshold? gradient can be very high
                selectEdges(pyramid[i], edgeMask, 50, selectedEdges);
            }

            // TODO: continue
        }

        psf = kernel;
    }
}