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
     * @param confidence result (r)
     * @param gradients  vector with x and y gradients (∇B)
     * @param width      width of window for Nh
     * @param mask       binary mask of region that should be computed
     */
    void computeGradientConfidence(Mat& confidence, const vector<Mat>& gradients, const int width,
                                   const Mat& mask) {

        int rows = gradients[0].rows;
        int cols = gradients[0].cols;
        confidence = Mat::zeros(rows, cols, CV_32F);

        // get range for Nh window
        int range = width / 2;

        // go through all pixels
        for (int x = width; x < (cols - width); x++) {
            for (int y = width; y < (rows - width); y++) {
                // check if inside region
                if (mask.at<uchar>(y, x) != 0) {
                    pair<float, float> sum = {0, 0};  // sum of the part: ||sum_y∈Nh(x) ∇B(y)||
                    float innerSum = 0;               // sum of the part: sum_y∈Nh(x) ||∇B(y)||

                    // sum all gradient values inside the window (width x width) around pixel
                    for (int xOffset = range * -1; xOffset <= range; xOffset++) {
                        for (int yOffset = range * -1; yOffset <= range; yOffset++) {
                            float xGradient = gradients[0].at<float>(y + yOffset, x + xOffset);
                            float yGradient = gradients[1].at<float>(y + yOffset, x + xOffset);

                            sum.first += xGradient;
                            sum.second += yGradient;

                            // norm of gradient
                            innerSum += norm(xGradient, yGradient);
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
            Mat copy;
            floatMat.copyTo(copy);
            
            // handling that floats could be negative
            copy -= min;

            // convert and show
            copy.convertTo(ucharMat, CV_8U, 255.0/(max-min));
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
     * where H is the Heaviside step function and M = = H(r - τ_r)
     * 
     * @param image      input image which will be shockfiltered (I)
     * @param confidence mask for ruling out some pixel (r)
     * @param r          threshold for edge mask (value should be in range [0,1]) (τ_r)
     * @param s          threshold for edge selection (value should be in range [0, 200]) (τ_s)
     * @param selection  result (∇I^s)
     */
    void selectEdges(const Mat& image, const Mat& confidence, const float r, const float s, vector<Mat>& selection) {
        // create mask for ruling out pixel belonging to small confidence-values
        // M = H(r - τ_r) where H is Heaviside step function
        Mat mask;
        inRange(confidence, r, 1, mask);
        imshow("edge mask", mask);

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

        // save gradients of the final selected edges
        selection = {Mat::zeros(image.rows, image.cols, CV_32F),
                     Mat::zeros(image.rows, image.cols, CV_32F)};

        for (int x = 0; x < gradients.cols; x++) {
            for (int y = 0; y < gradients.rows; y++) {
                // if the mask is zero at the current coordinate the result
                // of the equation (see method description) is zero too.
                // So nothing has to be computed for this case
                if (mask.at<uchar>(y, x) != 0) {
                    Vec2f gradient = gradients.at<Vec2f>(y, x);

                    // if the following equation doesn't hold the value
                    // is also zero and nothing has to be computed
                    if ((norm(gradient[0], gradient[1]) - s) > 0) {
                        selection[0].at<float>(y,x) = gradient[0];
                        selection[1].at<float>(y,x) = gradient[1];
                    }
                }
            }
        }

        #ifndef NDEBUG
            // display gradients
            convertFloatToUchar(xGradientsViewable, selection[0]);
            convertFloatToUchar(yGradientsViewable, selection[1]);
            imshow("x gradient selection", xGradientsViewable);
            imshow("y gradient selection", yGradientsViewable);
        #endif
    }


    /**
     * Applies DFT after expanding input image to optimal size for Fourier transformation
     * 
     * @param image   input image
     * @param complex result as 2 channel matrix with complex numbers
     */
    void FFT(const Mat& image, Mat& complex) {
        // for fast DFT expand image to optimal size
        Mat padded;
        int m = getOptimalDFTSize( image.rows );
        int n = getOptimalDFTSize( image.cols );

        // on the border add zero pixels
        copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));

        // Add to the expanded another plane with zeros
        Mat planes[] = {padded, Mat::zeros(padded.size(), CV_32F)};
        merge(planes, 2, complex);

        // this way the result may fit in the source matrix
        dft(complex, complex); 
    }


    /**
     * Displays a matrix with complex numbers stored as 2 channels
     * Copied from: http://docs.opencv.org/2.4/doc/tutorials/core/
     * discrete_fourier_transform/discrete_fourier_transform.html
     * 
     * @param windowName name of window
     * @param complex    matrix that should be displayed
     */
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

        // rearrange the quadrants of Fourier image  so that the origin is at the image center
        int cx = magI.cols/2;
        int cy = magI.rows/2;

        Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
        Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
        Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
        Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

        Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
        q2.copyTo(q1);
        tmp.copyTo(q2);

        normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
                                                // viewable image form (float between values 0 and 1).

        imshow(windowName, magI);
    }


    /**
     * Multiplication with complex numbers with real and imaginare part
     * 
     * w * z = (a + bi) * (c + di) = (a*c - b*d) + (a*d + c*b)i
     * 
     * @param  w first complex number
     * @param  z second complex number
     * @return   result of w * z as real and imaginary part
     */
    inline Vec2f compMult(Vec2f w, Vec2f z) {
        return {(w[0] * z[0] - w[1] * z[1]),
                (w[0] * z[1] + w[1] * z[0])};
    }


    /**
     * Addition with complex numbers with real and imaginary part
     *
     * w + z = (a + bi) + (c + di) = (a + c) + (b + d)i
     * 
     * @param  w first complex number
     * @param  z second complex number
     * @return   result of w + z as real and imaginary part
     */
    inline Vec2f compAdd(Vec2f w, Vec2f z) {
        return {w[0] + z[0], w[1] + z[1]};
    }


    /**
     * Division with complex numbers with real and imaginary part
     *
     * w   a*c + b*d + (c*b - a*d)i
     * - = ------------------------
     * z           c² + d²
     * 
     * @param  w first complex number
     * @param  z second complex number
     * @return   result of w / z as real and imaginary part
     */
    inline Vec2f compDiv(Vec2f w, Vec2f z) {
        float denominator = z[0] * z[0] + z[1] * z[1];
        return {(w[0] * z[0] + w[1] * z[1]) / denominator,
                (z[0] * w[1] - w[0] * z[1]) / denominator};
    }


    /**
     * With the critical edge selection, initial kernel erstimation can be accomplished quickly.
     * Objective function: E(k) = ||∇I^s ⊗ k - ∇B||² + γ||k||²
     * 
     * @param selectionGrads  vector of x and y gradients of final selected edges (∇I^s)
     * @param blurredGrads    vector of x and y gradients of blurred image (∇B)
     * @param kernel          result (k)
     */
    void fastKernelEstimation(const vector<Mat>& selectionGrads, const vector<Mat>& blurredGrads, Mat kernel) {
        assert(selectionGrads[0].rows == blurredGrads[0].rows && "matrixes have to be of same size!");
        assert(selectionGrads[0].cols == blurredGrads[0].cols && "matrixes have to be of same size!");

        int rows = selectionGrads[0].rows;
        int cols = selectionGrads[0].cols;

        // based on Perseval's theorem, perform FFT
        //                __________              __________
        //             (  F(∂_x I^s) * F(∂_x B) + F(∂_y I^s) * F(∂_y B) )
        // k = F^-1 * ( ----------------------------------------------   )
        //             (         F(∂_x I^s)² + F(∂_y I^s)² + γ          )
        // where * is pointwise multiplication
        // 
        // here: F(∂_x I^s) = xS
        //       F(∂_x B)   = xB
        //       F(∂_y I^s) = yS
        //       F(∂_y B)   = yB
        
        // compute FFTs
        // the result are stored as 2 channel matrices: Re(FFT(I)), Im(FFT(I))
        Mat xS, xB, yS, yB;
        FFT(selectionGrads[0], xS);
        FFT(blurredGrads[0], xB);
        FFT(selectionGrads[1], yS);
        FFT(blurredGrads[1], yB);

        #ifndef NDEBUG
            showComplexImage("spectrum magnitude xS", xS);
            showComplexImage("spectrum magnitude yS", yS);
            showComplexImage("spectrum magnitude xB", xB);
            showComplexImage("spectrum magnitude yB", yB);
        #endif

        // go through all pixel and calculate the value in the brackets of the equation
        Mat brackets = Mat::zeros(xS.size(), xS.type());
        
        for (int x = 0; x < cols; x++) {
            for (int y = 0; y < rows; y++) {
                // conjugate complex number
                Vec2f conjXS = {xS.at<Vec2f>(y, x)[0], -1 * xS.at<Vec2f>(y, x)[1]};
                Vec2f conjYS = {yS.at<Vec2f>(y, x)[0], -1 * yS.at<Vec2f>(y, x)[1]};

                Vec2f numerator = compAdd(compMult(conjXS, xB.at<Vec2f>(x, y)),
                                          compMult(conjYS, yB.at<Vec2f>(x, y)));
                // TODO: add weight
                Vec2f denominator = compAdd(compMult(xS.at<Vec2f>(x, y), xS.at<Vec2f>(x, y)),
                                            compMult(yS.at<Vec2f>(x, y), yS.at<Vec2f>(x, y)));

                brackets.at<Vec2f>(y, x) = compDiv(numerator, denominator);
            }
        }

        // compute inverse FFT of the result in brackets
        Mat result;
        dft(brackets, result, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);

        #ifndef NDEBUG
            // print confidence matrix
            Mat resultviewable;
            convertFloatToUchar(resultviewable, result);
            imshow("result", resultviewable);
        #endif

        // TODO: set kernel
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

            // remove borders of region through erosion of the mask
            Mat element = getStructuringElement(MORPH_CROSS, Size(7, 7), Point(3, 3));
            Mat erodedMask;
            erode(mask, erodedMask, element);
            imshow("eroded mask", erodedMask);

            // save x and y gradients in a vector and erase borders because of region
            Mat erodedXGradients, erodedYGradients;
            xGradients.copyTo(erodedXGradients, erodedMask);
            yGradients.copyTo(erodedYGradients, erodedMask);
            vector<Mat> gradients = {erodedXGradients, erodedYGradients};

            #ifndef NDEBUG
                // display gradients
                Mat xGradientsViewable, yGradientsViewable;
                convertFloatToUchar(xGradientsViewable, erodedXGradients);
                convertFloatToUchar(yGradientsViewable, erodedYGradients);
                imshow("x gradient", xGradientsViewable);
                imshow("y gradient", yGradientsViewable);
            #endif

            // compute gradient confidence for al pixels
            Mat gradientConfidence;
            computeGradientConfidence(gradientConfidence, gradients, width, erodedMask);

            #ifndef NDEBUG
                // print confidence matrix
                Mat confidenceUchar;
                convertFloatToUchar(confidenceUchar, gradientConfidence);
                imshow("confidence", confidenceUchar);
            #endif

            // thresholds τ_r and τ_s will be decreased in each iteration
            // to include more and more edges
            // TDOO: parameter for this
            float thresholdR = 0.25;
            float thresholdS = 50;

            int iterations = 1;  // TODO: add parameter for this
            for (int i = 0; i < iterations; i++) {
                // select edges for kernel estimation
                vector<Mat> selectedEdges;
                selectEdges(pyramid[i], gradientConfidence, thresholdR, thresholdS, selectedEdges);

                // estimate kernel with gaussian prior
                fastKernelEstimation(selectedEdges, gradients, kernel);


                // decrease thresholds
                thresholdR = thresholdR / 1.1;
                thresholdS = thresholdS / 1.1;
            }

            // TODO: continue
        }

        psf = kernel;
    }
}