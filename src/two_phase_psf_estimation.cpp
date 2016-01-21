#include <iostream>                     // cout, cerr, endl
#include <cmath>                        // sqrt
#include <complex>                      // complex numbers
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
    *  Author:WinCoder@qq.com with editing by me
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
    void coherenceFilter(const Mat& img, const int sigma, const int str_sigma,
                         const float blend, const int iter, Mat& shockImage) {

        img.copyTo(shockImage);

        int height = shockImage.rows;
        int width  = shockImage.cols;

        for(int i = 0;i <iter; i++) {
            Mat gray;
            
            if (img.type() != CV_8U) {
                cvtColor(shockImage, gray, COLOR_BGR2GRAY);
            } else {
                img.copyTo(gray);
            }

            Mat eigen;
            cornerEigenValsAndVecs(gray, eigen, str_sigma, 3);

            vector<Mat> vec;
            split(eigen,vec);

            Mat x,y;
            x = vec[2];
            y = vec[3];

            Mat gxx,gxy,gyy;
            Sobel(gray, gxx, CV_32F, 2, 0, sigma);
            Sobel(gray, gxy, CV_32F, 1, 1, sigma);
            Sobel(gray, gyy, CV_32F, 0, 2, sigma);

            Mat ero;
            Mat dil;
            erode(shockImage, ero, Mat());
            dilate(shockImage, dil, Mat());

            Mat img1 = ero;
            for(int nY = 0;nY<height;nY++) {
                for(int nX = 0;nX<width;nX++) {
                    if(x.at<float>(nY,nX)* x.at<float>(nY,nX)* gxx.at<float>(nY,nX)
                        + 2*x.at<float>(nY,nX)* y.at<float>(nY,nX)* gxy.at<float>(nY,nX)
                        + y.at<float>(nY,nX)* y.at<float>(nY,nX)* gyy.at<float>(nY,nX)<0) {

                            if (img.type() != CV_8U)
                                img1.at<Vec3b>(nY,nX) = dil.at<Vec3b>(nY,nX);
                            else
                                img1.at<uchar>(nY,nX) = dil.at<uchar>(nY,nX);
                    }
                }
            }

            shockImage = shockImage * (1.0 - blend) + img1 * blend;
        }
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
        assert(image.type() == CV_8U && "gray value image needed");

        // create mask for ruling out pixel belonging to small confidence-values
        // M = H(r - τ_r) where H is Heaviside step function
        Mat mask;
        inRange(confidence, r + 0.00001, 1, mask);  // add a very small value to r to exclude zero values
        imshow("edge mask", mask);

        // shock filter the input image
        Mat shockImage;
        coherenceFilter(image, 11, 11, 0.5, 4, shockImage);
        imshow("shock filter", shockImage);

        // gradients of shock filtered image
        int delta = 0;
        int ddepth = CV_32F;
        int ksize = 3;
        int scale = 1;

        // compute gradients
        Mat xGradients, yGradients;
        Sobel(shockImage, xGradients, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
        Sobel(shockImage, yGradients, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

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
     * @param image   input image with 1 channel
     * @param complex result as 2 channel matrix with complex numbers
     */
    void FFT(const Mat& image, Mat& complex) {
        assert(image.type() == CV_32F && "fft works on 32FC1-images");

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
        dft(complex, complex, DFT_COMPLEX_OUTPUT);

        assert(padded.size() == complex.size() && "Resulting complex matrix must be of same size");
    }


    /**
     * Rearrange quadrants of an image so that the origin is at the image center.
     * This is useful for fourier images. 
     */
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

        swapQuadrants(magI);

        normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
                                                // viewable image form (float between values 0 and 1).

        imshow(windowName, magI);
    }


    /**
     * With the critical edge selection, initial kernel erstimation can be accomplished quickly.
     * Objective function: E(k) = ||∇I^s ⊗ k - ∇B||² + γ||k||²
     * 
     * @param selectionGrads  vector of x and y gradients of final selected edges (∇I^s)
     * @param blurredGrads    vector of x and y gradients of blurred image (∇B)
     * @param kernel          result (k)
     */
    void fastKernelEstimation(const vector<Mat>& selectionGrads, const vector<Mat>& blurredGrads, Mat& kernel) {
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

        // #ifndef NDEBUG
        //     showComplexImage("spectrum magnitude xS", xS);
        //     showComplexImage("spectrum magnitude yS", yS);
        //     showComplexImage("spectrum magnitude xB", xB);
        //     showComplexImage("spectrum magnitude yB", yB);
        // #endif

        // go through all pixel and calculate the value in the brackets of the equation
        Mat kernelFourier = Mat::zeros(xS.size(), xS.type());

        assert(xS.type() == CV_32FC2);
        assert(yS.type() == CV_32FC2);
        assert(xB.type() == CV_32FC2);
        assert(yB.type() == CV_32FC2);

        for (int x = 0; x < xS.cols; x++) {
            for (int y = 0; y < xS.rows; y++) {
                // complex entries at the current position
                complex<float> xs(xS.at<Vec2f>(y, x)[0], xS.at<Vec2f>(y, x)[1]);
                complex<float> ys(yS.at<Vec2f>(y, x)[0], yS.at<Vec2f>(y, x)[1]);

                complex<float> xb(xB.at<Vec2f>(y, x)[0], xB.at<Vec2f>(y, x)[1]);
                complex<float> yb(yB.at<Vec2f>(y, x)[0], yB.at<Vec2f>(y, x)[1]);

                complex<float> weight(1.0e+20, 0.0);

                // kernel entry in the Fourier space
                complex<float> k = (conj(xs) * xb + conj(ys) * yb) /
                                   (xs * xs + ys * ys + weight);
                
                kernelFourier.at<Vec2f>(y, x) = { real(k), imag(k) };
            }
        }

        // compute inverse FFT of the kernel in frequency domain
        Mat kernelResult;
        dft(kernelFourier, kernelResult, DFT_INVERSE);

        // swap quadrants
        vector<Mat> channels(2);
        split(kernelResult, channels);
        swapQuadrants(channels[0]);

        channels[0].copyTo(kernel);
    }


    /**
     * The predicted sharp edge gradient ∇I^s is used as a spatial prior to guide
     * the recovery of a coarse version of the latent image.
     * Objective function: E(I) = ||I ⊗ k - B||² + λ||∇I - ∇I^s||²
     * 
     * @param blurredImage   blurred image
     * @param kernel         kernel in image size
     * @param selectionGrads gradients of selected edges
     * @param latentImage    resulting image
     */
    void coarseImageEstimation(const Mat& blurredImage, const Mat& kernel,
                               const vector<Mat>& selectionGrads, Mat& latentImage) {
        //                ____              ______                ______
        //             (  F(k) * F(B) + λ * F(∂_x) * F(∂_x I^s) + F(∂_y) * F(∂_y I^s) )
        // I = F^-1 * ( -------------------------------------------------------------  )
        //            (     ____              ______            ______                 )
        //             (    F(k) * F(k) + λ * F(∂_x) * F(∂_x) + F(∂_y) * F(∂_y)       )
        // where * is pointwise multiplication
        // 
        // here: F(k)       = k
        //       F(∂_x I^s) = xS
        //       F(∂_y I^s) = yS
        //       F(∂_x)     = dx
        //       F(∂_y)     = dy
        //       F(B)       = B

        // compute FFTs
        // the result are stored as 2 channel matrices: Re(FFT(I)), Im(FFT(I))
        Mat K, xS, yS, B;
        FFT(kernel, K);
        FFT(selectionGrads[0], xS);
        FFT(selectionGrads[1], yS);

        Mat blurredFloat;
        assert(blurredImage.type() == CV_8U && "gray value image needed");
        blurredImage.convertTo(blurredFloat, CV_32F);
        FFT(blurredFloat, B);

        // go through fourier transformed image
        Mat imageFourier = Mat::zeros(xS.size(), xS.type());

        for (int x = 0; x < xS.cols; x++) {
            for (int y = 0; y < xS.rows; y++) {
                // complex entries at the current position
                complex<float> k(K.at<Vec2f>(y, x)[0], K.at<Vec2f>(y, x)[1]);
                complex<float> xs(xS.at<Vec2f>(y, x)[0], xS.at<Vec2f>(y, x)[1]);
                complex<float> ys(yS.at<Vec2f>(y, x)[0], yS.at<Vec2f>(y, x)[1]);
                complex<float> b(B.at<Vec2f>(y, x)[0], B.at<Vec2f>(y, x)[1]);

                // F(∂_x) is the factor: 2 * π * i * x
                complex<float> dx(0, 2 * M_PI * x);
                // F(∂_y) is the factor: 2 * π * i * y
                complex<float> dy(0, 2 * M_PI * y);

                // weight from paper
                complex<float> weight(2.0e-4, 0.0);

                // image deblurring in the Fourier space
                complex<float> i = (conj(k) * b + weight * conj(dx) * xs + conj(dy) * ys) /
                                   (conj(k) * k + weight * conj(dx) * dx + conj(dy) * dy);
                
                imageFourier.at<Vec2f>(y, x) = { real(i), imag(i) };
            }
        }

        // compute inverse FFT of the latent image in frequency domain
        Mat imageResult;
        dft(imageFourier, imageResult, DFT_INVERSE);

        // split complex matrix where the result of the dft is stored in channel 1
        vector<Mat> channels(2);
        split(imageResult, channels);
        // swapQuadrants(channels[0]);

        channels[0].copyTo(latentImage);
    }


    void estimateKernel(Mat& psf, const Mat& blurredImage, const int psfWidth, const Mat& mask) {
        // set expected kernel witdh to odd number
        int width = (psfWidth % 2 == 0) ? psfWidth + 1 : psfWidth;

        // phase one: initialize kernel
        // 
        // all-zer kernel
        Mat kernel = Mat::zeros(psfWidth, psfWidth, CV_8U);

        // in the iterations this kernel is used
        Mat tmpKernel;

        // convert blurred image to gray
        Mat blurredGray;
        cvtColor(blurredImage, blurredGray, CV_BGR2GRAY);

        // build an image pyramid
        int level = 1;  // TODO: add parameter for this
        vector<Mat> pyramid;
        pyramid.push_back(blurredGray);

        for (int i = 0; i < (level - 1); i++) {
            Mat downImage;
            pyrDown(pyramid[i], downImage, Size(pyramid[i].cols/2, pyramid[i].rows/2));

            pyramid.push_back(pyramid[i]);
        }

        // go through image image pyramid
        for (int i = 0; i < pyramid.size(); i++) {
            imshow("pyr " + i, pyramid[i]);
            // compute image gradient for x and y direction
            // 
            // gaussian blur
            GaussianBlur(pyramid[i], pyramid[i], Size(3,3), 0, 0, BORDER_DEFAULT);

            Mat xGradients, yGradients;
            int delta = 0;
            int ddepth = CV_32F;
            int ksize = 3;
            int scale = 1;

            // gradient x
            assert(pyramid[i].type() == CV_8U && "sobel on gray value image");
            Sobel(pyramid[i], xGradients, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);

            // gradient y
            Sobel(pyramid[i], yGradients, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);


            // remove borders of region through erosion of the mask
            assert(mask.type() == CV_8U && "mask should be binary image");
            Mat element = getStructuringElement(MORPH_CROSS, Size(7, 7), Point(3, 3));
            Mat erodedMask;
            erode(mask, erodedMask, element);

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
                fastKernelEstimation(selectedEdges, gradients, tmpKernel);

                #ifndef NDEBUG
                    // print kernel
                    Mat kernelUchar;
                    convertFloatToUchar(kernelUchar, tmpKernel);
                    imshow("tmp kernel", kernelUchar);
                #endif
                
                // coarse image estimation with a spatial prior
                Mat latentImage;
                coarseImageEstimation(pyramid[i], tmpKernel, selectedEdges, latentImage);

                #ifndef NDEBUG
                    // print kernel
                    Mat imageUchar;
                    convertFloatToUchar(imageUchar, latentImage);
                    imshow("tmp latent image", imageUchar);
                #endif

                // decrease thresholds
                thresholdR = thresholdR / 1.1;
                thresholdS = thresholdS / 1.1;
            }

            // TODO: continue
        }

        // cut of kernel in middle of the temporary kernel
        int x = tmpKernel.cols / 2 - kernel.cols / 2;
        int y = tmpKernel.rows / 2 - kernel.rows / 2;
        Mat kernelROI = tmpKernel(Rect(x, y, kernel.cols, kernel.rows));

        // convert kernel to uchar
        Mat resultUchar;
        convertFloatToUchar(resultUchar, kernelROI);
        resultUchar.copyTo(psf);

        #ifndef NDEBUG
            imshow("kernel", psf);
        #endif
    }


    void estimateKernel(Mat& psf, const Mat& image, const int psfWidth) {
        Mat mask = Mat(image.rows, image.cols, CV_8U, 1);
        estimateKernel(psf, image, psfWidth, mask);
    }
}