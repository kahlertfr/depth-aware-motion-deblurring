#include <iostream>                     // cout, cerr, endl
#include <complex>                      // complex numbers
#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite

#include "utils.hpp"
#include "coherence_filter.hpp"

#include "kernel_initialization.hpp"

using namespace std;
using namespace cv;
using namespace deblur;


namespace TwoPhaseKernelEstimation {
    
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
     * The final selected edges for kernel estimation are determined as:
     * ∇I^s = ∇I · H (M * ||∇I||_2 − τ_s )
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
        // imshow("edge mask", mask);

        // shock filter the input image
        Mat shockImage;
        coherenceFilter(image, showImage);
        // imshow("shock filter", shockImage);

        // gradients of shock filtered image
        const int delta = 0;
        const int ddepth = CV_32F;
        const int ksize = 3;
        const int scale = 1;

        // compute gradients
        Mat xGradients, yGradients;
        Sobel(shockImage, xGradients, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
        Sobel(shockImage, yGradients, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

        // normalize gradients
        Mat normedXGradients, normedYGradients;
        normalize(xGradients, normedXGradients, -255, 255);
        normalize(yGradients, normedYGradients, -255, 255);
        

        // merge the gradients of x- and y-direction to one matrix
        Mat gradients;
        vector<Mat> grads = {normedXGradients, normedYGradients};
        // vector<Mat> grads = {xGradients, yGradients};
        merge(grads, gradients);

        // #ifndef NDEBUG
        //     // display gradients
        //     Mat xGradientsViewable, yGradientsViewable;
        //     convertFloatToUchar(xGradientsViewable, xGradients);
        //     convertFloatToUchar(yGradientsViewable, yGradients);
        //     imshow("x gradient shock", xGradientsViewable);
        //     imshow("y gradient shock", yGradientsViewable);
        // #endif

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
        //                   __________
        // and F(∂_x I^s)² = F(∂_x I^s) * F(∂_x I^s) ! because they mean the norm
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

        Mat kernelFourier = Mat::zeros(xS.size(), xS.type());

        assert(xS.type() == CV_32FC2);
        assert(yS.type() == CV_32FC2);
        assert(xB.type() == CV_32FC2);
        assert(yB.type() == CV_32FC2);

        // delta function as one white pixel in black image
        Mat deltaFloat = Mat::zeros(xS.size(), CV_32F);
        deltaFloat.at<float>(xS.rows / 2, xS.cols / 2) = 1;
        Mat delta;
        FFT(deltaFloat, delta);

        // go through all pixel and calculate the value in the brackets of the equation
        for (int x = 0; x < xS.cols; x++) {
            for (int y = 0; y < xS.rows; y++) {
                // complex entries at the current position
                complex<float> xs(xS.at<Vec2f>(y, x)[0], xS.at<Vec2f>(y, x)[1]);
                complex<float> ys(yS.at<Vec2f>(y, x)[0], yS.at<Vec2f>(y, x)[1]);

                complex<float> xb(xB.at<Vec2f>(y, x)[0], xB.at<Vec2f>(y, x)[1]);
                complex<float> yb(yB.at<Vec2f>(y, x)[0], yB.at<Vec2f>(y, x)[1]);

                complex<float> d(delta.at<Vec2f>(y, x)[0], delta.at<Vec2f>(y, x)[1]);

                complex<float> weight(0.50, 0.0);

                // kernel entry in the Fourier space
                // complex<float> k = (conj(xs) * xb + conj(ys) * yb + xs * conj(xb) + ys * conj(yb)) /
                complex<float> k = (conj(xs) * xb + conj(ys) * yb) /
                                   // (conj(xs) * xs + conj(ys) * ys + weight );
                                   (conj(xs) * xs + conj(ys) * ys + weight * d);
                                   // (abs(xs) * abs(xs) + abs(ys) * abs(ys) + weight); // equivalent
                
                kernelFourier.at<Vec2f>(y, x) = { real(k), imag(k) };
            }
        }

        // compute inverse FFT of the kernel in frequency domain
        Mat kernelResult;
        dft(kernelFourier, kernelResult, DFT_INVERSE);

        // swap quadrants
        vector<Mat> channels(2);
        split(kernelResult, channels);
        // swapQuadrants(channels[0]);

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

        double min, max;
        minMaxLoc(blurredFloat, &min, &max);
        cout << "blurred float " << min << " " << max << endl;
        
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
                complex<float> weight(2.0e-3, 0.0);

                // image deblurring in the Fourier space
                complex<float> i = (conj(k) * b + weight * (conj(dx) * xs + conj(dy) * ys)) /
                                   (conj(k) * k + weight * (conj(dx) * dx + conj(dy) * dy));
                
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

    void initKernel(Mat& kernel, const Mat& blurredGray, const int width, const Mat& mask, 
                    const int pyrLevel, const int iterations, float thresholdR, float thresholdS) {
        
        assert(blurredGray.type() == CV_8U && "gray value image needed");
        assert(mask.type() == CV_8U && "mask should be binary image");
        
        imshow("blurred", blurredGray);

        kernel = Mat::zeros(width, width, CV_8U);

        // build an image pyramid
        vector<Mat> pyramid;
        pyramid.push_back(blurredGray);

        double min; double max;
        minMaxLoc(blurredGray, &min, &max);
        cout << "blurred image " << min << " " << max << endl;

        for (int i = 0; i < (pyrLevel - 1); i++) {
            Mat downImage;
            pyrDown(pyramid[i], downImage, Size(pyramid[i].cols/2, pyramid[i].rows/2));

            pyramid.push_back(pyramid[i]);
        }

        // in the iterations this kernel is used
        Mat tmpKernel;

        // go through image image pyramid
        for (int l = 0; l < pyramid.size(); l++) {
            // compute image gradient for x and y direction
            // 
            // gaussian blur
            GaussianBlur(pyramid[l], pyramid[l], Size(3,3), 0, 0, BORDER_DEFAULT);

            Mat xGradients, yGradients;
            const int delta = 0;
            const int ddepth = CV_32F;
            const int ksize = 3;
            const int scale = 1;

            // gradient x
            assert(pyramid[l].type() == CV_8U && "sobel on gray value image");
            Sobel(pyramid[l], xGradients, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);

            // gradient y
            Sobel(pyramid[l], yGradients, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

            // normalize gradients
            Mat normedXGradients, normedYGradients;
            normalize(xGradients, normedXGradients, -255, 255);
            normalize(yGradients, normedYGradients, -255, 255);

            minMaxLoc(normedXGradients, &min, &max);
            cout << "gradients x " << min << " " << max << endl;
            minMaxLoc(normedYGradients, &min, &max);
            cout << "gradients y " << min << " " << max << endl;

            // remove borders of region through erosion of the mask
            Mat element = getStructuringElement(MORPH_CROSS, Size(7, 7), Point(3, 3));
            Mat erodedMask;
            erode(mask, erodedMask, element);

            // save x and y gradients in a vector and erase borders because of region
            Mat erodedXGradients, erodedYGradients;
            normedXGradients.copyTo(erodedXGradients, erodedMask);
            normedYGradients.copyTo(erodedYGradients, erodedMask);

            vector<Mat> gradients = {erodedXGradients, erodedYGradients};

            // #ifndef NDEBUG
            //     // display gradients
            //     Mat xGradientsViewable, yGradientsViewable;
            //     convertFloatToUchar(xGradientsViewable, erodedXGradients);
            //     convertFloatToUchar(yGradientsViewable, erodedYGradients);
            //     imshow("x gradient", xGradientsViewable);
            //     imshow("y gradient", yGradientsViewable);
            // #endif

            // compute gradient confidence for al pixels
            Mat gradientConfidence;
            computeGradientConfidence(gradientConfidence, gradients, width, erodedMask);
            // #ifndef NDEBUG
            //     // print confidence matrix
            //     Mat confidenceUchar;
            //     convertFloatToUchar(confidenceUchar, gradientConfidence);
            //     imshow("confidence", confidenceUchar);
            // #endif

            Mat currentImage;
            pyramid[l].copyTo(currentImage);

            for (int i = 0; i < iterations; i++) {
                // select edges for kernel estimation
                vector<Mat> selectedEdges;
                selectEdges(currentImage, gradientConfidence, thresholdR, thresholdS, selectedEdges);

                minMaxLoc(selectedEdges[0], &min, &max);
                cout << "selectedEdges x " << min << " " << max << endl;
                minMaxLoc(selectedEdges[1], &min, &max);
                cout << "selectedEdges y " << min << " " << max << endl;

                 #ifndef NDEBUG
                    Mat xGradientsViewable, yGradientsViewable;
                    // display gradients
                    convertFloatToUchar(xGradientsViewable, selectedEdges[0]);
                    convertFloatToUchar(yGradientsViewable, selectedEdges[1]);
                    imshow("x gradient selection " + i, xGradientsViewable);
                    imshow("y gradient selection " + i, yGradientsViewable);
                #endif

                // estimate kernel with gaussian prior
                fastKernelEstimation(selectedEdges, gradients, tmpKernel);


                minMaxLoc(tmpKernel, &min, &max);
                cout << "kernel " << min << " " << max << endl;

                // Mat kernelmask;
                // inRange(tmpKernel, min + max/10, max, kernelmask);
                // Mat newKernel; 
                // tmpKernel.copyTo(newKernel, kernelmask);
                // // blur(newKernel, newKernel, Size(3,3));

                #ifndef NDEBUG
                    // print kernel
                    Mat kernelUchar;
                    // convertFloatToUchar(kernelUchar, newKernel);
                    convertFloatToUchar(kernelUchar, tmpKernel);
                    swapQuadrants(kernelUchar);
                    imshow("tmp kernel " + i, kernelUchar);
                    imwrite("kernel.jpg", kernelUchar);
                #endif
                
                // // coarse image estimation with a spatial prior
                Mat latentImage;
                // coarseImageEstimation(pyramid[l], newKernel, selectedEdges, latentImage);
                coarseImageEstimation(pyramid[l], tmpKernel, selectedEdges, latentImage);

                minMaxLoc(latentImage, &min, &max);
                cout << "latentImage " << min << " " << max << endl;
                
                // // cut of kernel in middle of the temporary kernel
                // int x = tmpKernel.cols / 2 - kernel.cols / 2;
                // int y = tmpKernel.rows / 2 - kernel.rows / 2;
                // swapQuadrants(tmpKernel);
                // kernel = tmpKernel(Rect(x, y, kernel.cols, kernel.rows));
                // double min; double max;
                // minMaxLoc(kernel, &min, &max);
                // cout << min << " " << max << endl;
                // Mat kernelmask;
                // inRange(kernel, min + max/12, max, kernelmask);
                // Mat newKernel; 
                // kernel.copyTo(newKernel, kernelmask);
                // Mat newUchar;
                // minMaxLoc(newKernel, &min, &max);
                // cout << min << " " << max << endl;
                // convertFloatToUchar(newUchar, newKernel);
                // imshow("new kernel", newUchar);
                // filter2D(pyramid[l], latentImage, CV_32F, newKernel);
                // Mat latentUchar;
                // convertFloatToUchar(latentUchar, latentImage);
                // imshow("latent", latentUchar);

                // set current image to coarse latent image
                Mat imageUchar;
                convertFloatToUchar(imageUchar, latentImage);

                // the latent image is some pixel larger than the original one therefore
                // cut it out in the right size
                currentImage = imageUchar(Rect((imageUchar.cols - pyramid[l].cols) / 2,
                                               (imageUchar.rows - pyramid[l].rows) / 2,
                                               pyramid[l].cols, pyramid[l].rows));
                
                #ifndef NDEBUG
                    // print latent image
                    imshow("tmp latent image " + i, imageUchar);
                    imwrite("latent.jpg", imageUchar);
                #endif

                // decrease thresholds τ_r and τ_s will to include more and more edges
                thresholdR = thresholdR / 1.1;
                thresholdS = thresholdS / 1.1;
            }
        }

        // #ifndef NDEBUG
        //     // print kernel
        //     Mat kernelUchar;
        //     convertFloatToUchar(kernelUchar, tmpKernel);
        //     imshow("kernel result", kernelUchar);
        // #endif

        // // cut of kernel in middle of the temporary kernel
        // int x = tmpKernel.cols / 2 - kernel.cols / 2;
        // int y = tmpKernel.rows / 2 - kernel.rows / 2;
        // swapQuadrants(tmpKernel);
        // Mat kernelROI = tmpKernel(Rect(x, y, kernel.cols, kernel.rows));

        // // convert kernel to uchar
        // Mat resultUchar;
        // convertFloatToUchar(resultUchar, kernelROI);
        // // convertFloatToUchar(resultUchar, kernel);

        // resultUchar.copyTo(kernel);
    }
}