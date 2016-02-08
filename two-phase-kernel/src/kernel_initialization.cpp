#include <iostream>                     // cout, cerr, endl
#include <array>                        // array
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
     * @param gradients  array with x and y gradients (∇B)
     * @param width      width of window for Nh
     * @param mask       binary mask of region that should be computed
     */
    void computeGradientConfidence(Mat& confidence, const array<Mat,2>& gradients, const int width,
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
    void selectEdges(const Mat& image, const Mat& confidence, const float r, const float s, array<Mat,2>& selection) {
        assert(image.type() == CV_8U && "gray value image needed");

        // create mask for ruling out pixel belonging to small confidence-values
        // M = H(r - τ_r) where H is Heaviside step function
        Mat mask;
        inRange(confidence, r + 0.00001, 1, mask);  // add a very small value to r to exclude zero values
        // imshow("edge mask", mask);

        // shock filter the input image
        Mat shockImage;
        coherenceFilter(image, shockImage);
        // imshow("shock filter", shockImage);

        // gradients of shock filtered image
        const int delta = 0;
        const int ddepth = CV_32F;
        const int ksize = 3;
        const int scale = 1;

        // compute gradients x/y
        array<Mat,2> gradients;
        Sobel(shockImage, gradients[0], ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
        Sobel(shockImage, gradients[1], ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

        // normalize gradients
        normalizeOne(gradients);

        // #ifndef NDEBUG
        //     // display gradients
        //     Mat xGradientsViewable, yGradientsViewable;
        //     convertFloatToUchar(gradients[0], xGradientsViewable);
        //     convertFloatToUchar(gradients[1], yGradientsViewable);
        //     imshow("x gradient shock", xGradientsViewable);
        //     imshow("y gradient shock", yGradientsViewable);
        // #endif

        // save gradients of the final selected edges
        selection = { Mat::zeros(image.rows, image.cols, CV_32F),
                      Mat::zeros(image.rows, image.cols, CV_32F) };

        for (int x = 0; x < gradients[0].cols; x++) {
            for (int y = 0; y < gradients[0].rows; y++) {
                // if the mask is zero at the current coordinate the result
                // of the equation (see method description) is zero too.
                // So nothing has to be computed for this case
                if (mask.at<uchar>(y, x) != 0) {
                    Vec2f gradient = { gradients[0].at<float>(y,x),
                                       gradients[1].at<float>(y,x) };

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
     * @param selectionGrads  array of x and y gradients of final selected edges (∇I^s)
     * @param blurredGrads    array of x and y gradients of blurred image (∇B)
     * @param kernel          result (k)
     */
    void fastKernelEstimation(const array<Mat,2>& selectionGrads, const array<Mat,2>& blurredGrads, Mat& kernel) {
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

        // // delta function as one white pixel in black image
        // Mat deltaFloat = Mat::zeros(xS.size(), CV_32F);
        // deltaFloat.at<float>(xS.rows / 2, xS.cols / 2) = 1;
        // Mat delta;
        // FFT(deltaFloat, delta);

        // go through all pixel and calculate the value in the brackets of the equation
        for (int y = 0; y < xS.rows; y++) {
            for (int x = 0; x < xS.cols; x++) { 
                // complex entries at the current position
                complex<float> xs(xS.at<Vec2f>(y, x)[0], xS.at<Vec2f>(y, x)[1]);
                complex<float> ys(yS.at<Vec2f>(y, x)[0], yS.at<Vec2f>(y, x)[1]);

                complex<float> xb(xB.at<Vec2f>(y, x)[0], xB.at<Vec2f>(y, x)[1]);
                complex<float> yb(yB.at<Vec2f>(y, x)[0], yB.at<Vec2f>(y, x)[1]);

                // complex<float> d(delta.at<Vec2f>(y, x)[0], delta.at<Vec2f>(y, x)[1]);

                complex<float> weight(0.0, 0.0);

                // kernel entry in the Fourier space
                // complex<float> k = (conj(xs) * xb + conj(ys) * yb + xs * conj(xb) + ys * conj(yb)) /
                complex<float> k = (conj(xs) * xb + conj(ys) * yb) /
                                   (conj(xs) * xs + conj(ys) * ys + weight );
                                   // (conj(xs) * xs + conj(ys) * ys + weight * d);
                                   // (abs(xs) * abs(xs) + abs(ys) * abs(ys) + weight); // equivalent
                
                kernelFourier.at<Vec2f>(y, x) = { real(k), imag(k) };
                // kernelFourier.at<Vec2f>(y, x) = xS.at<Vec2f>(y, x);
                // kernelFourier.at<Vec2f>(y, x) = { real(xs), imag(xs) };
            }
        }

        // kernelFourier.copyTo(kernel);
        dft(kernelFourier, kernel, DFT_INVERSE);

        // // // only use the real part of the complex output
        // dft(kernelFourier, kernel, DFT_INVERSE | DFT_REAL_OUTPUT);

        // Mat kernelUchar;
        // convertFloatToUchar(kernel, kernelUchar);
        // imshow("kernel (real part)", kernelUchar);

        // // // compute inverse FFT of the kernel in frequency domain
        // // Mat kernelResult;
        // // // dft(kernelFourier, kernelResult, DFT_INVERSE);

        vector<Mat> channels(2);
        split(kernel, channels);
        // // // swapQuadrants(channels[0]);
        
        // // // showComplexImage("complex kernel my", kernelResult);

        // normalizeOne(channels);

        // // convertFloatToUchar(channels[0], kernelUchar);
        // // // swapQuadrants(kernelUchar);
        // // imshow("kernel (real part)", kernelUchar);

        double min, max;
        minMaxLoc(channels[0], &min, &max);
        cout << "kernel real: " << min << " " << max << endl;
        minMaxLoc(channels[1], &min, &max);
        cout << "kernel imag: " << min << " " << max << endl;

        merge(channels, kernel);

        // // convertFloatToUchar(channels[1], kernelUchar);
        // // // swapQuadrants(kernelUchar);
        // // imshow("kernel (imag part)", kernelUchar);

        // waitKey(0);

        // // // only copy real part of the complex output
        // // channels[0].copyTo(kernel);
    }


    /**
     * The predicted sharp edge gradient ∇I^s is used as a spatial prior to guide
     * the recovery of a coarse version of the latent image.
     * Objective function: E(I) = ||I ⊗ k - B||² + λ||∇I - ∇I^s||²
     * 
     * @param blurredImage   blurred image
     * @param kernel         kernel in image size
     * @param selectionGrads gradients of selected edges (x and y direction)
     * @param latentImage    resulting image
     */
    void coarseImageEstimation(const Mat& blurredImage, const Mat& kernel,
                               const array<Mat,2>& selectionGrads, Mat& latentImage) {
        //                ____               ______                ______
        //             (  F(k) * F(B) + λ * (F(∂_x) * F(∂_x I^s) + F(∂_y) * F(∂_y I^s)) )
        // I = F^-1 * ( ---------------------------------------------------------------  )
        //            (     ____               ______            ______                  )
        //             (    F(k) * F(k) + λ * (F(∂_x) * F(∂_x) + F(∂_y) * F(∂_y))       )
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

        assert(kernel.type() == CV_32FC2 && "Kernel must be complex IFFT");

        FFT(kernel, K);
        // kernel.copyTo(K);
        
        vector<Mat> channels(2);
        split(K, channels);

        double min, max;
        minMaxLoc(channels[0], &min, &max);
        cout << "K real: " << min << " " << max << endl;
        minMaxLoc(channels[1], &min, &max);
        cout << "K imag: " << min << " " << max << endl;


        FFT(selectionGrads[0], xS);
        FFT(selectionGrads[1], yS);

        Mat blurredFloat;
        assert(blurredImage.type() == CV_8U && "gray value image needed");
        blurredImage.convertTo(blurredFloat, CV_32F);

        // normalizes blurred input image into range [-1, 1]
        normalizeOne(blurredFloat);

        #ifndef NDEBUG
            // double min, max;
            minMaxLoc(blurredFloat, &min, &max);
            cout << "blurred float " << min << " " << max << endl;
        #endif
        
        FFT(blurredFloat, B);

        // go through fourier transformed image
        Mat imageFourier = Mat::zeros(xS.size(), xS.type());


        // gradients
        Mat sobelx = Mat::zeros(blurredImage.size(), CV_32F);
        sobelx.at<float>(0,0) = -1;
        sobelx.at<float>(0,1) = 1;
        Mat sobely = Mat::zeros(blurredImage.size(), CV_32F);
        sobely.at<float>(0,0) = -1;
        sobely.at<float>(1,0) = 1;

        Mat Dx, Dy;
        FFT(sobelx, Dx);
        FFT(sobely, Dy);

        for (int x = 0; x < xS.cols; x++) {
            for (int y = 0; y < xS.rows; y++) {
                // complex entries at the current position
                complex<float> k(K.at<Vec2f>(y, x)[0], K.at<Vec2f>(y, x)[1]);
                complex<float> xs(xS.at<Vec2f>(y, x)[0], xS.at<Vec2f>(y, x)[1]);
                complex<float> ys(yS.at<Vec2f>(y, x)[0], yS.at<Vec2f>(y, x)[1]);
                complex<float> b(B.at<Vec2f>(y, x)[0], B.at<Vec2f>(y, x)[1]);

                // // F(∂_x) is the factor: 2 * π * i * x
                // complex<float> dx(0, 2 * M_PI * x);
                // // F(∂_y) is the factor: 2 * π * i * y
                // complex<float> dy(0, 2 * M_PI * y);
                complex<float> dx(Dx.at<Vec2f>(y, x)[0], Dx.at<Vec2f>(y, x)[1]);
                complex<float> dy(Dy.at<Vec2f>(y, x)[0], Dy.at<Vec2f>(y, x)[1]);

                // weight from paper
                complex<float> weight(2.0e-3, 0.0);
                // complex<float> weight(0.0, 0.0);

                // image deblurring in the Fourier space
                complex<float> i = (conj(k) * b ) /
                // complex<float> i = (conj(k) * b + weight * (conj(dx) * xs + conj(dy) * ys)) /
                                   (conj(k) * k + weight * (conj(dx) * dx + conj(dy) * dy));
                
                imageFourier.at<Vec2f>(y, x) = { real(i), imag(i) };
            }
        }
        Mat _imageFourier;
        normalizeOne(imageFourier, _imageFourier);
        showComplexImage("fimage foureir", _imageFourier);

        Mat imageUchar;
        split(imageFourier, channels);
        convertFloatToUchar(channels[0], imageUchar);
        imshow("real image fourier", imageUchar);

        convertFloatToUchar(channels[1], imageUchar);
        imshow("imag image fourier", imageUchar);


        // compute inverse FFT of the latent image in frequency domain
        Mat imageResult;
        dft(imageFourier, imageResult, DFT_INVERSE);
        imageResult.copyTo(latentImage);

        // Mat imageUchar;
        // convertFloatToUchar(imageResult, imageUchar);
        // imshow("result", imageUchar);

        showComplexImage("complex", imageResult);

        // split complex matrix where the result of the dft is stored in channel 1
        // vector<Mat> channels(2);
        split(imageResult, channels);
        // swapQuadrants(channels[0]);
        // 
        convertFloatToUchar(channels[0], imageUchar);
        imshow("real latent", imageUchar);

        convertFloatToUchar(channels[1], imageUchar);
        imshow("imag latent", imageUchar);

        channels[0].copyTo(latentImage);


        FFT(latentImage, imageFourier);
        normalizeOne(imageFourier);
        showComplexImage("image foureir 2",imageFourier);

    }

    void initKernel(Mat& kernel, const Mat& blurredGray, const int width, const Mat& mask, 
                    const int pyrLevel, const int iterations, float thresholdR, float thresholdS) {
        
        assert(blurredGray.type() == CV_8U && "gray value image needed");
        assert(mask.type() == CV_8U && "mask should be binary image");
        
        imshow("blurred", blurredGray);

        // // DEBUG
        // Mat _blurred, _fblurred;
        // double _min; double _max;
        // minMaxLoc(blurredGray, &_min, &_max);
        // blurredGray.convertTo(_blurred, CV_32F, 1.0 / (_max - _min));

        // minMaxLoc(_blurred, &_min, &_max);
        // cout << "min = " << _min << ", max = " << _max << endl;

        // FFT(_blurred, _fblurred);
        // dft(_fblurred, _blurred, DFT_INVERSE);

        // // split complex matrix where the result of the dft is stored in channel 1
        // vector<Mat> channels(2);
        // split(_blurred, channels);
        // // swapQuadrants(channels[0]);
        // // 
        // Mat imageUchar;
        // convertFloatToUchar(channels[1], imageUchar);
        // imshow("imaginary", imageUchar);

        // minMaxLoc(channels[0], &_min, &_max);
        // cout << "inverse real: min = " << _min << ", max = " << _max << endl;
        // minMaxLoc(channels[1], &_min, &_max);
        // cout << "inverse imag: min = " << _min << ", max = " << _max << endl;

        // convertFloatToUchar(channels[0], imageUchar);
        // imshow("real", imageUchar);
        // return;

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

        assert(pyramid.size() == 1 && "Implement multiple pyramid levels");

        // go through image pyramid from small to large
        for (int l = pyramid.size() - 1; l >= 0; l--) {
            // compute image gradient for x and y direction
            // 
            // gaussian blur (in-place operation is supported)
            GaussianBlur(pyramid[l], pyramid[l], Size(3,3), 0, 0, BORDER_DEFAULT);

            const int delta = 0;
            const int ddepth = CV_32F;
            const int ksize = 3;
            const int scale = 1;

            assert(pyramid[l].type() == CV_8U && "sobel on gray value image");

            array<Mat,2> gradients;
            // gradient x / y
            Sobel(pyramid[l], gradients[0], ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
            Sobel(pyramid[l], gradients[1], ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

            // normalize gradients into range [-1,1]
            normalizeOne(gradients);

            // // FIXME(?): scale maks to current pyramid level
            // //           (do erosion once, than scale the eroded mask)
            // //        
            // // remove borders of region through erosion of the mask
            // Mat element = getStructuringElement(MORPH_CROSS, Size(7, 7), Point(3, 3));
            // Mat erodedMask;
            // erode(mask, erodedMask, element);

            // // save x and y gradients in a vector and erase borders because of region
            // Mat erodedXGradients, erodedYGradients;

            // gradients[0].copyTo(ero)

            // gradients[0].copyTo(erodedXGradients, erodedMask);
            // gradients[1].copyTo(erodedYGradients, erodedMask);

            // vector<Mat> gradients = {erodedXGradients, erodedYGradients};

            // #ifndef NDEBUG
            //     // display gradients
            //     Mat xGradientsViewable, yGradientsViewable;
            //     convertFloatToUchar(erodedXGradients, xGradientsViewable);
            //     convertFloatToUchar(erodedYGradients, yGradientsViewable);
            //     imshow("x gradient", xGradientsViewable);
            //     imshow("y gradient", yGradientsViewable);
            // #endif

            // compute gradient confidence for al pixels
            Mat gradientConfidence;
            // FIXME: Scale mask to the current pyramid level!
            computeGradientConfidence(gradientConfidence, gradients, width, mask);
            // #ifndef NDEBUG
            //     // print confidence matrix
            //     Mat confidenceUchar;
            //     convertFloatToUchar(gradientConfidence, confidenceUchar);
            //     imshow("confidence", confidenceUchar);
            // #endif

            Mat currentImage;
            pyramid[l].copyTo(currentImage);

            for (int i = 0; i < iterations; i++) {
                // select edges for kernel estimation
                array<Mat,2> selectedEdges;
                selectEdges(currentImage, gradientConfidence, thresholdR, thresholdS, selectedEdges);

                minMaxLoc(selectedEdges[0], &min, &max);
                cout << "selectedEdges x " << min << " " << max << endl;
                minMaxLoc(selectedEdges[1], &min, &max);
                cout << "selectedEdges y " << min << " " << max << endl;

                 #ifndef NDEBUG
                    Mat xGradientsViewable, yGradientsViewable;
                    // display gradients
                    convertFloatToUchar(selectedEdges[0], xGradientsViewable);
                    convertFloatToUchar(selectedEdges[1], yGradientsViewable);
                    imshow("x gradient selection " + i, xGradientsViewable);
                    imshow("y gradient selection " + i, yGradientsViewable);
                #endif
                    cout << "here" << endl;
                // estimate kernel with gaussian prior
                fastKernelEstimation(selectedEdges, gradients, tmpKernel);


                // minMaxLoc(tmpKernel, &min, &max);
                // cout << "kernel " << min << " " << max << endl;

                // Mat kernelmask;
                // inRange(tmpKernel, min + max/10, max, kernelmask);
                // Mat newKernel; 
                // tmpKernel.copyTo(newKernel, kernelmask);
                // // blur(newKernel, newKernel, Size(3,3));

                // // DEBUG: Show real-part-only kernel
                // #ifndef NDEBUG
                //     // print kernel
                //     Mat kernelUchar;
                //     // convertFloatToUchar(newKernel, kernelUchar);
                //     convertFloatToUchar(tmpKernel, kernelUchar);
                //     swapQuadrants(kernelUchar);
                //     imshow("tmp kernel " + i, kernelUchar);
                //     imwrite("kernel.jpg", kernelUchar);
                // #endif

                // DEBUG: Show complex kernel
                #ifndef NDEBUG
                    showComplexImage("complex kernel", tmpKernel);

                    // // print kernel
                    // Mat kernelUchar;

                    // // split complex matrix where the result of the dft is stored in channel 1
                    // vector<Mat> channels(2);
                    // split(tmpKernel, channels);

                    // // convertFloatToUchar(newKernel, kernelUchar);
                    // convertFloatToUchar(channels[0], kernelUchar);
                    // swapQuadrants(kernelUchar);
                    // imshow("kernel (real part) " + i, kernelUchar);

                    // convertFloatToUchar(channels[1], kernelUchar);
                    // swapQuadrants(kernelUchar);
                    // imshow("kernel (imag part) " + i, kernelUchar);
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
                // convertFloatToUchar(newKernel, newUchar);
                // imshow("new kernel", newUchar);
                // filter2D(pyramid[l], latentImage, CV_32F, newKernel);
                // Mat latentUchar;
                // convertFloatToUchar(latentImage, latentUchar);
                // imshow("latent", latentUchar);

                // set current image to coarse latent image
                Mat imageUchar;
                convertFloatToUchar(latentImage, imageUchar);

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
        //     convertFloatToUchar(tmpKernel, kernelUchar);
        //     imshow("kernel result", kernelUchar);
        // #endif

        // // cut of kernel in middle of the temporary kernel
        // int x = tmpKernel.cols / 2 - kernel.cols / 2;
        // int y = tmpKernel.rows / 2 - kernel.rows / 2;
        // swapQuadrants(tmpKernel);
        // Mat kernelROI = tmpKernel(Rect(x, y, kernel.cols, kernel.rows));

        // // convert kernel to uchar
        // Mat resultUchar;
        // convertFloatToUchar(kernelROI, resultUchar);
        // // convertFloatToUchar(kernel, resultUchar);

        // resultUchar.copyTo(kernel);
    }
}