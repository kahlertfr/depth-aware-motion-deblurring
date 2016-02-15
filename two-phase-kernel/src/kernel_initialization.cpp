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

        // shock filter the input image
        Mat shockImage;
        coherenceFilter(image, shockImage);

        // #ifndef NDEBUG
        //      imshow("shock filter", shockImage);
        // #endif

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
        //     showFloat("x gradient shock", gradients[0]);
        //     showFloat("y gradient shock", gradients[1]);
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
     * @param selectionGrads  array of x and y gradients of final selected edges (∇I^s) [-1, 1]
     * @param blurredGrads    array of x and y gradients of blurred image (∇B) [-1, 1]
     * @param kernel          energy preserving kernel (k)
     */
    void fastKernelEstimation(const array<Mat,2>& selectionGrads,
                              const array<Mat,2>& blurredGrads, Mat& kernel,
                              const float weight = 1e-2) {

        assert(selectionGrads[0].rows == blurredGrads[0].rows && "matrixes have to be of same size!");
        assert(selectionGrads[0].cols == blurredGrads[0].cols && "matrixes have to be of same size!");

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
        deblur::dft(selectionGrads[0], xS);
        deblur::dft(blurredGrads[0], xB);
        deblur::dft(selectionGrads[1], yS);
        deblur::dft(blurredGrads[1], yB);

        complex<float> we(weight, 0.0);

        // kernel in Fourier domain
        Mat K = Mat::zeros(xS.size(), xS.type());

        // pixelwise computation of kernel
        for (int y = 0; y < K.rows; y++) {
            for (int x = 0; x < K.cols; x++) { 
                // complex entries at the current position
                complex<float> xs(xS.at<Vec2f>(y, x)[0], xS.at<Vec2f>(y, x)[1]);
                complex<float> ys(yS.at<Vec2f>(y, x)[0], yS.at<Vec2f>(y, x)[1]);

                complex<float> xb(xB.at<Vec2f>(y, x)[0], xB.at<Vec2f>(y, x)[1]);
                complex<float> yb(yB.at<Vec2f>(y, x)[0], yB.at<Vec2f>(y, x)[1]);


                // kernel entry in the Fourier space
                complex<float> k = (conj(xs) * xb + conj(ys) * yb) /
                                   (conj(xs) * xs + conj(ys) * ys + we);
                                   // (abs(xs) * abs(xs) + abs(ys) * abs(ys) + we); // equivalent
                
                K.at<Vec2f>(y, x) = { real(k), imag(k) };
            }
        }

        // only use the real part of the complex output
        Mat kernelBig;
        dft(K, kernelBig, DFT_INVERSE | DFT_REAL_OUTPUT);

        // FIXME: find kernel inside image (kind of bounding box) instead of force user to
        // approximate a correct kernel-width (otherwise some information are lost)

        // cut of kernel in middle of the temporary kernel
        int x = kernelBig.cols / 2 - kernel.cols / 2;
        int y = kernelBig.rows / 2 - kernel.rows / 2;
        swapQuadrants(kernelBig);
        Mat kernelROI = kernelBig(Rect(x, y, kernel.cols, kernel.rows));

        // copy the ROI to the kernel to avoid that some OpenCV functions accidently
        // uses the information outside of the ROI (like copyMakeBorder())
        kernelROI.copyTo(kernel);

        // threshold kernel to erease negative values
        threshold(kernel, kernel, 0.0, -1, THRESH_TOZERO);

        // // kernel has to be energy preserving
        // // this means: sum(kernel) = 1
        kernel /= sum(kernel)[0];
    }


    /**
     * The predicted sharp edge gradient ∇I^s is used as a spatial prior to guide
     * the recovery of a coarse version of the latent image.
     * Objective function: E(I) = ||I ⊗ k - B||² + λ||∇I - ∇I^s||²
     * 
     * @param blurred        blurred grayvalue image (B)
     * @param kernel         energy presserving kernel (k)
     * @param selectionGrads gradients of selected edges (x and y direction) (∇I^s)
     * @param latent         resulting image (I)
     * @param weight         λ, default is 2.0e-3 (weight from paper)
     */
    void coarseImageEstimation(Mat blurred, const Mat& kernel,
                               const array<Mat,2>& selectionGrads, Mat& latent,
                               const float weight = 2.0e-3) {

        assert(kernel.type() == CV_32F && "works with energy preserving float kernel");
        assert(blurred.type() == CV_8U && "works with gray valued blurred image");

        // convert grayvalue image to float and normalize it to [0,1]
        blurred.convertTo(blurred, CV_32F);
        blurred /= 255.0;

        // fill kernel with zeros to get to the blurred image size
        // it's important to use BORDER_ISOLATED flag if the kernel is an ROI of a greater image!
        Mat pkernel;
        copyMakeBorder(kernel, pkernel, 0,
                       blurred.rows - kernel.rows, 0,
                       blurred.cols - kernel.cols,
                       BORDER_CONSTANT, Scalar::all(0));

        // using sobel filter as gradients dx and dy
        Mat sobelx = Mat::zeros(blurred.size(), CV_32F);
        sobelx.at<float>(0,0) = -1;
        sobelx.at<float>(0,1) = 1;
        Mat sobely = Mat::zeros(blurred.size(), CV_32F);
        sobely.at<float>(0,0) = -1;
        sobely.at<float>(1,0) = 1;


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

        // compute DFT (withoud padding)
        // the result are stored as 2 channel matrices: Re(FFT(I)), Im(FFT(I))
        Mat K, xS, yS, B, Dx, Dy;
        deblur::dft(pkernel, K);
        deblur::dft(selectionGrads[0], xS);
        deblur::dft(selectionGrads[1], yS);
        deblur::dft(blurred, B);
        deblur::dft(sobelx, Dx);
        deblur::dft(sobely, Dy);

        // weight from paper
        complex<float> we(weight, 0.0);

        // latent image in fourier domain
        Mat I = Mat::zeros(xS.size(), xS.type());

        // pointwise computation of I
        for (int x = 0; x < xS.cols; x++) {
            for (int y = 0; y < xS.rows; y++) {
                // complex entries at the current position
                complex<float> b(B.at<Vec2f>(y, x)[0], B.at<Vec2f>(y, x)[1]);
                complex<float> k(K.at<Vec2f>(y, x)[0], K.at<Vec2f>(y, x)[1]);

                complex<float> xs(xS.at<Vec2f>(y, x)[0], xS.at<Vec2f>(y, x)[1]);
                complex<float> ys(yS.at<Vec2f>(y, x)[0], yS.at<Vec2f>(y, x)[1]);

                complex<float> dx(Dx.at<Vec2f>(y, x)[0], Dx.at<Vec2f>(y, x)[1]);
                complex<float> dy(Dy.at<Vec2f>(y, x)[0], Dy.at<Vec2f>(y, x)[1]);

                // compute current point of latent image in fourier domain
                complex<float> i = (conj(k) * b + we * (conj(dx) * xs + conj(dy) * ys)) /
                                   (conj(k) * k + we * (conj(dx) * dx + conj(dy) * dy));
                
                I.at<Vec2f>(y, x) = { real(i), imag(i) };
            }
        }

        // compute inverse DFT of the latent image
        dft(I, latent, DFT_INVERSE | DFT_REAL_OUTPUT);


        // threshold the result because it has large negative and positive values
        // which would result in a very grayish image
        threshold(latent, latent, 0.0, -1, THRESH_TOZERO);


        // swap slices of the result
        // because the image is shifted to the upper-left corner (why??)
        int x = latent.cols;
        int y = latent.rows;
        int hs1 = (kernel.cols - 1) / 2;
        int hs2 = (kernel.rows - 1) / 2;

        // create rects per image slice
        //  __________
        // |      |   |
        // |   0  | 1 |
        // |      |   |
        // |------|---|
        // |   2  | 3 |
        // |______|___|
        // 
        // rect gets the coordinates of the top-left corner, width and height
        Mat q0(latent, Rect(0, 0, x - hs1, y - hs2));      // Top-Left
        Mat q1(latent, Rect(x - hs1, 0, hs1, y - hs2));    // Top-Right
        Mat q2(latent, Rect(0, y - hs2, x - hs1, hs2));    // Bottom-Left
        Mat q3(latent, Rect(x - hs1, y - hs2, hs1, hs2));  // Bottom-Right

        Mat latentSwap;
        cv::hconcat(q3, q2, latentSwap);
        Mat tmp;
        cv::hconcat(q1, q0, tmp);
        cv::vconcat(latentSwap, tmp, latentSwap);


        // convert result to uchar image
        convertFloatToUchar(latentSwap, latent);

        assert(blurred.size() == latent.size()
               && "Something went wrong - latent and blurred size has to be equal");
    }


    void initKernel(Mat& kernel, const Mat& blurredGray, const int width, const Mat& mask, 
                    const int pyrLevel, const int iterations, float thresholdR, float thresholdS) {
        
        assert(blurredGray.type() == CV_8U && "gray value image needed");
        assert(mask.type() == CV_8U && "mask should be binary image");
        
        // #ifndef NDEBUG
        //     imshow("blurred", blurredGray);
        // #endif

        // build an image pyramid with gray value images
        vector<Mat> pyramid;
        pyramid.push_back(blurredGray);

        for (int i = 0; i < (pyrLevel - 1); i++) {
            Mat downImage;
            pyrDown(pyramid[i], downImage, Size(pyramid[i].cols/2, pyramid[i].rows/2));

            pyramid.push_back(pyramid[i]);
        }

        // init kernel but in the iterations the tmp-kernel is used
        kernel = Mat::zeros(width, width, CV_8U);
        Mat tmpKernel;

        assert(pyramid.size() == 1 && "Implement multiple pyramid levels");

        // go through image pyramid from small to large
        for (int l = pyramid.size() - 1; l >= 0; l--) {
            // compute image gradient for x and y direction
            // 
            // gaussian blur (in-place operation is supported)
            GaussianBlur(pyramid[l], pyramid[l], Size(3,3), 0, 0, BORDER_DEFAULT);

            // parameter for sobel filtering to obtain gradients
            array<Mat,2> gradients, tmpGradients;
            const int delta = 0;
            const int ddepth = CV_32F;
            const int ksize = 3;
            const int scale = 1;

            // gradient x and y
            Sobel(pyramid[l], tmpGradients[0], ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
            Sobel(pyramid[l], tmpGradients[1], ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

            // cut off gradients outside the mask
            // FIXME: Scale mask to the current pyramid level!
            tmpGradients[0].copyTo(gradients[0], mask);
            tmpGradients[1].copyTo(gradients[1], mask);

            // normalize gradients into range [-1,1]
            normalizeOne(gradients);

            //  #ifndef NDEBUG
            //     showFloat("x gradient", gradients[0]);
            //     showFloat("y gradient", gradients[1]);
            // #endif
            

            // compute gradient confidence for al pixels
            Mat gradientConfidence;
            computeGradientConfidence(gradientConfidence, gradients, width, mask);

            // #ifndef NDEBUG
            //     showFloat("confidence", gradientConfidence);
            // #endif


            // each iterations works on an updated image
            Mat currentImage;
            pyramid[l].copyTo(currentImage);

            assert(iterations == 1 && "Implement multiple iterations");

            for (int i = 0; i < iterations; i++) {
                // select edges for kernel estimation (normalized gradients [-1,1])
                array<Mat,2> selectedEdges;
                selectEdges(currentImage, gradientConfidence, thresholdR, thresholdS, selectedEdges);

                // #ifndef NDEBUG
                //     showFloat("x gradient selection", selectedEdges[0]);
                //     showFloat("y gradient selection", selectedEdges[1]);
                // #endif


                // estimate kernel with gaussian prior
                fastKernelEstimation(selectedEdges, gradients, kernel, 0.0);

                // #ifndef NDEBUG                  
                //     showFloat("tmp-kernel", kernel, true);
                // #endif


                // coarse image estimation with a spatial prior
                Mat latentImage;
                // FIXME: it looks like there are some edges of the gradients in the latent image.
                //        with more iterations it becomes worse
                coarseImageEstimation(pyramid[l], kernel, selectedEdges, latentImage);

                // #ifndef NDEBUG
                //     string name = "tmp-latent" + i;
                //     imshow(name, latentImage);
                //     string filename = name + ".png";
                //     imwrite(filename, latentImage);
                // #endif


                // set current image to coarse latent image
                latentImage.copyTo(currentImage);

                // decrease thresholds τ_r and τ_s will to include more and more edges
                thresholdR = thresholdR / 1.1;
                thresholdS = thresholdS / 1.1;
            }
        }

        // #ifndef NDEBUG
        //     imshow("kernel", kernel);
        // #endif
    }
}