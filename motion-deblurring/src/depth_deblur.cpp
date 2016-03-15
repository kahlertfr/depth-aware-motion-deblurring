#include <iostream>                     // cout, cerr, endl
#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite
#include <queue>                        // FIFO queue
#include <cmath>                        // log

#include "utils.hpp"
#include "disparity_estimation.hpp"     // SGBM, fillOcclusions, quantize
#include "region_tree.hpp"
#include "edge_map.hpp"
#include "two_phase_psf_estimation.hpp"
#include "deconvolution.hpp"
#include "coherence_filter.hpp"

#include "depth_deblur.hpp"


using namespace cv;
using namespace std;


namespace deblur {

    DepthDeblur::DepthDeblur(Mat& imageLeft, Mat& imageRight, const int width)
                            : psfWidth((width % 2 == 0) ? width - 1 : width)    // odd psf-width needed
                            , layers((width % 2 == 0) ? width : width - 1)      // even layer number needed
                            , images({imageLeft, imageRight})
                            , current(0)
    {
        // convert color images to gray value images
        if (imageLeft.type() == CV_8UC3) {
            cvtColor(imageLeft, grayImages[LEFT], CV_BGR2GRAY);
        } else {
            grayImages[LEFT] = imageLeft;
        }

        if (imageRight.type() == CV_8UC3) {
            cvtColor(imageRight, grayImages[RIGHT], CV_BGR2GRAY);
        } else {
            grayImages[RIGHT] = imageRight;
        }
    }


    void DepthDeblur::disparityEstimation() {
        // quantized disparity maps for both directions (left-right and right-left)
        quantizedDisparityEstimation(grayImages[LEFT], grayImages[RIGHT], layers, disparityMaps[LEFT]);
        quantizedDisparityEstimation(grayImages[RIGHT], grayImages[LEFT], layers, disparityMaps[RIGHT], true);
    }


    void DepthDeblur::regionTreeReconstruction(const int maxTopLevelNodes) {
        // create a region tree
        regionTree.create(disparityMaps[LEFT], disparityMaps[RIGHT], layers,
                          &grayImages[LEFT], &grayImages[RIGHT], maxTopLevelNodes);
    }


    void DepthDeblur::toplevelKernelEstimation() {
        // go through each top-level node
        for (int i = 0; i < regionTree.topLevelNodeIds.size(); i++) {
            int id = regionTree.topLevelNodeIds[i];

            // // get the mask of the top-level region
            // Mat region, mask;
            // regionTree.getRegionImage(id, region, mask, LEFT);

            // // edge tapering to remove high frequencies at the border of the region
            // Mat taperedRegion;
            // regionTree.edgeTaper(taperedRegion, region, mask, grayImages[LEFT]);

            // // compute kernel
            // TwoPhaseKernelEstimation::estimateKernel(regionTree[id].psf, grayImages[LEFT], psfWidth, mask);
            // // TwoPhaseKernelEstimation::estimateKernel(regionTree[id].psf, taperedRegion, psfWidth);

            // #ifdef IMWRITE
            //     // top-level region
            //     string filename = "top-" + to_string(id) + "-mask.png";
            //     imwrite(filename, mask);

            //     // tapered image
            //     filename = "top-" + to_string(id) + "-tapered.png";
            //     imwrite(filename, taperedRegion);

            //     // top-level region
            //     grayImages[LEFT].copyTo(region, mask);
            //     filename = "top-" + to_string(id) + ".png";
            //     imwrite(filename, region);

            //     // kernel
            //     Mat tmp;
            //     regionTree[id].psf.copyTo(tmp);
            //     tmp *= 1000;
            //     convertFloatToUchar(tmp, tmp);
            //     filename = "top-" + to_string(id) + "-kernel.png";
            //     imwrite(filename, tmp);
            // #endif


            // // WORKAROUND because of deferred two-phase kernel estimation
            // // use the next two steps after each other
            // //
            // // 1. save the tappered region images for the exe of two-phase kernel estimation
            // // get an image of the top-level region
            // Mat region, mask;
            // regionTree.getRegionImage(id, region, mask, LEFT);
            
            // // edge tapering to remove high frequencies at the border of the region
            // Mat taperedRegion;
            // regionTree.edgeTaper(taperedRegion, region, mask, grayImages[LEFT]);

            // // use this images for example for the .exe of the two-phase kernel estimation
            // string name = "tapered" + to_string(i) + ".jpg";
            // imwrite(name, taperedRegion);
            
            // 2. load kernel images generated with the exe for toplevels
            // load the kernel images which should be named left/right-kerneli.png
            // they should be located in the folder where this algorithm is started
            string filename = "kernel" + to_string(i) + ".png";
            Mat kernelImage = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

            if (!kernelImage.data) {
                throw runtime_error("Can not load kernel!");
            }
            
            // convert kernel-image to energy preserving float kernel
            kernelImage.convertTo(kernelImage, CV_32F);
            kernelImage /= sum(kernelImage)[0];

            // save the psf
            kernelImage.copyTo(regionTree[id].psf);
        }
    }


    void DepthDeblur::jointPSFEstimation(const Mat& maskLeft, const Mat& maskRight, 
                                         const array<Mat,2>& salientEdgesLeft,
                                         const array<Mat,2>& salientEdgesRight,
                                         Mat& psf) {

        // compute Objective function: E(k) = sum_i( ||∇S_i ⊗ k - ∇B||² + γ||k||² )
        // where i ∈ {r, m}, and S_i is the region for reference and matching view 
        // and k is the psf-kernel
        // 
        // perform FFT
        //                      __________                     __________
        //             (  sum_i(F(∂_x S_i) * F(∂_x B)) + sum_i(F(∂_y S_i) * F(∂_y B)) )
        // k = F^-1 * ( ------------------------------------------------------------   )
        //             (         sum( F(∂_x S_i)² + F(∂_y S_i)²) + γ F_1              )
        // where * is pointwise multiplication
        //                   __________
        // and F(∂_x S_i)² = F(∂_x S_i) * F(∂_x S_i)
        // and F_1 is the fourier transform of a delta function with a uniform 
        // energy distribution - they probably use this to transform the scalar weight
        // to a complex matrix
        // 
        // here: F(∂_x S_i) = xSr / xSm
        //       F(∂_x B)   = xB
        //       F(∂_y S_i) = xSr / xSm
        //       F(∂_y B)   = yB
        
        // the result are stored as 2 channel matrices: Re(FFT(I)), Im(FFT(I))
        Mat xSr, xSm, ySr, ySm;  // fourier transform of region gradients
        Mat xBr, xBm, yBr, yBm;  // fourier transform of blurred images
        
        fft(salientEdgesLeft[0], xSm);
        fft(salientEdgesLeft[1], ySm);
        fft(salientEdgesRight[0], xSr);
        fft(salientEdgesRight[1], ySr);
        fft(gradsLeft[0], xBm);
        fft(gradsLeft[1], yBm);
        fft(gradsRight[0], xBr);
        fft(gradsRight[1], yBr);


        // delta function as one white pixel in black image
        Mat deltaFloat = Mat::zeros(xSm.size(), CV_32F);
        deltaFloat.at<float>(xSm.rows / 2, xSm.cols / 2) = 1;
        Mat delta;
        fft(deltaFloat, delta);

        // kernel in Fourier domain
        Mat K = Mat::zeros(xSm.size(), xSm.type());

        // go through all pixel and calculate the value in the brackets of the equation
        for (int x = 0; x < xSm.cols; x++) {
            for (int y = 0; y < xSm.rows; y++) {
                // complex entries at the current position
                complex<float> xsr(xSr.at<Vec2f>(y, x)[0], xSr.at<Vec2f>(y, x)[1]);
                complex<float> ysr(ySr.at<Vec2f>(y, x)[0], ySr.at<Vec2f>(y, x)[1]);
                complex<float> xsm(xSm.at<Vec2f>(y, x)[0], xSm.at<Vec2f>(y, x)[1]);
                complex<float> ysm(ySm.at<Vec2f>(y, x)[0], ySm.at<Vec2f>(y, x)[1]);

                complex<float> xbr(xBr.at<Vec2f>(y, x)[0], xBr.at<Vec2f>(y, x)[1]);
                complex<float> ybr(yBr.at<Vec2f>(y, x)[0], yBr.at<Vec2f>(y, x)[1]);
                complex<float> xbm(xBm.at<Vec2f>(y, x)[0], xBm.at<Vec2f>(y, x)[1]);
                complex<float> ybm(yBm.at<Vec2f>(y, x)[0], yBm.at<Vec2f>(y, x)[1]);

                complex<float> d(delta.at<Vec2f>(y, x)[0], delta.at<Vec2f>(y, x)[1]);

                complex<float> weight(10000.50, 0.0);

                // kernel entry in the Fourier space
                complex<float> k = ( (conj(xsr) * xbr + conj(xsm) * xbm) +
                                     (conj(ysr) * ybr + conj(ysm) * ybm) ) /
                                     ( (conj(xsr) * xsr + conj(ysr) * ysr) + 
                                       (conj(xsm) * xsm + conj(ysm) * ysm) + weight );
                                       // (conj(xsm) * xsm + conj(ysm) * ysm) + weight * conj(d) * d );
                
                K.at<Vec2f>(y, x) = { real(k), imag(k) };
            }
        }

        // compute inverse FFT of the kernel
        Mat kernel;
        dft(K, kernel, DFT_INVERSE | DFT_REAL_OUTPUT);

        // threshold kernel to erease negative values
        // this is done because otherwise the resulting kernel is very grayish
        threshold(kernel, kernel, 0.0, -1, THRESH_TOZERO);

        // kernel has to be energy preserving
        // this means: sum(kernel) = 1
        kernel /= sum(kernel)[0];

        // cut of the psf-kernel
        int x = kernel.cols / 2 - psfWidth / 2;
        int y = kernel.rows / 2 - psfWidth / 2;
        swapQuadrants(kernel);
        Mat kernelROI = kernel(Rect(x, y, psfWidth, psfWidth));

        // important to copy the roi - otherwise for padding the originial image
        // will be used (we don't want this behavior)
        kernelROI.copyTo(psf);

        // #ifndef NDEBUG
        //     Mat kernelUchar;
        //     convertFloatToUchar(kernel, kernelUchar);
        //     imshow("full psf", kernelUchar);
        //     waitKey(0);
        // #endif
    }


    void DepthDeblur::computeBlurredGradients() {
        // compute simple gradients for blurred images
        std::array<cv::Mat,2> gradsR, gradsL;

        // parameter for sobel gradient computation
        const int delta = 0;
        const int ddepth = CV_32F;
        const int ksize = 3;
        const int scale = 1;

        // gradients of left image
        Sobel(grayImages[LEFT], gradsL[0],
              ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
        Sobel(grayImages[LEFT], gradsL[1],
              ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

        // gradients of right image
        Sobel(grayImages[RIGHT], gradsR[0],
              ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
        Sobel(grayImages[RIGHT], gradsR[1],
              ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

        // norm the gradients
        normalize(gradsR[0], gradsRight[0], -1, 1);
        normalize(gradsR[1], gradsRight[1], -1, 1);
        normalize(gradsL[0], gradsLeft[0], -1, 1);
        normalize(gradsL[1], gradsLeft[1], -1, 1);
    }


    void DepthDeblur::estimateChildPSF(int id) {
        // get masks for regions of both views
        Mat maskM, maskR;
        regionTree.getMasks(id, maskM, maskR);

        // get parent id
        int parent = regionTree[id].parent;

        // compute salient edge map ∇S_i for region
        // 
        // deblur the current views with psf from parent
        Mat deblurredLeft, deblurredRight;
        deconvolveFFT(grayImages[LEFT], deblurredLeft, regionTree[parent].psf);
        deconvolveFFT(grayImages[RIGHT], deblurredRight, regionTree[parent].psf);
        // FIXME: strong ringing artifacts in deconvoled image

        // #ifdef IMWRITE
        //     imshow("devonv left", deblurredLeft);
        //     waitKey();
        // #endif

        // compute a gradient image with salient edge (they are normalized to [-1, 1])
        array<Mat,2> salientEdgesLeft, salientEdgesRight;
        // FIXME: not just inside mask??
        computeSalientEdgeMap(deblurredLeft, salientEdgesLeft, psfWidth, maskM);
        computeSalientEdgeMap(deblurredRight, salientEdgesRight, psfWidth, maskR);

        // #ifdef IMWRITE
        //     showGradients("salient edges left x", salientEdgesLeft[0]);
        //     showGradients("salient edges right x", salientEdgesRight[0]);
        //     waitKey();
        // #endif

        // estimate psf for the first child node
        jointPSFEstimation(maskM, maskR, salientEdgesLeft, salientEdgesRight, regionTree[id].psf);

        #ifdef IMWRITE
            // region images
            Mat region;
            grayImages[LEFT].copyTo(region, maskM);
            string filename = "mid-" + to_string(id) + "-left.png";
            imwrite(filename, region);

            // kernels
            Mat tmp;
            regionTree[id].psf.copyTo(tmp);
            tmp *= 1000;
            convertFloatToUchar(tmp, tmp);
            filename = "mid-" + to_string(id) + "-kernel-init.png";
            imwrite(filename, tmp);
        #endif
    }


    float DepthDeblur::computeEntropy(Mat& kernel) {
        assert(kernel.type() == CV_32F && "works with float values");

        float entropy = 0.0;

        // go through all pixel of the kernel
        for (int row = 0; row < kernel.rows; row++) {
            for (int col = 0; col < kernel.cols; col++) {
                float x = kernel.at<float>(row, col);
                
                // prevent caculation of log(0)
                if (x > 0) {
                    entropy += x * log(x);
                }
            }
        }

        entropy = -1 * entropy;

        return entropy; 
    }


    void DepthDeblur::candidateSelection(vector<Mat>& candiates, int id, int sid) {
        // own psf is added as candidate
        candiates.push_back(regionTree[id].psf);

        // psf of parent is added as candidate
        int pid = regionTree[id].parent;
        candiates.push_back(regionTree[pid].psf);

        // add sibbling psf just if it is reliable
        // this means: entropy - mean < threshold
        float mean = (regionTree[id].entropy + regionTree[sid].entropy) / 2.0;

        // empirically choosen threshold
        float threshold = 0.2 * mean;

        if (regionTree[sid].entropy - mean < threshold) {
            candiates.push_back(regionTree[sid].psf);
        }
    }


    void DepthDeblur::psfSelection(vector<Mat>& candiates, int id) {
        float minEnergy = 2;
        int winner = 0;
        
        for (int i = 0; i < candiates.size(); i++) {
            // get mask of this region
            Mat mask;
            regionTree.getMask(id, mask, LEFT);

            // compute latent image
            Mat latent;
            // FIXME: latent image just of one view?
            deconvolveFFT(grayImages[LEFT], latent, candiates[i]);

            // slightly Gaussian smoothed
            // use the complete image to avoid unwanted effects at the borders
            Mat smoothed;
            GaussianBlur(latent, smoothed, Size(5, 5), 0, 0, BORDER_DEFAULT);
            
            // shock filtered
            Mat shockFiltered;
            coherenceFilter(smoothed, shockFiltered);
            
            // compute correlation of the latent image and the shockfiltered image
            float energy = 1 - gradientCorrelation(latent, shockFiltered, mask);

            #ifdef IMWRITE
                cout << "    energy for " << i << ": " << energy << endl;
            #endif

            if (energy < minEnergy) {
                winner = i;
            }
        }

        // save the winner of the psf selection in the current node
        candiates[winner].copyTo(regionTree[id].psf);
    }


    float DepthDeblur::gradientCorrelation(Mat& image1, Mat& image2, Mat& mask) {
        assert(mask.type() == CV_8U && "mask is uchar image with zeros and ones");

        // #ifdef IMWRITE
        //     imshow("image1", image1);
        //     imshow("image2", image2);
        //     waitKey();
        // #endif

        // compute gradients
        // parameter for sobel filtering to obtain gradients
        array<Mat,2> tmpGrads1, tmpGrads2;
        const int delta = 0;
        const int ddepth = CV_32F;
        const int ksize = 3;
        const int scale = 1;

        // gradient x and y for both images
        Sobel(image1, tmpGrads1[0], ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
        Sobel(image1, tmpGrads1[1], ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
        Sobel(image2, tmpGrads2[0], ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
        Sobel(image2, tmpGrads2[1], ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

        // compute single channel gradient image
        Mat gradients1, gradients2;
        normedGradients(tmpGrads1, gradients1);
        normedGradients(tmpGrads2, gradients2);

        // norm gradients to [0,1]
        double min; double max;
        minMaxLoc(gradients1, &min, &max);
        gradients1 /= max;
        minMaxLoc(gradients2, &min, &max);
        gradients2 /= max;

        // cut of region
        Mat X, Y;
        gradients1.copyTo(X, mask);
        gradients2.copyTo(Y, mask);

       
        // compute correlation
        //
        // compute mean of the matrices
        // use just the pixel inside the mask
        float meanX = 0;
        float meanY = 0;
        float N = 0;

        for (int row = 0; row < X.rows; row++) {
            for (int col = 0; col < X.cols; col++) {
                // compute if inside mask (0 - ouside, 255 -inside)
                if (mask.at<uchar>(row, col) > 0) {
                    // expected values                
                    meanX += X.at<float>(row, col);
                    meanY += Y.at<float>(row, col);
                    N++;
                }
            }
        }

        meanX /= N;
        meanY /= N;
        
        // expected value can be computed using the mean:
        // E(X - μx) = 1/N * sum_x(x - μx) ... denoted as Ex

        // FIXME: does the paper use the corr2 function of matlab? I think so
        float E = 0;

        // deviation = sqrt(1/N * sum_x(x - μx)²) -> do not use 1/N 
        float deviationX = 0;
        float deviationY = 0;

        assert(X.size() == Y.size() && "images of same size");
        
        // go through each gradient map and
        // compute the sums in the computation of expedted values and deviations
        for (int row = 0; row < X.rows; row++) {
            for (int col = 0; col < X.cols; col++) {
                // compute if inside mask
                if (mask.at<uchar>(row, col) > 0) {
                    float valueX = X.at<float>(row, col) - meanX;
                    float valueY = Y.at<float>(row, col) - meanY;

                    // expected values (the way matlab calculates it)              
                    E += valueX * valueY;

                    // deviation
                    deviationX += (valueX * valueX);
                    deviationY += (valueY * valueY);
                }
            }
        }
           
        deviationX = sqrt(deviationX);
        deviationY = sqrt(deviationY);

        float correlation = E / (deviationX * deviationY);

        return correlation;
    }


    void DepthDeblur::midLevelKernelEstimation() {
        // we can compute the gradients for each blurred image only ones
        computeBlurredGradients();

        // go through all nodes of the region tree in a top-down manner
        // 
        // the current node is responsible for the PSF computation of its children
        // because later the information from the parent and the children are needed for 
        // PSF candidate selection
        // 
        // for storing the future "current nodes" a queue is used (FIFO) this fits the
        // levelwise computation of the paper
        
        queue<int> remainingNodes;

        // init queue with the top-level node IDs
        for (int i = 0; i < regionTree.topLevelNodeIds.size(); i++) {
            remainingNodes.push(regionTree.topLevelNodeIds[i]);
        }


        while(!remainingNodes.empty()) {
            // pop id of current node from the front of the queue
            int id = remainingNodes.front();
            remainingNodes.pop();

            cout << "  at node: " << id;

            // get IDs of the child nodes
            int cid1 = regionTree[id].children.first;
            int cid2 = regionTree[id].children.second;

            cout << " with " << cid1 << " " << cid2 << endl;

            // do PSF computation for a middle node with its children
            // (leaf nodes doesn't have any children)
            if (cid1 != -1 && cid2 != -1) {
                // // add children ids to the back of the queue
                // remainingNodes.push(regionTree[id].children.first);
                // remainingNodes.push(regionTree[id].children.second);

                // PSF estimation for each children
                // (salient edge map computation and joint psf estimation)
                estimateChildPSF(cid1);
                estimateChildPSF(cid2);

                // to eliminate errors
                //
                // calucate entropy of the found psf
                regionTree[cid1].entropy = computeEntropy(regionTree[cid1].psf);
                regionTree[cid2].entropy = computeEntropy(regionTree[cid2].psf);

                cout << "  entropies: " << regionTree[cid1].entropy << " " << regionTree[cid2].entropy << endl;

                // candiate selection
                vector<Mat> candiates1, candiates2;
                candidateSelection(candiates1, cid1, cid2);
                candidateSelection(candiates2, cid2, cid1);

                // final psf selection
                psfSelection(candiates1, cid1);
                psfSelection(candiates2, cid2);
            }
        }
    }


    void DepthDeblur::deconvolve(Mat& dst, view view, bool color) {
        // make a deconvolution for each disparity layer
        for (int i = 0; i < psfWidth; i++) {         
            // get mask of the disparity level
            Mat mask;
            regionTree.getMask(i, mask, view);

            Mat tmpDeconv;

            if (color) {
                deconvolveIRLS(images[view], tmpDeconv, regionTree[i].psf);
            } else {
                deconvolveIRLS(grayImages[view], tmpDeconv, regionTree[i].psf);
            }

            tmpDeconv.copyTo(dst, mask);

            #ifdef IMWRITE
                string filename = "tmp-deconv-" + to_string(view) + "-" + to_string(i) + ".png";
                imwrite(filename, dst);
            #endif
        }

        #ifdef IMWRITE
            string filename = "deconv-" + to_string(view) + ".png";
            imwrite(filename, dst);
        #endif
    }
}