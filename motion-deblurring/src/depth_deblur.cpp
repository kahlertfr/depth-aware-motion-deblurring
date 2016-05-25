#include <iostream>                     // cout, cerr, endl
#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite
#include <cmath>                        // log
#include <thread>

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

    DepthDeblur::DepthDeblur(const Mat& imageLeft, const Mat& imageRight, const int width, const int _layers)
                            : psfWidth((width % 2 == 0) ? width - 1 : width)       // odd psf-width needed
                            , layers((_layers % 2 == 0) ? _layers : _layers - 1)   // psf width should be larger - even layer number needed
                            , images({imageLeft, imageRight})
    {
        assert(imageLeft.type() == imageRight.type() && "images of same type necessary");

        // use gray values for disparity estimation
        if (images[LEFT].type() == CV_8UC3) {
            cvtColor(images[LEFT], grayImages[LEFT], CV_BGR2GRAY);
            cvtColor(images[RIGHT], grayImages[RIGHT], CV_BGR2GRAY);
        } else {
            grayImages[LEFT] = images[LEFT];
            grayImages[RIGHT] = images[RIGHT];
        }

        // convert images to floats and scale to range [0,1]
        grayImages[LEFT].convertTo(floatImages[LEFT], CV_32F, 1 / 255.0);
        grayImages[RIGHT].convertTo(floatImages[RIGHT], CV_32F, 1 / 255.0);
    }


    void DepthDeblur::disparityEstimation(const array<Mat, 2>& input, const disparityAlgo algorithm,
                                          const int maxDisparity) {
        array<Mat, 2> views;

        // use gray values for disparity estimation for SGBM
        if (algorithm == SGBM && input[LEFT].type() == CV_8UC3) {
            cvtColor(input[LEFT], views[LEFT], CV_BGR2GRAY);
            cvtColor(input[RIGHT], views[RIGHT], CV_BGR2GRAY);
        } else {
            views[LEFT] = input[LEFT];
            views[RIGHT] = input[RIGHT];
        }

        // down sample images to roughly reduce blur for disparity estimation
        array<Mat, 2> small;

        // because we checked that both images are of the same size
        // the new size is the same for both too
        // (down sampling ratio is 2)
        Size downsampledSize = Size(views[LEFT].cols / 2, views[RIGHT].rows / 2);

        // down sample with Gaussian pyramid
        pyrDown(views[LEFT], small[LEFT], downsampledSize);
        pyrDown(views[RIGHT], small[RIGHT], downsampledSize);

        array<Mat, 2> smallDMaps = { Mat::zeros(small[LEFT].size(), CV_8U),
                                     Mat::zeros(small[RIGHT].size(), CV_8U)};
      
        if (algorithm == SGBM) {
            // disparity map with occlusions as black regions
            // here a different algorithm as the paper approach is used
            // because it is more convenient to use a OpenCV implementation.
            disparityFilledSGBM(small, smallDMaps);
        } else if (algorithm == MATCH) {
            // disparity estimation algorithm from the paper
            disparityFilledMatch(small, smallDMaps, maxDisparity);
        } else {
            throw runtime_error("Invalid disparity algorithm");
        }

        // #ifdef IMWRITE
        //     // convert quantized image to be displayable
        //     Mat disparityViewableAlgo;
        //     double min1; double max1;
        //     minMaxLoc(smallDMaps[LEFT], &min1, &max1);
        //     cout << "min: " << min1 << " max: " << max1 << endl;
        //     smallDMaps[LEFT].convertTo(disparityViewableAlgo, CV_8U, 255.0/(max1-min1));

        //     string filenameAlgo = "dmap-algo-left.png";
        //     imwrite(filenameAlgo, disparityViewableAlgo);
        // #endif

        // quantize the image
        array<Mat, 2> quantizedDMaps;
        quantizeImage(smallDMaps, layers, quantizedDMaps);

        #ifdef IMWRITE
            // convert quantized image to be displayable
            Mat disparityViewable;
            double min; double max;
            minMaxLoc(quantizedDMaps[LEFT], &min, &max);
            quantizedDMaps[LEFT].convertTo(disparityViewable, CV_8U, 255.0/(max-min));

            // imshow("quantized disparity map " + prefix, disparityViewable);
            string filename = "dmap-left.png";
            imwrite(filename, disparityViewable);

            minMaxLoc(quantizedDMaps[RIGHT], &min, &max);
            quantizedDMaps[RIGHT].convertTo(disparityViewable, CV_8U, 255.0/(max-min));

            // imshow("quantized disparity map " + prefix, disparityViewable);
            filename = "dmap-right.png";
            imwrite(filename, disparityViewable);
        #endif

        // up sample disparity map to original resolution without interpolation
        resize(quantizedDMaps[LEFT], disparityMaps[LEFT], Size(views[LEFT].cols, views[LEFT].rows), 0, 0, INTER_NEAREST);      
        resize(quantizedDMaps[RIGHT], disparityMaps[RIGHT], Size(views[RIGHT].cols, views[RIGHT].rows), 0, 0, INTER_NEAREST);      
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
            // Mat regionUchar, taperedRegion;
            // region.convertTo(regionUchar, CV_8U);
            // edgeTaper(regionUchar, taperedRegion, mask, grayImages[LEFT]);

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
            // regionTree.getRegionImage(id, region, mask, RIGHT);
            // string name = "mask-right" + to_string(i) + ".jpg";
            // imwrite(name, mask * 255);

            // Mat regionLeft, maskLeft;
            // regionTree.getRegionImage(id, regionLeft, maskLeft, LEFT);
            // name = "mask-left" + to_string(i) + ".jpg";
            // imwrite(name, maskLeft * 255);
            
            // // edge tapering to remove high frequencies at the border of the region
            // Mat regionUchar, taperedRegion;
            // regionLeft.convertTo(regionUchar, CV_8U);
            // edgeTaper(regionUchar, taperedRegion, maskLeft, grayImages[LEFT]);

            // // use this images for example for the .exe of the two-phase kernel estimation
            // name = "tapered" + to_string(i) + ".jpg";
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

            #ifdef IMWRITE
                Mat region, mask, regionUchar;
                regionTree.getRegionImage(id, region, mask, LEFT);
                region.convertTo(regionUchar, CV_8U);
                string name = "top-" + to_string(i) + "-left.jpg";
                imwrite(name, regionUchar);
            #endif
        }
    }


    void DepthDeblur::jointPSFEstimation(const array<Mat, 2>& masks, const array<Mat,2>& salientEdgesLeft,
                                         const array<Mat,2>& salientEdgesRight, Mat& psf) {

        // get gradients of current region only
        array<Mat,2> regionGradsLeft, regionGradsRight;
        gradsLeft[0].copyTo(regionGradsLeft[0], masks[LEFT]);
        gradsLeft[1].copyTo(regionGradsLeft[1], masks[LEFT]);
        gradsRight[0].copyTo(regionGradsRight[0], masks[RIGHT]);
        gradsRight[1].copyTo(regionGradsRight[1], masks[RIGHT]);

        // showGradients("region-grads-left-x", regionGradsLeft[0], true);
        // showGradients("region-grads-right-x", regionGradsRight[0], true);
        // showGradients("salient-left-x", salientEdgesLeft[0], true);
        // showGradients("salient-right-x", salientEdgesRight[0], true);
        // showGradients("region-grads-left-y", regionGradsLeft[1], true);
        // showGradients("region-grads-right-y", regionGradsRight[1], true);
        // showGradients("salient-left-y", salientEdgesLeft[1], true);
        // showGradients("salient-right-y", salientEdgesRight[1], true);
        // waitKey();

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

        dft(salientEdgesLeft[0], xSm);
        dft(salientEdgesLeft[1], ySm);
        dft(salientEdgesRight[0], xSr);
        dft(salientEdgesRight[1], ySr);

        dft(regionGradsLeft[0], xBm);
        dft(regionGradsLeft[1], yBm);
        dft(regionGradsRight[0], xBr);
        dft(regionGradsRight[1], yBr);

        // delta function as one white pixel in black image
        Mat deltaFloat = Mat::zeros(xSm.size(), CV_32F);
        deltaFloat.at<float>(0, 0) = 1;
        Mat delta;
        dft(deltaFloat, delta);

        // Mat Br, Bm;
        // array<Mat, 2> blurredRegion;
        // floatImages[LEFT].copyTo(blurredRegion[LEFT], masks[LEFT]);
        // floatImages[RIGHT].copyTo(blurredRegion[RIGHT], masks[RIGHT]);
        // dft(blurredRegion[LEFT], Bm);
        // dft(blurredRegion[RIGHT], Br);
        
        // // sobel gradients for x and y direction
        // Mat sobelx = Mat::zeros(xSm.size(), CV_32F);
        // sobelx.at<float>(0,0) = -1;
        // sobelx.at<float>(0,1) = 1;

        // Mat sobely = Mat::zeros(xSm.size(), CV_32F);
        // sobely.at<float>(0,0) = -1;
        // sobely.at<float>(1,0) = 1;

        // Mat Gx, Gy;
        // dft(sobelx, Gx);
        // dft(sobely, Gy);

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

                // complex<float> bm(Br.at<Vec2f>(y, x)[0], Br.at<Vec2f>(y, x)[1]);
                // complex<float> br(Bm.at<Vec2f>(y, x)[0], Bm.at<Vec2f>(y, x)[1]);
                // complex<float> gx(Gx.at<Vec2f>(y, x)[0], Gx.at<Vec2f>(y, x)[1]);
                // complex<float> gy(Gy.at<Vec2f>(y, x)[0], Gy.at<Vec2f>(y, x)[1]);

                complex<float> d(delta.at<Vec2f>(y, x)[0], delta.at<Vec2f>(y, x)[1]);

                // weight from paper wk = 1
                complex<float> weight(1, 0.0);

                // kernel entry in the Fourier space
                // we are using the Fourier transform of the gradients of the blurred region
                // instead of transforming the sobel filter and the blurred region separately in
                // frequency domain. Because this will prevent the huge gradients at the
                // region boundary
                complex<float> k = ( (conj(xsr) * xbr + conj(xsm) * xbm) +
                                     (conj(ysr) * ybr + conj(ysm) * ybm) ) /
                                     ( (conj(xsr) * xsr + conj(ysr) * ysr) + 
                                     (conj(xsm) * xsm + conj(ysm) * ysm) + weight * conj(d) * d );

                // // kernel entry in the Fourier space
                // complex<float> k = ( (conj(xsr) * gx * br + conj(xsm) * gx * bm) +
                //                      (conj(ysr) * gy * br + conj(ysm) * gy * bm) ) /
                //                      ( (conj(xsr) * xsr + conj(ysr) * ysr) + 
                //                      (conj(xsm) * xsm + conj(ysm) * ysm) + weight * conj(d) * d );
                
                K.at<Vec2f>(y, x) = { real(k), imag(k) };
            }
        }

        // compute inverse FFT of the kernel
        Mat kernel;
        dft(K, kernel, DFT_INVERSE | DFT_REAL_OUTPUT);

        // threshold kernel to erease negative values
        // this is done because otherwise the resulting kernel is very grayish
        // the negative results are noise
        threshold(kernel, kernel, 0.0, -1, THRESH_TOZERO);

        // swap slices of the result
        // because the image is shifted to the upper-left corner
        int x = kernel.cols;
        int y = kernel.rows;
        int hs1 = (psfWidth - 1) / 2;
        int hs2 = (psfWidth - 1) / 2;

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
        Mat q0(kernel, Rect(0, 0, x - hs1, y - hs2));      // Top-Left
        Mat q1(kernel, Rect(x - hs1, 0, hs1, y - hs2));    // Top-Right
        Mat q2(kernel, Rect(0, y - hs2, x - hs1, hs2));    // Bottom-Left
        Mat q3(kernel, Rect(x - hs1, y - hs2, hs1, hs2));  // Bottom-Right

        Mat kernelSwap;
        hconcat(q3, q2, kernelSwap);
        Mat tmp;
        hconcat(q1, q0, tmp);
        vconcat(kernelSwap, tmp, kernelSwap);
        kernelSwap.copyTo(kernel);

        // cut of the psf-kernel
        Mat kernelROI = kernel(Rect(0, 0, psfWidth, psfWidth));

        // important to copy the roi - otherwise for padding the originial image
        // will be used (we don't want this behavior)
        kernelROI.copyTo(psf);

        // alternative thresholding idea:
        double min; double max;
        minMaxLoc(psf, &min, &max);
        threshold(psf, psf, max / 7, -1, THRESH_TOZERO); 
            
        // kernel has to be energy preserving
        // this means: sum(kernel) = 1
        psf /= sum(psf)[0];

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

        // showGradients("grads-blur-left-x.png", gradsL[0], true);
        // showGradients("grads-blur-left-y.png", gradsL[1], true);
        // showGradients("grads-blur-right-x.png", gradsR[0], true);
        // showGradients("grads-blur-right-y.png", gradsR[1], true);
        // waitKey();
    }


    void DepthDeblur::estimateChildPSF(const Mat& parentPSF, Mat& psf, const array<Mat, 2>& masks,
                                       const int id) {

        // compute salient edge map ∇S_i for region
        // 
        // deblur the current views with psf from parent
        array<Mat, 2> deconv;
        deconvolveFFT(floatImages[LEFT], deconv[LEFT], parentPSF);
        deconvolveFFT(floatImages[RIGHT], deconv[RIGHT], parentPSF);

        // FIXME: strong ringing artifacts in deconvoled image (with fft)
    
        // #ifdef IMWRITE
        //     imshow("devonv left", deconv[LEFT]);
        //     waitKey();
        // #endif
        
        // compute a gradient image with salient edge (they are normalized to [-1, 1])
        array<Mat,2> salientEdgesLeft, salientEdgesRight;
        computeSalientEdgeMap(deconv[LEFT], salientEdgesLeft, psfWidth, masks[LEFT]);
        computeSalientEdgeMap(deconv[RIGHT], salientEdgesRight, psfWidth, masks[RIGHT]);

        // #ifdef IMWRITE
        //     showGradients("salient-edges-left-x", salientEdgesLeft[0], true);
        //     showGradients("salient-edges-left-y", salientEdgesLeft[1], true);
        //     // waitKey();
        // #endif

        // estimate psf for the first child node
        jointPSFEstimation(masks, salientEdgesLeft, salientEdgesRight, psf);

        #ifdef IMWRITE
            // region images
            Mat region;
            grayImages[LEFT].copyTo(region, masks[LEFT]);
            string filename = "mid-" + to_string(id) + "-left.png";
            imwrite(filename, region);

            Mat regionR;
            grayImages[RIGHT].copyTo(regionR, masks[RIGHT]);
            filename = "mid-" + to_string(id) + "-right.png";
            imwrite(filename, regionR);

            // kernels
            Mat tmp;
            psf.copyTo(tmp);
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
            deconvolveFFT(floatImages[LEFT], latent, candiates[i]);

            // slightly Gaussian smoothed
            // use the complete image to avoid unwanted effects at the borders
            Mat smoothed;
            GaussianBlur(latent, smoothed, Size(5, 5), 0, 0, BORDER_DEFAULT);
            
            // shock filtered
            Mat shockFiltered;
            coherenceFilter(smoothed, shockFiltered);
            
            // compute correlation of the latent image and the shockfiltered image
            float energy = 1 - gradientCorrelation(latent, shockFiltered, mask);

            // #ifdef IMWRITE
            //     cout << "    energy for " << i << ": " << energy << endl;
            // #endif

            if (energy < minEnergy) {
                winner = i;
            }
        }

        // save the winner of the psf selection in the current node
        candiates[winner].copyTo(regionTree[id].psf);
            
        #ifdef IMWRITE
            // kernels
            Mat tmp;
            regionTree[id].psf.copyTo(tmp);
            tmp *= 1000;
            convertFloatToUchar(tmp, tmp);
            string filename = "mid-" + to_string(id) + "-kernel-selection.png";
            imwrite(filename, tmp);
        #endif
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


    bool DepthDeblur::safeQueueAccess(queue<int>* sharedQueue, int& item) {
        // this lock guard calls lock and if the end of the scope is reached
        // it calls unlock automatically
        lock_guard<mutex> g(m);

        // now we can access the stack without collisions
        if (sharedQueue->empty() != true) {
            item = sharedQueue->front();
            sharedQueue->pop();
            return true;
        } else {
            return false;
        }
    }


    void DepthDeblur::midLevelKernelEstimationNode(){
        int id;

        while(visitedLeafs != layers) {
            if (safeQueueAccess(&remainingNodes, id)) {
                // get IDs of the child nodes
                int cid1 = regionTree[id].children.first;
                int cid2 = regionTree[id].children.second;

                // do PSF computation for a middle node with its children
                // (leaf nodes doesn't have any children)
                if (cid1 != -1 && cid2 != -1) {
                    // PSF estimation for each children
                    // (salient edge map computation and joint psf estimation)
                    
                    // initial psf estimation child 1
                    // get masks for regions of both views
                    array<Mat, 2> masks;
                    regionTree.getMasks(cid1, masks);

                    // check if one of the masks is empty because then the joint estimation is not working
                    // (this could happen when the depth value is appears just in one disparity map)
                    if (sum(masks[LEFT])[0] != 0 && sum(masks[RIGHT])[0] != 0) {
                        estimateChildPSF(regionTree[id].psf, regionTree[cid1].psf, masks, cid1);
                    } else {
                        // set the child psf to the parents one if one mask is empty
                        regionTree[cid1].psf = regionTree[id].psf;
                    }
                    

                    // initial psf estimation child 2
                    // get masks for regions of both views
                    regionTree.getMasks(cid2, masks);

                    // check if one of the masks is empty because then the joint estimation is not working
                    // (this could happen when the depth value is appears just in one disparity map)
                    if (sum(masks[LEFT])[0] != 0 && sum(masks[RIGHT])[0] != 0) {
                        estimateChildPSF(regionTree[id].psf, regionTree[cid2].psf, masks, cid2);
                    } else {
                        // set the child psf to the parents one if one mask is empty
                        regionTree[cid2].psf = regionTree[id].psf;
                    }


                    // to eliminate errors
                    //
                    // calucate entropy of the found psf
                    regionTree[cid1].entropy = computeEntropy(regionTree[cid1].psf);
                    regionTree[cid2].entropy = computeEntropy(regionTree[cid2].psf);

                    // candiate selection
                    vector<Mat> candiates1, candiates2;
                    candidateSelection(candiates1, cid1, cid2);
                    candidateSelection(candiates2, cid2, cid1);

                    // final psf selection
                    psfSelection(candiates1, cid1);
                    psfSelection(candiates2, cid2);


                    // add children ids to the back of the queue (this has to be thread save)
                    m.lock();
                    remainingNodes.push(cid1);
                    remainingNodes.push(cid2);
                    m.unlock();

                } else {
                    mCounter.lock();
                    visitedLeafs++;
                    mCounter.unlock();
                }
            }
        }
    }


    void DepthDeblur::midLevelKernelEstimation(int nThreads) {
        // // // debug --------------------------------------------------------------
        // Mat src, kernel, dst;
        // // src = imread("conv-texture-bw.jpg", CV_LOAD_IMAGE_GRAYSCALE);
        // // // src = imread("conv-texture-wall-ausschnitt.jpg", CV_LOAD_IMAGE_GRAYSCALE);
        // // src.convertTo(src, CV_32F);
        // // src /= 255;

        // kernel = imread("kernel0.png", CV_LOAD_IMAGE_GRAYSCALE);
        // kernel.convertTo(kernel, CV_32F);
        // kernel /= sum(kernel)[0];  // mouse kernel is not energy preserving

        // computeBlurredGradients();

        // // get masks for regions of both views
        // array<Mat, 2> masks;
        // masks[LEFT] = imread("mask-left0.jpg", CV_LOAD_IMAGE_GRAYSCALE);
        // masks[LEFT] /= 255;
        // masks[RIGHT] = imread("mask-right0.jpg", CV_LOAD_IMAGE_GRAYSCALE);
        // masks[RIGHT] /= 255;

        // // masks[LEFT] = Mat::ones(floatImages[LEFT].size(), CV_8U);
        // // masks[RIGHT] = Mat::ones(floatImages[LEFT].size(), CV_8U);
        

        // Mat parentPSF, psf;
        // estimateChildPSF(kernel, psf, masks, 0);

        // deconvolveFFT(floatImages[LEFT], dst, psf);
        // threshold(dst, dst, 0.0, -1, THRESH_TOZERO);
        // threshold(dst, dst, 1.0, -1, THRESH_TRUNC);
        // dst.convertTo(dst, CV_8U, 255);
        // imwrite("deconv-estimate-fft.png", dst);

        // deconvolveIRLS(floatImages[LEFT], dst, psf, masks[LEFT]);
        // threshold(dst, dst, 0.0, -1, THRESH_TOZERO);
        // threshold(dst, dst, 1.0, -1, THRESH_TRUNC);
        // dst.convertTo(dst, CV_8U, 255);
        // imwrite("deconv-estimate-irls.png", dst);
        
        // // deconvolveFFT(floatImages[LEFT], dst, kernel);
        // // threshold(dst, dst, 0.0, -1, THRESH_TOZERO);
        // // threshold(dst, dst, 1.0, -1, THRESH_TRUNC);
        // // dst.convertTo(dst, CV_8U, 255);
        // // imwrite("deconv-original-fft.png", dst);
        
        // // // add border with zeros to the mask
        // // copyMakeBorder(floatImages[LEFT], floatImages[LEFT], 130, 130, 130, 130,
        // //                BORDER_CONSTANT, Scalar::all(0));
        // // // add border with zeros to the mask
        // // copyMakeBorder(masks[LEFT], masks[LEFT], 130, 130, 130, 130,
        // //                BORDER_CONSTANT, Scalar::all(0));

        // // deconvolveIRLS(floatImages[LEFT], dst, kernel, masks[LEFT]);
        // // threshold(dst, dst, 0.0, -1, THRESH_TOZERO);
        // // threshold(dst, dst, 1.0, -1, THRESH_TRUNC);
        // // dst.convertTo(dst, CV_8U, 255);
        // // imwrite("deconv-original-irls.png", dst);

        // // // end debug ----------------------------------------------------------------



        visitedLeafs = 0;

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
        
        // init queue with the top-level node IDs
        for (int i = 0; i < regionTree.topLevelNodeIds.size(); i++) {
            remainingNodes.push(regionTree.topLevelNodeIds[i]);
        }

        // create worker threads
        int nrOfWorker = nThreads - 1;
        thread threads[nrOfWorker];

        for (int id = 0; id < nrOfWorker; id++) {
            // each worker gets the deconvolveRegion method with the regionStack
            threads[id] = thread(&DepthDeblur::midLevelKernelEstimationNode, this);
        }

        midLevelKernelEstimationNode();

        // wait for all threads to finish
        for (int id = 0; id < nrOfWorker; id++) {
            threads[id].join();
        }
    }


    bool DepthDeblur::safeStackAccess(stack<int>* sharedStack, int& item) {
        // this lock guard calls lock and if the end of the scope is reached
        // it calls unlock automatically
        lock_guard<mutex> g(m);

        // now we can access the stack without collisions
        if (sharedStack->empty() != true) {
            item = sharedStack->top();
            sharedStack->pop();
            return true;
        } else {
            return false;
        }
    }


    void DepthDeblur::deconvolveRegion(const view view, const bool color) {
        // region index
        int i;

        // work as long as there is something on the stack
        while (safeStackAccess(&regionStack, i)) {
            // get mask of the disparity level
            Mat mask;
            regionTree.getMask(i, mask, view);

            // get region image (color or grayvalue as float)
            Mat region;

            if (color) {
                images[view].copyTo(region, mask);
            } else {
                floatImages[view].copyTo(region, mask);
            }

            deconvolveIRLS(region, regionDeconv[i], regionTree[i].psf, mask);
        }
    }


    void DepthDeblur::deconvolve(Mat& dst, view view, int nThreads, bool color) {
        // deconvolve in parallel
        // reset storage for deconvolved images
        regionDeconv.resize(layers);

        // set up stack with regions that have to be calculated
        // store leaf node region index
        for (int nr = 0; nr < layers; nr++) {
            regionStack.push(nr);
        }

        // create worker threads
        int nrOfWorker = nThreads - 1;
        thread threads[nrOfWorker];

        for (int id = 0; id < nrOfWorker; id++) {
            // each worker gets the deconvolveRegion method with the regionStack
            threads[id] = thread(&DepthDeblur::deconvolveRegion, this, view, color);
        }

        // let the main thread do some work too
        deconvolveRegion(view, color);

        // wait for all threads to finish
        for (int id = 0; id < nrOfWorker; id++) {
            threads[id].join();
        }

        // add all region deconvs
        for (int i = 0; i < regionDeconv.size(); i++) {
            Mat mask;
            // the index of the region in regionDeconv and regionTree are the same
            regionTree.getMask(i, mask, view);

            regionDeconv[i].copyTo(dst, mask);
        }

        // show and save the deblurred image
        //
        // threshold the result because it has large negative and positive values
        // which would result in a very grayish image
        threshold(dst, dst, 0.0, -1, THRESH_TOZERO);
        threshold(dst, dst, 1.0, -1, THRESH_TRUNC);
        dst.convertTo(dst, CV_8U, 255);

        #ifdef IMWRITE
            string filename = "deconv-" + to_string(view) + ".png";
            imwrite(filename, dst);
        #endif
    }
}
