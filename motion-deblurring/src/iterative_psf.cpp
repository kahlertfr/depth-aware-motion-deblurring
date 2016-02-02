#include <iostream>                     // cout, cerr, endl
#include <opencv2/highgui/highgui.hpp>  // imread, imshow, imwrite
#include <queue>                        // FIFO queue

#include "utils.hpp"
#include "region_tree.hpp"
#include "edge_map.hpp"
// #include "two_phase_psf_estimation.hpp"

#include "iterative_psf.hpp"


using namespace cv;
using namespace std;
using namespace deblur;


namespace DepthAwareDeblurring {

    IterativePSF::IterativePSF(const Mat& disparityMapM, const Mat& disparityMapR,
                               const int layers, Mat* imageLeft, Mat* imageRight,
                               const int maxTopLevelNodes, const int width)
                               : psfWidth(width)
    {
        // create a region tree
        regionTree.create(disparityMapM, disparityMapR, layers,
                          imageLeft, imageRight, maxTopLevelNodes);
    }


    void IterativePSF::toplevelKernelEstimation(const string filePrefix) {
        // go through each top-level node
        for (int i = 0; i < regionTree.topLevelNodeIds.size(); i++) {
            int id = regionTree.topLevelNodeIds[i];

            // get an image of the top-level region
            Mat region, mask;
            regionTree.getRegionImage(id, region, mask, RegionTree::LEFT);


            // // edge tapering to remove high frequencies at the border of the region
            // Mat taperedRegion;
            // regionTree.edgeTaper(taperedRegion, region, mask, *(regionTree.images[LEFT]));

            // // use this images for example for the .exe of the two-phase kernel estimation
            // string name = "tapered" + to_string(i) + ".jpg";
            // imwrite(name, taperedRegion);
            

            // // calculate PSF with two-phase kernel estimation (deferred)
            // TwoPhaseKernelEstimation::estimateKernel(regionTree[id].psf[0], region, psfWidth, mask);
            // 
            // because this algorithm won't work
            // load the kernel images which should be named left/right-kerneli.png
            // they should be located in the folder where this algorithm is started
            string filename = filePrefix + "-kernel" + to_string(i) + ".png";
            regionTree[id].psf = imread(filename, 1);

            if (!regionTree[id].psf.data) {
                throw runtime_error("Can not load kernel!");
            }
        }
    }


    void IterativePSF::jointPSFEstimation(const Mat& maskLeft, const Mat& maskRight, Mat& psf) {

        // compute a gradient image with salient edge
        array<Mat,2> salientEdgesLeft, salientEdgesRight;
        thresholdGradients(enhancedGradsLeft, salientEdgesLeft, psfWidth, maskLeft);
        thresholdGradients(enhancedGradsRight, salientEdgesRight, psfWidth, maskRight);

        #ifndef NDEBUG
            // display salient edges
            Mat _display;
            convertFloatToUchar(_display, salientEdgesLeft[0]);
            imshow("salient edges x left", _display);
        #endif

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
        // energy distribution
        // 
        // here: F(∂_x S_i) = xSr / xSm
        //       F(∂_x B)   = xB
        //       F(∂_y S_i) = xSr / xSm
        //       F(∂_y B)   = yB
        
        // the result are stored as 2 channel matrices: Re(FFT(I)), Im(FFT(I))
        Mat xSr, xSm, ySr, ySm;  // fourier transform of region gradients
        Mat xBr, xBm, yBr, yBm;  // fourier transform of blurred images
        
        FFT(salientEdgesLeft[0], xSm);
        FFT(salientEdgesLeft[1], ySm);
        FFT(salientEdgesRight[0], xSr);
        FFT(salientEdgesRight[1], ySr);
        FFT(gradsLeft[0], xBm);
        FFT(gradsLeft[1], yBm);
        FFT(gradsRight[0], xBr);
        FFT(gradsRight[1], yBr);

        Mat kernelFourier = Mat::zeros(xSm.size(), xSm.type());

        // // delta function as one white pixel in black image
        // Mat deltaFloat = Mat::zeros(xSm.size(), CV_32F);
        // deltaFloat.at<float>(xSm.rows / 2, xSm.cols / 2) = 1;
        // Mat delta;
        // FFT(deltaFloat, delta);

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

                // complex<float> d(delta.at<Vec2f>(y, x)[0], delta.at<Vec2f>(y, x)[1]);

                complex<float> weight(0.50, 0.0);

                // kernel entry in the Fourier space
                complex<float> k = ( (conj(xsr) * xbr + conj(xsm) * xbm) +
                                     (conj(ysr) * ybr + conj(ysm) * ybm) ) /
                                     ( (conj(xsr) * xsr + conj(ysr) * ysr) + 
                                       (conj(xsm) * xsm + conj(ysm) * ysm) + weight);
                
                kernelFourier.at<Vec2f>(y, x) = { real(k), imag(k) };
            }
        }

        // compute inverse FFT of the kernel in frequency domain
        Mat kernelResult;
        dft(kernelFourier, kernelResult, DFT_INVERSE);

        // the real value result of the kernel is stored in the first channel
        vector<Mat> channels(2);
        split(kernelResult, channels);

        channels[0].copyTo(psf);
    }


    void IterativePSF::gradientComputation() {
        // compute enhanced gradients for regions
        gradientMaps(*(regionTree.images[RegionTree::LEFT]), enhancedGradsLeft);
        gradientMaps(*(regionTree.images[RegionTree::LEFT]), enhancedGradsRight);

        // compute simple gradients for blurred images
        const int delta = 0;
        const int ddepth = CV_32F;
        const int ksize = 3;
        const int scale = 1;

        // gradients of left image
        Sobel(*(regionTree.images[RegionTree::LEFT]), gradsLeft[0],
              ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
        Sobel(*(regionTree.images[RegionTree::LEFT]), gradsLeft[1],
              ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

        // gradients of right image
        Sobel(*(regionTree.images[RegionTree::RIGHT]), gradsRight[0],
              ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
        Sobel(*(regionTree.images[RegionTree::RIGHT]), gradsRight[1],
              ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

        // TODO: normalize gradients
    }


    void IterativePSF::midLevelKernelEstimation() {
        // we can compute the enhanced gradients for each blurred image ones
        // and later cut off the necessary regions ∇S_i
        gradientComputation();

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


        // while(!remainingNodes.empty()) {
        //     // pop id of current node from the front of the queue
        //     int id = remainingNodes.front();
        //     remainingNodes.pop();

        //     // do PSF computation for a middle node with its children
        //     // (leaf nodes doesn't have any children)
        //     if (regionTree[id].children.first != -1 && regionTree[id].children.second != -1) {
        //         // add children ids to the back of the queue
        //         remainingNodes.push(regionTree[id].children.first);
        //         remainingNodes.push(regionTree[id].children.second);

        //         // PSF estimation for each children
                    
                Mat maskM, maskR;
                regionTree.getMasks(44, maskM, maskR);

                Mat psf;
                jointPSFEstimation(maskM, maskR, psf);

                // save PSFs
                #ifndef NDEBUG
                    // print kernel
                    Mat kernelUchar;
                    convertFloatToUchar(kernelUchar, psf);
                    swapQuadrants(kernelUchar);
                    imshow("node psf", kernelUchar);
                #endif
        //     }
        // }
    }
}