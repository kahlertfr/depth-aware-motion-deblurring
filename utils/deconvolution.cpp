#include <cmath>

#include "utils.hpp"

#include "deconvolution.hpp"


using namespace cv;
using namespace std;


namespace deblur {

    void deconvolveFFT(const Mat& src, Mat& dst, const Mat& kernel, const float weight){
        assert(src.type() == CV_32F && "works on floating point images [0,1]");
        assert(kernel.type() == CV_32F && "works with energy preserving kernel");

        // important: do not flipp the kernel
        // fill kernel with zeros to get to blurred image size
        Mat pkernel;
        copyMakeBorder(kernel, pkernel,
                       0, src.rows - kernel.rows,
                       0, src.cols - kernel.cols,
                       BORDER_CONSTANT, Scalar::all(0));

        // sobel gradients for x and y direction
        Mat sobelx = Mat::zeros(src.size(), CV_32F);
        sobelx.at<float>(0,0) = -1;
        sobelx.at<float>(0,1) = 1;

        Mat sobely = Mat::zeros(src.size(), CV_32F);
        sobely.at<float>(0,0) = -1;
        sobely.at<float>(1,0) = 1;

        // matrices for fourier transformed images
        Mat Gx, Gy, F, I;

        // DFT without padding
        dft(sobelx, Gx);
        dft(sobely, Gy);
        dft(pkernel, F);
        dft(src, I);

        // FIXME
        // weight from paper
        complex<float> we(weight, 0.0);

        // deblurred image in fourier domain
        Mat X = Mat::zeros(I.size(), CV_32FC2);

        // pointwise computation of X
        for (int col = 0; col < I.cols; col++) {
            for (int row = 0; row < I.rows; row++) {
                // complex entries at the current position
                complex<float> gx(Gx.at<Vec2f>(row, col)[0],
                                  Gx.at<Vec2f>(row, col)[1]);
                complex<float> gy(Gy.at<Vec2f>(row, col)[0],
                                  Gy.at<Vec2f>(row, col)[1]);
                complex<float> f(F.at<Vec2f>(row, col)[0],
                                 F.at<Vec2f>(row, col)[1]);
                complex<float> i(I.at<Vec2f>(row, col)[0],
                                 I.at<Vec2f>(row, col)[1]);

                complex<float> b = conj(f) * i;
                complex<float> a = conj(f) * f 
                                  + we * (conj(gx) * gx + conj(gy) * gy);

                complex<float> x = b / a;
                
                X.at<Vec2f>(row, col) = { real(x), imag(x) };
            }
        }

        // inverse dft with real output
        // does exactly the same as returning the complex matrix
        // and cropping the real channel
        Mat deconv;
        dft(X, deconv, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

        // swap slices of the result
        // because the image is shifted to the upper-left corner
        int x = deconv.cols;
        int y = deconv.rows;
        int hs1 = (kernel.cols - 1) / 2;
        int hs2 = (kernel.rows - 1) / 2;

        // create rects per image slice
        //  __________
        // |      |   |
        // |   0  | 1 |
        // |      |   |
        // |------|---|
        // |   2  | 3 | <- size of kernel
        // |______|___|
        // 
        // After cropping some image information from the top will be at the bottom.
        // This is due to the repeated pattern of the image in the frequency domain
        // so the region of the half filter size at the bottom and right edge have many
        // visual artifacts.
        // 
        // rect gets the coordinates of the top-left corner, width and height
        Mat q0(deconv, Rect(0, 0, x - hs1, y - hs2));      // Top-Left
        Mat q1(deconv, Rect(x - hs1, 0, hs1, y - hs2));    // Top-Right
        Mat q2(deconv, Rect(0, y - hs2, x - hs1, hs2));    // Bottom-Left
        Mat q3(deconv, Rect(x - hs1, y - hs2, hs1, hs2));  // Bottom-Right

        Mat deconvSwap;
        hconcat(q3, q2, deconvSwap);
        Mat tmp;
        hconcat(q1, q0, tmp);
        vconcat(deconvSwap, tmp, deconvSwap);
        deconvSwap.copyTo(deconv);
      
        deconv.copyTo(dst);
    }

    /**
     * Does the following matlab snippet:
     * Ax = Ax + we * conv2(weight_x .* conv2(x, fliplr(flipud(dxf)), 'valid'), dxf);
     * 
     * @param src     x
     * @param dst     Ax
     * @param kernel  dxf
     * @param fkernel fliplr(flipud(dxf))
     * @param weight  weight_x
     * @param we      we
     */
    void conv2add(const Mat& src, Mat& dst, const Mat& kernel, const Mat& fkernel, const Mat& weight,
                  const float we) {
        Mat tmp;

        conv2(src, tmp, fkernel, VALID);
        tmp = tmp.mul(weight);
        conv2(tmp, tmp, kernel, FULL);

        dst += tmp * we;
    }


    /**
     * Convolves the matrix A with a kernel and adds weighted convolutions 
     * with derivation filters.
     * 
     * @param src     blurred image
     * @param dst     resulting A
     * @param kernel  kernel for convolution
     * @param fkernel flipped kernel
     * @param mask    mask of region
     * @param df      filter of first and second order derivations
     * @param weights weights for derivation filters
     * @param we      weight
     */
    void computeA(const Mat& src, Mat& dst, Mat& kernel, Mat& fkernel, Mat& mask,
                  const derivationFilter& df, const weights& weights, const float we) {
        // matlab: Ax = conv2(conv2(x, fliplr(flipud(filt1)), 'same') .* mask,  filt1, 'same');
        Mat tmpAx;
        conv2(src, tmpAx, fkernel, SAME);
        tmpAx = tmpAx.mul(mask);     
        conv2(tmpAx, dst, kernel, SAME);

        // add weighted gradients to Ax
        conv2add(src, dst, df.x, df.xf, weights.x, we);
        conv2add(src, dst, df.y, df.yf, weights.y, we);
        conv2add(src, dst, df.xx, df.xxf, weights.xx, we);
        conv2add(src, dst, df.yy, df.yyf, weights.yy, we);
        conv2add(src, dst, df.xy, df.xyf, weights.xy, we);
    }


    /**
     * Deconvolution used from deconvolveIRLS
     * 
     * @param src      blurred grayvalue image
     * @param dst      latent image
     * @param kernel   energy preserving kernel
     * @param mask     mask
     * @param we       weight
     * @param df       filter of first and second order derivations
     * @param maxIt    number of iterations
     * @param weights  weights of first and second order derivatives
     */
    void deconvL2w(const Mat& src, Mat& dst, Mat& kernel, Mat& mask, const weights& weights,
                   const derivationFilter& df, const float we = 0.001, const int maxIt = 200) {

        // half filter size
        int hfsX = kernel.cols / 2;
        int hfsY = kernel.rows / 2;

        Mat zeroPaddedSrc;
        copyMakeBorder(src, zeroPaddedSrc, hfsY, hfsY, hfsX, hfsX,
                       BORDER_CONSTANT, Scalar::all(0));

        // matlab: b = conv2(x .* mask, filt1, 'same');
        Mat b;
        zeroPaddedSrc = zeroPaddedSrc.mul(mask);
        conv2(zeroPaddedSrc, b, kernel, SAME);

        // flip kernel
        Mat fkernel;
        flip(kernel, fkernel, -1);
        
        Mat x, Ax;
        // padding around image such that the border will be replicated from the pixel
        // values at the edges of the original image
        copyMakeBorder(src, x, hfsY, hfsY, hfsX, hfsX, BORDER_REPLICATE, 0);
        computeA(x, Ax, kernel, fkernel, mask, df, weights, we);

        // matlab: r = b - Ax;
        Mat r;
        r = b - Ax;
        
        Mat p;
        r.copyTo(p);

        float rhoPrev;

        for (int i = 0; i < maxIt; i++) {
            // matlab: rho = (r(:)'*r(:));
            float rho = r.dot(r);

            if (i > 0) {
                float beta = rho / rhoPrev;
                p *= beta;
                p += r;
            }

            // Ap = conv2(conv2(p, fliplr(flipud(filt1)), 'same') .* mask,  filt1,'same');
            // and so on
            Mat Ap;
            computeA(p, Ap, kernel, fkernel, mask, df, weights, we);

            // matlab:  q = Ap; alpha = rho / (p(:)'*q(:) );
            float alpha = rho / p.dot(Ap);

            x = x + (alpha * p);
            r = r - (alpha * Ap);
            
            rhoPrev = rho;
        }

        x.copyTo(dst);
    }


    void updateWeight(Mat& weight, const Mat& gradient, const Mat& boundaries, const float factor = 1) {
        // some parameters (see levin paper for details)
        float w0 = exp(-3);  // Levin: 0.1;
        float exp_a = 0.8;
        float thr_e = 0.01;  // for avoiding zero division

        for (int row = 0; row < weight.rows; row++) {
            for (int col = 0; col < weight.cols; col++) {
                if (abs(gradient.at<float>(row, col)) > thr_e) {
                    float value = abs(gradient.at<float>(row, col));
                    
                    // to suppress visual artifacts set the weight 3 times larger in boundary areas
                    if (boundaries.at<float>(row, col) != 0)
                        weight.at<float>(row, col) = factor * 3 * w0 * pow(value, exp_a - 2);
                    else
                        weight.at<float>(row, col) = factor * w0 * pow(value, exp_a - 2);
                } else {
                    weight.at<float>(row, col) = factor * w0 * pow(thr_e, exp_a - 2);
                }
            }
        }
    }


    /**
     * The spatial deconvolution algorithm for one channel.
     * 
     * @param src    blurred grayvalue image
     * @param dst    latent image
     * @param kernel energy preserving kernel
     * @param we     weight
     * @param maxIt  number of iterations
     */
    void deconvolveChannelIRLS(const Mat& src, Mat& dst, Mat& kernel, const Mat& regionMask,
                               const float we, const int maxIt) {
        assert(src.type() == CV_32F && "works on floating point images [0,1]");

        // half filter size
        int hfsX = kernel.cols / 2;
        int hfsY = kernel.rows / 2;

        // new image dimensions = old + filter size
        int m = 2 * hfsX + src.cols;
        int n = 2 * hfsY + src.rows;

        // create mask with m columns and n rows with ones except for a boundary
        // of the half filter size in all directions
        Mat tmpMask, mask;

        if (regionMask.empty()) {
            // mask with ones of image size
            tmpMask = Mat::ones(src.size(), CV_32F);
        } else {
            // mask with range [0, 1] needed
            // it will be converted to float and set to 0 and 1
            regionMask.convertTo(tmpMask, CV_32F);
        }

        // add border with zeros to the mask
        copyMakeBorder(tmpMask, mask, hfsY, hfsY, hfsX, hfsX,
                       BORDER_CONSTANT, Scalar::all(0));


        // create mask for region boundaries
        // because for pixels with their distant to the region boundaries smaller
        // than the kernel size the weight is set 3 times larger
        Mat tmp, boundaries;
        Mat structElement = Mat::ones(kernel.rows * 2, kernel.cols * 2, CV_32F);
        erode(mask, tmp, structElement);

        boundaries = mask - tmp;

        // get first and second order derivations in x and y direction as sobel filter
        derivationFilter df;
        sobelDerivations(df);

        // weights for the derivation filter
        weights weights;
        weights.x = Mat::ones(n, m - 1, CV_32F);
        weights.y = Mat::ones(n - 1, m, CV_32F);
        weights.xx = Mat::ones(n, m - 2, CV_32F);
        weights.yy = Mat::ones(n - 2, m, CV_32F);
        weights.xy = Mat::ones(n - 1, m - 1, CV_32F);

        // first deconvolution of the src image
        Mat x;
        deconvL2w(src, x, kernel, mask, weights, df, we, maxIt);

        for (int i = 0; i < 2; i++) {
            // compute first and second order gradients
            Mat dx, dy, dxx, dyy, dxy;
            conv2(x, dx, df.xf, VALID);
            conv2(x, dy, df.yf, VALID);
            conv2(x, dxx, df.xxf, VALID);
            conv2(x, dyy, df.yyf, VALID);
            conv2(x, dxy, df.xyf, VALID);

            updateWeight(weights.x, dx, boundaries);
            updateWeight(weights.y, dy, boundaries);
            updateWeight(weights.xx, dxx, boundaries, 0.25);
            updateWeight(weights.yy, dyy, boundaries, 0.25);
            updateWeight(weights.xy, dxy, boundaries, 0.25);

            deconvL2w(src, x, kernel, mask, weights, df, we, maxIt);
        }

        // crop result
        Mat cropped;
        x(Rect(
            hfsX,
            hfsY,
            src.cols,
            src.rows
        )).copyTo(dst);
    }


    void deconvolveIRLS(const Mat& src, Mat& dst, Mat& kernel, const Mat& regionMask,
                        const float we, const int maxIt) {
        assert(kernel.type() == CV_32F && "works with energy preserving kernel");
        assert((src.type() == CV_32FC3 || src.type() == CV_32F) && "works with energy preserving kernel");

        assert(kernel.rows % 2 == 1 && "odd kernel expected");
        assert(kernel.cols % 2 == 1 && "odd kernel expected");


        if (src.channels() == 3) {
            // deconvolve each channel of a color image
            vector<Mat> channels(3), tmp(3);
            split(src, channels);


            for (int i = 0; i < channels.size(); i++) {
                deconvolveChannelIRLS(channels[i], tmp[i], kernel, regionMask,
                                      we, maxIt);
            }

            merge(tmp, dst);

        } else if (src.channels() == 1) {
            // deconvolve gray value image
            deconvolveChannelIRLS(src, dst, kernel, regionMask, we, maxIt);

        } else {
            throw runtime_error("Cannot convolve this image type");
        } 
    }
}