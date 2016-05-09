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
     * First and second order derivations in x and y direction
     * as sobel filter
     * 
     * @param df   filter of first and second order derivations
     */
    void sobelDerivations(derivationFilter& df) {
        // sobel gradients of first order for x and y direction
        df.x = Mat::zeros(1, 2, CV_32F);
        (df.x).at<float>(0,0) = 1;
        (df.x).at<float>(0,1) = -1;

        df.y = Mat::zeros(2, 1, CV_32F);
        (df.y).at<float>(0,0) = 1;
        (df.y).at<float>(1,0) = -1;

        // sobel gradients of second order for x and y direction
        df.xx = Mat::zeros(1, 3, CV_32F);
        (df.xx).at<float>(0,0) = -1;
        (df.xx).at<float>(0,1) = 2;
        (df.xx).at<float>(0,2) = -1;

        df.yy = Mat::zeros(3, 1, CV_32F);
        (df.yy).at<float>(0,0) = -1;
        (df.yy).at<float>(1,0) = 2;
        (df.yy).at<float>(2,0) = -1;

        df.xy = Mat::zeros(2, 2, CV_32F);
        (df.xy).at<float>(0,0) = -1;
        (df.xy).at<float>(0,1) = 1;
        (df.xy).at<float>(1,0) = 1;
        (df.xy).at<float>(1,1) = -1;

        // flip all gradients
        flip(df.x, df.xf, -1);
        flip(df.y, df.yf, -1);
        flip(df.xx, df.xxf, -1);
        flip(df.yy, df.yyf, -1);
        flip(df.xy, df.xyf, -1);
    }


    enum ConvShape {
        FULL,
        SAME,
        VALID,
    };

    /**
     * Works like matlab conv2
     *
     * The shape parameter controls the result matrix size:
     * 
     *  - FULL  Returns the full two-dimensional convolution
     *  - SAME  Returns the central part of the convolution of the same size as A
     *  - VALID Returns only those parts of the convolution that are computed without
     *          the zero-padded edges
     */
    void conv2(const Mat& src, Mat& dst, const Mat& kernel, ConvShape shape = FULL) {
        int padSizeX = kernel.cols - 1;
        int padSizeY = kernel.rows - 1;

        Mat zeroPadded;
        copyMakeBorder(src, zeroPadded, padSizeY, padSizeY, padSizeX, padSizeX,
                       BORDER_CONSTANT, Scalar::all(0));
        
        Point anchor(0, 0);

        // // openCV is doing a correlation in their filter2D function ...
        // Mat kernel;
        // flip(kernel, fkernel, -1);

        Mat tmp;
        filter2D(zeroPadded, tmp, -1, kernel, anchor);

        // src =
        //     1 2 3 4
        //     1 2 3 4
        //     1 2 3 4
        // 
        // zeroPadded =
        //     0 0 1 2 3 4 0 0
        //     0 0 1 2 3 4 0 0
        //     0 0 1 2 3 4 0 0
        // 
        // kernel =
        //     0.5 0 0.5
        // 
        // tmp =
        //     0.5 1 2 3 1.5 2 0 2
        //     0.5 1 2 3 1.5 2 0 2
        //     0.5 1 2 3 1.5 2 0 2
        //     |<----------->|      full
        //         |<---->|         same
        //           |-|            valid
        // 
        // the last column is complete rubbish, because openCV's
        // filter2D uses reflected borders (101) by default.
        
        // crop padding
        Mat cropped;

        // variables cannot be declared in case statements
        int width  = -1;
        int height = -1;

        switch(shape) {
            case FULL:
                cropped = tmp(Rect(0, 0,
                                   tmp.cols - padSizeX,
                                   tmp.rows - padSizeY));
                break;

            case SAME:
                cropped = tmp(Rect((tmp.cols - padSizeX - src.cols + 1) / 2,  // +1 for ceil
                                   (tmp.rows - padSizeY - src.rows + 1) / 2,  // +1 for ceil
                                   src.cols,
                                   src.rows));
                break;

            case VALID:
                width  = src.cols - kernel.cols + 1;
                height = src.rows - kernel.rows + 1;
                cropped = tmp(Rect((tmp.cols - padSizeX - width) / 2,
                                   (tmp.rows - padSizeY - height) / 2,
                                   width,
                                   height));
                break;

            default:
                throw runtime_error("Invalid shape");
                break;
        }

        cropped.copyTo(dst);
    }


    /**
     * Just for debugging.
     *
     */
    void test() {
        Mat I = (Mat_<float>(3,4) << 1,2,3,4,1,2,3,4,1,2,3,4);
        cout << endl << "I: " << endl;
        for (int row = 0; row < I.rows; row++) {
            for (int col = 0; col < I.cols; col++) {
                cout << " " << I.at<float>(row, col);
            }
            cout << endl;
        }

        Mat k = (Mat_<float>(1,3) << 0.3, 0, 0.7);
        // Mat k = (Mat_<float>(1,4) << 0.5, 0, 0, 0.5);
        cout << endl << "k: " << endl;
        for (int row = 0; row < k.rows; row++) {
            for (int col = 0; col < k.cols; col++) {
                cout << " " << k.at<float>(row, col);
            }
            cout << endl;
        }

        Mat normal;
        filter2D(I, normal, -1, k);
        cout << endl << "normal (reflected border): " << endl;
        for (int row = 0; row < normal.rows; row++) {
            for (int col = 0; col < normal.cols; col++) {
                cout << " " << normal.at<float>(row, col);
            }
            cout << endl;
        }

        Mat full, same, valid;

        conv2(I, full, k, FULL);
        cout << endl << "full: " << endl;
        for (int row = 0; row < full.rows; row++) {
            for (int col = 0; col < full.cols; col++) {
                cout << " " << full.at<float>(row, col);
            }
            cout << endl;
        }

        conv2(I, same, k, SAME);
        cout << endl << "same: " << endl;
        for (int row = 0; row < same.rows; row++) {
            for (int col = 0; col < same.cols; col++) {
                cout << " " << same.at<float>(row, col);
            }
            cout << endl;
        }

        conv2(I, valid, k, VALID);
        cout << endl << "valid: " << endl;
        for (int row = 0; row < valid.rows; row++) {
            for (int col = 0; col < valid.cols; col++) {
                cout << " " << valid.at<float>(row, col);
            }
            cout << endl;
        }

    }


    void conv2add(const Mat& src, Mat& dst, const Mat& kernel, const Mat& fkernel, const Mat& weight,
                  const float we) {
        Mat tmp;

        // matlab: Ax = Ax + we * conv2(weight_x .* conv2(x, fliplr(flipud(dxf)), 'valid'), dxf);
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
            // because region here is a CV_8U with 0 and 255 values
            // it will be converted to float and set to 0 and 1
            regionMask.convertTo(tmpMask, CV_32F);
            tmpMask /= 255;
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