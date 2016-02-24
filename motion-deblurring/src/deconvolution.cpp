#include <cmath>

#include "utils.hpp"

#include "deconvolution.hpp"


using namespace cv;
using namespace std;


namespace deblur {

    void deconvolveFFT(Mat src, Mat& dst, Mat& kernel, const float weight){
        assert(src.type() == CV_8U && "works on gray value images");
        assert(kernel.type() == CV_32F && "works with energy preserving kernel");

        // convert input image to floats and normalize it to [0,1]
        src.convertTo(src, CV_32F);
        src /= 255.0;

        // fill kernel with zeros to get to blurred image size
        Mat pkernel;
        copyMakeBorder(kernel, pkernel, 0,
                       src.rows - kernel.rows, 0,
                       src.cols - kernel.cols,
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

                complex<float> x = b / real(a);
                
                X.at<Vec2f>(row, col) = { real(x), imag(x) };
            }
        }


        // // inverse dft with complex output
        // dft(X, deconv, DFT_INVERSE);
        // showComplexImage("result complex", deconv, false);

        // inverse dft with real output
        Mat deconv;
        dft(X, deconv, DFT_INVERSE | DFT_REAL_OUTPUT);


        // swap slices of the result
        // because the image is shifted to the upper-left corner (why??)
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
        // |   2  | 3 |
        // |______|___|
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


        // show and save the deblurred image
        //
        // threshold the result because it has large negative and positive values
        // which would result in a very grayish image
        threshold(deconv, deconv, 0.0, -1, THRESH_TOZERO);

        convertFloatToUchar(deconv, dst);
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
    void computeA(Mat& src, Mat& dst, Mat& kernel, Mat& fkernel, Mat& mask,
                  const derivationFilter& df, const weights& weights, const float we) {
        // matlab: Ax = conv2(conv2(x, fliplr(flipud(filt1)), 'same') .* mask,  filt1, 'same');
        Mat tmpAx;
        filter2D(src, tmpAx, -1, fkernel);
        tmpAx = tmpAx.mul(mask);     
        filter2D(tmpAx, dst, -1, kernel);

        double min, max;
        minMaxLoc(dst, &min, &max);
        cout << "Ax: " << min << " " << max << dst.size() << endl;

        // add weighted gradients to Ax
        Mat tmp, res;

        // matlab: Ax = Ax + we * conv2(weight_x .* conv2(x, fliplr(flipud(dxf)), 'valid'), dxf);
        filter2D(src, tmp, -1, df.xf);
        // FIXME: have to cut out correct image part
        //        maybe use matrices as weights -> no need for cropping
        tmp = tmp.mul(weights.x);
        filter2D(tmp, res, -1, df.x);
        cout << res.size() << endl;
        res *= we;
        dst += res;

        // matlab: Ax = Ax + we * conv2(weight_y .* conv2(x, fliplr(flipud(dyf)), 'valid'), dyf);
        filter2D(src, tmp, -1, df.yf);
        tmp = tmp.mul(weights.y);
        filter2D(tmp, res, -1, df.y);
        res *= we;
        dst += res;

        // matlab: Ax = Ax + we * conv2(weight_xx .* conv2(x, fliplr(flipud(dxxf)), 'valid'), dxxf);
        filter2D(src, tmp, -1, df.xxf);
        tmp = tmp.mul(weights.xx);
        filter2D(tmp, res, -1, df.xx);
        res *= we;
        dst += res;

        // matlab: Ax = Ax + we * conv2(weight_yy .* conv2(x, fliplr(flipud(dyyf)), 'valid'), dyyf);
        filter2D(src, tmp, -1, df.yyf);
        tmp = tmp.mul(weights.yy);
        filter2D(tmp, res, -1, df.yy);
        res *= we;
        dst += res;

        // matlab: Ax = Ax + we * conv2(weight_xy .* conv2(x, fliplr(flipud(dxyf)), 'valid'), dxyf);
        filter2D(src, tmp, -1, df.xyf);
        tmp = tmp.mul(weights.xy);
        filter2D(tmp, res, -1, df.xy);
        res *= we;
        dst += res;
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
    void deconvL2w(Mat& src, Mat& dst, Mat& kernel, Mat& mask, const weights& weights,
                   const derivationFilter& df, const float we = 0.001, const int maxIt = 200) {

        // half filter size
        int hfsX = kernel.cols / 2;
        int hfsY = kernel.rows / 2;

        Mat zeroPaddedSrc;
        copyMakeBorder(src, zeroPaddedSrc, hfsY, hfsY, hfsX, hfsX,
                       BORDER_CONSTANT, Scalar::all(0));

        // matlab: b = conv2(x .* mask, filt1, 'same');
        Mat b;
        filter2D(zeroPaddedSrc, b, -1, kernel);
        showFloat("b", b, true);


        double min; double max;
        minMaxLoc(src, &min, &max);
        // cout << "src: " << min << " " << max << endl;
        minMaxLoc(kernel, &min, &max);
        // cout << "kernel: " << min << " " << max << endl;
        minMaxLoc(b, &min, &max);
        // cout << endl << "b: " << min << " " << max << endl;

        // flip kernel
        Mat fkernel;
        flip(kernel, fkernel, -1);
        
        Mat x, Ax;
        // padding around image such that the border will be replicated from the pixel
        // values at the edges of the original image
        copyMakeBorder(src, x, hfsY, hfsY, hfsX, hfsX, BORDER_REPLICATE, 0);
        computeA(x, Ax, kernel, fkernel, mask, df, weights, we);

        showFloat("Ax", Ax, true);

        minMaxLoc(Ax, &min, &max);
        cout << "Ax: " << min << " " << max << endl;

        // // matlab: r = b - Ax;
        // Mat r;
        // r = b - Ax;

        // minMaxLoc(r, &min, &max);
        // // cout << min << " " << max << endl;
        // showFloat("r", r, true);
        
        // Mat p;
        // r.copyTo(p);

        // float rhoPrev;

        // for (int i = 0; i < maxIt; i++) {
        //     // matlab: rho = (r(:)'*r(:));
        //     float rho = r.dot(r);
        //     // cout << "rho: " << rho << endl;


        //     if (i > 0) {
        //         float beta = rho / rhoPrev;
        //         p *= beta;
        //         p += r;
        //     }

        //     // Ap = conv2(conv2(p, fliplr(flipud(filt1)), 'same') .* mask,  filt1,'same');
        //     // and so on
        //     Mat Ap;
        //     computeA(p, Ap, kernel, fkernel, mask, df, weights, we);

        //     showFloat("Ap", Ap, true);
        //     minMaxLoc(Ap, &min, &max);
        //     // cout << "Ap: " << min << " " << max << endl;

        //     // matlab:  q = Ap; alpha = rho / (p(:)'*q(:) );
        //     float alpha = rho / p.dot(Ap);

        //     // cout << "alpha: " << alpha << endl;

        //     x = x + (alpha * p);
        //     minMaxLoc(x, &min, &max);
        //     // cout << "x: " << min << " " << max << endl;
        //     showFloat("x-new", x, true);
        //     r = r - (alpha * Ap);

        //     showFloat("r2", r, true);
        //     minMaxLoc(r, &min, &max);
        //     // cout << "r2: " << min << " " << max << endl;
            

        //     rhoPrev = rho;
        // }

        x.copyTo(dst);
    }


    void deconvolveIRLS(Mat src, Mat& dst, Mat& kernel, const float we, const int maxIt) {

        assert(src.type() == CV_8U && "works on gray value images");
        assert(kernel.type() == CV_32F && "works with energy preserving kernel");

        assert(kernel.rows % 2 == 1 && "odd kernel expected");
        assert(kernel.cols % 2 == 1 && "odd kernel expected");

        // convert input image to floats and normalize it to [0,1]
        src.convertTo(src, CV_32F);
        src /= 255.0;

        // half filter size
        int hfsX = kernel.cols / 2;
        int hfsY = kernel.rows / 2;

        // new image dimensions = old + filter size
        int m = 2 * hfsX + src.cols;
        int n = 2 * hfsY + src.rows;

        // create mask with m columns and n rows with ones except for a boundary
        // of the half filter size in all directions
        // 
        // mask with ones of image size
        Mat tmpMask = Mat::ones(src.size(), CV_32F);

        // add border with zeros to the mask
        Mat mask;
        copyMakeBorder(tmpMask, mask, hfsY, hfsY, hfsX, hfsX,
                       BORDER_CONSTANT, Scalar::all(0));

        // // add border with zeros to the image
        // Mat paddedSrc;
        // copyMakeBorder(src, paddedSrc, hfsY, hfsY, hfsX, hfsX,
        //                BORDER_CONSTANT, Scalar::all(0));


        // get first and second order derivations in x and y direction as sobel filter
        derivationFilter df;
        sobelDerivations(df);


        // weights for the derivation filter
        // FIXME: different sizes in levin
        weights weights;
        weights.x = Mat::ones(n,m, CV_32F);
        weights.y = Mat::ones(n,m, CV_32F);
        weights.xx = Mat::ones(n,m, CV_32F);
        weights.yy = Mat::ones(n,m, CV_32F);
        weights.xy = Mat::ones(n,m, CV_32F);

        // first deconvolution of the src image
        Mat x;
        deconvL2w(src, x, kernel, mask, weights, df, we, maxIt);

        showFloat("intermediate result x", x, true);

        // // some parameters (see levin paper for details)
        // float w0 = 0.1;
        // float exp_a = 0.8;
        // float thr_e = 0.01;

        // for (int i = 0; i < 2; i++) {
        //     Mat dx, dy, dxx, dyy, dxy;
        //     filter2D(x, dx, -1, df.xf);
        //     filter2D(x, dy, -1, df.yf);
        //     filter2D(x, dxx, -1, df.xxf);
        //     filter2D(x, dyy, -1, df.yyf);
        //     filter2D(x, dxy, -1, df.xyf);

        //     // set new weights
        //     for (int row = 0; row < (weights.x).rows; row++) {
        //         for (int col = 0; col < (weights.x).cols; col++) {
        //             float value;

        //             if (abs(dx.at<float>(row, col)) > thr_e)
        //                 value = abs(dx.at<float>(row, col));
        //             else
        //                 value = thr_e;
        //             weights.x.at<float>(row, col) = w0 * pow(value, exp_a - 2);

        //             if (abs(dy.at<float>(row, col)) > thr_e)
        //                 value = abs(dy.at<float>(row, col));
        //             else
        //                 value = thr_e;
        //             weights.y.at<float>(row, col) = w0 * pow(value, exp_a - 2);

        //             if (abs(dxx.at<float>(row, col)) > thr_e)
        //                 value = abs(dxx.at<float>(row, col));
        //             else
        //                 value = thr_e;
        //             weights.xx.at<float>(row, col) = 0.25 * pow(value, exp_a - 2);

        //             if (abs(dyy.at<float>(row, col)) > thr_e)
        //                 value = abs(dyy.at<float>(row, col));
        //             else
        //                 value = thr_e;
        //             weights.yy.at<float>(row, col) = 0.25 * pow(value, exp_a - 2);

        //             if (abs(dxy.at<float>(row, col)) > thr_e)
        //                 value = abs(dxy.at<float>(row, col));
        //             else
        //                 value = thr_e;
        //             weights.xy.at<float>(row, col) = 0.25 * pow(value, exp_a - 2);
        //         }
        //     }

        //     deconvL2w(src, x, kernel, mask, weights, df, we, maxIt);
        // }

        // showFloat("result", x);

        // // crop result and convert to uchar
        // // TODO
    }
}