#include "utils.hpp"

#include "deconvolution.hpp"


using namespace cv;
using namespace std;


namespace deblur {

    void deconvolveFFT(Mat src, Mat& dst, Mat& kernel, float weight){
        assert(src.type() == CV_8U && "works on gray value images");
        assert(kernel.type() == CV_32F && "works with energy preserving kernel");

        // convert input images to floats
        src.convertTo(src, CV_32F);

        // normalize blurred image to [0,1]
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


    void deconvolveIRLS(Mat src, Mat& dst, Mat& kernel) {
        assert(src.type() == CV_8U && "works on gray value images");
        assert(kernel.type() == CV_32F && "works with energy preserving kernel");

        // half filter size
        // FIXME: kernel size has to be odd
        int hfsX = kernel.cols / 2;
        int hfsY = kernel.rows / 2;

        // new image dimensions = old + filter size
        int m = 2 * hfsX + src.cols;
        int n = 2 * hfsY + src.rows;

        // create mask with m columns and n rows with ones except for a boundary
        // of the half filter size in all directions
        // 
        // mask with ones of image size
        Mat tmpMask = Mat::ones(src.size(), CV_32FC2);

        // add boundary with zeros
        Mat mask;
        copyMakeBorder(tmpMask, mask, 0, n, 0, m,
                       BORDER_CONSTANT, Scalar::all(0));

        showFloat("mask", mask);
    }
}