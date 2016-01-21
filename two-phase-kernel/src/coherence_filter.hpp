/***********************************************************************
 * Author:       WinCoder@qq.com & Franziska Krüger
 * Reference:    http://blog.csdn.net/bluecol/article/details/49924739
 * Requirements: OpenCV 3
 *
 * Description:
 * ------------
 * inspired by
 *  Joachim Weickert "Coherence-Enhancing Shock Filters"
 *  http://www.mia.uni-saarland.de/Publications/weickert-dagm03.pdf
 *
 * Example:
 *    Mat dst = CoherenceFilter(I,11,11,0.5,4);
 *   imshow("shock filter",dst);
 * 
 ***********************************************************************
 */

#ifndef COHERENCE_FILTER_H
#define COHERENCE_FILTER_H

#include <opencv2/opencv.hpp>


/**
* Shock filters an image.
* 
* @param img        input image ranging value from 0 to 255.
* @param sigma      sobel kernel size.
* @param str_sigma  neighborhood size,see detail in reference[2]
* @param blend      blending coefficient.default value 0.5.
* @param iter       number of iteration.
* @param shockImage filtered image
*/
inline void coherenceFilter(const cv::Mat& img, const int sigma, const int str_sigma,
                     const float blend, const int iter, cv::Mat& shockImage) {

    img.copyTo(shockImage);

    int height = shockImage.rows;
    int width  = shockImage.cols;

    for(int i = 0;i <iter; i++) {
        cv::Mat gray;
        
        if (img.type() != CV_8U) {
            cv::cvtColor(shockImage, gray, cv::COLOR_BGR2GRAY);
        } else {
            img.copyTo(gray);
        }

        cv::Mat eigen;
        cv::cornerEigenValsAndVecs(gray, eigen, str_sigma, 3);

        std::vector<cv::Mat> vec;
        cv::split(eigen,vec);

        cv::Mat x,y;
        x = vec[2];
        y = vec[3];

        cv::Mat gxx,gxy,gyy;
        cv::Sobel(gray, gxx, CV_32F, 2, 0, sigma);
        cv::Sobel(gray, gxy, CV_32F, 1, 1, sigma);
        cv::Sobel(gray, gyy, CV_32F, 0, 2, sigma);

        cv::Mat ero;
        cv::Mat dil;
        cv::erode(shockImage, ero, cv::Mat());
        cv::dilate(shockImage, dil, cv::Mat());

        cv::Mat img1 = ero;
        for(int nY = 0; nY < height; nY++) {
            for(int nX = 0; nX < width; nX++) {
                if(x.at<float>(nY, nX) * x.at<float>(nY, nX) * gxx.at<float>(nY, nX)
                    + 2 * x.at<float>(nY, nX) * y.at<float>(nY, nX)* gxy.at<float>(nY, nX)
                    + y.at<float>(nY, nX) * y.at<float>(nY, nX) * gyy.at<float>(nY, nX) < 0) {

                        if (img.type() != CV_8U)
                            img1.at<cv::Vec3b>(nY,nX) = dil.at<cv::Vec3b>(nY,nX);
                        else
                            img1.at<uchar>(nY,nX) = dil.at<uchar>(nY,nX);
                }
            }
        }

        shockImage = shockImage * (1.0 - blend) + img1 * blend;
    }
}

#endif