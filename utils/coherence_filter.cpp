#include "coherence_filter.hpp"

using namespace std;
using namespace cv;


namespace deblur {

    void coherenceFilter(const Mat& img, Mat& shockImage,
                         const int sigma, const int str_sigma,
                         const float blend, const int iter) {
        
        assert(img.type() == CV_32F && "works on float images");

        // FIXME: color isn't working
        // assert(img.type() == CV_32F || img.type() == CV_32FC3 && "works on float images");

        img.copyTo(shockImage);

        int height = shockImage.rows;
        int width  = shockImage.cols;

        for(int i = 0;i <iter; i++) {
            Mat gray;
            if (shockImage.channels() == 3) {
                cvtColor(shockImage, gray, COLOR_BGR2GRAY);
            } else {
                gray = shockImage;
            }

            Mat eigen;
            cornerEigenValsAndVecs(gray, eigen, str_sigma, 3);

            vector<Mat> vec;
            split(eigen,vec);

            Mat x,y;
            x = vec[2];
            y = vec[3];

            Mat gxx,gxy,gyy;
            Sobel(shockImage, gxx, CV_32F, 2, 0, sigma);
            Sobel(shockImage, gxy, CV_32F, 1, 1, sigma);
            Sobel(shockImage, gyy, CV_32F, 0, 2, sigma);

            Mat ero;
            Mat dil;
            erode(shockImage, ero, Mat());
            dilate(shockImage, dil, Mat());

            Mat img1 = ero;
            for(int nY = 0; nY < height; nY++) {
                for(int nX = 0; nX < width; nX++) {
                    if(x.at<float>(nY, nX) * x.at<float>(nY, nX) * gxx.at<float>(nY, nX)
                        + 2 * x.at<float>(nY, nX) * y.at<float>(nY, nX)* gxy.at<float>(nY, nX)
                        + y.at<float>(nY, nX) * y.at<float>(nY, nX) * gyy.at<float>(nY, nX) < 0) {

                            if (img.channels() == 3)
                                img1.at<Vec3f>(nY,nX) = dil.at<Vec3f>(nY,nX);
                            else
                                img1.at<float>(nY,nX) = dil.at<float>(nY,nX);
                    }
                }
            }

            shockImage = shockImage * (1.0 - blend) + img1 * blend;
        }
    }

}
