#include "edge_map.hpp"
#include "coherence_filter.hpp"
#include "utils.hpp" // convertFloatToUchar

using namespace cv;
using namespace std;
using namespace deblur;


namespace DepthAwareDeblurring {

    void gradientMaps(const Mat& image, array<Mat, 2>& gradients) {
        assert(image.type() == CV_8U && "Input image must be grayscaled");

        Mat bilateral;
        bilateralFilter(image, bilateral,
                        5,     // diamter / support size in pxiel
                        0.5,   // range (color) sigma
                        2.0);  // spatial sigma

        // #ifndef NDEBUG
        //     imshow("bilateral", bilateral);
        // #endif

        Mat shock;
        coherenceFilter(bilateral, shock);

        // #ifndef NDEBUG
        //     imshow("shock", shock);
        // #endif
        
        const int delta = 0;
        const int ddepth = CV_32F;
        const int ksize = 3;
        const int scale = 1;

        Sobel(shock, gradients[0], ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
        Sobel(shock, gradients[1], ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

        // #ifndef NDEBUG
        //     Mat normalized;
        //     convertFloatToUchar(normalized, gradients[0]);
        //     imshow("x-gradient", normalized);
        // #endif
    }

}