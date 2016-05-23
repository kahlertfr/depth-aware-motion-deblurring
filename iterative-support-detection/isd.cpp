#include "utils.hpp"

#include "isd.hpp"


using namespace cv;
using namespace std;


namespace deblur {

    void isd(const Mat& src, Mat& dst, cosnt Mat& image){
        Mat kernel;
        src.copyTo(kernel);

        // iterations
        // previous kernel is used to form a partial support
        
        kernel.copyTo(dst);
    }
}