#include "utils.hpp"

#include "isd.hpp"


using namespace cv;
using namespace std;


namespace deblur {

    void isd(const Mat& src, Mat& dst){
        src.copyTo(dst);
    }
}