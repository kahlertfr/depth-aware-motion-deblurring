/***********************************************************************
 * Author:       Franziska Krüger
 * Using:        argtable3 - http://www.argtable.org/
 *
 * Description:
 * ------------
 * Start twp phase kernel estimation algorithm from command lin.
 *
 ************************************************************************
*/

#include <iostream>     // cout, cerr, endl
#include <string>       // stoi
#include <stdexcept>
#include <opencv2/opencv.hpp>

#include "argtable3.h"  // cross platform command line parsing
#include "two_phase_psf_estimation.hpp"

using namespace std;
using namespace cv;

// global structs for command line parsing
struct arg_lit *help;
struct arg_file *image, *mask;
struct arg_end *end_args;
struct arg_int *psf_width;


/**
 * Saves the user input in given variables.
 * On error while parsing or used help option this function returns false.
 */
static bool parse_commandline_args(int argc, char** argv, 
                                   string &imageName,
                                   string &maskName,
                                   int &psfWidth,
                                   int &exitcode) {
    
    // command line options
    // the global arg_xxx structs are initialized within the argtable
    void *argtable[] = {
        help        = arg_litn("h", "help", 0, 1, "display this help and exit"),
        psf_width   = arg_intn ("w", "psf-width", "<n>", 0, 1, "approximate PSF width. Default: 24"),
        mask        = arg_filen ("m", "mask", "<path>", 0, 1, "mask of an image region. Default: complete image"),
        image       = arg_filen(nullptr, nullptr, "<image>",  1, 1, "image"),
        end_args    = arg_end(20),
    };

    // default values (they weren't set if there is a value given)
    psf_width->ival[0] = 25;
    mask->filename[0] = "none";

    // parsing arguments
    int nerrors = arg_parse(argc,argv,argtable);

    // special case: '--help' takes precedence over error reporting
    if (help->count > 0)
    {
        cout << "Usage: " << argv[0];
        arg_print_syntax(stdout, argtable, "\n");
        cout << "Two-phase kernel estimation." << endl << endl;
        arg_print_glossary(stdout, argtable, "  %-25s %s\n");
        
        // deallocate each non-null entry in argtable[]
        arg_freetable(argtable, sizeof(argtable) / sizeof(argtable[0]));
        exitcode = 0;
        return false;
    }

    // If the parser returned any errors then display them and exit
    if (nerrors > 0)
    {
        arg_print_errors(stdout, end_args, argv[0]);
        cout << "Try '" << argv[0] << "--help' for more information." << endl;
        
        // deallocate each non-null entry in argtable[]
        arg_freetable(argtable, sizeof(argtable) / sizeof(argtable[0]));
        exitcode = 1;
        return false;
    }

    // saving arguments in variables
    // path to input model
    imageName = image->filename[0];
    maskName = mask->filename[0];
    psfWidth = psf_width->ival[0];

    // deallocate each non-null entry in argtable[]
    arg_freetable(argtable, sizeof(argtable) / sizeof(argtable[0]));

    return true;
}


int main(int argc, char** argv) {
    // path to models and other parameter
    string imageName;
    string maskName;
    int psfWidth;

    // parse command line arguments
    int exitcode = 0;
    bool success = parse_commandline_args(argc, argv, imageName, maskName, psfWidth, exitcode);

    if (success == false) {
        return exitcode;
    }

    // run algorithm
    cout << "Start Depth-Aware Motion Deblurring with" << endl;
    cout << "   image:               " << imageName << endl;
    cout << "   approx. PSF width:   " << psfWidth << endl;
    cout << endl;

    try {
        Mat psf, mask;

        // load image
        Mat image = imread(imageName, 1);

        // load mask
        if (maskName != "none") {
            mask = imread(maskName, CV_LOAD_IMAGE_GRAYSCALE);
            mask /= 255;
            TwoPhaseKernelEstimation::estimateKernel(psf, image, psfWidth, mask);
        } else {
            TwoPhaseKernelEstimation::estimateKernel(psf, image, psfWidth);
        }

        
        #ifndef NDEBUG
            // Wait for a key stroke
            waitKey(0);
        #endif
    }
    catch(const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    
    return 0;
}