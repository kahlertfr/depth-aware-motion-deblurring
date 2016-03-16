/***********************************************************************
 * Author:       Franziska Krüger
 * Using:        argtable3 - http://www.argtable.org/
 *
 * Description:
 * ------------
 * Reference Implementation of the Depth-Aware Motion Deblurring
 * Algorithm by Xu and Jia.
 *
 ************************************************************************
*/

#include <iostream>     // cout, cerr, endl
#include <string>       // stoi
#include <stdexcept>

#include "argtable3.h"  // cross platform command line parsing
#include "depth_aware_deblurring.hpp"

using namespace std;

// global structs for command line parsing
struct arg_lit *help;
struct arg_file *left_image, *right_image;
struct arg_end *end_args;
struct arg_int *psf_width, *max_toplevel_nodes;


/**
 * Saves the user input in given variables.
 * On error while parsing or used help option this function returns false.
 */
static bool parse_commandline_args(int argc, char** argv, 
                                   string &left, string &right,
                                   int &psfWidth, int &maxTopLevelNodes,
                                   int &exitcode) {
    
    // command line options
    // the global arg_xxx structs are initialized within the argtable
    void *argtable[] = {
        help        = arg_litn("h", "help", 0, 1, "display this help and exit"),
        psf_width   = arg_intn ("w", "psf-width", "<n>",           0, 1, "approximate PSF width. Default: 24"),
        max_toplevel_nodes   = arg_intn ("m", "max-top-nodes", "<n>", 0, 1, "max top level nodes in region tree. Default: 3"),
        left_image  = arg_filen(nullptr, nullptr, "<left image>",  1, 1, "left image"),
        right_image = arg_filen(nullptr, nullptr, "<right image>", 1, 1, "right image"),
        end_args    = arg_end(20),
    };

    // default values (they weren't set if there is a value given)
    psf_width->ival[0] = 25;
    max_toplevel_nodes->ival[0] = 3;

    // parsing arguments
    int nerrors = arg_parse(argc,argv,argtable);

    // special case: '--help' takes precedence over error reporting
    if (help->count > 0)
    {
        cout << "Usage: " << argv[0];
        arg_print_syntax(stdout, argtable, "\n");
        cout << "Depth-Aware Motion Deblurring." << endl << endl;
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
    left = left_image->filename[0];
    right = right_image->filename[0];
    psfWidth = psf_width->ival[0];
    maxTopLevelNodes = max_toplevel_nodes->ival[0];

    // deallocate each non-null entry in argtable[]
    arg_freetable(argtable, sizeof(argtable) / sizeof(argtable[0]));

    return true;
}


int main(int argc, char** argv) {
    // path to models and other parameter
    string imageLeft;
    string imageRight;
    int psfWidth;
    int maxTopLevelNodes;

    // parse command line arguments
    int exitcode = 0;
    bool success = parse_commandline_args(argc, argv, imageLeft, imageRight, 
                                          psfWidth, maxTopLevelNodes, exitcode);

    if (success == false) {
        return exitcode;
    }

    // run algorithm
    cout << "Start Depth-Aware Motion Deblurring with" << endl;
    cout << "   image left:          " << imageLeft << endl;
    cout << "   image right:         " << imageRight << endl;
    cout << "   approx. PSF width:   " << psfWidth << endl;
    cout << "   max top level nodes: " << maxTopLevelNodes << endl;
    cout << endl;

    try {
        deblur::depthDeblur(imageLeft, imageRight, psfWidth, maxTopLevelNodes);
    }
    catch(const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    
    return 0;
}