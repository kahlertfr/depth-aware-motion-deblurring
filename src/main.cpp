/***********************************************************************
 * Author:       Franziska Krüger
 * Using:        argtable3 - http://www.argtable.org/
 *
 * Description:
 * ------------
 * Reference Implemenatation of Depth-Aware Motion Deblurring Algorithm
 * by Xu and Jia.
 *
 ************************************************************************
*/

#include <iostream>     // cout, cerr, endl
#include <string>       // stoi

#include "argtable3.h"  // cross platform command line parsing

using namespace std;

// global structs for command line parsing
struct arg_lit *help;
struct arg_file *left_image, *right_image;
struct arg_end *end_args;

/**
 * Saves the user input in given variables.
 * On error while parsing or used help option this function returns false.
 */
static bool parse_commandline_args(int argc, char** argv, 
                                   string &left, string &right,
                                   int &exitcode) {
    
    // command line options
    // the global arg_xxx structs are initialised within the argtable
    void *argtable[] = {
        help        = arg_litn("h", "help", 0, 1, "display this help and exit"),
        left_image  = arg_filen(nullptr, nullptr, "<left image>", 1, 1, "left image"),
        right_image = arg_filen(nullptr, nullptr, "<right image>", 1, 1, "right image"),
        end_args    = arg_end(20),
    };

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

    // deallocate each non-null entry in argtable[]
    arg_freetable(argtable, sizeof(argtable) / sizeof(argtable[0]));

    return true;
}


int main(int argc, char** argv) {
    // path to models
    string image_left;
    string image_right;

    // parse commandline arguments
    int exitcode = 0;
    bool success = parse_commandline_args(argc, argv, image_left, image_right, exitcode);

    if (success == false) {
        return exitcode;
    }

    // run algorithm
    cout << "Start Depth-Aware Motion Deblurring with" << endl;
    cout << "   image:  " << image_left << endl;
    cout << endl;

    // TODO: algorithm
    
    return 0;
}