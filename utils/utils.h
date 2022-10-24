#ifndef _UTILS_H_
#define _UTILS_H_

#include <png.h>

// Size of the image we use for testing
// TODO: Remove after figuring out how to dinamically compute it
#define HEIGHT 69
#define WIDTH 91

// We need one function to initialize all data structs from the libpng library and load our image
// (It returns an int because we have to check for errors when initializing).
int initialize_png(png_structp png_ptr, png_infop info_ptr, png_infop end_info, FILE *fp);

// We then need one function to read the image an translate all the data to an array of rows to then be able to
// later divide it in batches and send it to a GPU kernel for applying the sharpening filter.
void process_png(png_structp png_ptr, png_infop info_ptr, png_bytepp rows);

// Finally we have a function to read all the already modified rows of the png and write them to a new file
// so that we are able to se if the application of the filter was succesful.
void write_png();

#endif