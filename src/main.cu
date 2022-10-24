/*
    Instructions
Develop a CUDA implementation of an image convolutional algorithm for RGB or RGBA images
(RGB or RGBA it depends on the image format). This assignment can be accomplished in group (up to 2 students).

You can refer to the file format you prefer, for instance PNG, JPG, or TIFF. I suggest to use a software library
for loading (into a buffer)/saving images like libpng (http://www.libpng.org/pub/png/libpng.html). libpng is already
installed on the JPDM2 workstation.
The kernels must receive the image as a linear buffer representing the pixels color in the RBG or RGBA format, together
with the convolutional filter (e.g., the one reported below), apply the filter, and store the resulting image into an
output buffer. Eventually, the filtered image must be saved to disk for assessment purposes.

Sharpen convolutional filter (ref: https://en.wikipedia.org/wiki/Kernel_(image_processing))

 0 -1  0
-1  5 -1
 0 -1  0


Assessment

Run some experiments by using three block sizes, namely 8x8, 8x16, 16x8, 16x16, 16x32, 32x16 and 32x32 by profiling
the executions into a table reporting the elapsed times and the bytes accessed L1, L2 and DRAM memory systems.
*/


// TODO: Implement the algorithm from wikipedia
// TODO: Adapt the algorithm for using it as a GPU kernel
// TODO: Output the filtered image to a new file
// TODO: Test it with the different grid sizes
// TODO: Test the performance with the nvprof


#include <png.h>
#include <stdio.h>
#include <stdint.h>

#include "../utils/utils.h"

int main()
{
    // Loading file and initializing png structs
    FILE *fp = fopen("../images/pngtest.png", "rb");
    if (!fp)
        return 1;

    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL, end_info = NULL;
    png_bytepp rows = NULL;

    if (initialize_png(png_ptr, info_ptr, end_info, fp) != 0) {
        printf("Error initializing png!\n");
        return 1;
    }

    process_png(png_ptr, info_ptr, rows);

    // Sharpen convolutional filter
    int filter[3][3] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};

    return 0;
}