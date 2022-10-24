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

#include <stdio.h>
#include <stdint.h>
#include "../utils/spng/spng.h"

#include "../utils/utils.h"

// Input and output paths
#define INPATH "../input/pngtest.png"
#define OUTPATH "../output/sharpened_pngtest.png"

int main()
{
    // Loading png
    spng_ctx *ctx = load_png(INPATH);
    if (ctx == NULL)
        return 1;

    // Getting png information
    struct spng_ihdr ihdr;
    if (spng_get_ihdr(ctx, &ihdr))
    {
        printf("failed to get information header\n");
        return 1;
    }

    printf("width: %u\n"
           "height: %u\n"
           "bit depth: %u\n"
           "color type: %u\n", //  "color type: %u - %s\n"
           ihdr.width, ihdr.height, ihdr.bit_depth, ihdr.color_type /*, color_name*/);

    size_t image_size, image_width;
    int format = SPNG_FMT_PNG;
    if (ihdr.color_type == SPNG_COLOR_TYPE_INDEXED)
        format = SPNG_FMT_RGB8;

    if (spng_decoded_image_size(ctx, format, &image_size))
    {
        printf("decoding image size failed\n");
        return 1;
    }

    // Allocating memory for the image
    unsigned char *image = (unsigned char *)malloc(image_size);
    if (image == NULL)
    {
        printf("error allocating image memory\n");
    }

    // Decoding the image to get the RBGA values
    if (spng_decode_image(ctx, image, image_size, SPNG_FMT_RGBA8, 0))
    {
        printf("error decoding image\n");
        return 1;
    }

    // 4 values for each one of the pixels in the row (RGBA)
    image_width = image_size / ihdr.height;

    // Sharpen convolutional filter
    int filter[3][3] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};

    spng_color_type color_type = get_color_type(ihdr.color_type);

    encode_png(image, image_size, ihdr.width, ihdr.height, color_type, ihdr.bit_depth);

    write_png(OUTPATH, image_size, image);

    spng_ctx_free(ctx);
    free(image);

    return 0;
}
