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

int encode_image(void *image, size_t length, uint32_t width, uint32_t height, enum spng_color_type color_type, int bit_depth)
{
    int format;
    int ret = 0;
    spng_ctx *ctx = NULL;
    struct spng_ihdr ihdr = {0};

    ctx = spng_ctx_new(SPNG_CTX_ENCODER);
    spng_set_option(ctx, SPNG_ENCODE_TO_BUFFER, 1);

    /* Alternatively we can set an output FILE* or stream with spng_set_png_file() or spng_set_png_stream() */

    ihdr.width = width;
    ihdr.height = height;
    ihdr.color_type = color_type;
    ihdr.bit_depth = bit_depth;

    spng_set_ihdr(ctx, &ihdr);

    format = SPNG_FMT_PNG;

    /* SPNG_ENCODE_FINALIZE will finalize the PNG with the end-of-file marker */
    ret = spng_encode_image(ctx, image, length, format, SPNG_ENCODE_FINALIZE);
    if(ret)
    {
        printf("spng_encode_image() error: %s\n", spng_strerror(ret));
        spng_ctx_free(ctx);
        return ret;
    }

    size_t png_size;
    void *png_buf = NULL;

    png_buf = spng_get_png_buffer(ctx, &png_size, &ret);
    if(png_buf == NULL)
    {
        printf("spng_get_png_buffer() error: %s\n", spng_strerror(ret));
    }

    free(png_buf);
}

int main()
{
    // Loading png
    spng_ctx *ctx = load_png("../images/pngtest.png");
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

    encode_image(image, image_size, ihdr.width, ihdr.height, ihdr.color_type, ihdr.bit_depth);

    fwrite(image, image_size);

    spng_ctx_free(ctx);
    free(image);

    return 0;
}
