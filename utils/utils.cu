
// This also need to be a .cu file and not just a c file for the makefile to work

#include "utils.h"

spng_ctx *load_png(char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp)
    {
        printf("Failed to open image file\n");
        return NULL;
    }

    spng_ctx *ctx = spng_ctx_new(0);
    if (!ctx)
    {
        printf("Failed to create spng context\n");
        return NULL;
    }

    spng_set_png_file(ctx, fp);

    return ctx;
}

spng_color_type get_color_type(uint8_t type)
{
    switch (type)
    {
    case 0:
        return SPNG_COLOR_TYPE_GRAYSCALE;
        break;
    case 2:
        return SPNG_COLOR_TYPE_TRUECOLOR;
        break;
    case 3:
        return SPNG_COLOR_TYPE_INDEXED;
        break;
    case 4:
        return SPNG_COLOR_TYPE_GRAYSCALE_ALPHA;
        break;
    case 6:
        return SPNG_COLOR_TYPE_TRUECOLOR_ALPHA;
        break;
    default:
        return SPNG_COLOR_TYPE_TRUECOLOR;
    }
}

int encode_png(void *image, size_t length, uint32_t width, uint32_t height, enum spng_color_type color_type, int bit_depth)
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
    if (ret)
    {
        printf("Spng_encode_image() error: %s\n", spng_strerror(ret));
        spng_ctx_free(ctx);
        return ret;
    }

    size_t png_size;
    void *png_buf = NULL;

    png_buf = spng_get_png_buffer(ctx, &png_size, &ret);
    if (png_buf == NULL)
    {
        printf("Spng_get_png_buffer() error: %s\n", spng_strerror(ret));
    }

    free(png_buf);
}

int write_png(char *path, size_t image_size, unsigned char *image)
{
    FILE *outfp = fopen(path, "w");
    if (!outfp)
    {
        printf("Error creating output file\n");
        return 1;
    }
    uint64_t no_of_elements = image_size / sizeof(unsigned char); // considering img as byte array -> no. of bytes
    if (fwrite(image, sizeof(unsigned char), no_of_elements, outfp) != no_of_elements)
    {
        printf("Error writing contents of output file\n");
        return 1;
    }
    fclose(outfp);
}