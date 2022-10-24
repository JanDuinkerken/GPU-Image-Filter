
// This also need to be a .cu file and not just a c file for the makefile to work

#include "utils.h"

spng_ctx *load_png(char * path)
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
