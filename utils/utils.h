#ifndef _UTILS_H_
#define _UTILS_H_

#include "spng/spng.h"

int decode_png(const char *path, spng_ctx *ctx, struct spng_ihdr *ihdr,
               unsigned char **image, size_t *image_size, size_t *image_width,
               spng_color_type *color_type);

int encode_png(unsigned char *image, size_t length, uint32_t width,
               uint32_t height, enum spng_color_type color_type, int bit_depth,
               const char *path);

#endif
