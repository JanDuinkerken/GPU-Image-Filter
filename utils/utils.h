#ifndef _UTILS_H_
#define _UTILS_H_

#include "spng/spng.h"

spng_ctx *load_png(char * path);
spng_color_type get_color_type(uint8_t type);
int encode_png(void *image, size_t length, uint32_t width, uint32_t height, enum spng_color_type color_type, int bit_depth);
int write_png(char * path, size_t image_size, unsigned char * image);

#endif