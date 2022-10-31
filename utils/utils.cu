
// This also needs to be a .cu file and not just a c file for the makefile to
// work

#include "utils.h"

spng_ctx *load_png(const char *path) {
  FILE *fp = fopen(path, "rb");
  if (!fp) {
    printf("Failed to open image file\n");
    return NULL;
  }

  spng_ctx *ctx = spng_ctx_new(0);
  if (!ctx) {
    printf("Failed to create spng context\n");
    return NULL;
  }

  spng_set_png_file(ctx, fp);

  return ctx;
}

spng_color_type get_color_type(uint8_t type) {
  switch (type) {
  case 0:
    return SPNG_COLOR_TYPE_GRAYSCALE;
  case 2:
    return SPNG_COLOR_TYPE_TRUECOLOR;
  case 3:
    return SPNG_COLOR_TYPE_INDEXED;
  case 4:
    return SPNG_COLOR_TYPE_GRAYSCALE_ALPHA;
  case 6:
    return SPNG_COLOR_TYPE_TRUECOLOR_ALPHA;
  default:
    return SPNG_COLOR_TYPE_TRUECOLOR;
  }
}

int decode_png(const char *path, spng_ctx *ctx, struct spng_ihdr *ihdr,
               unsigned char **image, size_t *image_size, size_t *image_width,
               spng_color_type *color_type) {
  ctx = load_png(path);
  if (ctx == NULL)
    return 1;

  // Getting png information
  if (spng_get_ihdr(ctx, ihdr)) {
    printf("Failed to get information header\n");
    return 1;
  }

  int format = SPNG_FMT_PNG;
  if ((*ihdr).color_type == SPNG_COLOR_TYPE_INDEXED)
    format = SPNG_FMT_RGB8;

  if (spng_decoded_image_size(ctx, format, image_size)) {
    printf("Decoding image size failed\n");
    return 1;
  }

  // Allocating memory for the image
  *image = (unsigned char *)malloc(*image_size);
  if (*image == NULL) {
    printf("Error allocating image memory\n");
    return 1;
  }

  // Decoding the image to get the RBGA values
  if (spng_decode_image(ctx, *image, *image_size, SPNG_FMT_RGBA8, 0)) {
    printf("Error decoding image\n");
    return 1;
  }

  // 4 values for each one of the pixels in the row (RGBA)
  *image_width = *image_size / (*ihdr).height;

  *color_type = get_color_type((*ihdr).color_type);

  return 0;
}

int write_png(FILE *outfp, size_t image_size, void *image) {
  uint64_t no_of_elements =
      image_size /
      sizeof(unsigned char); // considering img as byte array -> no. of bytes
  if (fwrite(image, sizeof(unsigned char), no_of_elements, outfp) !=
      no_of_elements) {
    printf("Error writing contents of output file\n");
    return 1;
  }
  fclose(outfp);
  return 0;
}

unsigned char *expand_image(unsigned char *image, size_t image_size,
                            size_t image_width, size_t height) {
  unsigned char *expanded_image;
  size_t expanded_size =
      image_size + sizeof(unsigned char) * (2 * (image_width + 8) + height * 8);
  expanded_image = (unsigned char *)malloc(expanded_size);
  int row = 1;
  // Populate expanded_image, first and second rows are equal and the last one
  // and the one before that are also equal
  for (int i = 0; i < height + 2; i++) {
    row = i + 1;
    for (int j = 0; j < image_width + 8; j++) {
      if (row < 2) {
        if (j <= 3) {
          expanded_image[j * row] = image[j * row];
        } else if (j == image_width + 3) {
          expanded_image[j * row] = image[(j - 8) * row];
        } else
          expanded_image[j] = image[(j - 4) * row];
      } else if (row == height + 1) {
        if (j <= 3) {
          expanded_image[j * row] = image[j * (row - 2)];
        } else if (j == image_width + 3) {
          expanded_image[j * row] = image[(j - 8) * (row - 2)];
        } else
          expanded_image[j] = image[(j - 4) * (row - 2)];
      } else {
        if (j <= 3) {
          expanded_image[j * row] = image[j * (row - 1)];
        } else if (j == image_width + 3) {
          expanded_image[j * row] = image[(j - 8) * (row - 1)];
        } else
          expanded_image[j] = image[(j - 4) * (row - 1)];
      }
    }
  }

  // printf("image: %s\n expanded image: %s\n", image, expanded_image);

  return expanded_image;
}

int encode_png(unsigned char *image, size_t length, uint32_t width,
               uint32_t height, enum spng_color_type color_type, int bit_depth,
               const char *path) {
  FILE *outfp = fopen(path, "w");
  if (!outfp) {
    printf("Error creating output file\n");
    return 1;
  }
  int format;
  int ret = 0;
  spng_ctx *ctx = NULL;
  struct spng_ihdr ihdr = {0};

  ctx = spng_ctx_new(SPNG_CTX_ENCODER);
  spng_set_option(ctx, SPNG_ENCODE_TO_BUFFER, 1);

  /* Alternatively we can set an output FILE* or stream with spng_set_png_file()
   * or spng_set_png_stream() */

  ihdr.width = width;
  ihdr.height = height;
  ihdr.color_type = color_type;
  ihdr.bit_depth = bit_depth;

  spng_set_ihdr(ctx, &ihdr);

  format = SPNG_FMT_PNG;

  /* SPNG_ENCODE_FINALIZE will finalize the PNG with the end-of-file marker */
  ret = spng_encode_image(ctx, image, length, format, SPNG_ENCODE_FINALIZE);
  if (ret) {
    printf("Spng_encode_image() error: %s\n", spng_strerror(ret));
    spng_ctx_free(ctx);
    return ret;
  }

  size_t png_size;

  void *png_buf = NULL;

  png_buf = spng_get_png_buffer(ctx, &png_size, &ret);
  if (png_buf == NULL) {
    printf("Spng_get_png_buffer() error: %s\n", spng_strerror(ret));
  }

  ret = write_png(outfp, png_size, png_buf);

  free(png_buf);

  return ret;
}
