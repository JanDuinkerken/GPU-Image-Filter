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

// TODO: Test it with the different grid sizes (args)
// TODO: Test the performance with the nvprof

#include <stdio.h>
#include <stdint.h>
#include "../utils/spng/spng.h"

#include "../utils/utils.h"

// Input and output paths
#define INPATH "../input/pngtest.png"
#define OUTPATH "../output/sharpened_pngtest.png"

#define SHARPEN_SIZE 1
#define F_EXPANSION (SHARPEN_SIZE * 2 + 1)
#define F_PITCH 3
#define COLOR_VALUES 4

__global__ void sharpenFilterKernel(unsigned char *d_image, unsigned char *d_mod_image, int offset,
                                    int filter[F_EXPANSION][F_EXPANSION], int width, int height)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column < width && row < height)
    {
        for (int sharpenRow = -SHARPEN_SIZE; sharpenRow < SHARPEN_SIZE + 1; sharpenRow++)
            for (int sharpenCol = -SHARPEN_SIZE; sharpenCol < SHARPEN_SIZE + 1; sharpenCol++)
            {
                int currentRow = row + sharpenRow;
                int currentColumn = column + sharpenCol;
                if (currentRow > -1 && currentRow < height && currentColumn > -1 && currentColumn < width)
                {
                    d_mod_image[COLOR_VALUES * (row * width + column)    ]
                        += d_image[COLOR_VALUES * (currentRow * width + currentColumn)    ]
                            * filter[sharpenRow][sharpenCol];  // R

                    d_mod_image[COLOR_VALUES * (row * width + column) + 1]
                        += d_image[COLOR_VALUES * (currentRow * width + currentColumn) + 1]
                            * filter[sharpenRow][sharpenCol];  // G

                    d_mod_image[COLOR_VALUES * (row * width + column) + 2]
                        += d_image[COLOR_VALUES * (currentRow * width + cucurrentColumnrCol) + 2]
                            * filter[sharpenRow][sharpenCol];  // B

                    d_mod_image[COLOR_VALUES * (row * width + column) + 3]
                        += d_image[COLOR_VALUES * (currentRow * width + currentColumn) + 3]
                            * filter[sharpenRow][sharpenCol];  // A
                }
            }
    }
}

// for error-handling on operations that return cudaError_t
void checkReturnedError(cudaError_t error, int line)
{
    if (error != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), __FILE__, line);
        // cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

void process_image(size_t image_size, unsigned char *image, unsigned char *mod_image,
                   int block_size_x, int block_size_y, int n, int h_height, int h_width)
{
    unsigned char *d_mod_image;
    unsigned char *d_image;
    dim3 block_size(block_size_x, block_size_y, 1);
    dim3 grid_size((int)ceil((float)n / block_size_x), (int)ceil((float)n / block_size_y), 1);

    cudaError_t error = cudaMalloc(&d_image, image_size);
    checkReturnedError(error, __LINE__);
    error = cudaMalloc(&d_mod_image, image_size);
    checkReturnedError(error, __LINE__);
    error = cudaMemset(&d_mod_image, 0, image_size);
    checkReturnedError(error, __LINE__);

    error = cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);
    checkReturnedError(error, __LINE__);

    // Sharpen convolutional filter
    int filter[F_EXPANSION][F_EXPANSION] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
    int d_filter[F_EXPANSION][F_EXPANSION];
    cudaMemcpy2D(d_filter, F_PITCH, filter, F_PITCH, F_EXPANSION * sizeof(int), F_EXPANSION * sizeof(int), cudaMemcpyHostToDevice);

    int *width;
    int *height;

    cudaMallocManaged(&width, sizeof(int));
    cudaMallocManaged(&height, sizeof(int));

    *width = h_width;
    *height = h_height;

    for (int i = 0; i < 4; i++)
        sharpenFilterKernel<<<grid_size, block_size>>>(d_image, d_mod_image, i, d_filter, *width, *height);

    error = cudaMemcpy(mod_image, d_mod_image, image_size, cudaMemcpyDeviceToHost);
    checkReturnedError(error, __LINE__);

    cudaFree(d_mod_image);
    cudaFree(d_image);
}

int main(int argc, char **argv)
{
    spng_ctx *ctx = NULL;
    struct spng_ihdr ihdr;
    spng_color_type color_type;
    size_t image_size, image_width;
    unsigned char *image = NULL;
    unsigned char *mod_image = NULL;
    int block_size_x, block_size_y, n;
    block_size_x = atoi(argv[1]);
    block_size_y = atoi(argv[2]);
    n = atoi(argv[3]);

    if (decode_png(INPATH, ctx, &ihdr, &image, &image_size, &image_width, &color_type))
        return 1;

    mod_image = (unsigned char *)malloc(image_size);
    if (mod_image == NULL)
    {
        printf("Error allocating the necessary memory to store the modified image");
        return 1;
    }
    process_image(image_size, image, mod_image, block_size_x, block_size_y, n, ihdr.height, image_width);

    encode_png(mod_image, image_size, ihdr.width, ihdr.height, color_type, ihdr.bit_depth, OUTPATH);

    spng_ctx_free(ctx);
    free(image);
    free(mod_image);

    return 0;
}
