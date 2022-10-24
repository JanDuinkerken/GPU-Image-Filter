
// This also need to be a .cu file and not just a c file for the makefile to work

#include "utils.h"

int initialize_png(png_structp png_ptr, png_infop info_ptr, png_infop end_info, FILE *fp)
{
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);  // equivalent for writing: ng_create_write_struct() && png_destroy_write_struct()
    if (!png_ptr)
        return (1);

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        png_destroy_read_struct(&png_ptr,
                                (png_infopp)NULL, (png_infopp)NULL);
        return (1);
    }
    // end_info = png_create_info_struct(png_ptr);
    // if (!end_info)
    // {
    //     png_destroy_read_struct(&png_ptr, &info_ptr,
    //                             (png_infopp)NULL);
    //     return (1);
    // }

    png_init_io(png_ptr, fp);

    return 0;
}

void process_png(png_structp png_ptr, png_infop info_ptr, png_bytepp rows)
{
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    // TODO: Find a way to compute the width and height of the image so that we dont have to hardcode it
    rows = png_get_rows(png_ptr, info_ptr); // Array of pointers to each of the rows.
    // size_t height = sizeof(rows) / sizeof(rows[0]);
    // size_t width = sizeof(rows[0]) / sizeof(rows[0][0]);

    // printf("Computed width: %d; Real width: %d \n", width, WIDTH);
    // printf("Computed height: %d; Real height: %d \n", height, HEIGHT);

    printf("%d\n", rows);
    for (int i = 0; i < HEIGHT; i++)
    {
        printf("row %d: ", i);
        for (int j = 0; j < WIDTH; j++)
        {
            printf("%d ", rows[i][j]);
        }
        printf("\n");
    }
}

void write_png()
{
    printf("Still under work XD");
}