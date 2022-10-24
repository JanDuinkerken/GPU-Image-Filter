# Sharpenning image filter CUDA

This CUDA program loads an image into memory and the applies a convolutional sharpenning filter on GPU in parallel

The kernels receive the image as a linear buffer representing the pixels color in the RBG or RGBA format, together
with the convolutional filter (e.g., the one reported below), apply the filter, and store the resulting image in disk

## Sharpen convolutional filter

 0 -1  0
-1  5 -1
 0 -1  0
