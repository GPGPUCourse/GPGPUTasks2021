#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void matrix_transpose(__global float* input,
                               __global float* output,
                               unsigned int width,
                               unsigned int height) {
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);

    if (x < width && y < height) {
        output[y + height * x] = input[x + width * y];
    }
}