#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void matrix_transpose(const __global float* input,
                               __global float* output,
                               unsigned int width,
                               unsigned int height,
                               unsigned int block_size) {
    unsigned int x = get_local_id(0);
    unsigned int y = get_local_id(1);
    unsigned int i = x + block_size * get_group_id(0);
    unsigned int j = y + block_size * get_group_id(1);

    __local float buffer[256];

    if (i < width && j < height) {
        buffer[x + block_size * y] = input[i + width * j];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int i1 = x + block_size * get_group_id(1);
    unsigned int j1 = y + block_size * get_group_id(0);

    if (i1 < height && j1 < width) {
        output[i1 + height * j1] = buffer[y + block_size * x];
    }
}