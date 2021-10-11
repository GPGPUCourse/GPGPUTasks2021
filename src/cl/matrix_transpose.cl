#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6


#define GROUP_SIDE_SIZE 16
#define GROUP_DIV 15
__kernel void matrix_transpose(__global float *as, __global float *as_t, unsigned int M, unsigned int K)
{
    __local float storage[GROUP_SIDE_SIZE][GROUP_SIDE_SIZE];

    unsigned int local_x = get_local_id(0);
    unsigned int local_y = get_local_id(1);
    unsigned int global_x = get_global_id(0);
    unsigned int global_y = get_global_id(1);

    if (global_y < M && global_x < K) {
        storage[local_y][(local_y + local_x) & GROUP_DIV] = as[global_y * K + global_x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_y < M && global_x < K) {
        as_t[(get_group_id(0) * GROUP_SIDE_SIZE + local_y) * M + (get_group_id(1) * GROUP_SIDE_SIZE + local_x)] = storage[local_x][(local_x + local_y) & GROUP_DIV];
    }
}