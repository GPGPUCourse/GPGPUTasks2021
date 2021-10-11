#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GROUP_SIZE 16

__kernel void matrix_transpose(__global const float* as, __global float* as_t,
                               unsigned int m, unsigned int k) {
    __local float buffer[GROUP_SIZE][GROUP_SIZE+1]; // add fantom element to fix banks conflict
    const unsigned int delta_x = get_local_id(0);
    const unsigned int delta_y = get_local_id(1);

    const unsigned int group_x = get_group_id(0);
    const unsigned int group_y = get_group_id(1);

    const unsigned int x = get_global_id(0); // group_x * GROUP_SIZE + delta_x
    const unsigned int y = get_global_id(1); // group_y * GROUP_SIZE + delta_y

    if (x < k && y < m) {
        buffer[delta_y][delta_x] = as[y * k + x];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // now buffer[0..GROUP_SIZE-1][0..GROUP_SIZE-1] contains transposed fragment of
    // as[group_x*GROUP_SIZE..group_x*GROUP_SIZE+GROUP_SIZE-1][group_y*GROUP_SIZE..group_y*GROUP_SIZE-1]

    const unsigned int new_x = group_y * GROUP_SIZE + delta_x;
    const unsigned int new_y = group_x * GROUP_SIZE + delta_y;

    if (new_x < m && new_y < k) {
        as_t[new_y * m + new_x] = buffer[delta_x][delta_y];
    }
}
