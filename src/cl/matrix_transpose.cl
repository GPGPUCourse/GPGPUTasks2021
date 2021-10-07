#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GROUP_SIZE_L 16

__kernel void matrix_transpose(__global const float* a,
                               __global       float* a_t,
                               unsigned int m,
                               unsigned int k)
{
    const unsigned int global_i = get_global_id(0);
    const unsigned int global_j = get_global_id(1);
    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);
    const unsigned int start_i = get_group_id(0) * GROUP_SIZE_L;
    const unsigned int start_j = get_group_id(1) * GROUP_SIZE_L;

    __local float local_store[GROUP_SIZE_L][GROUP_SIZE_L];

    local_store[local_j][(local_i + local_j) & 0b1111u] = (global_j < m && global_i < k) ? a[global_j * k + global_i] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_j < m && global_i < k) {
        a_t[(start_i + local_j) * m + (start_j + local_i)] = local_store[local_i][(local_i + local_j) & 0b1111u];
    }
}
