#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GROUP_SIZE_L 16

__kernel void matrix_multiplication(__global const float* a,
                                    __global const float* b,
                                    __global       float* c,
                                    unsigned int m,
                                    unsigned int k,
                                    unsigned int n)
{
    const unsigned int global_i = get_global_id(0);
    const unsigned int global_j = get_global_id(1);
    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    __local float a_local[GROUP_SIZE_L][GROUP_SIZE_L];
    __local float b_local[GROUP_SIZE_L][GROUP_SIZE_L];

    float result = 0.0f;

    for (unsigned int start = 0; start < k; start += GROUP_SIZE_L) {
        a_local[local_j][(local_i + local_j) & 0b1111u] = (global_j < m && (start + local_i) < k) ? a[global_j * k + (start + local_i)] : 0.0f;
        b_local[local_j][local_i] = ((start + local_j) < k && global_i < n) ? b[(start + local_j) * n + global_i] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int index = 0; index < GROUP_SIZE_L; index++) {
            result += a_local[local_j][(index + local_j) & 0b1111u] * b_local[index][local_i];
        }
    }

    if (global_j < m && global_i < n) {
        c[global_j * n + global_i] = result;
    }
}
