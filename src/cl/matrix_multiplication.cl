#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GROUP_SIZE 16

__kernel void matrix_multiplication(__global const float* as, __global const float* bs,
                                    __global float* cs, unsigned int m, unsigned int k, unsigned int n) {
    __local float buffer_a[GROUP_SIZE][GROUP_SIZE];
    __local float buffer_b[GROUP_SIZE][GROUP_SIZE];
    // mxk * kxn -> mxn

    unsigned int local_x = get_local_id(0);
    unsigned int global_x = get_global_id(0); // col idx 0..n

    unsigned int local_y = get_local_id(1);
    unsigned int global_y = get_global_id(1); // row idx 0..m

    float res = 0;
    for (unsigned int offset = 0; offset < k; offset += GROUP_SIZE) {
        // load buffers
        buffer_a[local_y][local_x] = as[global_y * k + offset + local_x];
        buffer_b[local_y][local_x] = bs[(offset + local_y) * n + global_x];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int delta = 0; delta < GROUP_SIZE; ++delta) {
            res += buffer_a[local_y][delta] * buffer_b[delta][local_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    cs[global_y * n + global_x] = res; // sum_{t = 0..k-1} as[global_y * k + t] * bs[t * n + global_x]
}