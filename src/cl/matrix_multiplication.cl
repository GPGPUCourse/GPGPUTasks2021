#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GROUP_SIDE_SIZE 16

__kernel void matrix_multiplication(__global const float *as, __global const float *bs, __global float *cs, unsigned int M, unsigned int K, unsigned int N) {
    __local float storage_a[GROUP_SIDE_SIZE][GROUP_SIDE_SIZE];
    __local float storage_b[GROUP_SIDE_SIZE][GROUP_SIDE_SIZE];

    unsigned int local_x = get_local_id(0);
    unsigned int local_y = get_local_id(1);
    unsigned int global_x = get_global_id(0);
    unsigned int global_y = get_global_id(1);

    float res = 0.0;
    for (unsigned int i = 0; i < K; i += GROUP_SIDE_SIZE) {
        unsigned int xa = i + local_x;
        unsigned int ya = global_y;
        if (xa < K && ya < M) {
            storage_a[local_y][local_x] = as[ya * K + xa];
        }

        unsigned int xb = global_x;
        unsigned int yb = i + local_y;
        if (xb < N && yb < K) {
            storage_b[local_y][local_x] = bs[yb * N + xb];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < GROUP_SIDE_SIZE; j++) {
            res += storage_a[local_y][j] * storage_b[j][local_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_x < N && global_y < M) {
        cs[global_y * N + global_x] = res;
    }
}