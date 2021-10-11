#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define TILE_SIZE 16

__kernel void matrix_multiplication(__global float * a, __global float * b, __global float * c,
                                    unsigned int M, unsigned int K, unsigned int N) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0;
    for (int tileK = 0; tileK * TILE_SIZE < K; tileK++) {
        if (j < M && (tileK * TILE_SIZE + local_i < K))
            tileA[local_j][local_i] = a[j * K + (tileK * TILE_SIZE + local_i)];
        else
            tileA[local_j][local_i] = 0;

        if (i < N && (tileK * TILE_SIZE + local_j < K))
            tileB[local_i][local_j] = b[i + (tileK * TILE_SIZE * N + local_j * N)];
        else
            tileB[local_i][local_j] = 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[local_j][k] * tileB[local_i][k];
        }

    }
    if (i < N && j < M) {
        c[j * N + i] = sum;
    }
}