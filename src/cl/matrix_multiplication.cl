#define TILE_SIZE 16

__kernel void matrix_multiplication(__global float *a, __global float *b, __global float *c,
                                    unsigned int M, unsigned int K, unsigned int N)
{
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    int i = get_global_id(0);
    int j = get_global_id(1);

    float sum = 0;

    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        if (tileK * TILE_SIZE + local_i < K && j < M)
            tileA[local_j][local_i] = a[j * K + (tileK * TILE_SIZE + local_i)];
        else
            tileA[local_j][local_i] = 0;

        if (i < N && tileK * TILE_SIZE + local_j < K)
            tileB[local_j][local_i] = b[(local_j * N + tileK * TILE_SIZE * N) + i];
        else
            tileB[local_j][local_i] = 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int iter_sum = 0; iter_sum < TILE_SIZE; ++iter_sum) {
            sum += tileA[local_j][iter_sum] * tileB[iter_sum][local_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[j * N + i] = sum;
}