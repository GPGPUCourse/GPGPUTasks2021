#define TILE_SIZE 16

__kernel void matrix_multiplication(__global float *as, __global float *bs, __global float *cs, 
                                    unsigned int M, unsigned int K, unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int lc_i = get_local_id(0);
    int lc_j = get_local_id(1);
    
    __local float lc_as[TILE_SIZE][TILE_SIZE];
    __local float lc_bs[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {

        lc_as[lc_j][lc_i] = (tileK * TILE_SIZE + lc_i < K && j < M) ?
                                as[j * K + (tileK * TILE_SIZE + lc_i)] : 0.0f;

        lc_bs[lc_j][lc_i] = (lc_j + tileK * TILE_SIZE < K && i < N) ?
                                bs[(lc_j * N + tileK * TILE_SIZE * N) + i] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += lc_as[lc_j][k] * lc_bs[k][lc_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    cs[j * N + i] = sum;
}