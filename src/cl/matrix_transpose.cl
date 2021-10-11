#define TILE_SIZE 16

__kernel void matrix_transpose(__global float *as, __global float *as_t, 
                                unsigned int M, unsigned int K)
{
    /* 
        |1....K| -> 1       |1.........M| -> 1
        |1....K| -> 2   ->  .............
        ........            |1.........M| -> K
        |1....K| -> M
    */
    __local float lc_as[TILE_SIZE][TILE_SIZE];

    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= K || j >= M) 
        return;

    int lc_i = get_local_id(0);
    int lc_j = get_local_id(1);

    const int pow4 = (1 << 4) - 1;
    lc_as[lc_j][(lc_i + lc_j) & pow4] = as[i + j * K];

    barrier(CLK_LOCAL_MEM_FENCE);

    int i_t = j - lc_j + lc_i;
    int j_t = i - lc_i + lc_j;

    if (i_t >= M || j_t >= K)
        return;
    
    as_t[i_t + j_t * M] = lc_as[lc_i][(lc_i + lc_j) & pow4];
}