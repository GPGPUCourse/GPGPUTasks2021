
#define GroupSizeX 16
#define GroupSizeY 16

__kernel void matrix_transpose(__global const float* as, __global float* as_t, unsigned int M, unsigned int K)
{
    __local float cache[GroupSizeX][GroupSizeY];

    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);
    cache[local_i][(local_j + local_i) % GroupSizeX] = (i < M && j < K ? as[i + j * K]: 0);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i >= M || j >= K)
        return;

    as_t[j - local_j + local_i + (i - local_i + local_j) * M] = cache[local_j][(local_j + local_i) % GroupSizeX];
}