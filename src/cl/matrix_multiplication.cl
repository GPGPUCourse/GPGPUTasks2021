
#define GroupSize 16
__kernel void matrix_multiplication(__global const float* as, __global const float* bs, __global float* cs, unsigned int M, 
                                    unsigned int K, unsigned int N)
{
    // TODO
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    __local float aCache[GroupSize][GroupSize];
    __local float bCache[GroupSize][GroupSize];
    float sum = .0f;
    for (int tileNum = 0; tileNum < (K + GroupSize - 1) / GroupSize; tileNum++) {
        if (tileNum > 0) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (i < N && j < M) {
            aCache[local_j][local_i] = as[local_i + tileNum * GroupSize + j * K];
            bCache[local_j][local_i] = bs[i + (local_j + tileNum * GroupSize) * N];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < GroupSize; k++) {
            sum += aCache[local_j][k] * bCache[k][local_i];
        }
    }
    if (i < N && j < M)
        cs[i + j * N] = sum;
}