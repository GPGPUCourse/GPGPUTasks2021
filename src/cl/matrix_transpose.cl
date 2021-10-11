#define GROUP_DIM_SIZE 16

__kernel void matrix_transpose(__global float* src, __global float* dst, unsigned int m, unsigned int k)
{
    unsigned i = get_global_id(0); // col number 0..k
    unsigned j = get_global_id(1); // row number 0..m
    unsigned li = get_local_id(0);
    unsigned lj = get_local_id(1);
    unsigned mod16 = GROUP_DIM_SIZE - 1;

    __local float buffer[GROUP_DIM_SIZE][GROUP_DIM_SIZE];
    buffer[li][(li + lj) & mod16] = src[j * k + i];

    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned id = (get_group_id(0) * GROUP_DIM_SIZE + lj) * m + (get_group_id(1) * GROUP_DIM_SIZE + li);
    dst[id] = buffer[lj][(li + lj) & mod16];
}
