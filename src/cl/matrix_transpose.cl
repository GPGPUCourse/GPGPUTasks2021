#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 16
__kernel void matrix_transpose(__global float * a, __global float * at, unsigned int m, unsigned int k)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    __local float tile[TILE_SIZE][TILE_SIZE+1];
    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    if (i < k && j < m)
        tile[local_j + 1][local_i] = a[j * m + i];
    else
        tile[local_j + 1][local_i] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    i = get_group_id(1) * TILE_SIZE + local_i;
    j = get_group_id(0) * TILE_SIZE + local_j;
    if (i < m && j < k)
        at[j*k + i] = tile[local_i  + 1][local_j];
}