#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define TILE_SIZE 16

__kernel void matrix_transpose(__global const float* a,
                               __global       float* at,
                               unsigned int M, /* размер по оси X*/
                               unsigned int K  /* размер по оси Y*/)
{
    __local float block[TILE_SIZE][TILE_SIZE]; // при размере 16*16 нет конфликта банок, не нужно что-то придумывать?

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    int i = get_global_id(0);
    int j = get_global_id(1);

    int hi = get_group_id(0) * TILE_SIZE; // смещение для транспонирования
    int hj = get_group_id(1) * TILE_SIZE;

    bool isInMatrix = local_i < K && j < M; // из-за барьера нельзя делать return

    if (isInMatrix) {
        block[local_j][local_i] = a[j * K + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (isInMatrix) {
        at[(hi + local_j) * M + (hj + local_i)] = block[local_i][local_j];
    }
}