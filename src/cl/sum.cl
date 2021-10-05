#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 256

#define LOCAL_MEM_BARRIER(n) if (n > WARP_SIZE) barrier(CLK_LOCAL_MEM_FENCE)

__kernel void sum(__global const unsigned int* xs,
                  __global       unsigned int* sum,
                  unsigned int n,
                  int last_iteration)
{
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);
    const unsigned int group_id = get_group_id(0);

    __local volatile unsigned int local_xs[WORK_GROUP_SIZE];

    if (global_id < n) {
        local_xs[local_id] = xs[global_id];
    } else {
        local_xs[local_id] = 0;
    }

    LOCAL_MEM_BARRIER(256);
    if (local_id < 128) {
        local_xs[local_id] += local_xs[local_id + 128];
    }

    LOCAL_MEM_BARRIER(128);
    if (local_id < 64) {
        local_xs[local_id] += local_xs[local_id + 64];
    }

    LOCAL_MEM_BARRIER(64);
    if (local_id < 32) {
        local_xs[local_id] += local_xs[local_id + 32];
    }

    LOCAL_MEM_BARRIER(32);
    if (local_id < 16) {
        local_xs[local_id] += local_xs[local_id + 16];
    }

    LOCAL_MEM_BARRIER(16);
    if (local_id < 8) {
        local_xs[local_id] += local_xs[local_id + 8];
    }

    LOCAL_MEM_BARRIER(8);
    if (local_id < 4) {
        local_xs[local_id] += local_xs[local_id + 4];
    }

    LOCAL_MEM_BARRIER(4);
    if (local_id < 2) {
        local_xs[local_id] += local_xs[local_id + 2];
    }

    LOCAL_MEM_BARRIER(2);
    if (local_id == 0) {
        local_xs[0] += local_xs[1];
        if (last_iteration) {
            atomic_add(sum, local_xs[0]);
        } else {
            sum[group_id] = local_xs[0];
        }
    }
}
