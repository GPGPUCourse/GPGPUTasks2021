#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 256

#define LOCAL_MEM_BARRIER(n) if (n > WARP_SIZE) barrier(CLK_LOCAL_MEM_FENCE)

__kernel void local_prefix(__global const int* xs,
                           __global       int* group_prefix_sum,
                           __global       int* group_sum,
                           unsigned int n)
{
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);
    const unsigned int group_id = get_group_id(0);

    __local int local_sum[WORK_GROUP_SIZE * 2];
    __local int local_prefix_sum[WORK_GROUP_SIZE];

    if (global_id < n) {
        local_sum[256 + local_id] = xs[global_id];
    } else {
        local_sum[256 + local_id] = 0;
    }

    LOCAL_MEM_BARRIER(256);
    if (local_id < 128) {
        local_sum[128 + local_id] = local_sum[256 + local_id * 2] + local_sum[257 + local_id * 2];
    }

    LOCAL_MEM_BARRIER(128);
    if (local_id < 64) {
        local_sum[64 + local_id] = local_sum[128 + local_id * 2] + local_sum[129 + local_id * 2];
    }

    LOCAL_MEM_BARRIER(64);
    if (local_id < 32) {
        local_sum[32 + local_id] = local_sum[64 + local_id * 2] + local_sum[65 + local_id * 2];
    }

    LOCAL_MEM_BARRIER(32);
    if (local_id < 16) {
        local_sum[16 + local_id] = local_sum[32 + local_id * 2] + local_sum[33 + local_id * 2];
    }

    LOCAL_MEM_BARRIER(16);
    if (local_id < 8) {
        local_sum[8 + local_id] = local_sum[16 + local_id * 2] + local_sum[17 + local_id * 2];
    }

    LOCAL_MEM_BARRIER(8);
    if (local_id < 4) {
        local_sum[4 + local_id] = local_sum[8 + local_id * 2] + local_sum[9 + local_id * 2];
    }

    LOCAL_MEM_BARRIER(4);
    if (local_id < 2) {
        local_sum[2 + local_id] = local_sum[4 + local_id * 2] + local_sum[5 + local_id * 2];
    }

    LOCAL_MEM_BARRIER(2);
    if (local_id == 0) {
        local_sum[1] = local_sum[2] + local_sum[3];
        group_sum[group_id] = local_sum[1];
    }

    LOCAL_MEM_BARRIER(256);

    local_prefix_sum[local_id] = 0;
    unsigned int index = local_id + 1;
    if (index & 1) {
        local_prefix_sum[local_id] += local_sum[255 + index];
    }
    if (index & 2) {
        local_prefix_sum[local_id] += local_sum[127 + (index >> 1)];
    }
    if (index & 4) {
        local_prefix_sum[local_id] += local_sum[63 + (index >> 2)];
    }
    if (index & 8) {
        local_prefix_sum[local_id] += local_sum[31 + (index >> 3)];
    }
    if (index & 16) {
        local_prefix_sum[local_id] += local_sum[15 + (index >> 4)];
    }
    if (index & 32) {
        local_prefix_sum[local_id] += local_sum[7 + (index >> 5)];
    }
    if (index & 64) {
        local_prefix_sum[local_id] += local_sum[3 + (index >> 6)];
    }
    if (index & 128) {
        local_prefix_sum[local_id] += local_sum[1 + (index >> 7)];
    }
    if (index & 256) {
        local_prefix_sum[local_id] += local_sum[1 + (index >> 8)];
    }
    group_prefix_sum[global_id] = local_prefix_sum[local_id];
}

__kernel void prefix_sum(__global const int* prefix_level_0,
                         __global const int* prefix_level_1,
                         __global const int* prefix_level_2,
                         __global       int* prefix_sum,
                         unsigned int n)
{
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);
    const unsigned int group_id = get_group_id(0);

    __local int local_prefix_sum[WORK_GROUP_SIZE];

    local_prefix_sum[local_id] = 0;

    unsigned int index = global_id + 1;
    if (index & 255) {
        local_prefix_sum[local_id] += prefix_level_0[index - 1];
    }
    if ((index >> 8) & 255) {
        local_prefix_sum[local_id] += prefix_level_1[(index >> 8) - 1];
    }
    if ((index >> 16) & 255) {
        local_prefix_sum[local_id] += prefix_level_2[(index >> 16) - 1];
    }
    prefix_sum[global_id] = local_prefix_sum[local_id];
}

__kernel void with_max_value(__global const int* keys,
                             __global const int* values,
                             __global       int* result_keys,
                             __global       int* result_values,
                           unsigned int n)
{
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);
    const unsigned int group_id = get_group_id(0);

    __local int local_keys[WORK_GROUP_SIZE];
    __local int local_values[WORK_GROUP_SIZE];

    if (global_id < n) {
        if (keys == 0) {
            local_keys[local_id] = global_id + 1;
        } else {
            local_keys[local_id] = keys[global_id];
        }
        local_values[local_id] = values[global_id];
    } else {
        local_keys[local_id] = n;
        local_values[local_id] = -2147483648;
    }

    LOCAL_MEM_BARRIER(256);
    if (local_id < 128 && local_values[local_id + 128] > local_values[local_id]) {
        local_keys[local_id] = local_keys[local_id + 128];
        local_values[local_id] = local_values[local_id + 128];
    }

    LOCAL_MEM_BARRIER(128);
    if (local_id < 64 && local_values[local_id + 64] > local_values[local_id]) {
        local_keys[local_id] = local_keys[local_id + 64];
        local_values[local_id] = local_values[local_id + 64];
    }

    LOCAL_MEM_BARRIER(64);
    if (local_id < 32 && local_values[local_id + 32] > local_values[local_id]) {
        local_keys[local_id] = local_keys[local_id + 32];
        local_values[local_id] = local_values[local_id + 32];
    }

    LOCAL_MEM_BARRIER(32);
    if (local_id < 16 && local_values[local_id + 16] > local_values[local_id]) {
        local_keys[local_id] = local_keys[local_id + 16];
        local_values[local_id] = local_values[local_id + 16];
    }

    LOCAL_MEM_BARRIER(16);
    if (local_id < 8 && local_values[local_id + 8] > local_values[local_id]) {
        local_keys[local_id] = local_keys[local_id + 8];
        local_values[local_id] = local_values[local_id + 8];
    }

    LOCAL_MEM_BARRIER(8);
    if (local_id < 4 && local_values[local_id + 4] > local_values[local_id]) {
        local_keys[local_id] = local_keys[local_id + 4];
        local_values[local_id] = local_values[local_id + 4];
    }

    LOCAL_MEM_BARRIER(4);
    if (local_id < 2 && local_values[local_id + 2] > local_values[local_id]) {
        local_keys[local_id] = local_keys[local_id + 2];
        local_values[local_id] = local_values[local_id + 2];
    }

    LOCAL_MEM_BARRIER(2);
    if (local_id == 0) {
        if (local_values[1] > local_values[0]) {
            result_keys[group_id] = local_keys[1];
            result_values[group_id] = local_values[1];
        } else {
            result_keys[group_id] = local_keys[0];
            result_values[group_id] = local_values[0];
        }
    }
}
