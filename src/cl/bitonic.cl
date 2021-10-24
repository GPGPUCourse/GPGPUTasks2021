#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define WORK_GROUP_SIZE 128

__kernel void bitonic_start_in_localmem(__global float *as, unsigned int n) {
    int globalId = get_global_id(0);
    int localId = get_local_id(0);
    __local float buf[WORK_GROUP_SIZE];
    if (globalId < n) {
        buf[localId] = as[globalId];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int i = 2; i <= WORK_GROUP_SIZE; i *= 2) {
        bool order = (globalId / i) % 2 == 0; // true for descending
        for (unsigned int j = i; j > 1; j /= 2) {
            if (localId + j / 2 < n && (localId % j) < j / 2) {
                float x = buf[localId];
                float y = buf[localId + j / 2];
                if ((x < y) != order) {
                    buf[localId] = y;
                    buf[localId + j / 2] = x;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    if (globalId < n) {
        as[globalId] = buf[localId];
    }
}

__kernel void bitonic_end_in_localmem(__global float *as, unsigned int n, unsigned int i) {
    int globalId = get_global_id(0);
    int localId = get_local_id(0);
    __local float buf[WORK_GROUP_SIZE];
    if (globalId < n) {
        buf[localId] = as[globalId];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    bool order = (globalId / i) % 2 == 0; // true for descending
    for (unsigned int j = WORK_GROUP_SIZE; j > 1; j /= 2) {
        if (localId + j / 2 < n && (localId % j) < j / 2) {
            float x = buf[localId];
            float y = buf[localId + j / 2];
            if ((x < y) != order) {
                buf[localId] = y;
                buf[localId + j / 2] = x;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (globalId < n) {
        as[globalId] = buf[localId];
    }
}

__kernel void bitonic(__global float *as, unsigned int n, unsigned int i, unsigned int j) {
    int id = get_global_id(0);
    bool order = (id / i) % 2 == 0; // true for descending
    if (id + j / 2 < n && ((id % j) < j / 2)) {
        float x = as[id];
        float y = as[id + j / 2];
        if ((x < y) != order) {
            as[id] = y;
            as[id + j / 2] = x;
        }
    }
}
