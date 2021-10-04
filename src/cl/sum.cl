#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 256

__kernel void sum(__global const unsigned int* input, __global unsigned int* sum, unsigned int n) {

    const unsigned int local_id = get_local_id(0);
    const unsigned int global_id = get_global_id(0);

    __local unsigned int buffer[WORK_GROUP_SIZE];
    buffer[local_id] = global_id < n ? input[global_id] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int workers = WORK_GROUP_SIZE >> 1; workers > 0; workers = workers >> 1) {
        if (local_id < workers) {
            buffer[local_id] += buffer[local_id + workers];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        atomic_add(sum, buffer[0]);
    }
}
