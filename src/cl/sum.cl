#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum(__global const unsigned int *input,
                  __global unsigned int *partialSums,
                  __local unsigned int *localSums) {
    unsigned int local_id = get_local_id(0);

    localSums[local_id] = input[get_global_id(0)];

    for (unsigned int i = get_local_size(0) / 2; i > 0; i /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < i) {
            localSums[local_id] += localSums[local_id + i];
        }            
    }

    if (local_id == 0) {
        partialSums[get_group_id(0)] = localSums[0];
    }
}