#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128

__kernel void sum(__global unsigned int* result, __global const unsigned int* as)
{
    const size_t local_id = get_local_id(0);

    __local int local_as[WORK_GROUP_SIZE];
    local_as[local_id] = as[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t i = WORK_GROUP_SIZE; i > 0; i /= 2) {
        if (i > local_id) {
            local_as[local_id] += local_as[local_id + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        atomic_add(result, local_as[0]);
    }
}
