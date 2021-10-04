#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128
__kernel void sum(__global const unsigned int* arr, __global unsigned int * res, const unsigned int arr_size)
{
    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    __local unsigned int local_res[WORK_GROUP_SIZE];
    if (global_id < arr_size) {
        local_res[local_id] = arr[global_id];
    } else {
        local_res[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int n = WORK_GROUP_SIZE/2; n > 0; n /= 2) {
        if (local_id < n)
            local_res[local_id] += local_res[local_id + n];

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0)
        atomic_add(res, local_res[0]);
}
