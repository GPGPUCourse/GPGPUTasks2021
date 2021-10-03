#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128

__kernel void sum(__global const unsigned int* arr, unsigned int n,
                  __global unsigned int* res) {
    unsigned int local_id = get_local_id(0),
                 local_size = get_local_size(0),
                 global_size = get_global_size(0),
                 global_id = get_global_id(0);

    if (global_id >= n) {
        return;
    }

    __local unsigned int arr_local[WORK_GROUP_SIZE];
    arr_local[local_id] = arr[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        float sum = 0;
        for (int i = 0; i < local_size; i++) {
            sum += arr_local[i];
        }
        atomic_add(res, sum);
    }
}