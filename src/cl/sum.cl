#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define WG_SIZE 128

#line 8

__kernel void sum(__global const uint* arr, uint n, __global uint* result)
{
    const uint g_ind = get_global_id(0);
    const uint l_ind = get_local_id(0);

    __local uint local_arr[WG_SIZE];

    if (g_ind == 0) {
        atomic_and(result, 0);
    }

    if (g_ind >= n)
        return;

    local_arr[l_ind] = arr[g_ind];

    barrier(CLK_LOCAL_MEM_FENCE);
    if (l_ind == 0) {
        uint local_sum = 0;
        for (int i = 0; i < WG_SIZE; i++) {
            local_sum += local_arr[i];
        }
        atomic_add(result, local_sum);
    }
}