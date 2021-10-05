#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define WORK_GROUP_SIZE 128

__kernel void sum(__global const unsigned int *xs, __global unsigned int* res, unsigned int n) {

    __local unsigned int xs_cache[WORK_GROUP_SIZE];
    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    if (global_id > n){
        return;
    }        
    xs_cache[local_id] = xs[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int elements_num = WORK_GROUP_SIZE; elements_num > 1; elements_num /= 2) {
        if (local_id < elements_num / 2) {
            unsigned int a = xs_cache[local_id];
            unsigned int b = xs_cache[local_id + elements_num / 2];
            xs_cache[local_id] = a + b;
        }
        if (elements_num > 2 * WARP_SIZE)
            barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        atomic_add(&res[0], xs_cache[0]);
    }
}
