#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128
__kernel void sum(__global unsigned int* result,
                  __global unsigned int* as,
                  unsigned int n) {
    unsigned int globalId = get_global_id(0);
    unsigned int localId = get_local_id(0);

    __local unsigned int local_as[WORK_GROUP_SIZE];
    unsigned int asi = 0;
    if (globalId < n) {
        asi = as[globalId];
    }
    local_as[localId] = asi;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId == 0) {
        unsigned int sum = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; i++) {
            sum += local_as[i];
        }
        atomic_add(result, sum);
    }
}