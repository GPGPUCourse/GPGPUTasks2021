#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define VALUES_PER_WORK_ITEM 128

__kernel void sum(__global unsigned int* xs, unsigned int n,
                  __global unsigned int* res)
{
    __local unsigned int localBuf[VALUES_PER_WORK_ITEM];
    unsigned int localId = get_local_id(0);
    unsigned int globalId = get_global_id(0);

    if(globalId < n) {
        localBuf[localId] = xs[globalId];
    }
    else {
        localBuf[localId] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int k = (VALUES_PER_WORK_ITEM >> 1); k > 0; k >>= 1) {
        if (localId < k) {
            localBuf[localId] += localBuf[localId + k];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(localId == 0) {
        atomic_add(res, localBuf[0]);
    }
}