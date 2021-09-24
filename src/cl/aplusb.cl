#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void aplusb(__global const float* a,
                     __global const float* b,
                     __global       float* c,
                     unsigned int n)
{
    const unsigned int index = get_global_id(0);

    if (index >= n)
        return;

    if (index == 0) {
        // Если бы printf был не под if, то printf попытался бы исполниться для всех запущенных workItems
        printf("Just example of printf usage: WARP_SIZE=%d\n", WARP_SIZE);
    }

    c[index] = a[index] + b[index];
}