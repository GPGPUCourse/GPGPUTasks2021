#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 256
#define TREE_STOP_SIZE 128

// Benchmarks on Intel(R) UHD Graphics 620 [0x5917] 12620Mb
// Average on 1000 iterations
// 3437.78 - main sum thread
// 1581.64 - full binary tree
// 2059.83 - tree with 002-stop
// 2283.96 - tree with 004-stop
// 2602.50 - tree with 008-stop
// 2824.53 - tree with 016-stop
// 3337.03 - tree with 032-stop
// 3759.41 - tree with 064-stop
// 3934.66 - tree with 128-stop
// 3565.83 - tree with 256-stop

__kernel void sum(__global const unsigned int* input, __global unsigned int* sum, unsigned int n) {

    const unsigned int local_id = get_local_id(0);
    const unsigned int global_id = get_global_id(0);

    __local unsigned int buffer[WORK_GROUP_SIZE];
    buffer[local_id] = global_id < n ? input[global_id] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int workers = WORK_GROUP_SIZE >> 1; workers >= TREE_STOP_SIZE; workers = workers >> 1) {
        if (local_id < workers) {
            buffer[local_id] += buffer[local_id + workers];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        unsigned int local_sum = 0;
        for (int i=0; i < TREE_STOP_SIZE; ++i) {
            local_sum += buffer[i];
        }
        atomic_add(sum, local_sum);
    }
}
