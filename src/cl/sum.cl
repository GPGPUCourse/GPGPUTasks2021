#define WORK_GROUP_SIZE 128
__kernel void sum (__global const int* xs,
                   __global int* res) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);

    __local int local_xs[WORK_GROUP_SIZE];
    local_xs[local_id] = xs[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int n = WORK_GROUP_SIZE; n > 1; n /= 2) {
        if (local_id < n/2) {
            local_xs[local_id] += local_xs[local_id + n/2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0)
        atomic_add(res, local_xs[0]);

}