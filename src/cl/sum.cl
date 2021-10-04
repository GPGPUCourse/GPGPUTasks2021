#define WORK_GROUP_SIZE 128

__kernel void sum(__global const int* as, __global int* result) {
    unsigned int lc_id = get_local_id(0);
    unsigned int gl_id = get_global_id(0);

    __local int lc_as[WORK_GROUP_SIZE];
    lc_as[lc_id] = as[gl_id];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (unsigned int n = WORK_GROUP_SIZE, half_n = n >> 1; n > 1;
             n = half_n, half_n >>= 1) {
        if (lc_id < half_n) {
            lc_as[lc_id] += lc_as[lc_id + half_n];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lc_id == 0) {
        atomic_add(result, lc_as[0]);
    }
} 