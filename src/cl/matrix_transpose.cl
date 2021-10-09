__kernel void matrix_transpose(
    __global const float* as,
    __global float* as_t,
    unsigned M, unsigned K
) {
    __local volatile float block[WORK_GROUP_SIZE][WORK_GROUP_SIZE];

    const unsigned x = get_global_id(0);
    const unsigned y = get_global_id(1);
    const unsigned gx = get_group_id(0) * WORK_GROUP_SIZE;
    const unsigned gy = get_group_id(1) * WORK_GROUP_SIZE;
    const unsigned idx = (get_local_id(0) + get_local_id(1)) % WORK_GROUP_SIZE;

    const bool active = x < K && y < M;

    if (active) {
        block[get_local_id(1)][idx] = as[y * K + x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (active) {
        as_t[(gx + get_local_id(1)) * M + gy + get_local_id(0)] = block[get_local_id(0)][idx];
    }
}
