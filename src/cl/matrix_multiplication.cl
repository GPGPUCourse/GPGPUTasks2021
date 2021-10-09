__kernel void matrix_multiplication(
    __global const float* as,
    __global const float* bs,
    __global float* cs,
    unsigned M, unsigned K, unsigned N
) {
    unsigned x = get_global_id(0);
    unsigned y = get_global_id(1);

    __local float work_as[WORK_GROUP_SIZE][WORK_GROUP_SIZE];
    __local float work_bs[WORK_GROUP_SIZE][WORK_GROUP_SIZE];

    float result = .0f;
    for (unsigned i = 0; i < K; i += WORK_GROUP_SIZE) {
        if (i > 0) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        unsigned z = i + get_local_id(0);
        if (y < M && z < K) {
            work_as[get_local_id(1)][get_local_id(0)] = (y < M && z < K) ? as[y * K + z] : .0f;
        }

        z = i + get_local_id(1);
        if (x < N && z < K) {
            work_bs[get_local_id(1)][get_local_id(0)] = (x < N && z < K) ? bs[z * N + x] : .0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned j = 0; j < WORK_GROUP_SIZE; ++j) {
            result += work_as[get_local_id(1)][j] * work_bs[j][get_local_id(0)];
        }
    }

    if (x < N && y < M) {
        cs[y * N + x] = result;
    }
}

