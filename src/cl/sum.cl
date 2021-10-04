#define WORK_GROUP_SIZE 128

__kernel void sum_gpu(__global const int* numbers, __global int* result) {
    atomic_add(result, numbers[get_global_id(0)]);
}

__kernel void sum_gpu2(__global const int* numbers, __global int* result) {
    int localId = get_local_id(0), globalId = get_global_id(0);
    __local int local_numbers[WORK_GROUP_SIZE];
    local_numbers[localId] = numbers[globalId];
    int sum = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId == 0) {
        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            sum += local_numbers[i];
        }
        atomic_add(result, sum);
    }
}

__kernel void sum_gpu3(__global const int* numbers, __global int* result) {
    int localId = get_local_id(0);
    int globalId = get_global_id(0);
    int groupSize = get_local_size(0); // 128 = WORK_GROUP_SIZE

    __local int local_numbers[WORK_GROUP_SIZE];
    local_numbers[localId] = numbers[globalId];

    for (unsigned int iteration = groupSize / 2; iteration > 0; iteration /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < iteration) {
            local_numbers[localId] += local_numbers[localId + iteration];
        }
    }
    if (localId == 0) {
        atomic_add(result, local_numbers[0]);
    }
}
