#define WORK_GROUP_SIZE 128
#define INT_MIN (-2147483648)
#define MINUS_INF INT_MIN


__kernel void prefix_sum(__global const int *numbers, unsigned size, __global int *result, unsigned power)
{
    unsigned gid = get_global_id(0);
    if (gid < (1 << power)) {
        result[gid] = numbers[gid];
    } else {
        result[gid] = numbers[gid] + numbers[gid - (1 << power)];
    }
}

__kernel void max_numbers(__global const int* numbers, unsigned size, __global int *max)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    __local int data[WORK_GROUP_SIZE];
    data[lid] = (gid < size ? numbers[gid] : MINUS_INF);

    for (int i = WORK_GROUP_SIZE / 2; i > 0; i /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < i && data[lid + i] > data[lid]) {
            data[lid] = data[lid + i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        atomic_max(max, data[lid]);
    }
}

__kernel void get_index(__global const int* numbers, unsigned size, int element, __global int *index) {
    int gid = get_global_id(0);
    if (gid >= size) return;
    if (numbers[gid] == element) {
        atomic_max(index, (gid + 1));
    }
}
