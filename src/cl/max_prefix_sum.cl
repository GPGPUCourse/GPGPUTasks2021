#define WORK_GROUP_SIZE 128

__kernel void max_prefix_sum(__global const int *numbers, __global int *result, __global int *max_pref_sum)
{
    int localId = get_local_id(0);
    int globalId = get_global_id(0);
    int groupSize = WORK_GROUP_SIZE;
    int groupNum = get_global_id(0) / groupSize;
    int numGroups = get_global_size(0) / WORK_GROUP_SIZE;
    __local int lresult[WORK_GROUP_SIZE];
    lresult[localId] = numbers[globalId];

    int offset = 2;
    for (unsigned int iteration = groupSize / 2; iteration > 0; iteration /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((localId + 1) % offset == 0) {
            lresult[localId] += lresult[localId - (offset / 2)];
        }
        offset *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId == 0) {
        lresult[WORK_GROUP_SIZE - 1] = 0;
    }

    for (unsigned int d = 1; d < groupSize; d *= 2) {
        offset /= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((localId + 1) % offset == 0) {
            int old = lresult[localId];
            lresult[localId] += lresult[localId - (offset / 2)];
            lresult[localId - (offset / 2)] = old;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    result[globalId] = lresult[localId] + numbers[globalId];


    barrier(CLK_LOCAL_MEM_FENCE);

    __local int lres[WORK_GROUP_SIZE];

    lres[localId] = result[globalId];

    // very slow part of code
    for (int i = 0; i < groupNum; ++i) {
        lres[localId] += result[(i + 1) * groupSize - 1];
    }

    // barrier(CLK_LOCAL_MEM_FENCE);
    // find maximum of local array
    for (unsigned int iteration = groupSize / 2; iteration > 0; iteration /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (localId < iteration) {
            lres[localId] =
                lres[localId] > lres[localId + iteration] ? lres[localId] : lres[localId + iteration];
        }
    }

    if (localId == 0) {
        atomic_max(max_pref_sum, lres[0]);
    }
}
