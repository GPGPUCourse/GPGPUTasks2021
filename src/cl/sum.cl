#define WARP_SIZE 1 // for some reason it breaks with WARP_SIZE >= 4 on nvidia, idk why
#define ITEMS_PER_WORK_GROUP_LOG 8
#define ITEMS_PER_WORK_GROUP (1 << ITEMS_PER_WORK_GROUP_LOG)

#define STEP(k) (1 << (ITEMS_PER_WORK_GROUP_LOG - 1 - k))
#define ITERATION(k) \
    do { \
        if (get_local_id(0) < STEP(k)) \
            tmp[get_local_id(0)] += tmp[get_local_id(0) + STEP(k)]; \
        if (STEP(k) > WARP_SIZE) \
            barrier(CLK_LOCAL_MEM_FENCE); \
    } while (0)

__kernel void sum(
    __global const unsigned *arr,
    __global unsigned *result
) {
    __local unsigned tmp[ITEMS_PER_WORK_GROUP];
    unsigned idx = get_local_id(0) + ITEMS_PER_WORK_GROUP * get_group_id(0);

    tmp[get_local_id(0)] = arr[idx] + arr[idx + STEP(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    ITERATION(1);
    ITERATION(2);
    ITERATION(3);
    ITERATION(4);
    ITERATION(5);
    ITERATION(6);
    ITERATION(7);
    
    if (get_local_id(0) == 0) {
        atomic_add(result, tmp[0]);
    }
}
