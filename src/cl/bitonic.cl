#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 256

__kernel void bitonic_local(__global float *as, unsigned int n,
                            unsigned int from_level, unsigned int to_level) {
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    const unsigned int block_start = (global_id >> 8) << 9;

    __local float local_storage[WORK_GROUP_SIZE * 2];

    local_storage[local_id] = as[block_start + local_id];
    local_storage[local_id + WORK_GROUP_SIZE] = as[block_start + local_id + WORK_GROUP_SIZE];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int level = from_level; level < to_level; ++level) {
        float sign = ((global_id >> (level - 1)) & 1) == 0 ? 1.0f : -1.0f;

        for (unsigned int in_level = min(level, 9u); in_level > 0; --in_level) {
            unsigned int step_size = 1 << (in_level - 1);
            unsigned int first_id = ((local_id >> (in_level - 1)) << in_level) + (local_id & (step_size - 1));
            unsigned int second_id = first_id + step_size;

            if ((local_storage[first_id] - local_storage[second_id]) * sign > 0.0f) {
                float tmp = local_storage[first_id];
                local_storage[first_id] = local_storage[second_id];
                local_storage[second_id] = tmp;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    as[block_start + local_id] = local_storage[local_id];
    as[block_start + local_id + WORK_GROUP_SIZE] = local_storage[local_id + WORK_GROUP_SIZE];
}

__kernel void bitonic_step(__global float *as, unsigned int n,
                           unsigned int level, unsigned int in_level) {
    const unsigned int global_id = get_global_id(0);

    const unsigned int index = global_id & ((1 << (in_level - 1)) - 1);
    const unsigned int block_start = (global_id >> (in_level - 1)) << in_level;

    float sign = ((global_id >> (level - 1)) & 1) == 0 ? 1.0f : -1.0f;

    unsigned int step_size = 1 << (in_level - 1);
    unsigned int first_id = block_start + index;
    unsigned int second_id = first_id + step_size;

    if ((as[first_id] - as[second_id]) * sign > 0.0f) {
        float tmp = as[first_id];
        as[first_id] = as[second_id];
        as[second_id] = tmp;
    }
}
