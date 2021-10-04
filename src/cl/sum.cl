
__kernel void sum(__global unsigned *input, unsigned n, __global unsigned *output) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);

    __local unsigned local_buffer[256];

    if (global_id < n) {
        local_buffer[local_id] = input[get_group_id(0) * get_local_size(0) + local_id];
    } else {
        local_buffer[local_id] = 0;
    }

    int workers_count = 128;
    
    while (workers_count) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < workers_count) {
            local_buffer[local_id] = local_buffer[local_id] + local_buffer[local_id + workers_count];
        }
        workers_count /= 2;
    }
    if (local_id == 0) {
        output[get_group_id(0)] = local_buffer[0];
    }
}
    
