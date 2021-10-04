__kernel void sum_reductor(__global int* data, unsigned read_offset, unsigned write_offset, unsigned current_reduction_size) {
    if (get_global_id(0) >= current_reduction_size) return;
    data[write_offset + get_global_id(0)] = data[read_offset + 2 * get_global_id(0)] + data[read_offset + 2 * get_global_id(0) + 1];
}

__kernel void sum_calculator(__global int* data, unsigned current_length) {
    unsigned id = get_global_id(0);
    if (id >= current_length) return;
    unsigned item = id + 1;
    unsigned current_binary_group_id = id;
    unsigned current_mask = 1;
    unsigned offset = 0;

    while (current_mask <= item) {
        if (current_mask == 1) {
            if ((current_mask & item) == 0) {
                data[id] = 0;
            }
        } else {
            if (current_mask & item) {
                data[id] += data[offset + current_binary_group_id];
            }
        }
        current_binary_group_id /= 2;
        current_binary_group_id = current_binary_group_id & 0b11111111111111111111111111111110;
        offset += current_length;
        current_length /= 2;
        current_mask = current_mask << 1;
    }
}
            
__kernel void min_reductor(__global int* data, unsigned read_offset, unsigned write_offset, unsigned work_size) {
    if (get_global_id(0) >= work_size) {
        return;
    }
    int a = data[read_offset + 2 * get_global_id(0)];
    int b = data[read_offset + 2 * get_global_id(0) + 1];
    data[write_offset + get_global_id(0)] = a > b ? a : b;
}

