#define k 2

__kernel void calc_c_init(__global unsigned int *as, __global unsigned *cs, unsigned n, unsigned stage) {
    __local unsigned local_cs[1 << k];
    if (get_local_id(0) < (1 << k)) { 
        local_cs[get_local_id(0)] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned mask = ((0xffffffff << (32 - k)) >> (32 - k)) << (stage * k);
    unsigned curr_val = (as[get_global_id(0)] & mask) >> (k * stage);
    atomic_add(local_cs + curr_val, 1);

    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) < (1 << k)) {
        unsigned offset = get_local_id(0) % (1 << k);
        cs[(1 << k) * get_group_id(0) + offset] = local_cs[offset];
    }
}

__kernel void calc_c_tree(__global int* cs, unsigned read_offset, unsigned write_offset, unsigned current_reduction_size) {
    if (get_global_id(0) >= current_reduction_size) return;
    cs[write_offset + get_global_id(0)] = cs[read_offset + 2 * get_global_id(0)] + cs[read_offset + 2 * get_global_id(0) + 1];
}

__kernel void matrix_transpose(__global unsigned* data, __global unsigned* output, unsigned w, unsigned h, unsigned work_group_size)
{
    __local unsigned mem[WARP_SIZE][WARP_SIZE];

    for (int i = 0; i != WARP_SIZE * WARP_SIZE / work_group_size; ++i) {
        unsigned mem_y = get_local_id(1) + i * get_local_size(1);
        unsigned mem_x = (get_local_id(0) + mem_y) % WARP_SIZE;

        unsigned in_x = get_global_id(0);
        unsigned in_y = get_group_id(1) * WARP_SIZE + i * get_local_size(1) + get_local_id(1);

        unsigned source_index = in_x + in_y * w; 
        if (in_x < w && in_y < h) {
            mem[mem_x][mem_y] = data[source_index];
        } else {
            mem[mem_x][mem_y] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);   

    unsigned tile_x_shift = get_group_id(1) * WARP_SIZE;
    unsigned tile_y_shift = get_group_id(0) * WARP_SIZE;

    for (int i = 0; i != WARP_SIZE * WARP_SIZE / work_group_size; ++i) {
        unsigned out_x = tile_x_shift + get_local_id(0);
        unsigned out_y = tile_y_shift + get_local_id(1) + i * get_local_size(1);

        unsigned mem_y = get_local_id(0);
        unsigned mem_x = (get_local_id(1) + get_local_size(1) * i + mem_y) % WARP_SIZE;

        unsigned output_index = out_x + out_y * h; 
        if (out_x < h && out_y < w) {
            output[output_index] = mem[mem_x][mem_y];
        }
    }
}

__kernel void calc_o(__global unsigned* c_tree, __global unsigned* o, unsigned current_length) {
    unsigned id = get_global_id(0);
    if (id >= current_length) return;
    unsigned item = id + 1;
    unsigned current_binary_group_id = id;
    unsigned current_mask = 1;
    unsigned offset = 0;
    unsigned result = 0;

    while (current_mask <= item) {
        if (current_mask & item) {
            result += c_tree[offset + current_binary_group_id];
        }
        current_binary_group_id /= 2;
        current_binary_group_id = current_binary_group_id & 0b11111111111111111111111111111110;
        offset += current_length;
        current_length /= 2;
        current_mask = current_mask << 1;
    }
    o[id] = result - c_tree[id];
}

__kernel void local_counting_sort(__global unsigned *as, unsigned n, stage) {
    if (get_global_id(0) >= n) {
        return;
    }
    __local int cs[1 << k];
    __local unsigned as_local[WORK_GROUP_SIZE];
    if (get_local_id(0) < (1 << k)) {
        cs[get_local_id(0)] = 0;
    }
    as_local[get_local_id(0)] = as[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned mask = ((0xffffffff << (32 - k)) >> (32 - k)) << (stage * k);
    unsigned value = as_local[get_local_id(0)];
    unsigned curr_val = (value & mask) >> (k * stage);
    atomic_add(cs + curr_val, 1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0) {
        for (int j = 1; j != (1 << k); ++j) {
            cs[j] = cs[j] + cs[j - 1];
        }
        for (int i = WORK_GROUP_SIZE - 1; i >= 0; --i) {
            value = as_local[i];
            curr_val = (value & mask) >> (k * stage);
            cs[curr_val] -= 1;
            as[get_local_size(0) * get_group_id(0) + cs[curr_val]] = as_local[i];
        }
    }
}

__kernel void radix(__global const unsigned *as, __global const unsigned* o, __global const unsigned* cs, __global unsigned* res, unsigned n, unsigned stage) {
    unsigned mask = ((0xffffffff << (32 - k)) >> (32 - k)) << (stage * k);
    unsigned value = as[get_global_id(0)];
    unsigned curr_val = (value & mask) >> (k * stage);
    unsigned global_offset = o[n / WORK_GROUP_SIZE * curr_val + get_group_id(0)];
    unsigned local_offset = get_local_id(0);
    for (unsigned i = 0; i != curr_val; ++i) {
        local_offset -= cs[get_group_id(0) * (1 << k) + i];
    }
    res[global_offset + local_offset] = value;
}

