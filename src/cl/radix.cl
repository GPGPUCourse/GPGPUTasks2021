#define k 2

__kernel void calc_c_init(__global unsigned int *as, __global unsigned *cs, unsigned n, unsigned stage) {
    __local unsigned local_cs[1 << k];
    local_cs[get_local_id(0)] = 0;

    unsigned mask = ((0xffffffff << (32 - k)) >> (32 - k)) << (stage * k);
    unsigned curr_val = (as[get_global_id(0)] & mask) >> (k * stage);
    atomic_add(local_cs + curr_val, 1);

    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) / (1 << k) == 0) {
        unsigned offset = get_local_id(0) % (1 << k);
        cs[(1 << k) * get_group_id(0) + offset] = local_cs[offset];
    }
}

__kernel void calc_c_tree(__global int* cs, unsigned read_offset, unsigned write_offset, unsigned current_reduction_size) {
    if (get_global_id(0) >= current_reduction_size) return;
    cs[write_offset + get_global_id(0)] = cs[read_offset + 2 * get_global_id(0)] + cs[read_offset + 2 * get_global_id(0) + 1];
}

__kernel void matrix_transpose(__global float* data, __global float* output, unsigned w, unsigned h, unsigned work_group_size)
{
    __local float mem[WARP_SIZE][WARP_SIZE];

    for (int i = 0; i != WARP_SIZE * WARP_SIZE / work_group_size; ++i) {
        unsigned mem_y = get_local_id(1) + i * get_local_size(1);
        unsigned mem_x = (get_local_id(0) + mem_y) % WARP_SIZE; //

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
