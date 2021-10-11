__kernel void matrix_transpose(__global float* data, __global float* output, unsigned w, unsigned h, unsigned work_group_size)
{
    __local float mem[WS][WS];

    for (int i = 0; i != WS * WS / work_group_size; ++i) {
        unsigned mem_y = get_local_id(1) + i * get_local_size(1);
        unsigned mem_x = (get_local_id(0) + mem_y) % WS; //

        unsigned in_x = get_global_id(0);
        unsigned in_y = get_group_id(1) * WS + i * get_local_size(1) + get_local_id(1);

        unsigned source_index = in_x + in_y * w; 
        if (in_x < w && in_y < h) {
            mem[mem_x][mem_y] = data[source_index];
        } else {
            mem[mem_x][mem_y] = 0;
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);   

    unsigned tile_x_shift = get_group_id(1) * WS;
    unsigned tile_y_shift = get_group_id(0) * WS;

    for (int i = 0; i != WS * WS / work_group_size; ++i) {
        unsigned out_x = tile_x_shift + get_local_id(0);
        unsigned out_y = tile_y_shift + get_local_id(1) + i * get_local_size(1);

        unsigned mem_y = get_local_id(0);
        unsigned mem_x = (get_local_id(1) + get_local_size(1) * i + mem_y) % WS;

        unsigned output_index = out_x + out_y * h; 
        if (out_x < h && out_y < w) {
            output[output_index] = mem[mem_x][mem_y];
        }
    }
}
