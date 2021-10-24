#define loc 256

__kernel void bitonic_local(__global float *as, unsigned n, unsigned block_number, int op_number) {
    if (get_global_id(0) * 2 >= n) return;
    unsigned gmem_i = get_group_id(0) * 256 + get_local_id(0);
    __local float mem[loc];
    mem[get_local_id(0)] = as[gmem_i];
    mem[get_local_id(0) + loc / 2] = as[gmem_i + loc / 2];
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for (; op_number > -1; --op_number) {
        unsigned elem = get_local_id(0) + (get_local_id(0) & ((((unsigned) 0) - 1) << op_number));
        unsigned offset = 1 << op_number;
        if ((get_global_id(0) >> block_number) & 1) {
            if (mem[elem] < mem[elem + offset]) {
                float temp = mem[elem];  
                mem[elem] = mem[elem + offset];
                mem[elem + offset] = temp;
            }
        } else {
            if (mem[elem] > mem[elem + offset]) {
                float temp = mem[elem];  
                mem[elem] = mem[elem + offset];
                mem[elem + offset] = temp;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    as[gmem_i] = mem[get_local_id(0)];
    as[gmem_i + loc / 2] = mem[get_local_id(0) + loc / 2];
}
__kernel void bitonic(__global float *as, unsigned n, unsigned block_number, unsigned op_number) {
    unsigned elem = get_global_id(0) + (get_global_id(0) & ((((unsigned) 0) - 1) << op_number));
    unsigned offset = 1 << op_number;
    if (elem > n) return;
    if ((get_global_id(0) >> block_number) & 1) {
        if (as[elem] < as[elem + offset]) {
            float temp = as[elem];  
            as[elem] = as[elem + offset];
            as[elem + offset] = temp;
        }
    } else {
        if (as[elem] > as[elem + offset]) {
            float temp = as[elem];  
            as[elem] = as[elem + offset];
            as[elem + offset] = temp;
        }
    }
}
