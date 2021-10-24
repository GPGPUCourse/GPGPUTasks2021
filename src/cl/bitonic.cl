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
