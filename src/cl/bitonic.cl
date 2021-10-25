#define WORK_GROUP_SIZE 128

__kernel void bitonic_start(__global float *as, unsigned int n, 
                            unsigned int step_in, unsigned int step_out) {
    unsigned int id = get_global_id(0);
    
    if (id + step_in >= n || id % (2 * step_in) >= step_in) 
        return;

    bool is_rise = id % (2 * step_out) < step_out;
    float lhs_elem = as[id];
    float rhs_elem = as[id + step_in];

    if (is_rise == true) {
        if (lhs_elem > rhs_elem) {
            as[id] = rhs_elem;
            as[id + step_in] = lhs_elem;
        }
    } else {
        if (lhs_elem < rhs_elem) {
            as[id] = rhs_elem;
            as[id + step_in] = lhs_elem;
        }
    }
}

__kernel void bitonic_end(__global float *as, unsigned int n, 
                            unsigned int step_in, unsigned int step_out) {
    unsigned int id = get_global_id(0); 
    unsigned int lc_id = get_local_id(0);

    __local float lc_as[WORK_GROUP_SIZE];
    if (id < n) {
        lc_as[lc_id] = as[id];
    }
    uint is_rise = id % (2 * step_out) < step_out;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (; step_in > 0; step_in /= 2) {
        if (id + step_in < n && id % (2 * step_in) < step_in) {
            float lhs_elem = lc_as[lc_id];
            float rhs_elem = lc_as[lc_id + step_in];
            
            if (is_rise == true) {
                if (lhs_elem > rhs_elem) {
                    lc_as[lc_id] = rhs_elem;
                    lc_as[lc_id + step_in] = lhs_elem;
                }
            } else {
                if (lhs_elem < rhs_elem) {
                    lc_as[lc_id] = rhs_elem;
                    lc_as[lc_id + step_in] = lhs_elem;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (id < n) {
        as[id] = lc_as[lc_id];
    }
}
