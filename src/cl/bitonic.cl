
__kernel void bitonic(__global float *as, unsigned int n, unsigned int k, unsigned int t) {
    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    if (global_id >= n / 2) {
        return;
    }

    __local float local_mem[256];

    if (t <= 7) {
        unsigned int r = global_id % (1 << t);
        unsigned int whole = (global_id / (1 << t)) * (1 << (t + 1));
        
        local_mem[local_id * 2] = as[whole + r * 2];
        local_mem[local_id * 2 + 1] = as[whole + r * 2 + 1];
        barrier(CLK_LOCAL_MEM_FENCE); 

        int orientation = ((global_id % (1 << k)) < (1 << (k - 1)) ? 1 : -1);

        for (int s = t; s >= 0; s--) {
            unsigned int local_block_num = local_id / (1 << s);
            whole = local_block_num * (1 << (s + 1));
            r = local_id % (1 << s);
            if (orientation == -1) {
                r += (1 << s);
            }
            
            unsigned int begin = whole + r;
            unsigned int end  = begin + orientation * (1 << s);
            if (local_mem[begin] > local_mem[end]) {
                float x = local_mem[begin];
                local_mem[begin] = local_mem[end];
                local_mem[end] = x;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        whole = (global_id / (1 << t)) * (1 << (t + 1));
        r = global_id % (1 << t);
        as[whole + r * 2] = local_mem[local_id * 2]; 
        as[whole + r * 2 + 1] = local_mem[local_id * 2 + 1];
    }
    else {
        unsigned int remainder = global_id % (1 << k);
        unsigned int whole = (global_id / (1 << k)) * (1 << (k + 1));
        unsigned int ind;
        unsigned int dest; 
        bool second_part = (remainder >= (1 << (k - 1)));
        unsigned int rr = remainder % (1 << t);
        unsigned int rw = (remainder / (1 << t)) * (1 << (t + 1));
        remainder = rw + rr;
        if (second_part) {
            remainder += (1 << t);
            ind = whole + remainder;
            dest = ind - (1 << t);
        }
        else{
            ind = whole + remainder;
            dest = ind + (1 << t);
        }

        if (as[ind] > as[dest]) {
            float x = as[ind];
            as[ind] = as[dest];
            as[dest] = x;
        }
    }
}
