#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

__kernel void merge(__global const float *as,
                    __global       float *buffer,
                    unsigned int k, unsigned int n) {
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = global_id % (2 * k);

    unsigned int l, r;
    if (local_id + 1 < k) {
        l = 0;
        r = local_id + 1;
    } else {
        l = local_id + 1 - k;
        r = k;
    }
    unsigned int l_start = (global_id / (2 * k)) * (2 * k);
    unsigned int r_start = global_id + k;
    float result;

    while (l < r) {
        unsigned int m = (l + r) / 2;
        if (as[l_start + m] <= as[r_start - m]) {
            l = m + 1;
        } else {
            r = m;
        }
    }

    if (l == 0) {
        buffer[global_id] = as[r_start - l];
    } else if (l == local_id + 1) {
        buffer[global_id] = as[l_start + l - 1];
    } else {
        buffer[global_id] = max(as[r_start - l], as[l_start + l - 1]);
    }
}