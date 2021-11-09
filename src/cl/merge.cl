unsigned calc_offset(unsigned my_index, __global const float* my_start, __global const float* reference_start, unsigned l, unsigned r, bool is_left) {
    while (l != r - 1) {
        unsigned middle = (r - l) / 2;
        if (is_left) {
            if (reference_start[l + middle - 1] < my_start[my_index]) {
                if (reference_start[l + middle] >= my_start[my_index]) {
                    return my_index + middle + l;
                } else {
                    l = l + middle;
                }
            } else {
                r = l + middle;
            }
        } else {
            if (reference_start[l + middle - 1] <= my_start[my_index]) {
                if (reference_start[l + middle] > my_start[my_index]) {
                    return my_index + middle + l;
                } else {
                    l = l + middle;
                }
            } else {
                r = l + middle;
            }
        }
    }
    if (is_left) {
        if (my_start[my_index] <= reference_start[l]) {
            return my_index + l;
        }
        return my_index + r;
    } else {
        if (my_start[my_index] < reference_start[l]) {
            return my_index + l;
        }
        return my_index + r;
    }
}

__kernel void merge(__global const float* a, __global float* b, unsigned merge_len) {
    bool is_left = (get_global_id(0) % merge_len) < (merge_len / 2);
    unsigned group_offset = (get_global_id(0) / merge_len) * merge_len;

    __global const float* my_start = is_left ? a + group_offset : a + group_offset + merge_len / 2;
    __global const float* reference_start = is_left ? a + group_offset + merge_len / 2 : a + group_offset;
    unsigned my_index = get_global_id(0) % (merge_len / 2);

    unsigned res = calc_offset(my_index, my_start, reference_start, 0, merge_len / 2, is_left);
    b[group_offset + res] = my_start[my_index];
}
