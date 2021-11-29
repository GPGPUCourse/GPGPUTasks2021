#define WORK_GROUP_len 128
#define quan 16

__kernel void count(__global const uint *as, __global uint *cs, uint offset, uint grps) {
    /* code */
    uint lc_id = get_local_id(0);
    uint gp_id = get_group_id(0);
    uint gl_id = get_global_id(0);

    __local unsigned lc_cs[WORK_GROUP_len][quan];

    for (int i = 0; i < quan; ++i) {
        lc_cs[lc_id][i] = 0;
    }

    uint num = (as[gl_id] >> offset) & (quan - 1);
    ++lc_cs[lc_id][num];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lc_id >= quan) {
        return;
    }

    for (int i = 1; i < WORK_GROUP_len; ++i) {
        lc_cs[0][lc_id] += lc_cs[i][lc_id];
    }

    cs[lc_id * grps + gp_id] = lc_cs[0][lc_id];
}

__kernel void prefix_scan_up(__global uint *cs, uint size, uint d) {
    uint gl_id = get_global_id(0);

    if (gl_id >= size / (1 << (d + 1))) {
        return;
    }

    uint indx = gl_id * (1 << (d + 1));
    cs[indx + (1 << (d + 1)) - 1] += cs[indx + (1 << d) - 1];
}

__kernel void prefix_scan_down(__global uint *cs, uint size, uint d) {
    uint gl_id = get_global_id(0);

    if (gl_id >= size / (1 << (d + 1))) {
        return;
    }

    uint indx = gl_id * (1 << (d + 1));
    uint tmp = cs[indx + (1 << d) - 1];
    cs[indx + (1 << d) - 1] = cs[indx + (1 << (d + 1)) - 1];
    cs[indx + (1 << (d + 1)) - 1] += tmp;
}

__kernel void prefix_scan_end(__global uint *cs, __global uint *out, uint size) {
    uint gl_id = get_global_id(0);

    if (gl_id < size - 1) {
        out[gl_id] = cs[gl_id + 1] - cs[0];
    } else {
        out[size - 1] = cs[0];
    }
}


__kernel void reorder(__global const uint *as, __global const uint *cs, __global uint *out, uint offset, uint grps) {
    /* code */
    uint lc_id = get_local_id(0);
    uint gp_id = get_group_id(0);
    uint gl_id = get_global_id(0);

    __local unsigned int lc_cs[WORK_GROUP_len][quan];

    for (int i = 0; i < quan; ++i) {
        lc_cs[lc_id][i] = 0;
    }

    uint num = (as[gl_id] >> offset) & (quan - 1);
    ++lc_cs[lc_id][num];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lc_id < quan) {
        for (int i = 1; i < WORK_GROUP_len; ++i) {
            lc_cs[i][lc_id] += lc_cs[i - 1][lc_id];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint i = cs[num * grps + gp_id] - lc_cs[WORK_GROUP_len - 1][num] + lc_cs[lc_id][num] - 1;
    out[i] = as[gl_id];
}