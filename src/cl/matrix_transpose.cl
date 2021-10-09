__kernel void matrix_transpose(__global const float* inp, __global float* out, unsigned int nx, unsigned int ny)
{
    // Work-item index info
    const unsigned int lx = get_local_id(0);
    const unsigned int gx = get_global_id(0);
    const unsigned int ly = get_local_id(1);
    const unsigned int gy = get_global_id(1);

    // Naive transpoce
    if (gy < ny && gx < nx) {
        out[gx * ny + gy] = inp[gy * nx + gx];
    }
}
