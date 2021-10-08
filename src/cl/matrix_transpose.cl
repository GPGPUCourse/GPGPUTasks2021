__kernel void matrix_transpose(__global const float* inp, __global float* out, unsigned int nr, unsigned int nc)
{
    // Work-item index info
    const unsigned int lr = get_local_id(0);
    const unsigned int gr = get_global_id(0);
    const unsigned int lc = get_local_id(1);
    const unsigned int gc = get_global_id(1);

    // Naive transpose
    if (gr < nr && gc < nc) {
        out[gc * nr + gr] = inp[gr * nc + gc];
    }
}
