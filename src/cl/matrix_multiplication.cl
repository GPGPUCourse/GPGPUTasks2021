#define GROUP_SIZE 16

__kernel void matrix_multiplication(__global float* a, __global float* b, __global float* c,
                                    uint m, uint k, uint n)
{
    // Произведение матриц M*K x K*N = M*N
    const uint i = get_global_id(0); // номер столбца 0..N
    const uint j = get_global_id(1); // номер строки 0..M
    const uint li = get_local_id(0);
    const uint lj = get_local_id(1);
    const uint mod16 = 16 - 1;

    __local float m1[16][16];
    __local float m2[16][16];
    float res = 0.f;

    for (unsigned s = 0; s < k; s += GROUP_SIZE) {
        m1[lj][li] = a[j * k + s + li];
        m2[lj][li] = b[(lj + s) * n + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned t = 0; t < GROUP_SIZE; ++t) {
            res += m1[lj][t] * m2[t][li];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[j * n + i] = res;
}
