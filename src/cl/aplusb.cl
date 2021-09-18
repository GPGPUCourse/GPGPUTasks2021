
__kernel void aplusb(__global const float* a, __global const float* b, __global float* c, unsigned n)
{
    unsigned index = get_global_id(0);

    if (index < n) {
        c[index] = a[index] + b[index];
    }
}
