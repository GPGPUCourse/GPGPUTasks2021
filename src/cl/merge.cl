#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void merge(__global float* a, __global float* b, unsigned int n, unsigned int i) {
    unsigned int ind = get_global_id(0);

    if (ind >= n) {
        return;
    }

    unsigned int a_begin = ind / (i * 2) * (i * 2);

    if (a_begin >= n){
        return;
    }

    unsigned int a_end = min(a_begin + i, n);
    unsigned int b_begin = a_end;
    unsigned int b_end = min(b_begin + i, n);

    int l = ind < a_end ? a_end : a_begin;
    int r = ind < a_end ? b_end : a_end;
    while (l < r) {
        int m = (l + r) / 2;
        if ((a[m] < a[ind]) || (m < b_begin && a[m] == a[ind])) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    
    b[ind + l - b_begin] = a[ind];
}