#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

__kernel void merge(__global float *as, unsigned int size, __global float *out, unsigned int block_size) {
    const unsigned int index = get_global_id(0);
    if (index >= size) {
        return;
    }

    const unsigned int merge_interval = 2 * block_size;
    const unsigned int l = (index / merge_interval) * merge_interval;
    const unsigned int r = l + block_size;
    const unsigned int end = min(r + block_size, size);
    // left block: [l, r); right block: [r, end)
    // internally blocks are sorted already

    const float current = as[index];
    if (r >= size) {
        out[index] = current; // merge is identity on the last block
        return;
    }

    unsigned int from, to;
    if (index < r) { // index is in the left block
        from = r;
        to = end;
    } else {         // index is in the right block
        from = l;
        to = r;
    }
    // [from, to) is the opposite block, i.e. block which does not contain index

    while (from < to) {
        const unsigned int mid = (to - from) / 2 + from;
        if (as[mid] < current) {
            from = mid + 1;
        } else {
            to = mid;
        }
    }
    const unsigned int pivot = from; // smallest index: as[pivot] >= current
    unsigned int next = pivot;       // smallest index: as[pivot] > current
    for (; current == as[next] && next < r; ++next) {}

    // currently: as[i] < current on [l, pivot), as[i] == current on [pivot, next), as[i] > current on [next, r)
    unsigned int count = 0; // how many elements were already merged in our group
    if (index < r) {
        // index is in the left block: l <= i < r; r <= p <= n < end
        count = (index - l) + (pivot - r);
    } else {
        // index is in the right block: l <= p <= n < r; r <= index < end
        count = (next - l) + (index - r);
    }
    out[l + count] = current;
}
