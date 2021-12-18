#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void merge(__global float *as,
                    __global float *buffer,
                    unsigned int n,
                    unsigned int block_size) {
  int id = get_global_id(0);
  if (id >= n) {
    return;
  }
  
  int offset1 = (id / (2 * block_size)) * 2 * block_size;
  int offset2 = offset1 + block_size;
  if (offset2 >= n) {
    buffer[id] = as[id];
    return;
  }

  int l, r;
  if (id < offset2) {
    l = offset2;
    r = offset2 + block_size;
  } else {
    l = offset1;
    r = offset2;
  }

  while (l < r) {
    int m = (l + r) / 2;
    if (as[m] < as[id]) {
      l = m + 1;
    } else {
      r = m;
    }
  }

  r = l;
  while (r < offset2 && as[r] == as[id]) {
    r++;
  }

  int diff, buf_idx;
  if (id < offset2) {
    diff = l - offset2;
    buf_idx = id + diff;
  } else {
    diff = offset2 - r;
    buf_idx = id - diff;
  }
  buffer[buf_idx] = as[id];
}
