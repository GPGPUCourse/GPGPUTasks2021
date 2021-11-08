#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void merge_cols(__global const float *as, __global float *buf, unsigned int n, unsigned int len) {
  const unsigned int id = get_global_id(0);
  const unsigned int block_start = (id / len) * len * 2;
  const unsigned int col = id % len;

  unsigned int l = -1;
  unsigned int r = len;
  while (r - l > 1) {
    unsigned int m = (l + r) / 2;
    if (as[block_start + m] < as[block_start + len + col]) {
      l = m;
    } else {
      r = m;
    }
  }
  buf[block_start + r + col] = as[block_start + len + col];
}

__kernel void merge_rows(__global const float *as, __global float *buf, unsigned int n, unsigned int len) {
  const unsigned int id = get_global_id(0);
  const unsigned int block_start = (id / len) * len * 2;
  const unsigned int row = id % len;

  unsigned int l = -1;
  unsigned int r = len;
  while (r - l > 1) {
    unsigned int m = (l + r) / 2;
    if (as[block_start + row] < as[block_start + len + m]) {
      r = m;
    } else {
      l = m;
    }
  }
  buf[block_start + row + r] = as[block_start + row];
}

