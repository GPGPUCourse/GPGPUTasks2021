
__kernel void merge(__global const float* as, unsigned int n, unsigned int block_size, __global float* bs) {
  unsigned int global_id = get_global_id(0);

  if (global_id > n) {
    return;
  }
  
  unsigned int b0, b1, b2;
  b0 = b1 = b2 = -1;

  unsigned int local_id_block = global_id % (1 << (block_size + 1));
  b0 = global_id - local_id_block;
  b1 = min(n, b0 + (1 << block_size));
  if (b1 < n) {
    b2 = min(n, b0 + (1 << (block_size + 1)));
  }
  if (b2 == -1) {
    bs[global_id] = as[global_id];
    return;
  }

  bool left = (local_id_block < (1 << block_size) ? true : false);
  float a = as[global_id];

  if (left) {
    unsigned int l = -1;
    unsigned int r = b2 - b1;
    while (r - l > 1) {
      unsigned int m = (r + l) / 2;
      if (a >= as[b1 + m]) {
          l = m;
      }
      else {
        r = m;
      }
    }
    bs[global_id + l + 1] = a;
  }
  else {
    unsigned int l = -1;
    unsigned int r = b1 - b0;
    while (r - l > 1) {
      unsigned int m = (r + l) / 2;
      if (a > as[b0 + m]) {
          l = m;
      }
      else {
        r = m;
      }
    }
    bs[b0 + l + 1 + global_id - b1] = a;
  }
}
