#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128

__kernel void bitonic(__global float *as,
                     unsigned int n,
                     unsigned int seq_length,
                     unsigned int block_length) {
  int id = get_global_id(0);
  int distance = block_length / 2;
  // id % block_length -- number in block, we should be in the first half
  if (id + distance  >= n || id % block_length >= block_length / 2) {
    return;
  }
  // alternate monotonicity
  bool asc = (id / seq_length) % 2 == 0;
  if (asc && (as[id] > as[id + distance]) || !asc && (as[id] < as[id + distance])) {
    float tmp = as[id];
    as[id] = as[id + distance];
    as[id + distance] = tmp;
  }
}

__kernel void bitonic_local(__global float *as,
                           unsigned int n,
                           unsigned int seq_length,
                           unsigned int block_length) {
  int id = get_global_id(0);
  int local_id = get_local_id(0);

  __local float buffer[WORK_GROUP_SIZE];
  if (id < n) {
    buffer[local_id] = as[id];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  bool asc = (id / seq_length) % 2 == 0;

  for (int distance = block_length / 2; distance >= 1; distance /= 2) {
    if (id + distance < n && id % (2 * distance) < distance) {
      if (asc && (buffer[local_id] > buffer[local_id + distance]) ||
         !asc && (buffer[local_id] < buffer[local_id + distance])) {
        float tmp = buffer[local_id];
        buffer[local_id] = buffer[local_id + distance];
        buffer[local_id + distance] = tmp;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (id < n) {
    as[id] = buffer[local_id];
  }
}
