#define LOCAL_SIZE 256

__kernel void bitonic(__global float *as, int n, int blockSize, int bitonicInterval) {
  int id = get_global_id(0);
  if (id * 2 >= n) {
    return;
  }
  int dist = blockSize / 2;
  int pos = id / dist * blockSize + id % dist;
  float a = as[pos];
  float b = as[pos + dist];
  bool less = a < b;
  bool shouldBeLess = (pos / bitonicInterval & 1) == 0;
  if (less ^ shouldBeLess) {
    as[pos] = b;
    as[pos + dist] = a;
  }
}

__kernel void bitonic_local(__global float *as, int n, int blockSize, int bitonicInterval) {
  int g_id = get_global_id(0);
  if (g_id * 2 >= n) {
    return;
  }

  __local volatile float as_local[LOCAL_SIZE];
  int l_id = g_id % (LOCAL_SIZE / 2);
  as_local[l_id] = as[g_id];
  as_local[l_id + LOCAL_SIZE / 2] = as[g_id + LOCAL_SIZE / 2];

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int bSize = blockSize; bSize >= 2; bSize /= 2) {
    int dist = bSize / 2;
    int g_pos = g_id / dist * bSize + g_id % dist;
    int l_pos = g_pos % (LOCAL_SIZE / 2);
    float a = as_local[l_pos];
    float b = as_local[l_pos + dist];
    bool less = a < b;
    bool shouldBeLess = (g_pos / bitonicInterval & 1) == 0;
    if (less ^ shouldBeLess) {
      as_local[l_pos] = b;
      as_local[l_pos + dist] = a;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  as[g_id] = as_local[l_id];
  as[g_id + LOCAL_SIZE / 2] = as_local[l_id + LOCAL_SIZE / 2];
}
