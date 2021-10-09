#define WORK_TILE_SIZE 16

__kernel void matrix_transpose(__global const float* inp, __global float* out, unsigned int nx, unsigned int ny)
{
    // Work-item index info
    const unsigned int lx = get_local_id(0);
    const unsigned int gx = get_global_id(0);
    const unsigned int ly = get_local_id(1);
    const unsigned int gy = get_global_id(1);

    // Tile top-left corner index
    const unsigned int sx = get_group_id(0) * WORK_TILE_SIZE;
    const unsigned int sy = get_group_id(1) * WORK_TILE_SIZE;

    // Target index for coalesced write phase
    const unsigned int tx = sy + lx;
    const unsigned int ty = sx + ly;

    // Transpose in local memory
    __local float buffer[WORK_TILE_SIZE*WORK_TILE_SIZE];

    // Phase 1: read + transpose
    buffer[lx * WORK_TILE_SIZE + ly] = (gx < nx && gy < ny) ? inp[gy * nx + gx] : 0.f;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: write to global memory
    if (tx < ny && ty < nx) {
        out[ty * ny + tx] = buffer[ly * WORK_TILE_SIZE + lx];
    }
}
