#define WORK_TILE_SIZE 16
#define WORK_TILE_SIZE_MOD 0b1111U
#define PRECISION double

__kernel void matrix_multiplication(
        __global const float* a_matrix,
        __global const float* b_matrix,
        __global float* c_matrix,
        const unsigned int sm,
        const unsigned int sk,
        const unsigned int sn
        )
{
    // Work-item index info
    const unsigned int lx = get_local_id(0);
    const unsigned int gx = get_global_id(0);
    const unsigned int ly = get_local_id(1);
    const unsigned int gy = get_global_id(1);

    // Tile buffers
    __local float a_tile[WORK_TILE_SIZE * WORK_TILE_SIZE];
    __local float b_tile[WORK_TILE_SIZE * WORK_TILE_SIZE];

    // Multiplication result for C[gy, gx]
    PRECISION sum = 0.f;

    // Iterating by tile pairs
    for (int tile_k = 0; tile_k * WORK_TILE_SIZE < sk; ++tile_k) {
        // Coalesced reading tiles
        const unsigned int shifted_wb_lx = (lx + ly) & WORK_TILE_SIZE_MOD;
        a_tile[ly * WORK_TILE_SIZE + lx] = a_matrix[gy * sk + (tile_k * WORK_TILE_SIZE + lx)];
        b_tile[ly * WORK_TILE_SIZE + shifted_wb_lx] = b_matrix[(tile_k * WORK_TILE_SIZE + ly) * sn + gx];
        barrier(CLK_LOCAL_MEM_FENCE);
        // Update result accumulator
        for (int ind = 0; ind < WORK_TILE_SIZE; ++ind) {
            const unsigned int shifted_rb_lx = (lx + ind) & WORK_TILE_SIZE_MOD;
            sum += (PRECISION) a_tile[ly * WORK_TILE_SIZE + ind] * b_tile[ind * WORK_TILE_SIZE + shifted_rb_lx];
        }
    }

    // Writting result to output
    c_matrix[gy * sn + gx] = (float) sum;
}