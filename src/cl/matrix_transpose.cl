#define WG_SIZE 16

__kernel void matrix_transpose(
    __global const float* in_mat,
    __global float* out_mat,
    unsigned rows, unsigned cols
) {
    __local float buffer[WG_SIZE][WG_SIZE];

    uint l_col = get_local_id(0);
    uint l_row = get_local_id(1);

    uint g_col = get_global_id(0);
    uint g_row = get_global_id(1);

    if (g_col < cols && g_row < rows) {
        buffer[l_row][l_col] = in_mat[g_row * cols + g_col];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (g_col < cols && g_row < rows) {
        out_mat[g_col * rows + g_row] = buffer[l_row][l_col];
    }
}