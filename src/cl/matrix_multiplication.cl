#define WG_SIZE 16

__kernel void matrix_multiplication(
    __global const float* as,
    __global const float* bs,
    __global float* cs,
    unsigned M, unsigned K, unsigned N
) {
    unsigned g_col = get_global_id(0);
    unsigned g_row = get_global_id(1);

    unsigned l_col = get_local_id(0);
    unsigned l_row = get_local_id(1);

    __local float as_buf[WG_SIZE][WG_SIZE];
    __local float bs_buf[WG_SIZE][WG_SIZE];

    float result = .0f;
    for (unsigned i = 0; i < K; i += WG_SIZE) {
        if (i > 0) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (g_col < N && g_row < M) {
            as_buf[l_row][l_col] = as[g_row * K + i + l_col];
            bs_buf[l_row][l_col] = bs[(i + l_row) * N + g_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint j = 0; j < WG_SIZE; ++j) {
            result += as_buf[l_row][j] * bs_buf[j][l_col];
        }
    }

    if (g_col < N && g_row < M) {
        cs[g_row * N + g_col] = result;
    }
}
