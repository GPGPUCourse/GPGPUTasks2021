void zero_C(__local float C_mem[WS][WS], unsigned work_group_size) {
    for (int i = 0; i != WS * WS / work_group_size; ++i) {
        unsigned C_mem_x = get_local_id(0);
        unsigned C_mem_y = get_local_id(1) + i * get_local_size(1);

        C_mem[C_mem_x][C_mem_y] = 0;
    }
}

void load_local(__global const float* A, __global const float* B, __local float A_mem[WS][WS], __local float B_mem[WS][WS], unsigned stage, unsigned work_group_size, unsigned M, unsigned K, unsigned N) {
    for (int i = 0; i != WS * WS / work_group_size; ++i) {
        unsigned A_mem_x = get_local_id(0);
        unsigned A_mem_y = get_local_id(1) + i * get_local_size(1);

        unsigned A_in_x = get_local_id(0) + stage * WS;
        unsigned A_in_y = get_group_id(1) * WS + get_local_id(1) + i * get_local_size(1);

        unsigned A_source_index = A_in_x + A_in_y * K; 
        if (A_in_x < K && A_in_y < M) {
            A_mem[A_mem_x][A_mem_y] = A[A_source_index];
        } else {
            A_mem[A_mem_x][A_mem_y] = 0;
        }

        unsigned B_mem_x = get_local_id(0);
        unsigned B_mem_y = get_local_id(1) + i * get_local_size(1);

        unsigned B_in_x = get_global_id(0);
        unsigned B_in_y = get_local_id(1) + stage * WS + i * get_local_size(1);

        unsigned B_source_index = B_in_x + B_in_y * N; 

        if (B_in_x < N && B_in_y < K) {
            B_mem[B_mem_x][B_mem_y] = B[B_source_index];
        } else {
            B_mem[B_mem_x][B_mem_y] = 0;
        }
    }
}

void multiply(__global const float* A, __global const float* B, __global float* C, __local float A_mem[WS][WS], __local float B_mem[WS][WS], __local float C_mem[WS][WS], unsigned M, unsigned K, unsigned N, unsigned work_group_size) {
    for (int i = 0; i != WS * WS / work_group_size; ++i) {
        unsigned C_mem_x = get_local_id(0);
        unsigned C_mem_y = get_local_id(1) + i * get_local_size(1);

        unsigned A_mem_y = C_mem_y;
        unsigned B_mem_x = C_mem_x;

        for (int j = 0; j != WS; ++j) {
            unsigned A_mem_x = j;

            unsigned B_mem_y = j;

            C_mem[C_mem_x][C_mem_y] += A_mem[A_mem_x][A_mem_y] * B_mem[B_mem_x][B_mem_y];
        }
    }
}
    
void store_local(__global float* C, __local float C_mem[WS][WS], unsigned M, unsigned N, unsigned work_group_size) {
    for (int i = 0; i != WS * WS / work_group_size; ++i) {
        unsigned C_mem_x = get_local_id(0);
        unsigned C_mem_y = get_local_id(1) + i * get_local_size(1);

        unsigned C_out_x = get_global_id(0);
        unsigned C_out_y = get_group_id(1) * WS + get_local_id(1) + i * get_local_size(1);

        unsigned C_store_index = C_out_x + C_out_y * N;

        if (C_out_x < N && C_out_y < M) {
            C[C_store_index] += C_mem[C_mem_x][C_mem_y];
        }
    }
}
    

__kernel void matrix_multiplication(__global const float* A, __global const float* B, __global float* C, unsigned M, unsigned K, unsigned N, unsigned work_group_size) 
{
    __local float A_mem[WS][WS];
    __local float B_mem[WS][WS];
    __local float C_mem[WS][WS];


    unsigned total_stages = K / WS + (K % WS ? 1 : 0);

    for (int stage = 0; stage != total_stages; ++stage) {
        zero_C(C_mem, work_group_size);
        load_local(A, B, A_mem, B_mem, stage, work_group_size, M, K, N);
        barrier(CLK_LOCAL_MEM_FENCE);
        multiply(A, B, C, A_mem, B_mem, C_mem, M, K, N, work_group_size);
        barrier(CLK_LOCAL_MEM_FENCE);
        store_local(C, C_mem, M, N, work_group_size);
    }
}
