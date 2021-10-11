#define BLOCK_SIZE 16


__kernel void matrix_multiplication(__global const float* as,
                                    __global const float* bs,
                                    __global float* cs,
                                    int M,
                                    int K,
                                    int N)
{

    __local float a_block[(BLOCK_SIZE + 1) * BLOCK_SIZE];
    __local float b_block[(BLOCK_SIZE + 1) * BLOCK_SIZE];
    __local float c_block[(BLOCK_SIZE + 1) * BLOCK_SIZE];

    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    const unsigned int blockX = get_group_id(0);
    const unsigned int blockY = get_group_id(1);

    const unsigned int localX = get_local_id(0);
    const unsigned int localY = get_local_id(1);

    c_block[localY * BLOCK_SIZE + localX] = 0.0;
    barrier(CLK_LOCAL_MEM_FENCE);

//    if(x == 0 && y == 0) {
//        for (int i = 0; i < 5; i++) {
//            for (int j = 0; j < 5; j++) {
//                printf("%f", bs[i * 16 + j]);
//            }
//        }
//        printf("\n");
//    }


    for (int w = 0; w < K; w += BLOCK_SIZE) {
        //copy to local memory

        a_block[localY * BLOCK_SIZE + localX] = as[(blockY * BLOCK_SIZE + localY) * K + localX + w];
        // matrix should be already transposed
        b_block[localY * BLOCK_SIZE + localX] = bs[(blockX * BLOCK_SIZE + localY) * K + localX + w];
        barrier(CLK_LOCAL_MEM_FENCE);


//        if(x == 0 && y == 0) {
//            for (int i = 0; i < 5; i++) {
//                for (int j = 0; j < 5; j++) {
//                    printf("%f", b_block[i * 16 + j]);
//                }
//            }
//            printf("\n");
//        }


        for (int localW = 0; localW < BLOCK_SIZE; localW++) {
            c_block[localY * BLOCK_SIZE + localX] += a_block[localY * BLOCK_SIZE + localW] * b_block[localX * BLOCK_SIZE + localW];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //copy to global memory
    cs[y * N + x] = c_block[localY * BLOCK_SIZE + localX];

}