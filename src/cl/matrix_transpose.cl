#define BLOCK_SIZE 16

__kernel void matrix_transpose(__global const float* matrix,
                               __global       float* matrixT,
                               unsigned int width,
                               unsigned int height)
{
    unsigned int col = get_global_id(0);
    unsigned int row = get_global_id(1);


    __local float swap[(BLOCK_SIZE + 1) * BLOCK_SIZE];

    //write to swap
    if (row < height && col < width) {
        swap[get_local_id(1) * (BLOCK_SIZE + 1) + get_local_id(0)] = matrix[row * width + col];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //write from swap to matrixT
    unsigned int newCol = get_group_id(1) * BLOCK_SIZE + get_local_id(0);
    unsigned int newRow = get_group_id(0) * BLOCK_SIZE + get_local_id(1);
    if(newCol < height && newRow < width) {
        matrixT[newRow * height + newCol] = swap[get_local_id(0) * (BLOCK_SIZE + 1) + get_local_id(1)];
    }


}