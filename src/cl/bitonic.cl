#define WORK_GROUP_SIZE 128

void load_buff(__global float *as, unsigned int as_size,
               __local float *buff)
{
    const unsigned int local_id  = get_local_id(0);
    const unsigned int global_id = get_global_id(0);

    if (global_id < as_size) {
        buff[local_id] = as[global_id];
    }
    else { // дополним нулями
        buff[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE); // ждем загрузку в __local
}

void save_buff(__global float *as, unsigned int as_size,
               __local float *buff)
{
    const unsigned int local_id  = get_local_id(0);
    const unsigned int global_id = get_global_id(0);

    if (global_id < as_size) {
        as[global_id] = buff[local_id];
    }
}

void swap(__local float *buff, unsigned int lhs, unsigned int rhs) {
    float temp = buff[lhs];
    buff[lhs] = buff[rhs];
    buff[rhs] = temp;
}

void do_bitonic(__local float *buff,
                unsigned int from, unsigned int i, unsigned int as_size)
{ // самое тяжелое в работе программиста -- придумывать названия функциям
    const unsigned int local_id  = get_local_id(0);
    const unsigned int global_id = get_global_id(0);

    bool is_increasing = (global_id / i) % 2 == 0; // граница направления сортировки

    for (unsigned int j = from; j > 1; j /= 2) {
        unsigned int lhs_index = local_id;
        unsigned int rhs_index = local_id + j / 2;

        if (rhs_index < as_size && (local_id % j) < j / 2) {
            bool is_less = buff[lhs_index] < buff[rhs_index];
            if (is_increasing != is_less) { // нужно отсортировать?
                swap(buff, lhs_index, rhs_index);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); // ждем завершение работы блока
    }
}

__kernel void bitonic_local_start(__global float *as, unsigned int as_size) {
    __local float buff[WORK_GROUP_SIZE];
    load_buff(as, as_size, buff);

    for (unsigned int i = 2; i <= WORK_GROUP_SIZE; i *= 2) {
        do_bitonic(buff, i, i, as_size);
    }

    save_buff(as, as_size, buff);
}

__kernel void bitonic_local_end(__global float *as, unsigned int as_size, unsigned int i) {
    __local float buff[WORK_GROUP_SIZE];
    load_buff(as, as_size, buff);

    do_bitonic(buff, WORK_GROUP_SIZE, i, as_size);

    save_buff(as, as_size, buff);
}

__kernel void bitonic(__global float *as, unsigned int as_size, unsigned int i, unsigned int j) {
    const unsigned int global_id = get_global_id(0);

    bool is_increasing = (global_id / i) % 2 == 0; // граница направления сортировки
    unsigned int lhs_index = global_id;
    unsigned int rhs_index = global_id + j / 2;

    if (rhs_index < as_size && (global_id % j) < j / 2) {
        float lhs = as[lhs_index];
        float rhs = as[rhs_index];

        bool is_less = lhs < rhs;
        if (is_increasing != is_less) { // нужно отсортировать?
            as[lhs_index] = rhs;
            as[rhs_index] = lhs;
        }
    }
}
