#ifdef __CLION_IDE__
    // Этот include виден только для CLion парсера, это позволяет IDE "знать" ключевые слова вроде __kernel, __global
    // а также уметь подсказывать OpenCL методы, описанные в данном инклюде (такие как get_global_id(...) и get_local_id(...))
    #include "clion_defines.cl"
#endif

#line 8// Седьмая строчка теперь восьмая (при ошибках компиляции в логе компиляции будут указаны корректные строчки благодаря этой директиве)

__kernel void aplusb(__global const float *a, __global const float *b, __global float *c)
{
    const unsigned int index = get_global_id(0);
    c[index] = a[index] + b[index];
}
