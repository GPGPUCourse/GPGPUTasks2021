#ifdef __CLION_IDE__
// Этот include виден только для CLion парсера, это позволяет IDE "знать" ключевые слова вроде __kernel, __global
// а также уметь подсказывать OpenCL методы, описанные в данном инклюде (такие как get_global_id(...) и get_local_id(...))
#include "clion_defines.cl"
#endif

#line 8 // Седьмая строчка теперь восьмая (при ошибках компиляции в логе компиляции будут указаны корректные строчки благодаря этой директиве)

__kernel void aplusb(__global float *as, __global float *bs, __global float *cs, unsigned int sz)
{
    unsigned int glId = get_global_id(0);
    if (glId >= sz) {
        return;
    }
    cs[glId] = as[glId] + bs[glId];
}
