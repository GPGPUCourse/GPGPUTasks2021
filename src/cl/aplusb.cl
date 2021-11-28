//#define __CLION_IDE__
#ifdef __CLION_IDE__
// Этот include виден только для CLion парсера, это позволяет IDE "знать" ключевые слова вроде __kernel, __global
// а также уметь подсказывать OpenCL методы, описанные в данном инклюде (такие как get_global_id(...) и get_local_id(...))
#include "clion_defines.cl"
#endif
 
#line 8 // Седьмая строчка теперь восьмая (при ошибках компиляции в логе компиляции будут указаны корректные строчки благодаря этой директиве)

// TODO 5 реализуйте кернел:
// - От обычной функции кернел отличается модификатором __kernel и тем, что возвращаемый тип всегда void
// - На вход дано три массива float чисел; единственное, чем они отличаются от обычных указателей - модификатором __global, т.к. это глобальная память устройства (видеопамять)
// - Четвертым и последним аргументом должно быть передано количество элементов в каждом массиве (unsigned int, главное, чтобы тип был согласован с типом в соответствующем clSetKernelArg в T0D0 10)

__kernel void aplusb(__global float* a, __global float* b, __global float* c, unsigned int n)
{
    unsigned int i = get_global_id(0) % n;

    // Узнать, какой workItem выполняется в этом потоке поможет функция get_global_id
    // см. в документации https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    // OpenCL Compiler -> Built-in Functions -> Work-Item Functions
    c[i] = a[i] + b[i];
    // P.S. В общем случае количество элементов для сложения может быть некратно размеру WorkGroup, тогда размер рабочего пространства округлен вверх от числа элементов до кратности на размер WorkGroup
    // и в таком случае, если сделать обращение к массиву просто по индексу=get_global_id(0), будет undefined behaviour (вплоть до повисания ОС)
    // поэтому нужно либо дополнить массив данных длиной до кратности размеру рабочей группы,
    // либо сделать return в кернеле до обращения к данным в тех WorkItems, где get_global_id(0) выходит за границы данных (явной проверкой)
}
