cmake_minimum_required(VERSION 3.1)

add_subdirectory(libs)

project(task3)

set(CMAKE_CXX_STANDARD 11)

# OpenMP позволит распараллеливать циклы на все ядра процессора простыми директивами

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(WARNING "OpenMP not found!")
endif()

# convertIntoHeader CMake-функция объявлена в libs/gpu/CMakeLists.txt:71
# Она считывает все байты из файла src/cl/aplusb.cl (т.е. весь исходный код кернела) и преобразует их в массив байтов в файле src/cl/aplusb_cl.h aplusb_kernel
# Обратите внимание что это происходит на этапе компиляции, кроме того необходимо чтобы файл src/cl/aplusb_cl.h был перечислен среди исходников для компиляции при вызове add_executable
convertIntoHeader(src/cl/aplusb.cl src/cl/aplusb_cl.h aplusb_kernel)
add_executable(aplusb src/main_aplusb.cpp src/cl/aplusb_cl.h)
target_link_libraries(aplusb libclew libgpu libutils)

# Аналогично для фрактала Мандельброта
convertIntoHeader(src/cl/mandelbrot.cl src/cl/mandelbrot_cl.h mandelbrot_kernel)
add_executable(mandelbrot src/main_mandelbrot.cpp src/cl/mandelbrot_cl.h)
target_link_libraries(mandelbrot libclew libgpu libimages libutils)

# Для вычисления суммы
convertIntoHeader(src/cl/sum.cl src/cl/sum_cl.h sum_kernel)
add_executable(sum src/main_sum.cpp src/cl/sum_cl.h)
target_link_libraries(sum libclew libgpu libutils)

# Для вычисления максимального по сумме префикса
convertIntoHeader(src/cl/max_prefix_sum.cl src/cl/max_prefix_sum_cl.h max_prefix_sum_kernel)
add_executable(max_prefix_sum src/main_max_prefix_sum.cpp src/cl/max_prefix_sum_cl.h)
target_link_libraries(max_prefix_sum libclew libgpu libutils)
