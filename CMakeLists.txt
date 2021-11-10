cmake_minimum_required(VERSION 3.1)

add_subdirectory(libs)

project(task6)

set(CMAKE_CXX_STANDARD 17)

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

convertIntoHeader(src/cl/merge.cl src/cl/merge_cl.h merge_kernel)
add_executable(merge src/main_merge.cpp src/cl/merge_cl.h)
target_link_libraries(merge libclew libgpu libutils)
