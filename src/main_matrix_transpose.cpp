#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/opencl/device_info.h>

#include "cl/matrix_transpose_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int rows_num = 1025;
    unsigned int cols_num = 1025;

    std::vector<float> as(rows_num*cols_num, 0);
    std::vector<float> as_t(rows_num*cols_num, 0);

    FastRandom r(rows_num+cols_num);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = i;
        //as[i] = r.nextf();
    }
    std::cout << "Data generated for M=" << rows_num << ", K=" << cols_num << "!" << std::endl;

    gpu::gpu_mem_32f as_gpu, as_t_gpu;
    as_gpu.resizeN(rows_num*cols_num);
    as_t_gpu.resizeN(cols_num*rows_num);
    as_gpu.writeN(as.data(), rows_num*cols_num);

    auto device_info = ocl::DeviceInfo();
    device_info.init(device.device_id_opencl);
    unsigned int work_group_size = 128;
    if (device_info.warp_size == 0) {
        device_info.warp_size = 1;
        work_group_size = 1;
    }

    ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "matrix_transpose", "-DWS=" + to_string(device_info.warp_size));

    matrix_transpose_kernel.compile(true);

    {

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // TODO
            unsigned int global_work_size = cols_num*rows_num;
            // Для этой задачи естественнее использовать двухмерный NDRange. Чтобы это сформулировать
            // в терминологии библиотеки - нужно вызвать другую вариацию конструктора WorkSize.
            // В CLion удобно смотреть какие есть вариант аргументов в конструкторах:
            // поставьте каретку редактирования кода внутри скобок конструктора WorkSize -> Ctrl+P -> заметьте что есть 2, 4 и 6 параметров
            // - для 1D, 2D и 3D рабочего пространства соответственно
            unsigned group_size_x = device_info.warp_size;
            unsigned group_size_y = work_group_size / device_info.warp_size;
            unsigned work_size_x = cols_num;
            unsigned work_size_y = std::ceil(static_cast<double>(rows_num) / group_size_y);

            gpu::WorkSize work_size (group_size_x, group_size_y, work_size_x, work_size_y);
            matrix_transpose_kernel.exec(work_size , as_gpu, as_t_gpu, cols_num, rows_num, work_group_size);

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << rows_num*cols_num/1000.0/1000.0 / t.lapAvg() << " millions/s" << std::endl;
    }

    as_t_gpu.readN(as_t.data(), rows_num*cols_num);

    // Проверяем корректность результатов
    for (int j = 0; j < rows_num; ++j) {
        for (int i = 0; i < cols_num; ++i) {
            float a = as[j * cols_num + i];
            float b = as_t[i * rows_num + j];
            if (a != b) {
                std::cerr << "Not the same!" << std::endl;
                return 1;
            }
        }
    }

    return 0;
}
