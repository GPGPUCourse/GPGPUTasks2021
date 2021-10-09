#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_transpose_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>
#include <iomanip>


#define DEBUG false
#define DISPLAY_COL_WIDTH 5


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int M = DEBUG ? 8 : 1024; // number of rows
    unsigned int K = DEBUG ? 16 : 2048; // number of columns

    std::vector<float> as(M*K, 0);
    std::vector<float> as_t(K*M, 0);

    if (DEBUG) {
        for (unsigned int i = 0; i < as.size(); ++i) {
            as[i] = (float) i;
        }
    } else {
        FastRandom random(M+K);
        for (float & a : as) {
            a = random.nextf();
        }
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << "!" << std::endl;

    gpu::gpu_mem_32f as_gpu, as_t_gpu;
    as_gpu.resizeN(M*K);
    as_t_gpu.resizeN(K*M);

    as_gpu.writeN(as.data(), M*K);

    ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "matrix_transpose");
    matrix_transpose_kernel.compile();

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // Для этой задачи естественнее использовать двухмерный NDRange. Чтобы это сформулировать
            // в терминологии библиотеки - нужно вызвать другую вариацию конструктора WorkSize.
            // В CLion удобно смотреть какие есть вариант аргументов в конструкторах:
            // поставьте каретку редактирования кода внутри скобок конструктора WorkSize -> Ctrl+P -> заметьте что есть 2, 4 и 6 параметров
            // - для 1D, 2D и 3D рабочего пространства соответственно
            unsigned int work_group_size = 16;
            auto work_size = gpu::WorkSize(work_group_size, work_group_size, K, M);
            matrix_transpose_kernel.exec(work_size, as_gpu, as_t_gpu, K, M);
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << M*K/1000.0/1000.0 / t.lapAvg() << " millions/s" << std::endl;
    }

    as_t_gpu.readN(as_t.data(), K*M);

    if (DEBUG) {

        // Выводим вход
        std::cout << "Input matrix" << std::endl;
        for (int r = 0; r < M; ++r) {
            for (int c = 0; c < K; ++c) {
                std::cout << std::setw(DISPLAY_COL_WIDTH) << as[r * K + c] << " ";
            }
            std::cout << std::endl;
        }

        // Выводим выход
        std::cout << "Output matrix" << std::endl;
        for (int r = 0; r < K; ++r) {
            for (int c = 0; c < M; ++c) {
                std::cout << std::setw(DISPLAY_COL_WIDTH) << as_t[r * M + c] << " ";
            }
            std::cout << std::endl;
        }

    }

    // Проверяем корректность результатов
    for (int j = 0; j < M; ++j) {
        for (int i = 0; i < K; ++i) {
            float a = as[j * K + i];
            float b = as_t[i * M + j];
            if (a != b) {
                std::cerr << "Not the same!" << std::endl;
                return 1;
            }
        }
    }

    return 0;
}
