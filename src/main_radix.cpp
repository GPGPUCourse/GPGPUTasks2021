#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <iostream>
#include <stdexcept>
#include <array>
#include <vector>
#include <iterator>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

void calc_c_tree(ocl::Kernel& calc_c_tree_kernel, gpu::gpu_mem_32u& c_tree_gpu, unsigned c_size, unsigned work_group_size) {
    unsigned current_reduction_size = c_size / 2;
    unsigned read_offset = 0;
    unsigned write_offset = current_reduction_size * 2;
    while (current_reduction_size) {
        calc_c_tree_kernel.exec(gpu::WorkSize(work_group_size, current_reduction_size), c_tree_gpu, read_offset, write_offset, current_reduction_size);
        read_offset = write_offset;
        write_offset += current_reduction_size;
        current_reduction_size /= 2;
    }
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 1;
    unsigned int n = 128;

    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (static_cast<double>(n) / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }
    std::array<gpu::gpu_mem_32u, 2> buffers;
    buffers[0].resizeN(n);
    buffers[1].resizeN(n);

    const unsigned int workGroupSize = 32;
    const unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
    const unsigned k = 2;
    unsigned int c_size = n / workGroupSize * (1 << k);
    unsigned c_tree_size = c_size * 2;

    //as = {128, 0, 1, 2, 5, 1, 0, 0, 3, 2, 4, 7, 1, 2, 1, 1, 1, 0, 1, 2, 5, 1, 0, 0, 3, 2, 4, 7, 1, 2, 1, 1, 1, 0, 1, 2, 5, 1, 0, 0, 3, 2, 4, 7, 1, 2, 1, 1, 1, 0, 1, 2, 5, 1, 0, 0, 3, 2, 4, 7, 1, 2, 1, 1, 1, 0, 1, 2, 5, 1, 0, 0, 3, 2, 4, 7, 1, 2, 1, 1, 1, 0, 1, 2, 5, 1, 0, 0, 3, 2, 4, 7, 1, 2, 1, 1, 1, 0, 1, 2, 5, 1, 0, 0, 3, 2, 4, 7, 1, 2, 1, 1, 1, 0, 1, 2, 5, 1, 0, 0, 3, 2, 4, 7, 1, 2, 1, 1};
    //as = {128, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 128};
    //as = {1024, 20120122, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    //as = {1024, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    gpu::gpu_mem_32u c_init_gpu;
    c_init_gpu.resizeN(c_size);

    gpu::gpu_mem_32u c_tree_gpu;
    c_tree_gpu.resizeN(c_tree_size);

    gpu::gpu_mem_32u o_gpu;
    o_gpu.resizeN(c_size);

    {
        ocl::Kernel calc_c_init(radix_kernel, radix_kernel_length, "calc_c_init", "-D WORK_GROUP_SIZE=" + std::to_string(workGroupSize));
        calc_c_init.compile();

        ocl::Kernel tranpose_init_c_kernel(radix_kernel, radix_kernel_length, "matrix_transpose", "-D WORK_GROUP_SIZE=" + std::to_string(workGroupSize));
        tranpose_init_c_kernel.compile();

        ocl::Kernel calc_c_tree_kernel(radix_kernel, radix_kernel_length, "calc_c_tree", "-D WORK_GROUP_SIZE=" + std::to_string(workGroupSize));
        calc_c_tree_kernel.compile();

        ocl::Kernel calc_o_kernel(radix_kernel, radix_kernel_length, "calc_o", "-D WORK_GROUP_SIZE=" + std::to_string(workGroupSize));
        calc_o_kernel.compile();

        ocl::Kernel local_counting_kernel(radix_kernel, radix_kernel_length, "local_counting_sort", "-D WORK_GROUP_SIZE=" + std::to_string(workGroupSize));
        local_counting_kernel.compile();

        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix", "-D WORK_GROUP_SIZE=" + std::to_string(workGroupSize));
        radix.compile();

    std::copy(as.begin(), as.end(), std::ostream_iterator<unsigned>(std::cout, " "));
    std::cout << std::endl;
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            buffers[0].writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            for (int stage = 0; stage < 32 / k; ++stage) {
    std::cout << stage << std::endl << std::endl;
                calc_c_init.exec(gpu::WorkSize(workGroupSize, global_work_size), buffers[0], c_init_gpu, n, stage);
                tranpose_init_c_kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), c_init_gpu, c_tree_gpu, (1 << k), c_size / (1 << k), workGroupSize);

                calc_c_tree(calc_c_tree_kernel, c_tree_gpu, c_size, workGroupSize);
                calc_o_kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), c_tree_gpu, o_gpu, c_size);
                local_counting_kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), buffers[0], n, stage);
                std::cout << "aoeu\n";
    buffers[0].readN(as.data(), n);
    for (int i = 0; i != 4; ++i) {
        std::copy(as.begin() + 32 * i, as.begin() + 32 * (i + 1), std::ostream_iterator<unsigned>(std::cout, " "));
        std::cout << std::endl;
    }
                radix.exec(gpu::WorkSize(workGroupSize, global_work_size), buffers[0], o_gpu, c_init_gpu, buffers[1], n, stage);
                std::swap(buffers[0], buffers[1]);

    c_init_gpu.readN(as.data(), c_size);
    std::copy(as.begin(), as.begin() + c_size, std::ostream_iterator<unsigned>(std::cout, " "));
    std::cout << std::endl;

    c_tree_gpu.readN(as.data(), c_tree_size);
    std::copy(as.begin(), as.begin() + c_tree_size, std::ostream_iterator<unsigned>(std::cout, " "));
    std::cout << std::endl;

    o_gpu.readN(as.data(), c_size);
    for (int i = 0; i != 4; ++i) {
        std::copy(as.begin() + 4 * i, as.begin() + 4 * (i + 1), std::ostream_iterator<unsigned>(std::cout, " "));
        std::cout << std::endl;
    }

    buffers[0].readN(as.data(), n);
    for (int i = 0; i != 4; ++i) {
        std::copy(as.begin() + 32 * i, as.begin() + 32 * (i + 1), std::ostream_iterator<unsigned>(std::cout, " "));
        std::cout << std::endl;
    }
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (static_cast<double>(n) / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        buffers[0].readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
