#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
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
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32u as_gpu;
    gpu::gpu_mem_32u out;
    gpu::gpu_mem_32u cs;
    gpu::gpu_mem_32u buff;

    constexpr uint workGroupSize = 128;
    constexpr uint quan = 1 << 4;

    const uint n_gpu = std::round((n - 1) / workGroupSize + 1) * workGroupSize;
    uint cs_size = n_gpu / workGroupSize * quan;

    as_gpu.resizeN(n);
    out.resizeN(n);
    cs.resizeN(cs_size);
    buff.resizeN(cs_size);

    {
        ocl::Kernel count(radix_kernel, radix_kernel_length, "count");
        count.compile();

        ocl::Kernel prefix_scan_up(radix_kernel, radix_kernel_length, "prefix_scan_up");
        prefix_scan_up.compile();

        ocl::Kernel prefix_scan_down(radix_kernel, radix_kernel_length, "prefix_scan_down");
        prefix_scan_down.compile();

        ocl::Kernel prefix_scan_end(radix_kernel, radix_kernel_length, "prefix_scan_end");
        prefix_scan_end.compile();

        ocl::Kernel reorder(radix_kernel, radix_kernel_length, "reorder");
        reorder.compile();


        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            for (uint i = 0; i < __CHAR_BIT__ * sizeof(int); i += sizeof(int)) {
                count.exec(gpu::WorkSize(workGroupSize, n_gpu), as_gpu, cs, i, n_gpu / workGroupSize);

                // prefix scan
                for (int d = 0; d < log2(cs_size); ++d) {
                    prefix_scan_up.exec(gpu::WorkSize(workGroupSize, n_gpu), cs, cs_size, d);
                }
                for (int d = log2(cs_size) - 1; d >= 0; --d) {
                    prefix_scan_down.exec(gpu::WorkSize(workGroupSize, n_gpu), cs, cs_size, d);
                }
                prefix_scan_end.exec(gpu::WorkSize(workGroupSize, n_gpu), cs, buff, cs_size);

                cs.swap(buff);

                reorder.exec(gpu::WorkSize(workGroupSize, n_gpu), as_gpu, cs, out, i, n_gpu / workGroupSize);
                std::swap(as_gpu, out);
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
