#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"

#include <array>

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        const unsigned work_group_size = 256;

        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum");
        kernel.compile(false);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // Учитываем время копирования данных в видеопамять
            gpu::gpu_mem_32u buffer1;
            gpu::gpu_mem_32u buffer2;
            buffer1.resizeN(n);
            buffer2.resizeN(n / work_group_size + 1);

            buffer1.writeN(as.data(), n);

            std::array<gpu::gpu_mem_32u*, 2> buffers {&buffer1, &buffer2};

            unsigned current_work_size = n;
            while (current_work_size > 1) {
                kernel.exec(gpu::WorkSize(work_group_size, current_work_size), *buffers[0], current_work_size, *buffers[1]);
                std::swap(buffers[0], buffers[1]);
                current_work_size = static_cast<unsigned>(std::lround(std::ceil(static_cast<double>(current_work_size) / work_group_size)));
            }

            unsigned result = 0;
            buffer1.readN(&result, 1);

            EXPECT_THE_SAME(reference_sum, result, "Unmatched results");
            t.nextLap();
        }
        std::cout << "GPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
}
