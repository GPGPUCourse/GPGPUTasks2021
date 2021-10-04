#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"

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
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;

    unsigned int workGroupSize = 128;
    unsigned int globalWorkSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

    std::vector<unsigned int> as(globalWorkSize, 0);
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
    std::cout << " \n --- OpenCL results: --- \n";
    {
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum_gpu");
        sum.compile();

        gpu::gpu_mem_32u numbers_gpu = gpu::gpu_mem_32u::createN(n);
        gpu::gpu_mem_32u result_gpu = gpu::gpu_mem_32u::createN(1);
        numbers_gpu.writeN(as.data(), n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int res = 0;
            result_gpu.writeN(&res, 1);
            sum.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                     numbers_gpu, result_gpu);
            result_gpu.readN(&res, 1);
            EXPECT_THE_SAME(res, reference_sum, "GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU(1): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU(1): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
    {
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum_gpu2");
        sum.compile();

        gpu::gpu_mem_32u numbers_gpu = gpu::gpu_mem_32u::createN(n);
        gpu::gpu_mem_32u result_gpu = gpu::gpu_mem_32u::createN(1);
        numbers_gpu.writeN(as.data(), n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int res = 0;
            result_gpu.writeN(&res, 1);
            sum.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                     numbers_gpu, result_gpu);
            result_gpu.readN(&res, 1);
            EXPECT_THE_SAME(res, reference_sum, "GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU(2): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU(2): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
    {
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum_gpu3");
        sum.compile();

        gpu::gpu_mem_32u numbers_gpu = gpu::gpu_mem_32u::createN(n);
        gpu::gpu_mem_32u result_gpu = gpu::gpu_mem_32u::createN(1);
        numbers_gpu.writeN(as.data(), n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int res = 0;
            result_gpu.writeN(&res, 1);
            sum.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                     numbers_gpu, result_gpu);
            result_gpu.readN(&res, 1);
            EXPECT_THE_SAME(res, reference_sum, "GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU(3): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU(3): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
}
