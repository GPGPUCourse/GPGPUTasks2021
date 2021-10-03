#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
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
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum");

        bool printLog = false;
        kernel.compile(printLog);
        unsigned int groupSize = 128;
        unsigned int workSize = (n + groupSize - 1) / groupSize * groupSize;
        gpu::gpu_mem_32u gpu_mem_results;
        gpu::gpu_mem_32u gpu_as;
        gpu_mem_results.resizeN(1);
        const uint32_t kek = 0;
        gpu_as.resizeN(n);
        gpu_as.writeN(as.data(), n);
        uint32_t gpu_result = 0;
        timer t;

        for (int i = 0; i < benchmarkingIters; ++i) {
            gpu_mem_results.write(&kek, sizeof(gpu_result));
            kernel.exec(gpu::WorkSize(groupSize, workSize),
                        gpu_mem_results,
                        gpu_as,
                        n);
            t.nextLap();
        }

        std::cout << "GPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;

        gpu_mem_results.read(&gpu_result, sizeof(gpu_result));

        EXPECT_THE_SAME(reference_sum, gpu_result, "GPU result should be consistent!");
    }
}
