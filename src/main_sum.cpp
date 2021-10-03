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
        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);

        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        unsigned int group_size = 256;

        gpu::gpu_mem_32u xs_gpu, ys_gpu, zs_gpu;

        xs_gpu.resizeN(n);
        ys_gpu.resizeN((n + group_size - 1) / group_size);
        zs_gpu.resizeN((n + group_size - 1) / group_size);

        xs_gpu.writeN(as.data(), n);

        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum");
        kernel.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            gpu::gpu_mem_32u sum_gpu;
            sum_gpu.resizeN(1);
            sum_gpu.writeN(&sum , 1);

            gpu::gpu_mem_32u* xs_ptr = &xs_gpu;
            gpu::gpu_mem_32u* sum_ptr = &ys_gpu;

            for (unsigned int current_size = n; ; current_size = (current_size + group_size - 1) / group_size) {
                unsigned int work_size = (current_size + group_size - 1) / group_size * group_size;

                if (current_size < group_size * group_size) {
                    kernel.exec(gpu::WorkSize(group_size, work_size), *xs_ptr, sum_gpu, current_size, 1);
                    break;
                }

                kernel.exec(gpu::WorkSize(group_size, work_size), *xs_ptr, *sum_ptr, current_size, 0);
                if (current_size == n) {
                    xs_ptr = sum_ptr;
                    sum_ptr = &zs_gpu;
                } else {
                    std::swap(xs_ptr, sum_ptr);
                }
            }

            sum_gpu.readN(&sum, 1);

            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
}
