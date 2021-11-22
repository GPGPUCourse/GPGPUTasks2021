#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/max_prefix_sum_cl.h"


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
    int max_n = 4 * 1024 * 1024;

    for (int log2_n = 1, n = 2; n <= max_n; n *= 2, log2_n++) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = r.next(-values_range, values_range) % 10;
        }

        int reference_max_sum;
        int reference_result;
        {

            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum >= max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")"
                  << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum >= max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
        {
            const unsigned int workGroupSize = 128;
            unsigned int globalWorkSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            gpu::Device device = gpu::chooseGPUDevice(argc, argv);
            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();
            ocl::Kernel prefix_sum(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "prefix_sum");
            prefix_sum.compile();
            ocl::Kernel max_number(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_numbers");
            max_number.compile();

            ocl::Kernel get_index(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "get_index");

            gpu::gpu_mem_32i as_gpu = gpu::gpu_mem_32i::createN(n);
            gpu::gpu_mem_32i buffer_gpu = gpu::gpu_mem_32i::createN(n);

            gpu::gpu_mem_32i max_el_gpu = gpu::gpu_mem_32i::createN(1);
            gpu::gpu_mem_32i max_id_gpu = gpu::gpu_mem_32i::createN(1);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.writeN(as.data(), n);
                int minus_inf = -2147483648, id_minus_inf = -2147483648;
                max_el_gpu.writeN(&minus_inf, 1);
                max_id_gpu.writeN(&id_minus_inf, 1);

                for (unsigned i = 0; i < log2_n; i++) {
                    prefix_sum.exec(gpu::WorkSize(workGroupSize, globalWorkSize), as_gpu, n, buffer_gpu, i);
                    as_gpu.swap(buffer_gpu);
                }

                max_number.exec(gpu::WorkSize(workGroupSize, globalWorkSize), as_gpu, n, max_el_gpu);
                int max_el, max_id{0};
                max_el_gpu.readN(&max_el, 1);
                if (max_el < 0) {
                    max_el = 0;
                    max_id = 0;
                } else {
                    get_index.exec(gpu::WorkSize(workGroupSize, globalWorkSize), as_gpu, (unsigned)as.size(), max_el, max_id_gpu);
                    max_id_gpu.readN(&max_id, 1);
                }

                EXPECT_THE_SAME(reference_max_sum, max_el, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, max_id, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
