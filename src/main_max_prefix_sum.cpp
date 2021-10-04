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

unsigned int round_up(unsigned int n, unsigned int block) {
    return (n + block - 1) / block * block;
}

int main(int argc, char **argv)
{
    int benchmarkingIters = 10;
    int max_n = (1 << 24);

    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")" << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
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
            // TODO: implement on OpenCL
            gpu::Device device = gpu::chooseGPUDevice(argc, argv);

            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();

            unsigned int group_size = 256;
            unsigned int level_0_size = round_up(n, group_size);
            unsigned int level_1_size = round_up(level_0_size / 256, group_size);
            unsigned int level_2_size = round_up(level_1_size / 256, group_size);

            gpu::gpu_mem_32i xs_gpu;
            gpu::gpu_mem_32i prefix_level_0_gpu;
            gpu::gpu_mem_32i prefix_level_1_gpu;
            gpu::gpu_mem_32i prefix_level_2_gpu;
            gpu::gpu_mem_32i block_sum_1_gpu;
            gpu::gpu_mem_32i block_sum_2_gpu;
            gpu::gpu_mem_32i result_sum_0_gpu;
            gpu::gpu_mem_32i result_sum_1_gpu;
            gpu::gpu_mem_32i result_sum_2_gpu;
            gpu::gpu_mem_32i result_index_1_gpu;
            gpu::gpu_mem_32i result_index_2_gpu;

            xs_gpu.resizeN(n);
            xs_gpu.writeN(as.data(), n);

            prefix_level_0_gpu.resizeN(level_0_size);
            prefix_level_1_gpu.resizeN(level_1_size);
            prefix_level_2_gpu.resizeN(level_2_size);

            block_sum_1_gpu.resizeN(level_1_size);
            block_sum_2_gpu.resizeN(level_2_size);

            result_sum_0_gpu.resizeN(level_0_size);
            result_sum_1_gpu.resizeN(level_1_size);
            result_sum_2_gpu.resizeN(level_2_size);

            result_index_1_gpu.resizeN(level_1_size);
            result_index_2_gpu.resizeN(level_2_size);

            ocl::Kernel local_prefix(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "local_prefix");
            local_prefix.compile();

            ocl::Kernel prefix_sum(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "prefix_sum");
            prefix_sum.compile();

            ocl::Kernel with_max_value(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "with_max_value");
            with_max_value.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                local_prefix.exec(gpu::WorkSize(group_size, level_0_size), xs_gpu, prefix_level_0_gpu, block_sum_1_gpu, n);
                local_prefix.exec(gpu::WorkSize(group_size, level_1_size), block_sum_1_gpu, prefix_level_1_gpu, block_sum_2_gpu, level_0_size / 256);
                local_prefix.exec(gpu::WorkSize(group_size, level_2_size), block_sum_2_gpu, prefix_level_2_gpu, block_sum_1_gpu, level_1_size / 256);

                prefix_sum.exec(gpu::WorkSize(group_size, level_0_size), prefix_level_0_gpu, prefix_level_1_gpu, prefix_level_2_gpu, result_sum_0_gpu, n);

                with_max_value.exec(gpu::WorkSize(group_size, level_0_size), nullptr, result_sum_0_gpu, result_index_1_gpu, result_sum_1_gpu, n);
                with_max_value.exec(gpu::WorkSize(group_size, level_1_size), result_index_1_gpu, result_sum_1_gpu, result_index_2_gpu, result_sum_2_gpu, level_0_size / 256);
                with_max_value.exec(gpu::WorkSize(group_size, level_2_size), result_index_2_gpu, result_sum_2_gpu, result_index_1_gpu, result_sum_1_gpu, level_1_size / 256);

                int max_sum = 0;
                int result = 0;

                result_sum_1_gpu.readN(&max_sum, 1);
                result_index_1_gpu.readN(&result, 1);
                if (max_sum < 0) {
                    max_sum = 0;
                    result = 0;
                }

                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
