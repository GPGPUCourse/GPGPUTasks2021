#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/max_prefix_sum_cl.h"

#include <numeric>
#include <iterator>

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
    int max_n = (1 << 24);

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    ocl::Kernel sum_reductor(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "sum_reductor");
    sum_reductor.compile(true);

    ocl::Kernel sum_calculator(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "sum_calculator");
    sum_calculator.compile(true);

    ocl::Kernel min_reductor(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "min_reductor");
    min_reductor.compile(true);

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
            const unsigned work_group_size = 128;

            std::vector<int> xa(n * 2, 0); 
            std::copy(as.begin(), as.end(), xa.begin());

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {

                gpu::gpu_mem_32i buffer;
                buffer.resizeN(xa.size() + 1); 
                buffer.writeN(xa.data(), xa.size()); 

                unsigned current_reduction_size = static_cast<unsigned>(xa.size()) / 4;
                unsigned read_offset = 0;
                unsigned write_offset = current_reduction_size * 2;

                while (current_reduction_size) {
                    sum_reductor.exec(gpu::WorkSize(work_group_size, current_reduction_size), buffer, read_offset, write_offset, current_reduction_size);
                    read_offset = write_offset;
                    write_offset += current_reduction_size;
                    current_reduction_size /= 2;
                }
                
                sum_calculator.exec(gpu::WorkSize(128, xa.size() / 2), buffer, static_cast<unsigned>(xa.size()) / 2);

                current_reduction_size = static_cast<unsigned>(xa.size()) / 4;
                read_offset = 0;
                write_offset = current_reduction_size * 2;

                while (current_reduction_size) {
                    min_reductor.exec(gpu::WorkSize(work_group_size, current_reduction_size), buffer, read_offset, write_offset);
                    read_offset = write_offset;
                    write_offset += current_reduction_size;
                    current_reduction_size /= 2;
                }
                
                int max_sum = 0;
                buffer.readN(&max_sum, 1, xa.size() - 2);
                max_sum = std::max(max_sum, 0);
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
