#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

template<typename T>
std::string to_string(T value)
{
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line)
{
    if (CL_SUCCESS == err) return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

class ClContext
{
public:
    ClContext(const ClContext &) = delete;

    ClContext &operator=(const ClContext &) = delete;

    ClContext(cl_platform_id platform, cl_device_id device)
    {
        cl_int err;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        OCL_SAFE_CALL(err);
    }

    cl_context operator*() const { return context; }

    ~ClContext() { OCL_SAFE_CALL(clReleaseContext(context)); }

private:
    cl_context context;
};

class ClCommandQueue
{
public:
    ClCommandQueue(const ClCommandQueue &) = delete;

    ClCommandQueue &operator=(const ClCommandQueue &) = delete;

    ClCommandQueue(cl_context context, cl_device_id device)
    {
        cl_int err;
        commandQueue = clCreateCommandQueue(context, device, 0, &err);
        OCL_SAFE_CALL(err);
    }

    cl_command_queue operator*() const { return commandQueue; }

    ~ClCommandQueue() { OCL_SAFE_CALL(clReleaseCommandQueue(commandQueue)); }

private:
    cl_command_queue commandQueue;
};

class ClBuffer
{
public:
    ClBuffer(const ClBuffer &) = delete;

    ClBuffer &operator=(const ClBuffer &) = delete;

    ClBuffer(cl_context context, cl_mem_flags flags, unsigned int size, void *data_ptr)
    {
        cl_int err;
        buffer = clCreateBuffer(context, flags, size, data_ptr, &err);
        OCL_SAFE_CALL(err);
    }

    cl_mem operator*() const { return buffer; }

    cl_mem *AsPtr() { return &buffer; }

    ~ClBuffer() { OCL_SAFE_CALL(clReleaseMemObject(buffer)); }

private:
    cl_mem buffer;
};

class ClProgram
{
public:
    ClProgram(const ClProgram &) = delete;

    ClProgram &operator=(const ClProgram &) = delete;

    ClProgram(cl_context context, const char *string, size_t length)
    {
        cl_int err;
        program = clCreateProgramWithSource(context, 1, &string, &length, &err);
        OCL_SAFE_CALL(err);
    }

    void Build(cl_device_id device)
    {
        cl_int err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        size_t logSize = 0;
        OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize));
        if (logSize > 1) {
            std::vector<char> log(logSize, '\0');
            OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr));
            std::cout << "Log:\n" << log.data() << std::endl;
        }
        OCL_SAFE_CALL(err);
    }

    cl_program operator*() const { return program; }

    ~ClProgram() { OCL_SAFE_CALL(clReleaseProgram(program)); }

private:
    cl_program program;
};

class ClKernel
{
public:
    ClKernel(const ClKernel &) = delete;

    ClKernel &operator=(const ClKernel &) = delete;

    ClKernel(cl_program program, const char *kernelName)
    {
        cl_int err;
        kernel = clCreateKernel(program, kernelName, &err);
        OCL_SAFE_CALL(err);
    }

    cl_kernel operator*() const { return kernel; }

    ~ClKernel() { OCL_SAFE_CALL(clReleaseKernel(kernel)); }

private:
    cl_kernel kernel;
};

size_t RoundUp(size_t n, size_t group_size) { return (n + group_size - 1) / group_size * group_size; }

void DeviceMain(cl_platform_id platform, cl_device_id device, size_t n)
{
    size_t work_group_size = 0;
    {
        OCL_SAFE_CALL(
                clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &work_group_size, nullptr));
        size_t deviceNameCount = 0;
        OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameCount));
        std::vector<unsigned char> deviceName(deviceNameCount, '\0');
        OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameCount, deviceName.data(), nullptr));
        std::cout << "Device name: " << deviceName.data() << std::endl;
        std::cout << "Work Group size: " << work_group_size << std::endl;
    }
    ClContext context(platform, device);
    ClCommandQueue queue(*context, device);

    auto global_work_size = RoundUp(n, work_group_size);
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(global_work_size, 0);
    std::vector<float> bs(global_work_size, 0);
    std::vector<float> cs(global_work_size, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    ClBuffer as_gpu(*context, CL_MEM_READ_ONLY, sizeof(float) * global_work_size, nullptr);
    ClBuffer bs_gpu(*context, CL_MEM_READ_ONLY, sizeof(float) * global_work_size, nullptr);
    ClBuffer cs_gpu(*context, CL_MEM_WRITE_ONLY, sizeof(float) * global_work_size, nullptr);
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueWriteBuffer(*queue, *as_gpu, CL_FALSE, 0, sizeof(float) * global_work_size,
                                               as.data(), 0, nullptr, nullptr));
            OCL_SAFE_CALL(clEnqueueWriteBuffer(*queue, *bs_gpu, CL_FALSE, 0, sizeof(float) * global_work_size,
                                               bs.data(), 0, nullptr, nullptr));
            OCL_SAFE_CALL(clEnqueueWriteBuffer(*queue, *cs_gpu, CL_TRUE, 0, sizeof(float) * global_work_size, cs.data(),
                                               0, nullptr, nullptr));
            t.nextLap();
        }
        const auto avg_time = t.lapAvg();
        std::cout << "Result data transfer time: " << avg_time << "+-" << t.lapStd() << " s\n"
                  << "RAM -> VRAM bandwidth: "
                  << double(3 * global_work_size * sizeof(float)) / avg_time / (1024 * 1024 * 1024) << " GB/s"
                  << std::endl;
    }

    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.empty()) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
    }

    ClProgram program(*context, kernel_sources.data(), kernel_sources.size());
    program.Build(device);

    ClKernel kernel(*program, "aplusb");
    {
        unsigned int i = 0;
        OCL_SAFE_CALL(clSetKernelArg(*kernel, i++, sizeof(cl_mem), (void *) as_gpu.AsPtr()));
        OCL_SAFE_CALL(clSetKernelArg(*kernel, i++, sizeof(cl_mem), (void *) bs_gpu.AsPtr()));
        OCL_SAFE_CALL(clSetKernelArg(*kernel, i++, sizeof(cl_mem), (void *) cs_gpu.AsPtr()));
    }
    {
        timer t;// Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(*queue, *kernel, 1, nullptr, &global_work_size, &work_group_size, 0,
                                                 nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            OCL_SAFE_CALL(clReleaseEvent(event));
            t.nextLap();// При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        const auto avg_time = t.lapAvg();
        std::cout << "Kernel average time: " << avg_time << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GFlops: " << double(global_work_size) / 1e9 / avg_time << std::endl;
        std::cout << "VRAM bandwidth: "
                  << double(3 * global_work_size * sizeof(float)) / avg_time / (1024 * 1024 * 1024) << " GB/s"
                  << std::endl;
    }

    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(*queue, *cs_gpu, CL_TRUE, 0, sizeof(float) * global_work_size, cs.data(),
                                              0, nullptr, nullptr));
            t.nextLap();
        }
        const auto avg_time = t.lapAvg();
        std::cout << "Result data transfer time: " << avg_time << "+-" << t.lapStd() << " s\n"
                  << "VRAM -> RAM bandwidth: "
                  << double(global_work_size * sizeof(float)) / avg_time / (1024 * 1024 * 1024) << " GB/s" << std::endl;
    }

    for (unsigned int i = 0; i < global_work_size; ++i) {
        auto sum = as[i] + bs[i];
        if (std::abs(cs[i] - sum) > std::numeric_limits<float>::epsilon()) {
            throw std::runtime_error("CPU and GPU results differ:"
                                     " i = " +
                                     std::to_string(i) + ", a = " + std::to_string(as[i]) +
                                     ", b = " + std::to_string(bs[i]) + ", expected sum = " + std::to_string(sum) +
                                     ", computed sum = " + std::to_string(cs[i]));
        }
    }
}

int main()
{
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init()) { throw std::runtime_error("Can't init OpenCL driver!"); }

    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));
    std::vector<cl_device_id> devices;
    for (auto n : {1U * 1000 * 1000, 100U * 1000 * 1000}) {
        for (auto platform : platforms) {
            cl_uint devicesCount = 0;
            OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
            devices.resize(devicesCount);
            OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
            for (auto device : devices) {
                DeviceMain(platform, device, n);
                std::cout << std::endl;
            }
        }
    }

    return 0;
}
