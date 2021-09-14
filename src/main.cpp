#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cassert>


template <typename T>
std::string to_string(T value)
{
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line)
{
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

cl_device_id pickDevice()
{
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    cl_device_type activeDeviceType = 0;
    cl_device_id activeDevice;

    for (auto platform : platforms) {
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (auto device : devices) {
            cl_device_type deviceType;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof deviceType, &deviceType, nullptr));
            if (deviceType == CL_DEVICE_TYPE_GPU || (deviceType == CL_DEVICE_TYPE_CPU && activeDeviceType == 0)) {
                activeDevice = device;
                activeDeviceType = deviceType;
            }
        }
    }

    if (activeDeviceType == 0) {
        throw std::runtime_error("No suitable OpenCL device found");
    }

    return activeDevice;
}

int main()
{
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    auto device = pickDevice();
    cl_int err;

    auto context = clCreateContext(nullptr, 1, &device, [](
                const char *errinfo,
                const void *private_info, size_t cb,
                void *user_data) {
        if (errinfo) {
            throw std::runtime_error(to_string("Error creating OpenCL context: ") + errinfo);
        }
    }, nullptr, &err);
    OCL_SAFE_CALL(err);

    auto cmdq = clCreateCommandQueue(context, device, 0, &err);
    OCL_SAFE_CALL(err);

    unsigned int n = 100*1000*1000;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    auto asBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), as.data(), &err);
    OCL_SAFE_CALL(err);
    auto bsBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), bs.data(), &err);
    OCL_SAFE_CALL(err);
    auto csBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, &err);
    OCL_SAFE_CALL(err);

    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
    }

    const char *srcptr = kernel_sources.c_str();
    auto program = clCreateProgramWithSource(context, 1, &srcptr, nullptr, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(program, 0, nullptr, "-Isrc/cl/", nullptr, nullptr);

    size_t log_size = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
    std::vector<char> log(log_size, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));
    if (log_size > 1) {
        std::cerr << log.data() << std::endl;
    }

    OCL_SAFE_CALL(err);

    auto kernel = clCreateKernel(program, "aplusb", &err);
    OCL_SAFE_CALL(err);

    {
        unsigned int i = 0;
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(float*), &asBuf));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(float*), &bsBuf));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(float*), &csBuf));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(unsigned), &n));
    }

    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t; // Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(cmdq, kernel, 1, nullptr, &global_work_size, &workGroupSize, 0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            t.nextLap(); // При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        
        std::cout << "GFlops: " << n / t.lapAvg() / 1e9 << std::endl;

        std::cout << "VRAM bandwidth: " << 3 * n * sizeof(float) / t.lapAvg() / (1 << 30) << " GB/s" << std::endl;
    }

    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueReadBuffer(cmdq, csBuf, CL_TRUE, 0, sizeof(float) * n, cs.data(), 0, nullptr, &event));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << n * sizeof(float) / t.lapAvg() / (1 << 30) << " GB/s" << std::endl;
    }

    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(csBuf);
    clReleaseMemObject(bsBuf);
    clReleaseMemObject(asBuf);
    clReleaseCommandQueue(cmdq);
    clReleaseContext(context);
    return 0;
}
