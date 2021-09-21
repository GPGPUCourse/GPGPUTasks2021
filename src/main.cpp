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

cl_device_id deviceHelper(cl_uint platformsCount, const std::vector<cl_platform_id>& platforms) {
    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        cl_platform_id platform = platforms[platformIndex];

        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
        cl_device_id deviceId;
        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            deviceId = devices[deviceIndex];

            size_t type_size = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, 0, nullptr, &type_size));
            cl_device_type device_type;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, type_size, &device_type, nullptr));

            if (device_type == CL_DEVICE_TYPE_GPU)  {
                return deviceId;
            }
        }
        if (platformIndex == platformsCount - 1) {
            return deviceId;
        }
    }
    return nullptr;
}

int main()
{
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    auto deviceId = deviceHelper(platformsCount, platforms);

    cl_int rc;
    auto context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, &rc);
    OCL_SAFE_CALL(rc);

    auto queue = clCreateCommandQueue(context, deviceId, 0, &rc);
    OCL_SAFE_CALL(rc);

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

    auto asBuff = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, n * sizeof(float), as.data(), &rc);
    OCL_SAFE_CALL(rc);
    auto bsBuff = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, n * sizeof(float), bs.data(), &rc);
    OCL_SAFE_CALL(rc);
    auto csBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, &rc);
    OCL_SAFE_CALL(rc);

    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.empty()) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        //std::cout << kernel_sources << std::endl;
    }

    size_t sz = kernel_sources.size();
    auto src = kernel_sources.c_str();
    auto program = clCreateProgramWithSource(context, 1, &src, &sz, &rc);
    OCL_SAFE_CALL(rc);
    OCL_SAFE_CALL(clBuildProgram(program, 0, nullptr, "-Isrc/cl/", nullptr, nullptr));

    size_t log_size = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
    std::vector<char> log(log_size, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));
    if (log_size > 1) {
        std::cout << "Log:" << std::endl;
        std::cout << log.data() << std::endl;
    }

    auto kernel = clCreateKernel(program, "aplusb", &rc);
    OCL_SAFE_CALL(rc);

    {
        unsigned int i = 0;
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(float*), &asBuff));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(float*), &bsBuff));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(float*), &csBuff));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(unsigned int), &n));
    }

    // TODO 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)
    // у меня CL_MEM_OBJECT_ALLOCATION_FAILURE так получается в 148 строчке :(

    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t; // Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, &workGroupSize, 0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            OCL_SAFE_CALL(clReleaseEvent(event));
            t.nextLap(); // При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        
        std::cout << "GFlops: " << ((double) n / t.lapAvg()) / 1e9 << std::endl;

        std::cout << "VRAM bandwidth: " << ((double) 3 * n * sizeof(float) / t.lapAvg()) / (1024 * 1024 * 1024) << " GB/s" << std::endl;
    }

    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueReadBuffer(queue, csBuff, true, 0, n * sizeof(float), cs.data(), 0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            OCL_SAFE_CALL(clReleaseEvent(event));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << ((double) n * sizeof(float) / t.lapAvg()) / (1024 * 1024 * 1024) << " GB/s" << std::endl;
    }

    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    OCL_SAFE_CALL(clReleaseKernel(kernel));
    OCL_SAFE_CALL(clReleaseProgram(program));
    OCL_SAFE_CALL(clReleaseMemObject(csBuff));
    OCL_SAFE_CALL(clReleaseMemObject(bsBuff));
    OCL_SAFE_CALL(clReleaseMemObject(asBuff));
    OCL_SAFE_CALL(clReleaseCommandQueue(queue));
    OCL_SAFE_CALL(clReleaseContext(context));
    return 0;
}
