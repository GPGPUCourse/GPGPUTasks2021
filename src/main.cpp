#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>


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

#define REPORT_ERROR(expr) reportError(expr, __FILE__, __LINE__)
#define OCL_SAFE_CALL(expr) REPORT_ERROR(expr)

cl_device_id select_device() {
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        cl_platform_id platform = platforms[platformIndex];

        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        cl_device_id cpu_device = nullptr;

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            cl_device_id device = devices[deviceIndex];
            cl_device_type deviceType;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));

            if (deviceType & CL_DEVICE_TYPE_GPU) {
                return device;
            } else if (deviceType & CL_DEVICE_TYPE_GPU) {
                cpu_device = device;
            }
        }
        if (cpu_device){
            return cpu_device;
        }
    }
    return nullptr;
}

int main()
{
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    cl_device_id device = select_device();
    if (device == nullptr) {
        std::cerr << "No GPU or CPU device found" << std::endl;
        return 0;
    }

    cl_int error;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
    REPORT_ERROR(error);

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);
    REPORT_ERROR(error);

    unsigned int n = 100 * 1000 * 1000;
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

    cl_mem as_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, as.data(), &error);
    REPORT_ERROR(error);

    cl_mem bs_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, bs.data(), &error);
    REPORT_ERROR(error);

    cl_mem cs_gpu = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n, nullptr, &error);
    REPORT_ERROR(error);


    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.empty()) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
    }

    const char *source_string = kernel_sources.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &source_string, nullptr, &error);
    REPORT_ERROR(error);

    error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    if (error != CL_BUILD_PROGRAM_FAILURE) {
        REPORT_ERROR(error);
    } else {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size, 0);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        if (log_size > 1) {
            std::cout << "Log:" << std::endl;
            std::cout << log.data() << std::endl;
        }
        if (error) {
            return 0;
        }
    }


    cl_kernel kernel = clCreateKernel(program, "aplusb", &error);
    REPORT_ERROR(error);

    {
         unsigned int i = 0;
         clSetKernelArg(kernel, i++, sizeof(float *), &as_gpu);
         clSetKernelArg(kernel, i++, sizeof(float *), &bs_gpu);
         clSetKernelArg(kernel, i++, sizeof(float *), &cs_gpu);
         clSetKernelArg(kernel, i, sizeof(unsigned int), &n);
    }

    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t; // Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, &workGroupSize, 0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            t.nextLap(); // При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GFlops: " << (n / t.lapAvg()) / 1e9 << std::endl;
        std::cout << "VRAM bandwidth: " << 3.0 * n * sizeof(float) / t.lapAvg() / (1 << 30) << " GB/s" << std::endl;
    }

    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(queue, cs_gpu, CL_TRUE, 0, sizeof(float) * n, cs.data(), 0, nullptr, nullptr));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << (double) (n * sizeof(float)) / t.lapAvg() / (1 << 30) << " GB/s" << std::endl;
    }

    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    OCL_SAFE_CALL(clReleaseKernel(kernel));
    OCL_SAFE_CALL(clReleaseProgram(program));
    OCL_SAFE_CALL(clReleaseMemObject(cs_gpu));
    OCL_SAFE_CALL(clReleaseMemObject(bs_gpu));
    OCL_SAFE_CALL(clReleaseMemObject(as_gpu));
    OCL_SAFE_CALL(clReleaseCommandQueue(queue));
    OCL_SAFE_CALL(clReleaseContext(context));

    return 0;
}
