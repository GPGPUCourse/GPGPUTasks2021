#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <vector>
#include <map>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cassert>
#include <iomanip>


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

    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


cl_device_id get_best_device() {
    cl_device_id any_device = nullptr;

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
        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            cl_device_id device = devices[deviceIndex];

            if (any_device == nullptr) {
                any_device = device;
            }

            cl_device_type deviceType = 0;

            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));

            if (deviceType & CL_DEVICE_TYPE_GPU) {
                return device;
            }
        }
    }

    return any_device;
}

int main()
{
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    cl_device_id device = get_best_device();

    cl_int error_create_context;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error_create_context);
    reportError(error_create_context, __FILE__, __LINE__ - 1);

    cl_int error_creating_command_queue;
    cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, &error_creating_command_queue);
    reportError(error_creating_command_queue, __FILE__, __LINE__ - 1);

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

    cl_int error_creating_buff;
    cl_mem mem_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * n, as.data(), &error_creating_buff);
    reportError(error_creating_buff, __FILE__, __LINE__ - 1);
    cl_mem mem_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * n, bs.data(), &error_creating_buff);
    reportError(error_creating_buff, __FILE__, __LINE__ - 1);
    cl_mem mem_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n, nullptr, &error_creating_buff);
    reportError(error_creating_buff, __FILE__, __LINE__ - 1);

    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        //std::cout << kernel_sources << std::endl;
    }

    const char *kernel_sources_c_str = kernel_sources.c_str();
    cl_int error_creating_program;
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_sources_c_str, nullptr, &error_creating_program);
    reportError(error_creating_program, __FILE__, __LINE__ - 1);

    OCL_SAFE_CALL(clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr));

    size_t log_size;
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
    std::vector<unsigned char> log(log_size, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));

    if (log_size > 1) {
        std::cout << "Log:" << std::endl;
        std::cout << log.data() << std::endl;
    }

    cl_int error_creating_kernel;
    cl_kernel kernel = clCreateKernel(program, "aplusb", &error_creating_kernel);
    reportError(error_creating_kernel, __FILE__, __LINE__ - 1);

    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
        unsigned int i = 0;
        clSetKernelArg(kernel, i++, sizeof(float *), &mem_a);
        clSetKernelArg(kernel, i++, sizeof(float *), &mem_b);
        clSetKernelArg(kernel, i++, sizeof(float *), &mem_c);
        clSetKernelArg(kernel, i++, sizeof(unsigned int), &n);
    }

    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t; // Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &global_work_size, &workGroupSize, 0,
                                                 nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            t.nextLap(); // При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        
        // TODO 13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "GFlops: " << (n / t.lapAvg()) / 1000000000 << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << (3 * n * sizeof(float) / t.lapAvg()) / (1024 * 1024 * 1024) << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueReadBuffer(command_queue, mem_c, CL_TRUE, 0, n * sizeof(float), cs.data(), 0, nullptr, &event));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << (n * sizeof(float) / t.lapAvg()) / (1024 * 1024 * 1024) << " GB/s" << std::endl;
    }

    for (unsigned int i = 0; i < n; ++i) {
        float expected = cs[i];
        float found = as[i] + bs[i];
        if (expected != found) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    OCL_SAFE_CALL(clReleaseKernel(kernel));
    OCL_SAFE_CALL(clReleaseProgram(program));
    OCL_SAFE_CALL(clReleaseMemObject(mem_c));
    OCL_SAFE_CALL(clReleaseMemObject(mem_b));
    OCL_SAFE_CALL(clReleaseMemObject(mem_a));
    OCL_SAFE_CALL(clReleaseCommandQueue(command_queue));
    OCL_SAFE_CALL(clReleaseContext(context));
    return 0;
}
