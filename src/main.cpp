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

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)
#define OCL_CHECK_ERR_CODE(err) reportError(err, __FILE__, __LINE__)

std::string getDeviceStrProperty(cl_device_id device, cl_device_info property)
{
    size_t devicePropertySize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, property, 0, nullptr, &devicePropertySize));
    std::vector<unsigned char> deviceProperty(devicePropertySize, 0);

    OCL_SAFE_CALL(clGetDeviceInfo(device, property, devicePropertySize, deviceProperty.data(), nullptr));
    std::string strPropertyVal(deviceProperty.begin(), --deviceProperty.end());
    return strPropertyVal;
}

template <typename T>
cl_mem createBufferWithData(cl_context context, T* userData, size_t memSize, cl_mem_flags memFlag){
    cl_int errcode;
    cl_mem buffer = clCreateBuffer(context, memFlag, memSize, userData, &errcode);
    return buffer;

}
int main()
{
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // TODO 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    assert(platformsCount >= 1 && "Platform count should be greater than 0");
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));
    cl_device_id cpuDevice = nullptr;
    cl_device_id gpuDevice = nullptr;
    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        cl_platform_id platform = platforms[platformIndex];
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        assert(devicesCount >= 1 && "Devices count should be greater than 0");

        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
        // try to find gpu
        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            cl_device_id device = devices[deviceIndex];
            size_t devicePropertySize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, 0, nullptr, &devicePropertySize));
            cl_device_type deviceProperty;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, devicePropertySize, &deviceProperty, NULL));
            if (deviceProperty == CL_DEVICE_TYPE_GPU)
                gpuDevice = device;
            else if (cpuDevice == nullptr && deviceProperty == CL_DEVICE_TYPE_CPU)
                cpuDevice = device; // берем хоть какой-то цпу девайс на случай если не найдем гпу
        }

    }
    cl_device_id device = (gpuDevice == nullptr) ? cpuDevice : gpuDevice;
    std::cout << "Choose device with name: " << getDeviceStrProperty(device, CL_DEVICE_NAME) << std::endl;
    // TODO 2 Создайте контекст с выбранным устройством

    cl_int errcode;
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &errcode);
    OCL_CHECK_ERR_CODE(errcode);

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &errcode);
    OCL_CHECK_ERR_CODE(errcode);

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

    // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    size_t vecsSize = sizeof(float) * n;

    cl_mem asBuffer = createBufferWithData<float>(context, as.data(), vecsSize,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    cl_mem bsBuffer = createBufferWithData<float>(context, bs.data(), vecsSize,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    cl_mem csBuffer = createBufferWithData<float>(context, cs.data(), vecsSize,CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR);

    // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь, что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания),
    // напечатав исходники в консоль (if проверяет, что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        // std::cout << kernel_sources << std::endl;
    }

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель
    const char * kernelCode = kernel_sources.c_str();
    cl_program prog = clCreateProgramWithSource(context, 1, &kernelCode, nullptr, &errcode);
    OCL_CHECK_ERR_CODE(errcode);
    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции

    errcode = clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr);
    size_t log_size = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0,  nullptr, &log_size));
    std::vector<char> log(log_size, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL));
    if (log_size > 1) {
        std::cout << "Log:" << std::endl;
        std::cout << log.data() << std::endl;
    }
    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
    cl_kernel kernel = clCreateKernel(prog, "aplusb", &errcode);
    OCL_CHECK_ERR_CODE(errcode);
    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
        unsigned int i = 0;
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(float *), &asBuffer));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(float *), &bsBuffer));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(float *), &csBuffer));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(unsigned int), &n));
    }

    // TODO 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

    // TODO 12 Запустите выполнения кернела:

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

        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        // TODO 13 Рассчитайте достигнутые гигафлопcы:

        double gflops = n / (t.lapAvg() * 1e9);
        std::cout << "GFlops: " << std::setprecision(3) << gflops << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)

        double vramBandwidth = 3 * n * sizeof(float) / ((1 << 30) * t.lapAvg());
        std::cout << "VRAM bandwidth: " << std::setprecision(3) << vramBandwidth << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(queue, csBuffer, CL_TRUE, 0, vecsSize, cs.data(), 0, nullptr, nullptr));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        double ramToVramBandwidth = 3 * n * sizeof(float) / ((1 << 30) * t.lapAvg());
        std::cout << "VRAM -> RAM bandwidth: " << std::setprecision(3) << ramToVramBandwidth << " GB/s" << std::endl;
    }

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }
    OCL_SAFE_CALL(clReleaseKernel(kernel));
    OCL_SAFE_CALL(clReleaseProgram(prog));
    OCL_SAFE_CALL(clReleaseMemObject(asBuffer));
    OCL_SAFE_CALL(clReleaseMemObject(bsBuffer));
    OCL_SAFE_CALL(clReleaseMemObject(csBuffer));
    OCL_SAFE_CALL(clReleaseCommandQueue(queue));
    OCL_SAFE_CALL(clReleaseContext(context));

    return 0;
}
