#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

void reportErrorAndFreeContext(cl_int err, const cl_context &ctx, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    OCL_SAFE_CALL(clReleaseContext(ctx));
    throw std::runtime_error(message);
}

#define OCL_CTX_SAFE_CALL(expr, ctx) reportErrorAndFreeContext(expr, ctx, __FILE__, __LINE__)

/**
 * Get device info which is in char[] format and print it
 * @param device [cl_device_id] id of device
 * @param info [cl_device_info] info to ge acquired
 * @param string_padding [size_t] number of spaces before debug info
 * @param print_string string to be printed before info
 */
void getDeviceInfoString(cl_device_id device, cl_device_info info, size_t string_padding, const std::string &print_string) {
    size_t deviceNameSize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, info, 0, nullptr, &deviceNameSize));
    std::vector<unsigned char> deviceName(deviceNameSize, 0);
    OCL_SAFE_CALL(clGetDeviceInfo(device, info, deviceNameSize, deviceName.data(), nullptr));
    std::cout << std::string(string_padding, ' ') << print_string << deviceName.data() << std::endl;
}

int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    int someDeviceIndex;
    cl_device_id someDevice;
    for (cl_uint platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        size_t platformNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));

        std::vector<unsigned char> platformName(platformNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
        std::cout << std::string(4, ' ') << "Platform name: " << platformName.data() << std::endl;

        size_t platformVendorSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
        std::vector<unsigned char> platformVendor(platformVendorSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
        std::cout << std::string(4, ' ') << "Platform vendor: " << platformVendor.data() << std::endl;

        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::cout << std::string(8, ' ') << "Number of devices for current platform: "
                  << devicesCount << std::endl;

        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            std::cout << std::string(12, ' ') << "Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
            cl_device_id device = devices[deviceIndex];
            someDeviceIndex = deviceIndex;
            someDevice = device;

            getDeviceInfoString(device, CL_DEVICE_NAME, 16, "Device name: ");

            cl_device_type deviceType;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
            std::cout << std::string(16, ' ') << "Device type: " << deviceType << " [Consult cl.h:172 for more info]" << std::endl;
            cl_ulong deviceMem;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &deviceMem, nullptr));
            std::cout << std::string(16, ' ') << "Device memory in MB: " << deviceMem / 1000000 << std::endl;

            getDeviceInfoString(device, CL_DEVICE_PROFILE, 16, "Device profile (consult documentation for more info): ");
            getDeviceInfoString(device, CL_DRIVER_VERSION, 16, "Device OpenCL driver version (consult documentation for more info): ");
        }
    }

    // 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)
    cl_int err;
    cl_context_properties cps[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[0],
        0};
    cl_context ctx = clCreateContext(nullptr, 1, &someDevice, nullptr, nullptr, &err);
    OCL_CTX_SAFE_CALL(err, ctx);

    // 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать ресурсы)
    cl_command_queue qu = clCreateCommandQueue(ctx, someDevice, 0, &err);
    if (err != CL_SUCCESS) {
        OCL_SAFE_CALL(clReleaseCommandQueue(qu));
    }
    OCL_CTX_SAFE_CALL(err, ctx);

    unsigned int n = 100 * 1000 * 1000;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    std::cout << "Starting data generation:" << std::endl;
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    // 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n штук
    // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг, на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично, все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)
    cl_mem_flags flags_ro = CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY;
    cl_mem buf1 = clCreateBuffer(ctx, flags_ro, sizeof(float) * n, as.data(), &err);
    if (CL_SUCCESS != err) {
        OCL_SAFE_CALL(clReleaseMemObject(buf1));
        OCL_SAFE_CALL(clReleaseCommandQueue(qu));
    }
    OCL_CTX_SAFE_CALL(err, ctx);

    cl_mem buf2 = clCreateBuffer(ctx, flags_ro, sizeof(float) * n, bs.data(), &err);
    if (CL_SUCCESS != err) {
        OCL_SAFE_CALL(clReleaseMemObject(buf2));
        OCL_SAFE_CALL(clReleaseMemObject(buf1));
        OCL_SAFE_CALL(clReleaseCommandQueue(qu));
    }
    OCL_CTX_SAFE_CALL(err, ctx);

    cl_mem_flags flags_wo = CL_MEM_WRITE_ONLY;
    cl_mem buf3 = clCreateBuffer(ctx, flags_wo, sizeof(float) * n, nullptr, &err);
    if (CL_SUCCESS != err) {
        // order is reversed creation order
        OCL_SAFE_CALL(clReleaseMemObject(buf3));
        OCL_SAFE_CALL(clReleaseMemObject(buf2));
        OCL_SAFE_CALL(clReleaseMemObject(buf1));
        OCL_SAFE_CALL(clReleaseCommandQueue(qu));
    }
    OCL_CTX_SAFE_CALL(err, ctx);

    // 6 Выполните 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь, что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания),
    // напечатав исходники в консоль (if проверяет, что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
         std::cout << kernel_sources << std::endl;
    }

    // 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель
    const char *c_kernel_sources = kernel_sources.c_str();
    const size_t c_kernel_sources_length = kernel_sources.length();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &c_kernel_sources, nullptr, &err);
    if (CL_SUCCESS != err) {
        OCL_SAFE_CALL(clReleaseProgram(prog));
        OCL_SAFE_CALL(clReleaseMemObject(buf3));
        OCL_SAFE_CALL(clReleaseMemObject(buf2));
        OCL_SAFE_CALL(clReleaseMemObject(buf1));
        OCL_SAFE_CALL(clReleaseCommandQueue(qu));
    }
    OCL_CTX_SAFE_CALL(err, ctx);

    // 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    cl_int err1, err2, err3;
    err1 = clBuildProgram(prog, 1, &someDevice, nullptr, nullptr, nullptr);

    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo
    size_t log_size = 0;

    err2 = clGetProgramBuildInfo(prog, someDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

    std::vector<char> log(log_size, 0);
    err3 = clGetProgramBuildInfo(prog, someDevice, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);

    if (log_size > 1) {
        std::cout << "Log:" << std::endl;
        std::cout << log.data() << std::endl;
    } else {
        std::cout << "Build log is empty" << std::endl;
    }

    if (CL_SUCCESS != err1) {
        OCL_SAFE_CALL(clReleaseProgram(prog));
        OCL_SAFE_CALL(clReleaseMemObject(buf3));
        OCL_SAFE_CALL(clReleaseMemObject(buf2));
        OCL_SAFE_CALL(clReleaseMemObject(buf1));
        OCL_SAFE_CALL(clReleaseCommandQueue(qu));
    }
    OCL_CTX_SAFE_CALL(err1, ctx);
    if (CL_SUCCESS != err2) {
        OCL_SAFE_CALL(clReleaseProgram(prog));
        OCL_SAFE_CALL(clReleaseMemObject(buf3));
        OCL_SAFE_CALL(clReleaseMemObject(buf2));
        OCL_SAFE_CALL(clReleaseMemObject(buf1));
        OCL_SAFE_CALL(clReleaseCommandQueue(qu));
    }
    OCL_CTX_SAFE_CALL(err2, ctx);
    if (CL_SUCCESS != err3) {
        OCL_SAFE_CALL(clReleaseProgram(prog));
        OCL_SAFE_CALL(clReleaseMemObject(buf3));
        OCL_SAFE_CALL(clReleaseMemObject(buf2));
        OCL_SAFE_CALL(clReleaseMemObject(buf1));
        OCL_SAFE_CALL(clReleaseCommandQueue(qu));
    }
    OCL_CTX_SAFE_CALL(err3, ctx);
    // 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
    cl_kernel kernel = clCreateKernel(prog, "aplusb", &err);
    if (CL_SUCCESS != err) {
        OCL_SAFE_CALL(clReleaseProgram(prog));
        OCL_SAFE_CALL(clReleaseMemObject(buf3));
        OCL_SAFE_CALL(clReleaseMemObject(buf2));
        OCL_SAFE_CALL(clReleaseMemObject(buf1));
        OCL_SAFE_CALL(clReleaseCommandQueue(qu));
    }
    OCL_CTX_SAFE_CALL(err, ctx);

    // 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
        unsigned int i = 0;
        // todo this todo is a reminder that I had a pesky error here as I used as.data() as last argument which resulted in a SIGSEGV down the line
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(float *), &buf1));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(float *), &buf2));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(float *), &buf3));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(unsigned int), &n));
    }

    // 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

    std::cout << "Starting kernel computations" << std::endl;
    // 12 Запустите выполнения кернела:
    // - С одномерной рабочей группой размера 128
    // - В одномерном рабочем пространстве размера roundedUpN, где roundedUpN - наименьшее число, кратное 128 и при этом не меньшее n
    // - см. clEnqueueNDRangeKernel
    // - Обратите внимание, что, чтобы дождаться окончания вычислений (чтобы знать, когда можно смотреть результаты в cs_gpu) нужно:
    //   - Сохранить событие "кернел запущен" (см. аргумент "cl_event *event")
    //   - Дождаться завершения полунного события - см. в документации подходящий метод среди Event Objects
    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;// Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            // clEnqueueNDRangeKernel...
            cl_event kernel_start_event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(qu, kernel, 1, nullptr, &global_work_size, &workGroupSize, 0, nullptr, &kernel_start_event));
            // clWaitForEvents...

            OCL_SAFE_CALL(clWaitForEvents(1, &kernel_start_event));

            t.nextLap();// При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Finished kernel computations" << std::endl;
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        // 13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "GFlops: " << n / t.lapAvg() / 1000000000 << std::endl;

        // 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << 3.0 * n * sizeof(float) / t.lapAvg() / (1024 * 1024 * 1024) << " GB/s" << std::endl;
    }

    // 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            // clEnqueueReadBuffer...
            OCL_SAFE_CALL(clEnqueueReadBuffer(qu, buf3, CL_TRUE, 0, n * sizeof(float), cs.data(), 0, nullptr, nullptr));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << 1.0 * n * sizeof(float) / t.lapAvg() / (1024 * 1024 * 1024) << " GB/s" << std::endl;
    }

    // 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    return 0;
}
