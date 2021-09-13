#include <CL/cl.h>
#include <libclew/ocl_init.h>

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
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


void printPlatformName(cl_platform_id platform)
{
    size_t platformNameSize = 0;
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
    std::vector<unsigned char> platformName(platformNameSize, 0);
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
    std::cout << "    Platform name: " << platformName.data() << std::endl;
}

void printPlatformVendor(cl_platform_id platform)
{
    size_t platformVendorSize = 0;
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
    std::vector<unsigned char> platformVendor(platformVendorSize, 0);
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
    std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;
}

std::vector<cl_device_id> printDevicesCount(cl_platform_id platform)
{
    cl_uint devicesCount = 0;
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
    std::cout << "    Number of OpenCL devices: " << devicesCount << std::endl;
    std::vector<cl_device_id> devices(devicesCount);
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
    return devices;
}

void printDeviceName(cl_device_id device)
{
    size_t deviceNameSize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
    std::vector<unsigned char> deviceName(deviceNameSize, 0);
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
    std::cout << "        Platform device name: " << deviceName.data() << std::endl;
}

void printDeviceType(cl_device_id device)
{
    cl_device_type deviceType = CL_DEVICE_TYPE_ALL;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
    std::cout << "        Platform device type: ";
    switch (deviceType) {
        case CL_DEVICE_TYPE_DEFAULT:
            std::cout << "DEFAULT";
            break;
        case CL_DEVICE_TYPE_CPU:
            std::cout << "CPU";
            break;
        case CL_DEVICE_TYPE_GPU:
            std::cout << "GPU";
            break;
        case CL_DEVICE_TYPE_ACCELERATOR:
            std::cout << "ACCELERATOR";
            break;
        case CL_DEVICE_TYPE_ALL:
            std::cout << "ALL";
            break;
        default:
            std::cout << "UNKNOWN";
            break;
    }
    std::cout << std::endl;
}

void printDeviceMemorySize(cl_device_id device)
{
    cl_ulong deviceMemorySize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &deviceMemorySize, nullptr));
    std::cout << "        Platform device global memory size: " << deviceMemorySize / 1024 / 1024 << " MB" << std::endl;
}

void printDeviceMaxMemAllocSize(cl_device_id device)
{
    cl_ulong deviceMaxMemAllocSize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &deviceMaxMemAllocSize, nullptr));
    std::cout << "        Platform device max memory alloc size: " << deviceMaxMemAllocSize / 1024 / 1024 << " MB" << std::endl;
}

void printDeviceMaxWorkGroupSize(cl_device_id device)
{
    cl_ulong deviceMaxWorkGroupSize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_ulong), &deviceMaxWorkGroupSize, nullptr));
    std::cout << "        Platform device max work group size: " << deviceMaxWorkGroupSize << std::endl;
}

int main()
{
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // Откройте
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    // Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
    // Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    // Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        // Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
        // Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL

        // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        printPlatformName(platform);
        // Запросите и напечатайте так же в консоль вендора данной платформы
        printPlatformVendor(platform);

        // Запросите число доступных устройств данной платформы (аналогично тому как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        auto devices = printDevicesCount(platform);
        for (cl_device_id device : devices) {
            // Запросите и напечатайте в консоль:
            // - Название устройства
            printDeviceName(device);
            // - Тип устройства (видеокарта/процессор/что-то странное)
            printDeviceType(device);
            // - Размер памяти устройства в мегабайтах
            printDeviceMemorySize(device);
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            printDeviceMaxMemAllocSize(device);
            printDeviceMaxWorkGroupSize(device);
        }
    }

    return 0;
}
