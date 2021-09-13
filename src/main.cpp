#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>


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
        size_t platformNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));

        std::vector<unsigned char> platformName(platformNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        size_t platformVendorSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
        std::vector<unsigned char> platformVendor(platformVendorSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
        std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            auto deviceId = devices[deviceIndex];
            std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;

            size_t deviceNameSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
            std::vector<unsigned char> deviceName(deviceNameSize);
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
            std::cout << "        Name: " << deviceName.data() << std::endl;

            cl_device_type deviceType;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, sizeof deviceType, &deviceType, nullptr));

            std::string deviceTypeString = "Unknown";
            switch (deviceType) {
                case CL_DEVICE_TYPE_CPU: deviceTypeString = "CPU"; break;
                case CL_DEVICE_TYPE_GPU: deviceTypeString = "GPU"; break;
                case CL_DEVICE_TYPE_ACCELERATOR: deviceTypeString = "Accelerator"; break;
                case CL_DEVICE_TYPE_DEFAULT: deviceTypeString = "Default"; break;
            }

            std::cout << "        Type: " << deviceTypeString << std::endl;

            cl_ulong memorySize;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof memorySize, &memorySize, nullptr));
            std::cout << "        Memory size: " << memorySize / (1 << 20) << " MiB" << std::endl;

            size_t deviceProfileSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_PROFILE, 0, nullptr, &deviceProfileSize));
            std::vector<unsigned char> deviceProfile(deviceProfileSize);
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_PROFILE, deviceProfileSize, deviceProfile.data(), nullptr));
            std::cout << "        Profile: " << deviceProfile.data() << std::endl;

            size_t deviceCVersionSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_OPENCL_C_VERSION, 0, nullptr, &deviceCVersionSize));
            std::vector<unsigned char> deviceCVersion(deviceCVersionSize);
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_OPENCL_C_VERSION, deviceCVersionSize, deviceCVersion.data(), nullptr));
            std::cout << "        OpenCL C version: " << deviceCVersion.data() << std::endl;
        }
    }

    return 0;
}
