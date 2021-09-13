#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
//#include <string>
//#include <algorithm>


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


int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // Откройте 
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    // Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
    // Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(
            clGetPlatformIDs(10 /* If platforms is not NULL, the num_entries must be greater than zero. */, nullptr,
                             &platformsCount));
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
        // TODO 1.1
        // Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
        // Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
        // Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
        // Откройте таблицу с кодами ошибок:
        // libs/clew/CL/cl.h:103
        // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
        // Найдите там нужный код ошибки и ее название
        // CL_INVALID_VALUE -30
        // Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
        // CL_INVALID_VALUE if param_name is not one of the supported values or if size in bytes specified by param_value_size is less than size of return type and param_value is not a NULL value.
        // в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
        // Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

        // TODO 1.2
        // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        std::vector<unsigned char> platformName(platformNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        size_t platformVendorNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorNameSize));
        std::vector<unsigned char> platformVendorName(platformVendorNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorNameSize, platformVendorName.data(),
                                        nullptr));
        std::cout << "    Platform vendor: " << platformVendorName.data() << std::endl;


        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platforms[platformIndex], CL_DEVICE_TYPE_ALL, 1, nullptr, &devicesCount));
        std::cout << "    Number of " << platformVendorName.data() << " devices: " << devicesCount << std::endl;

        // devices IDs
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(
                clGetDeviceIDs(platforms[platformIndex], CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            // TODO 2.2
            // Запросите и напечатайте в консоль:
            std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
            cl_device_id device = devices[deviceIndex];
            // - Название устройства
            size_t deviceNameSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));

            std::vector<unsigned char> deviceName(deviceNameSize, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
            std::cout << "        Device name: " << deviceName.data() << std::endl;
            // - Тип устройства (видеокарта/процессор/что-то странное)
            cl_device_type deviceType = CL_DEVICE_TYPE_ALL;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
            switch (deviceType) {
                case CL_DEVICE_TYPE_CPU:
                    std::cout << "        CPU" << std::endl;
                    break;
                case CL_DEVICE_TYPE_GPU:
                    std::cout << "        GPU" << std::endl;
                    break;
                default:
                    std::cout << "        OTHER" << std::endl;
                    break;
            }
            // - Размер памяти устройства в мегабайтах
            cl_ulong memorySize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memorySize, nullptr));
            std::cout << "        Memory size: " << memorySize / (1024 * 1024) << " MB" << std::endl;
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            cl_ulong cacheSize = 0;
            OCL_SAFE_CALL(
                    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &cacheSize, nullptr));
            std::cout << "        Cache size: " << cacheSize / 1024 << " KB" << std::endl;

            size_t clVersionSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, nullptr, &clVersionSize));

            std::vector<unsigned char> clVersion(clVersionSize, 0);
            OCL_SAFE_CALL(
                    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, clVersionSize, clVersion.data(), nullptr));
            std::cout << "        OpenCL Version: " << clVersion.data() << std::endl;

            size_t driverSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, nullptr, &driverSize));

            std::vector<unsigned char> driver(driverSize, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DRIVER_VERSION, driverSize, driver.data(), nullptr));
            std::cout << "        Driver Version: " << driver.data() << std::endl;

            size_t extensionsSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &extensionsSize));

            std::vector<unsigned char> extensions(extensionsSize, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, extensionsSize, extensions.data(), nullptr));
            // std::cout << "        Extensions: " << extensions.data() << std::endl;
        }
    }

    return 0;
}

// My output:
//Number of OpenCL platforms: 2
//Platform #1/2
//    Platform name: Intel(R) CPU Runtime for OpenCL(TM) Applications
//    Platform vendor: Intel(R) Corporation
//    Number of Intel(R) Corporation devices: 1
//    Device #1/1
//        Device name: Intel(R) Core(TM) i5-6300HQ CPU @ 2.30GHz
//        CPU
//        Memory size: 15883 MB
//        Cache size: 256 KB
//        OpenCL Version: OpenCL C 2.0
//        Driver Version: 18.1.0.0920
//Platform #2/2
//    Platform name: NVIDIA CUDA
//    Platform vendor: NVIDIA Corporation
//    Number of NVIDIA Corporation devices: 1
//    Device #1/1
//        Device name: NVIDIA GeForce GTX 970M
//        GPU
//        Memory size: 3024 MB
//        Cache size: 480 KB
//        OpenCL Version: OpenCL C 1.2
//        Driver Version: 470.57.02
