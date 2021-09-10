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
        // TODO 1.1
        // Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
        // Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
        // Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
        // Откройте таблицу с кодами ошибок:
        // libs/clew/CL/cl.h:103
        // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
        // Найдите там нужный код ошибки и ее название
        // Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
        // в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
        // Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевозможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

        // TODO 1.2
        // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        std::vector<unsigned char> platformName(platformNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        size_t vendorSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorSize));
        std::vector<unsigned char> platformVendor(vendorSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorSize, platformName.data(), nullptr));
        std::cout << "    Vendor name: " << platformName.data() << std::endl;

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        cl_device_id deviceId = nullptr;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &deviceId, &devicesCount));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            size_t retSize = 0;
            constexpr size_t mb = 1024 * 1024;
            clGetDeviceInfo(deviceId, CL_DEVICE_NAME, 0, nullptr, &retSize);

            std::vector<char> deviceInfo(retSize, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_NAME, retSize, deviceInfo.data(), &retSize));
            std::cout << deviceInfo.data() << std::endl;

            cl_device_type deviceType;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, &retSize));
            std::cout << "DEVICE TYPE: ";
            if (deviceType == CL_DEVICE_TYPE_CPU) {
                std::cout << "CPU" << std::endl;
            } else if (deviceType == CL_DEVICE_TYPE_GPU) {
                std::cout << "GPU" << std::endl;
            } else if (deviceType == CL_DEVICE_TYPE_ACCELERATOR) {
                std::cout << "ACCELERATOR" << std::endl;
            } else if (deviceType == CL_DEVICE_TYPE_DEFAULT) {
                std::cout << "DEFAULT" << std::endl;
            }

            cl_ulong global_mem_size = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_device_type), &global_mem_size,
                                          &retSize));
            std::cout << "GLOBAL MEM SIZE: " << global_mem_size / mb << std::endl;

            cl_bool image_support = false;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_device_type), &image_support,
                                          &retSize));
            std::cout << "IMAGE SUPPORT: " << image_support << std::endl;

            size_t max_2d_HEIGHT = 0;
            size_t max_2d_WIDTH = 0;

            OCL_SAFE_CALL(
                    clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(cl_device_type), &max_2d_HEIGHT,
                                    &retSize));
            OCL_SAFE_CALL(
                    clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(cl_device_type), &max_2d_WIDTH,
                                    &retSize));

            std::cout << "MAX 2D HEIGHT and WIDTH: " << "H: " << max_2d_HEIGHT << ", W: " << max_2d_WIDTH << std::endl;

            size_t max_3d_HEIGHT = 0;
            size_t max_3d_WIDTH = 0;
            size_t max_3d_DEPTH = 0;

            OCL_SAFE_CALL(
                    clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(cl_device_type), &max_3d_HEIGHT,
                                    &retSize));
            OCL_SAFE_CALL(
                    clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(cl_device_type), &max_3d_WIDTH,
                                    &retSize));
            OCL_SAFE_CALL(
                    clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(cl_device_type), &max_3d_DEPTH,
                                    &retSize));

            std::cout << "MAX 3D HEIGHT, WIDTH and DEPTH: " << "H: " << max_3d_HEIGHT
                      << ", W: " << max_2d_WIDTH << ", D: " << max_3d_DEPTH << std::endl;
        }
    }

    return 0;
}
