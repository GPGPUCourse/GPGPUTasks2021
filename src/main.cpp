/*************************************************************************
 * This file has been reformatted with 2 spaces, as per Google Code Style
 * Sorry
 ************************************************************************/
#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>

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

  // todo shouldn't it be better to use cl_uint instead of int?
  for (cl_uint platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
    std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
    cl_platform_id platform = platforms[platformIndex];

    // Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
    // Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
    size_t platformNameSize = 0;
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
    // TODO 1.1
    // Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
    // todo я передам 30!!
    // Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
    // Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
    // Откройте таблицу с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    // Найдите там нужный код ошибки и ее название
    // todo CL_INVALID_VALUE - такая ошибка мне встретилась
    // в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
    // Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

    // TODO 1.2
    // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
    std::vector<unsigned char> platformName(platformNameSize, 0);
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
    std::cout << std::string(4, ' ') << "Platform name: " << platformName.data() << std::endl;

    // TODO 1.3
    // Запросите и напечатайте так же в консоль вендора данной платформы
    size_t platformVendorSize = 0;
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
    std::vector<unsigned char> platformVendor(platformVendorSize, 0);
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
    std::cout << std::string(4, ' ') << "Platform vendor: " << platformVendor.data() << std::endl;

    // TODO 2.1
    // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
    cl_uint devicesCount = 0;
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
    std::cout << std::string(8, ' ') << "Number of devices for current platform: "
              << devicesCount << std::endl;

    std::vector<cl_device_id> devices(devicesCount);
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

    for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
      // TODO 2.2
      // Запросите и напечатайте в консоль:
      // - Название устройства
      // - Тип устройства (видеокарта/процессор/что-то странное)
      // - Размер памяти устройства в мегабайтах
      // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
      std::cout << std::string(12, ' ') << "Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
      cl_device_id device = devices[deviceIndex];

      getDeviceInfoString(device, CL_DEVICE_NAME, 16, "Device name: ");

      cl_device_type deviceType;
      OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
      std::cout << std::string(16, ' ') << "Device type: "<< deviceType << " [Consult cl.h:172 for more info]" << std::endl;
      cl_ulong deviceMem;
      OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &deviceMem, nullptr));
      std::cout << std::string(16, ' ') << "Device memory in MB: "<< deviceMem / 1000000 << std::endl;

      getDeviceInfoString(device, CL_DEVICE_PROFILE, 16, "Device profile (consult documentation for more info): ");
      getDeviceInfoString(device, CL_DRIVER_VERSION, 16, "Device OpenCL driver version (consult documentation for more info): ");
    }
  }

  return 0;
}
