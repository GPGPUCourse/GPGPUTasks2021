В этом репозитории предложены задания для [курса по вычислениям на видеокартах в CSC](https://compscicenter.ru/courses/video_cards_computation/2021-autumn/).

[Остальные задания](https://github.com/GPGPUCourse/GPGPUTasks2021/).

# Задание 0. Вводное.

[![Build Status](https://app.travis-ci.com/GPGPUCourse/GPGPUTasks2021.svg?branch=task00)](https://app.travis-ci.com/GPGPUCourse/GPGPUTasks2021)

Установка OpenCL-драйвера для процессора
========================================

Установить OpenCL-драйвер для процессора полезно, даже если у вас есть видеокарта, т.к. на нем удобно тестировать приложение (драйвер видеокарты гораздо чаще может повиснуть вместе с ОС).

Windows
-------

1. Откройте https://software.intel.com/content/www/us/en/develop/tools/opencl-cpu-runtime.html
2. Скачайте (требует регистрацию, [прямая ссылка для Windows](http://registrationcenter-download.intel.com/akdlm/irc_nas/vcp/13794/opencl_runtime_18.1_x64_setup.msi) - если не качает - попробуйте из-под инкогнито или [отсюда](https://disk.yandex.ru/d/dlVbMoI3tsPZfw))
3. Установите

Linux (Рекомендуется Ubuntu 18.04 или 20.04)
----------------------------------

1. Откройте https://software.intel.com/content/www/us/en/develop/tools/opencl-cpu-runtime.html
2. Скачайте (требует регистрацию, [прямая ссылка для Ubuntu](http://registrationcenter-download.intel.com/akdlm/irc_nas/vcp/15532/l_opencl_p_18.1.0.015.tgz) - если не качает - попробуйте из-под инкогнито или [отсюда](https://disk.yandex.ru/d/dlVbMoI3tsPZfw))
3. ``apt-get install -yq cpio``
4. ``tar -xzf l_opencl_p_18.1.0.015.tgz``
5. ``sudo ./l_opencl_p_18.1.0.015/install.sh``
6. Проведите установку.

Если у вас довольно новый процессор, например i7-8550U, то драйвер может его не поддерживать - ```clCreateContext``` вернет ошибку ```CL_DEVICE_NOT_AVAILABLE```, в таком случае поставьте свежий драйвер [отсюда](https://github.com/intel/compute-runtime/releases) (включает в т.ч. драйвер для встроенной Intel GPU).

Если в процессе запуска этого задания процессор не виден как допустимое OpenCL-устройство - создайте **Issue** в этом репозитории с перечислением:

 - Версия OS
 - Вывод команды ``ls /etc/OpenCL/vendors``
 - Если там в т.ч. есть ``intel.icd`` файл - то его содержимое (это маленький текстовый файл)

Установка OpenCL-драйвера для видеокарты
========================================

Windows
-------

Поставьте драйвер стандартным образом - скачав инсталлятор с официального сайта вендора вашей видеокарты и установив.

Linux
-----

NVidia: ``sudo apt install nvidia-384``

AMD: [скачав](https://www.amd.com/en/support) и установив amdgpu-pro драйвер

Проверка окружения и начало выполнения задания
==============================================

Про работу под Windows см. в секции [Как работать под windows](#%D0%9A%D0%B0%D0%BA-%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%D1%82%D1%8C-%D0%BF%D0%BE%D0%B4-windows).

1. Сделайте fork этого репозитория
2. ``git clone ВАШ_ФОРК_РЕПОЗИТОРИЯ``
3. ``cd GPGPUTasks2021``
4. ``git checkout task00``
5. ``mkdir build``
6. ``cd build``
7. ``cmake ..``
8. ``make -j4``
9. ``./enumDevices`` должно увидеть хотя бы одну OpenCL-платформу:

```
Number of OpenCL platforms: 1
Platform #1/1
    Platform name: 
```

Если же вы видите ошибку:
```
terminate called after throwing an instance of 'std::runtime_error'
  what(): Can't init OpenCL driver!
Aborted (Core dumped)
```
То попробуйте установить ```sudo apt install ocl-icd-libopencl1``` и выполнить ``./enumDevices`` снова.

Если вы видите ошибку:
```
: CommandLine Error: Option 'polly' registered more than once!
LLVM ERROR: inconsistency in registered CommandLine options
```
То, наоборот, может помочь удалить пакет ```sudo apt remove ocl-icd-libopencl1``` и попробовать выполнить ``./enumDevices`` снова.

Если ``./enumDevices`` не показывает хотя бы одну платформу - создайте **Issue** с перечислением:

 - OS, процессор и видеокарта
 - Успешно ли прошла установка Intel-CPU драйвера
 - Какое было поведение до установки пакета ``ocl-icd-libopencl1`` и какое поведение стало после
 - Вывод ``./enumDevices``

Задание
=======

0. Сделать fork проекта
1. Прочитать все комментарии подряд и выполнить все **TODO** в файле ``src/main.cpp``. Для разработки под Linux рекомендуется использовать CLion. Под Windows рекомендуется использовать CLion+MSVC. Также под Windows можно использовать Visual Studio Community.
2. Отправить **Pull-request** с названием ```Task00 <Имя> <Фамилия> <Аффиляция>```. **Аффиляция** - например JetBrains/Yandex если вы из дружественной компании (а не студент) - чтобы я понимал, что вас проверять необязательно (иначе - **CSC**, или если вы из дружественного ВУЗа - **его название**).
3. В тексте **PR** укажите вывод программы при исполнении на Travis CI и на вашем компьютере (в **pre**-тэгах, чтобы сохранить форматирование, см. [пример](https://raw.githubusercontent.com/GPGPUCourse/GPGPUTasks2021/task00/.github/pull_request_example.md)). И ваш бранч должен называться так же, как и у меня - **task00**.
4. Убедиться что Travis continuous integration смог скомпилировать ваш код и что все хорошо (если нет - то поправить, пожалуйста, не используйте C++ из будущего, о котором не знает GCC 5.5)
5. Ждать комментарии проверки

**Дедлайн**: начало лекции 13 сентября. Но убедиться, что хотя бы одно OpenCL-устройство у вас обнаруживается, лучше как можно раньше (см. **Проверка окружения** выше), т.к., если с этим будут проблемы - я могу не успеть вам помочь достаточно быстро.

Как работать под Windows
========================

1. Используйте **64-битный компилятор**, т.е. [amd64](/.figures/clion_msvc_settings.png), а не x86. (Если при запуске видите ``Invalid Parameter - 100``, то вы все еще используете 32-битный компилятор)
2. Рекомендуется использовать CLion+MSVC.
3. Можно использовать Visual Studio 2017 Community или новее, она поддерживает CMake-проекты (``File`` -> ``Open`` -> ``Cmake...``). Разве что передавать аргументы запускаемой программе [неудобно](https://docs.microsoft.com/en-us/cpp/ide/cmake-tools-for-visual-cpp?view=vs-2017#configure-cmake-debugging-sessions).




