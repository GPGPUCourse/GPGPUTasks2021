#include <cmath>

#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libimages/images.h>

#include "cl/mandelbrot_cl.h"


void mandelbrotCPU(float* results,
                   unsigned int width, unsigned int height,
                   float fromX, float fromY,
                   float sizeX, float sizeY,
                   unsigned int iters, bool smoothing)
{
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;
    
    #pragma omp parallel for
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            float x0 = fromX + (i + 0.5f) * sizeX / width;
            float y0 = fromY + (j + 0.5f) * sizeY / height;

            float x = x0;
            float y = y0;

            int iter = 0;
            for (; iter < iters; ++iter) {
                float xPrev = x;
                x = x * x - y * y + x0;
                y = 2.0f * xPrev * y + y0;
                if ((x * x + y * y) > threshold2) {
                    break;
                }
            }
            float result = iter;
            if (smoothing && iter != iters) {
                result = result - logf(logf(sqrtf(x * x + y * y)) / logf(threshold)) / logf(2.0f);
            }

            result = 1.0f * result / iters;
            results[j * width + i] = result;
        }
    }
}

void renderToColor(const float* results, unsigned char* img_rgb, unsigned int width, unsigned int height);

void renderInWindow(float centralX, float centralY, unsigned int iterationsLimit, bool useGPU);


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    unsigned int benchmarkingIters = 10;

    unsigned int width = 2048;
    unsigned int height = 2048;
    unsigned int iterationsLimit = 256;

    float centralX = -0.789136f;
    float centralY = -0.150316f;
    float sizeX = 0.00239f;

//    // Менее красивый ракурс, но в этом ракурсе виден весь фрактал:
//    float centralX = -0.5f;
//    float centralY = 0.0f;
//    float sizeX = 2.0f;

    images::Image<float> cpu_results(width, height, 1);
    images::Image<float> gpu_results(width, height, 1);
    images::Image<unsigned char> image(width, height, 3);

    float sizeY = sizeX * height / width;

    {
        timer t;
        for (int i = 0; i < benchmarkingIters; ++i) {
            mandelbrotCPU(cpu_results.ptr(),
                          width, height,
                          centralX - sizeX / 2.0f, centralY - sizeY / 2.0f,
                          sizeX, sizeY,
                          iterationsLimit, false);
            t.nextLap();
        }
        size_t flopsInLoop = 10;
        size_t maxApproximateFlops = width * height * iterationsLimit * flopsInLoop;
        size_t gflops = 1000*1000*1000;
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << maxApproximateFlops / gflops / t.lapAvg() << " GFlops" << std::endl;

        double realIterationsFraction = 0.0;
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                realIterationsFraction += cpu_results.ptr()[j * width + i];
            }
        }
        std::cout << "    Real iterations fraction: " << 100.0 * realIterationsFraction / (width * height) << "%" << std::endl;

        renderToColor(cpu_results.ptr(), image.ptr(), width, height);
        image.savePNG("mandelbrot_cpu.png");
    }


//    // Раскомментируйте это:
//
//    gpu::Context context;
//    context.init(device.device_id_opencl);
//    context.activate();
//    {
//        ocl::Kernel kernel(mandelbrot_kernel, mandelbrot_kernel_length, "mandelbrot");
//        // Если у вас есть интеловский драйвер для запуска на процессоре - попробуйте запустить на нем и взгляните на лог,
//        // передав printLog=true - скорее всего, в логе будет строчка вроде
//        // Kernel <mandelbrot> was successfully vectorized (8)
//        // это означает, что драйвер смог векторизовать вычисления с помощью интринсик, и если множитель векторизации 8, то
//        // это означает, что одно ядро процессит сразу 8 workItems, а т.к. все вычисления в float, то
//        // это означает, что используются 8 x float регистры (т.е. 256-битные, т.е. AVX)
//        // обратите внимание, что и произвдительность относительно референсной ЦПУ реализации выросла почти в восемь раз
//        bool printLog = false;
//        kernel.compile(printLog);
//        // TODO близко к ЦПУ-версии, включая рассчет таймингов, гигафлопс, Real iterations fraction и сохранение в файл
//        // результат должен оказаться в gpu_results
//    }
//
//    {
//        double errorAvg = 0.0;
//        for (int j = 0; j < height; ++j) {
//            for (int i = 0; i < width; ++i) {
//                errorAvg += fabs(gpu_results.ptr()[j * width + i] - cpu_results.ptr()[j * width + i]);
//            }
//        }
//        errorAvg /= width * height;
//        std::cout << "GPU vs CPU average results difference: " << 100.0 * errorAvg << "%" << std::endl;
//
//        if (errorAvg > 0.03) {
//            throw std::runtime_error("Too high difference between CPU and GPU results!");
//        }
//    }

    // Это бонус в виде интерактивной отрисовки, не забудьте запустить на ГПУ, чтобы посмотреть, в какой момент числа итераций/точности single float перестанет хватать
    // Кликами мышки можно смещать ракурс
    // Но в Pull-request эти две строки должны быть закомментированы, т.к. на автоматическом тестировании нет оконной подсистемы 
//    bool useGPU = false;
//    renderInWindow(centralX, centralY, iterationsLimit, useGPU);

    return 0;
}

void renderInWindow(float centralX, float centralY, unsigned int iterationsLimit, bool useGPU)
{
    images::ImageWindow window("Mandelbrot");

    unsigned int width = 1024;
    unsigned int height = 1024;

    float sizeX = 2.0f;
    float sizeY = sizeX * height / width;

    float zoomingSpeed = 1.005f;

    images::Image<float> results(width, height, 1);
    images::Image<unsigned char> image(width, height, 3);

    ocl::Kernel kernel(mandelbrot_kernel, mandelbrot_kernel_length, "mandelbrot");
    gpu::gpu_mem_32f results_vram;
    if (useGPU) {
        kernel.compile();
        results_vram.resizeN(width * height);
    }

    do {
        if (!useGPU) {
            mandelbrotCPU(results.ptr(), width, height,
                          centralX - sizeX / 2.0f, centralY - sizeY / 2.0f,
                          sizeX, sizeY,
                          iterationsLimit, true);
        } else {
            kernel.exec(gpu::WorkSize(16, 16, width, height),
                        results_vram, width, height,
                        centralX - sizeX / 2.0f, centralY - sizeY / 2.0f,
                        sizeX, sizeY,
                        iterationsLimit, 1);
            results_vram.readN(results.ptr(), width * height);
        }
        renderToColor(results.ptr(), image.ptr(), width, height);

        window.display(image);
        window.wait(30);

        if (window.getMouseClick() == MOUSE_LEFT) {
            centralX = centralX - sizeX * 0.5f + sizeX * window.getMouseX() / width;
            centralY = centralY - sizeY * 0.5f + sizeY * window.getMouseY() / height;
            std::cout << "Focus: " << centralX << " " << centralY  << " " << sizeX << std::endl;
        }
        if (window.isResized()) {
            window.resize();
            width = window.width();
            height = window.height();
            std::cout << "Resized to " << width << "x" << height << std::endl;

            sizeY = sizeX * height / width;

            results = images::Image<float>(width, height, 1);
            image = images::Image<unsigned char>(width, height, 3);

            if (useGPU) {
                results_vram.resizeN(width * height);
            }
        }
        sizeX /= zoomingSpeed;
        sizeY /= zoomingSpeed;
    } while (!window.isClosed());
}


struct vec3f {
    vec3f(float x, float y, float z) : x(x), y(y), z(z) {}

    float x; float y; float z;
};

vec3f operator+(const vec3f &a, const vec3f &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

vec3f operator*(const vec3f &a, const vec3f &b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

vec3f operator*(const vec3f &a, float t) {
    return {a.x * t, a.y * t, a.z * t};
}

vec3f operator*(float t, const vec3f &a) {
    return a * t;
}

vec3f sin(const vec3f &a) {
    return {sinf(a.x), sinf(a.y), sinf(a.z)};
}

vec3f cos(const vec3f &a) {
    return {cosf(a.x), cosf(a.y), cosf(a.z)};
}

void renderToColor(const float* results, unsigned char* img_rgb,
             unsigned int width, unsigned int height)
{
    #pragma omp parallel for
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            // Палитра взята отсюда: http://iquilezles.org/www/articles/palettes/palettes.htm
            float t = results[j * width + i];
            vec3f a(0.5, 0.5, 0.5);
            vec3f b(0.5, 0.5, 0.5);
            vec3f c(1.0, 0.7, 0.4);
            vec3f d(0.00, 0.15, 0.20);
            vec3f color = a + b * cos(2*3.14f*(c*t+d));
            img_rgb[j * 3 * width + i * 3 + 0] = (unsigned char) (color.x * 255);
            img_rgb[j * 3 * width + i * 3 + 1] = (unsigned char) (color.y * 255);
            img_rgb[j * 3 * width + i * 3 + 2] = (unsigned char) (color.z * 255);
        }
    }
}
