#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float * results,
                         unsigned int width, unsigned int height,
                         float fromX, float fromY,
                         float xScale, float yScale,
                         unsigned int iterationsLimit, int smoothing)
{
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    float x0 = fromX + (i + 0.5f) * xScale / width;
    float y0 = fromY + (j + 0.5f) * yScale / height;

    float x = x0;
    float y = y0;

    unsigned int iteration = 0;
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;
    while (x * x + y * y <= threshold2 && iteration < iterationsLimit) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        iteration = iteration + 1;
    }
    float result = iteration;
    if (smoothing && iteration != iterationsLimit) {
        result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
    }

    result = 1.0f * result / iterationsLimit;

    results[j * width + i] = result;
}
