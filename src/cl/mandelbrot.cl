#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float *results, unsigned int width, unsigned int height, float fromX, float fromY,
                         float sizeX, float sizeY, unsigned int iters, unsigned int smoothing) {
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    size_t i = get_global_id(0);
    size_t j = get_global_id(1);

    const float x0 = fromX + ((float) i + 0.5f) * sizeX / width;
    const float y0 = fromY + ((float) j + 0.5f) * sizeY / height;

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
    if (smoothing == 1 && iter != iters) {
        result -= log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
    }
    result = 1.0f * result / iters;
    results[j * width + i] = result;
}
