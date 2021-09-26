#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float *image, unsigned w, unsigned h, float fromX, float fromY, float sizeX, float sizeY, unsigned iters, int what)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);

    if (idx < w && idy < h) {
        float x0 = fromX + (idx + 0.5f) * sizeX / w;
        float y0 = fromY + (idy + 0.5f) * sizeY / h;

        float x = x0;
        float y = y0;

        int iter = 0;
        iters = 256;
        for (; iter < iters; ++iter) {
            float xPrev = x;
            x = x * x - y * y + x0;
            y = 2.0f * xPrev * y + y0;
            if ((x * x + y * y) > 65536) {
                break;
            }
        }
        image[idx + w * idy] = (float) iter / iters;
    }
}
