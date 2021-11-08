__kernel void merge(__global const float *as, __global float *buff, uint n, uint i) {
    const uint gl_id = get_global_id(0);
    const uint id = gl_id & (2 * i - 1);

    if (gl_id >= n) {
        return;
    }

    uint lhs =  0;
    uint rhs =  id + 1;

    if (id >= i) {
        lhs = id + 1 - i;
    }
    if (id >= i - 1) {
        rhs = i;
    }

    while (lhs < rhs) {
        const uint avg = (lhs + rhs) / 2;
        if (as[gl_id - id + avg] <= as[gl_id + i - avg]) {
            lhs = avg + 1;
        } else {
            rhs = avg;
        }
    }

    float m_f = -10000;
    float m_s = -10000;

    if (lhs != 0) {
        m_f = as[gl_id - id + lhs - 1];
    }
    if (lhs != id + 1) {
        m_s = as[gl_id + i - lhs];
    }

    buff[gl_id] = max(m_f, m_s);
}