void swap(__global float* f, __global float* s) {
    float tmp;
    tmp = *f;
    *f = *s;
    *s = tmp;
}


__kernel void bitonic(__global float *as, unsigned int n, unsigned int logdist, unsigned int logrepeat) {
    unsigned int ind = get_global_id(0);

    if ( ((ind >> logdist) & 1) == 0 && ind < n) {
        // if edge_orientation, sort pair ind, ind + 2**logdist is descending order,
        // otherwise in ascending order
        const int edge_orientation =(1 & (ind >> logdist >> logrepeat >> 1)) ? -1 : 1;
        if (edge_orientation == 1 && as[ind] > as[ind + (1 << logdist)]) {
            swap(as + ind, as + ind + (1 << logdist));
        }
        if (edge_orientation == -1 && as[ind] < as[ind + (1 << logdist)]) {
            swap(as + ind, as + ind + (1 << logdist));
        }
    }


}
