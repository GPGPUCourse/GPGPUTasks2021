__kernel void merge( __global float *a, __global float *b, unsigned int logn, unsigned int n) {



    int ind = get_global_id(0);

    if (ind >= n) {
        return;
    }

    int block_ind = (ind >> logn);
    int global_block_ind = (block_ind >> 1);
    int ind_in_block = (ind & ((1 << (logn + 1)) - 1));
    int lower_ind = (global_block_ind << (logn + 1));
    int upper_ind = lower_ind + (1 << (logn + 1));
    int mid_ind = lower_ind + (1 << logn);


    // binary search
    if (ind_in_block < (1 << logn)) {

        //printf("%d here", ind);
        // left side case
        int len = ind_in_block + 1;
        int l_ind = ind;
        int r_ind = mid_ind;

        if (a[l_ind] < a[r_ind]) {
            b[ind] = a[l_ind];
        } else if (a[l_ind - len + 1] > a[r_ind + len - 1]) {
            b[ind] = a[r_ind + len - 1];
        } else {
            int l = 0;
            int r = len - 1;
            while (l + 1 < r) {
                int m = (l + r) / 2;
                if (a[l_ind - m] < a[r_ind + m]) {
                    r = m;
                } else {
                    l = m;
                }
            }

            if (a[l_ind - r] < a[r_ind + l]) {
                b[ind] = a[r_ind + l];
            } else {
                b[ind] = a[l_ind - r];
            }
            //printf("at ind %d found len %d, side case; b[ind] = %f", ind, l, b[ind]);
        }
    } else {
        // bottom side case
        int len = (1 << (logn + 1)) - ind_in_block - 1;
        int l_ind = mid_ind - 1;
        int r_ind = ind + 1;

        if (len == 0) {
            if (a[l_ind] < a[r_ind - 1]) {
                b[ind] = a[r_ind - 1];
            } else {
                b[ind] = a[l_ind];
            }
        } else if(a[l_ind] < a[r_ind]){
            if (a[l_ind] < a[r_ind - 1]) {
                b[ind] = a[r_ind - 1];
            } else {
                b[ind] = a[l_ind];
            }
        } else if(a[l_ind - len + 1] > a[r_ind + len - 1]) {
            //b[ind] = a[l_ind - len + 1];
            if (a[l_ind - len] > a[r_ind + len - 1]) {
                b[ind] = a[l_ind - len];
            } else {
                b[ind] = a[r_ind + len - 1];
            }


        } else {
            int l = 0;
            int r = len - 1;
            while (l + 1 < r) {
                int m = (l + r) / 2;
                if (a[l_ind - m] < a[r_ind + m]) {
                    r = m;
                } else {
                    l = m;
                }
            }
            //printf("at ind %d found len %d, bottom case", ind, l);

            if (a[l_ind - r] < a[r_ind + l]) {
                b[ind] = a[r_ind + l];
            } else {
                b[ind] = a[l_ind - r];
            }
        }

    }

}