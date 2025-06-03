#include <stdint.h>
#include <stdlib.h>

#include "common.h"
#include "modular_bench.h"
#include "modular_test.h"
#include "dpf.h"
#include "fft.h"
#include "gr128_bench.h"
#include "gr128_trace_bench.h"

void gr128_trace_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time) {
    clock_t start_time = clock();

    struct Param *param = calloc(1, sizeof(struct Param));
    size_t m = 2;
    init_gr128_bench_params(param, n, c, t, m);
    
    struct FFT_GR128_Trace_A * fft_gr128_trace_a = calloc(1, sizeof(struct FFT_GR128_Trace_A));
    init_FFT_GR128_Trace_A(param, fft_gr128_trace_a);
    sample_gr128_trace_a_and_tensor(param, fft_gr128_trace_a);
    struct Keys *keys = calloc(1, sizeof(struct Keys));
    sample_gr128_DPF_keys(param, keys);
    size_t dpf_block_size = param->dpf_block_size;
    size_t poly_size = param->poly_size;
    if (t*t*dpf_block_size == poly_size) {
        printf("OK\n");
    } else {
        printf("Incorrect\n");
        exit(-1);
    }
    printf("Benchmarking PCG evaluation \n");

    struct GR128 *polys = calloc(c*c*t*t*m*dpf_block_size, sizeof(struct GR128));
    struct GR128 *poly_buf = calloc(c*c*t*t*m*dpf_block_size, sizeof(struct GR128));
    struct GR128 *z_poly0 = calloc(t*t*dpf_block_size, sizeof(struct GR128));
    struct GR128 *z_poly1 = calloc(t*t*dpf_block_size, sizeof(struct GR128));
    uint128_t *rlt0 = calloc(t*t*dpf_block_size, sizeof(uint128_t));
    uint128_t *rlt1 = calloc(t*t*dpf_block_size, sizeof(uint128_t));

    uint128_t *shares = calloc(dpf_block_size*2, sizeof(uint128_t));
    uint128_t *cache = calloc(dpf_block_size*2, sizeof(uint128_t));

    clock_t start_expand_time = clock();
    evaluate_gr128_DPF(param, keys, polys, shares, cache);
    convert_gr128_to_FFT(param, polys);
    multiply_gr128_FFT(param, fft_gr128_trace_a->fft_a_tensor_maps, polys, poly_buf);
    sum_gr128_FFT_polys(param, poly_buf, z_poly0);
    sum_gr128_FFT_polys_special(param, poly_buf, z_poly1);
    trace_gr128_FFT_polys(param, z_poly0, rlt0);
    trace_gr128_FFT_polys(param, z_poly1, rlt1);

    double end_expand_time = clock();
    double time_taken = ((double)(end_expand_time - start_expand_time)) / (CLOCKS_PER_SEC / 1000.0); // ms
    printf("Eval time (total) %f ms\n", time_taken);
    printf("DONE Benchmarking PCG evaluation\n\n");

    free_FFT_GR128_Trace_A(param, fft_gr128_trace_a);
    free_gr128_DPF_keys(param, keys);
    free(param);
    free(polys);
    free(poly_buf);
    free(z_poly0);
    free(z_poly1);
    free(rlt0);
    free(rlt1);
    free(shares);
    free(cache);

    clock_t end_time = clock();
    pcg_time->pp_time = ((double)(start_expand_time-start_time))/(CLOCKS_PER_SEC/1000.0);
    pcg_time->expand_time = time_taken;
    pcg_time->total_time = ((double)(end_time-start_time))/(CLOCKS_PER_SEC / 1000.0);
}

void init_FFT_GR128_Trace_A(const struct Param *param, struct FFT_GR128_Trace_A *fft_gr128_trace_a) {
    size_t poly_size = param->poly_size;
    size_t c = param->c;
    size_t m = param->m;
    struct GR128 **fft_a = xcalloc(c, sizeof(void*));
    for (size_t i = 0; i < c; ++i) {
        fft_a[i] = calloc(poly_size, sizeof(struct GR128));
    }
    struct GR128 **fft_a_maps = xcalloc(c, sizeof(void*));
    for (size_t i = 0; i < c; ++i) {
        fft_a_maps[i] = xcalloc(poly_size, sizeof(struct GR128));
    }

    struct GR128 **fft_a_tensor_maps = xcalloc(c*c, sizeof(void*));
    for(size_t i = 0; i < c*c; ++i) {
        fft_a_tensor_maps[i] = xcalloc(m*poly_size, sizeof(struct GR128));
    }

    fft_gr128_trace_a->fft_a = fft_a;
    fft_gr128_trace_a->fft_a_maps = fft_a_maps;
    fft_gr128_trace_a->fft_a_tensor_maps = fft_a_tensor_maps;
}

void free_FFT_GR128_Trace_A(const struct Param *param, struct FFT_GR128_Trace_A *fft_gr128_trace_a) {
    size_t c = param->c;

    for (size_t i = 0; i < c; ++i) {
        free(fft_gr128_trace_a->fft_a[i]);
    }
    // Free fft_a
    free(fft_gr128_trace_a->fft_a);

    for (size_t i = 0; i < c; ++i) {
        free(fft_gr128_trace_a->fft_a_maps[i]);
    }
    free(fft_gr128_trace_a->fft_a_maps);

    // Free the nested arrays in fft_a_tensor
    for (size_t i = 0; i < c * c; ++i) {
        free(fft_gr128_trace_a->fft_a_tensor_maps[i]);
    }
    free(fft_gr128_trace_a->fft_a_tensor_maps);
    free(fft_gr128_trace_a);
}

void sample_gr128_trace_a_and_tensor(const struct Param *param, struct FFT_GR128_Trace_A *fft_gr128_trace_a) {
    const size_t poly_size = param->poly_size;
    const size_t c = param->c;
    struct GR128 **fft_a = fft_gr128_trace_a->fft_a;
    struct GR128 **fft_a_maps = fft_gr128_trace_a->fft_a_maps;
    struct GR128 **fft_a_tensor_maps = fft_gr128_trace_a->fft_a_tensor_maps;
    for (size_t i = 1; i < c; ++i) {
        RAND_bytes((uint8_t *)fft_a[i], sizeof(struct GR128) * poly_size);
    }
    // make first a the identity polynomial (in FFT space) i.e., all 1s
    for (size_t i = 0; i < poly_size; ++i) {
        fft_a[0][i].c0 = 1;
        fft_a[0][i].c1 = 0;
    }

    // optimize this copy both and the later is better
    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < poly_size; j++) {
            struct GR128 *cur_a = &fft_a[i][j];
            struct GR128 *cur_a_maps = &fft_a_maps[i][j];
            cur_a_maps->c0 = cur_a->c0-cur_a->c1;
            cur_a_maps->c1 = -cur_a->c1;
        }
    }

    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t k = 0; k < poly_size; ++k) {
                mult_gr128(&fft_a[i][k], &fft_a[j][k], &fft_a_tensor_maps[i*c+j][2*k]);
                mult_gr128(&fft_a[i][k], &fft_a_maps[j][k], &fft_a_tensor_maps[i*c+j][2*k+1]);
            }
        }
    }
}

void trace_gr128_FFT_polys(const struct Param *param, const struct GR128 *src, uint128_t *rlt) {

    const size_t poly_size = param->poly_size;
    for (size_t i = 0; i < poly_size; ++i) {
        rlt[i] = 2*src[i].c0-src[i].c1;
    }
}

void init_FFT_GR128_d3_Trace_A(const struct Param *param, struct FFT_GR128_D3_Trace_A *fft_gr128_d3_trace_a) {

    size_t poly_size = param->poly_size;
    size_t c = param->c;
    size_t m = param->m;
    struct GR128_D3 **fft_a = xcalloc(m*c, sizeof(void*));
    for (size_t i = 0; i < m*c; ++i) {
        fft_a[i] = xcalloc(poly_size, sizeof(struct GR128_D3));
    }

    struct GR128_D3 **fft_a_tensor = calloc(m*c*c, sizeof(void*));
    for(size_t i = 0; i < m*c*c; ++i) {
        // m indicates the automorphisms
        fft_a_tensor[i] = xcalloc(m*c*c*poly_size, sizeof(struct GR128_D3));
    }
    fft_gr128_d3_trace_a->fft_a = fft_a;
    fft_gr128_d3_trace_a->fft_a_tensor = fft_a_tensor;
}


void init_FFT_GR128_d4_Trace_A(const struct Param *param, struct FFT_GR128_D4_Trace_A *fft_gr128_d4_trace_a) {

    size_t poly_size = param->poly_size;
    size_t c = param->c;
    size_t m = param->m;
    struct GR128_D4 **fft_a = xcalloc(m*c, sizeof(void*));
    for (size_t i = 0; i < m*c; ++i) {
        fft_a[i] = xcalloc(poly_size, sizeof(struct GR128_D4));
    }

    struct GR128_D4 **fft_a_tensor = calloc(m*c*c, sizeof(void*));
    for(size_t i = 0; i < m*c*c; ++i) {
        // m indicates the automorphisms
        fft_a_tensor[i] = xcalloc(m*c*c*poly_size, sizeof(struct GR128_D4));
    }
    fft_gr128_d4_trace_a->fft_a = fft_a;
    fft_gr128_d4_trace_a->fft_a_tensor = fft_a_tensor;
}

void free_FFT_GR128_d3_Trace_A(const struct Param *param, struct FFT_GR128_D3_Trace_A *fft_gr128_d3_trace_a) {
    size_t c = param->c;
    size_t m = param->m;

    for (size_t i = 0; i < m*c; ++i) {
        free(fft_gr128_d3_trace_a->fft_a[i]);
    }
    // Free fft_a
    free(fft_gr128_d3_trace_a->fft_a);

    // Free the nested arrays in fft_a_tensor
    for (size_t i = 0; i < c*c*m; ++i) {
        free(fft_gr128_d3_trace_a->fft_a_tensor[i]);
    }
    free(fft_gr128_d3_trace_a->fft_a_tensor);
    
    free(fft_gr128_d3_trace_a);
}

void free_FFT_GR128_d4_Trace_A(const struct Param *param, struct FFT_GR128_D4_Trace_A *fft_gr128_d4_trace_a) {
    size_t c = param->c;
    size_t m = param->m;

    for (size_t i = 0; i < m*c; ++i) {
        free(fft_gr128_d4_trace_a->fft_a[i]);
    }
    // Free fft_a
    free(fft_gr128_d4_trace_a->fft_a);

    // Free the nested arrays in fft_a_tensor
    for (size_t i = 0; i < c*c*m; ++i) {
        free(fft_gr128_d4_trace_a->fft_a_tensor[i]);
    }
    free(fft_gr128_d4_trace_a->fft_a_tensor);
    
    free(fft_gr128_d4_trace_a);
}
