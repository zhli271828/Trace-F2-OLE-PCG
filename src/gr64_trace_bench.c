#include <stdint.h>
#include <stdlib.h>

#include "common.h"
#include "modular_bench.h"
#include "modular_test.h"
#include "dpf.h"
#include "fft.h"
#include "gr64_bench.h"
#include "gr64_trace_bench.h"

void gr64_trace_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time) {

    clock_t start_time = clock();

    struct Param *param = calloc(1, sizeof(struct Param));
    const size_t m = 2;
    init_gr64_bench_params(param, n, c, t, m);
    
    struct FFT_GR64_Trace_A *fft_gr64_trace_a = calloc(1, sizeof(struct FFT_GR64_Trace_A));
    init_FFT_GR64_Trace_A(param, fft_gr64_trace_a);
    sample_gr64_trace_a_and_tensor(param, fft_gr64_trace_a);

    size_t dpf_block_size = param->dpf_block_size;
    size_t poly_size = param->poly_size;
    if (t*t*dpf_block_size == poly_size) {
        printf("OK\n");
    } else {
        printf("Incorrect\n");
        exit(-1);
    }
    printf("Benchmarking PCG evaluation \n");
    struct GR64_Trace_Prod *prod = calloc(1, sizeof(struct GR64_Trace_Prod));
    init_gr64_trace_prod(param, prod);

    clock_t start_expand_time = clock();
    run_gr64_trace_prod(param, prod, fft_gr64_trace_a->fft_a_tensor_maps);

    double end_expand_time = clock();
    double time_taken = ((double)(end_expand_time - start_expand_time)) / (CLOCKS_PER_SEC / 1000.0); // ms
    printf("Eval time (total) %f ms\n", time_taken);
    printf("DONE Benchmarking PCG evaluation\n\n");

    free_gr64_trace_prod(param, prod);
    free_FFT_GR64_Trace_A(param, fft_gr64_trace_a);
    free(param);
    clock_t end_time = clock();
    pcg_time->pp_time = ((double)(start_expand_time-start_time))/(CLOCKS_PER_SEC/1000.0);
    pcg_time->expand_time = time_taken;
    pcg_time->total_time = ((double)(end_time-start_time))/(CLOCKS_PER_SEC / 1000.0);
}

void init_FFT_GR64_Trace_A(const struct Param *param, struct FFT_GR64_Trace_A *fft_gr64_trace_a) {
    size_t poly_size = param->poly_size;
    size_t c = param->c;
    size_t m = param->m;
    struct GR64 **fft_a = calloc(c, sizeof(void*));
    for (size_t i = 0; i < c; ++i) {
        fft_a[i] = calloc(poly_size, sizeof(struct GR64));
    }
    struct GR64 **fft_a_maps = calloc(c, sizeof(void*));
    for (size_t i = 0; i < c; ++i) {
        fft_a_maps[i] = calloc(poly_size, sizeof(struct GR64));
    }

    struct GR64 **fft_a_tensor_maps = calloc(c*c, sizeof(void*));
    for(size_t i = 0; i < c*c; ++i) {
        // m indicates the automorphisms
        fft_a_tensor_maps[i] = calloc(m*poly_size, sizeof(struct GR64));
    }

    fft_gr64_trace_a->fft_a = fft_a;
    fft_gr64_trace_a->fft_a_maps = fft_a_maps;
    fft_gr64_trace_a->fft_a_tensor_maps = fft_a_tensor_maps;
}

void init_FFT_GR64_d3_Trace_A(const struct Param *param, struct FFT_GR64_D3_Trace_A *fft_gr64_d3_trace_a) {

    size_t poly_size = param->poly_size;
    size_t c = param->c;
    size_t m = param->m;
    struct GR64_D3 **fft_a = calloc(m*c, sizeof(void*));
    for (size_t i = 0; i < m*c; ++i) {
        fft_a[i] = calloc(poly_size, sizeof(struct GR64_D3));
    }

    struct GR64_D3 **fft_a_tensor = calloc(m*c*c, sizeof(void*));
    for(size_t i = 0; i < m*c*c; ++i) {
        // m indicates the automorphisms
        fft_a_tensor[i] = calloc(m*c*c*poly_size, sizeof(struct GR64_D3));
    }
    fft_gr64_d3_trace_a->fft_a = fft_a;
    fft_gr64_d3_trace_a->fft_a_tensor = fft_a_tensor;
}

void free_FFT_GR64_Trace_A(const struct Param *param, struct FFT_GR64_Trace_A *fft_gr64_trace_a) {
    size_t c = param->c;

    for (size_t i = 0; i < c; ++i) {
        free(fft_gr64_trace_a->fft_a[i]);
    }
    // Free fft_a
    free(fft_gr64_trace_a->fft_a);

    for (size_t i = 0; i < c; ++i) {
        free(fft_gr64_trace_a->fft_a_maps[i]);
    }
    free(fft_gr64_trace_a->fft_a_maps);

    // Free the nested arrays in fft_a_tensor
    for (size_t i = 0; i < c * c; ++i) {
        free(fft_gr64_trace_a->fft_a_tensor_maps[i]);
    }
    free(fft_gr64_trace_a->fft_a_tensor_maps);
    free(fft_gr64_trace_a);
}

void free_FFT_GR64_d3_Trace_A(const struct Param *param, struct FFT_GR64_D3_Trace_A *fft_gr64_d3_trace_a) {
    size_t c = param->c;
    size_t m = param->m;

    for (size_t i = 0; i < m*c; ++i) {
        free(fft_gr64_d3_trace_a->fft_a[i]);
    }
    // Free fft_a
    free(fft_gr64_d3_trace_a->fft_a);

    // Free the nested arrays in fft_a_tensor
    for (size_t i = 0; i < c*c*m; ++i) {
        free(fft_gr64_d3_trace_a->fft_a_tensor[i]);
    }
    free(fft_gr64_d3_trace_a->fft_a_tensor);
    
    free(fft_gr64_d3_trace_a);
}

void sample_gr64_trace_a_and_tensor(const struct Param *param, struct FFT_GR64_Trace_A *fft_gr64_trace_a) {
    const size_t poly_size = param->poly_size;
    const size_t c = param->c;
    struct GR64 **fft_a = fft_gr64_trace_a->fft_a;
    struct GR64 **fft_a_maps = fft_gr64_trace_a->fft_a_maps;
    struct GR64 **fft_a_tensor_maps = fft_gr64_trace_a->fft_a_tensor_maps;

    // randomize a first
    for (size_t i = 1; i < c; ++i) {
        RAND_bytes((uint8_t *)fft_a[i], sizeof(struct GR64) * poly_size);
    }
    // make first a the identity polynomial (in FFT space) i.e., all 1s
    for (size_t i = 0; i < poly_size; ++i) {
        fft_a[0][i].c0 = 1;
        fft_a[0][i].c1 = 0;
    }

    // compute \sigma^1(a)
    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < poly_size; j++) {
            struct GR64 *cur_a = &fft_a[i][j];
            struct GR64 *cur_a_maps = &fft_a_maps[i][j];
            cur_a_maps->c0 = cur_a->c0-cur_a->c1;
            cur_a_maps->c1 = -cur_a->c1;
        }
    }
    // compute ai*\sigma^0(aj) and ai*\sigma^1(aj)
    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t k = 0; k < poly_size; ++k) {
                // the two polynomials in fft_a_tensor_maps is crossed
                mult_gr64(&fft_a[i][k], &fft_a[j][k], &fft_a_tensor_maps[i*c+j][k]);
                mult_gr64(&fft_a[i][k], &fft_a_maps[j][k], &fft_a_tensor_maps[i*c+j][k+poly_size]);
            }
        }
    }
}

void trace_gr64_FFT_polys(const struct Param *param, const struct GR64 *src, uint64_t *rlt) {

    const size_t poly_size = param->poly_size;
    for (size_t i = 0; i < poly_size; ++i) {
        rlt[i] = 2*src[i].c0-src[i].c1;
    }
}

void init_gr64_trace_prod(const struct Param *param, struct GR64_Trace_Prod *prod) {

    size_t c = param->c;
    size_t t = param->t;
    size_t m = param->m;
    size_t dpf_block_size = param->dpf_block_size;
    size_t poly_size = param->poly_size;
    struct Keys *keys = calloc(1, sizeof(struct Keys));
    sample_gr64_DPF_keys(param, keys);
    struct GR64 *polys = calloc(c*c*t*t*m*dpf_block_size, sizeof(struct GR64));
    struct GR64 *poly_buf = calloc(c*c*t*t*m*dpf_block_size, sizeof(struct GR64));
    struct GR64 *z_poly0 = calloc(t*t*dpf_block_size, sizeof(struct GR64));
    struct GR64 *z_poly1 = calloc(t*t*dpf_block_size, sizeof(struct GR64));
    uint64_t *rlt0 = calloc(t*t*dpf_block_size, sizeof(uint64_t));
    uint64_t *rlt1 = calloc(t*t*dpf_block_size, sizeof(uint64_t));

    uint128_t *shares = calloc(dpf_block_size, sizeof(uint128_t));
    uint128_t *cache = calloc(dpf_block_size, sizeof(uint128_t));

    prod->keys = keys;
    prod->polys = polys;
    prod->poly_buf = poly_buf;
    prod->z_poly0 = z_poly0;
    prod->z_poly1 = z_poly1;
    prod->rlt0 = rlt0;
    prod->rlt1 = rlt1;
    prod->shares = shares;
    prod->cache = cache;
}

void free_gr64_trace_prod(const struct Param *param, struct GR64_Trace_Prod *prod) {
    free_gr64_DPF_keys(param, prod->keys);
    free(prod->polys);
    free(prod->poly_buf);
    free(prod->z_poly0);
    free(prod->z_poly1);
    free(prod->rlt0);
    free(prod->rlt1);
    free(prod->shares);
    free(prod->cache);
    free(prod);
}

void run_gr64_trace_prod(const struct Param *param, struct GR64_Trace_Prod *prod, struct GR64 **fft_a_tensor_maps) {

    struct Keys *keys = prod->keys;
    struct GR64 *polys = prod->polys;
    struct GR64 *poly_buf = prod->poly_buf;
    struct GR64 *z_poly0 = prod->z_poly0;
    struct GR64 *z_poly1 = prod->z_poly1;
    uint64_t *rlt0 = prod->rlt0;
    uint64_t *rlt1 = prod->rlt1;
    uint128_t *shares = prod->shares;
    uint128_t *cache = prod->cache;

    evaluate_gr64_DPF(param, keys, polys, shares, cache);
    convert_gr64_to_FFT(param, polys);
    multiply_gr64_FFT(param, fft_a_tensor_maps, polys, poly_buf);
    sum_gr64_FFT_polys(param, poly_buf, z_poly0);
    sum_gr64_FFT_polys_special(param, poly_buf, z_poly1);
    trace_gr64_FFT_polys(param, z_poly0, rlt0);
    trace_gr64_FFT_polys(param, z_poly1, rlt1);
}