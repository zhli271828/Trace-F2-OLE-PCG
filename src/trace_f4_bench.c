#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "test.h"
#include "modular_test.h"
#include "modular_bench.h"
#include "trace_bench.h"
#include "trace_f4_bench.h"
#include "dpf.h"
#include "prf.h"
#include "fft.h"
#include "utils.h"
#include "f4ops.h"

/**
 * For q=4, existing parameter only supports c<=9.
 * The program supports c<=16
 */
void trace_f4_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time) {
    if (c > 16) {
        printf("ERROR: currently only implemented for c <= 16");
        exit(-1);
    }
    clock_t start_time = clock();
    struct Param *param = calloc(1, sizeof(struct Param));
    init_f4_trace_bench_params(param, n, c, t);

    // struct FFT_Trace_A
    struct FFT_F4_Trace_A *fft_f4_trace_a = xcalloc(1, sizeof(struct FFT_F4_Trace_A));
    init_fft_f4_trace_a(param, fft_f4_trace_a);
    sample_f4_trace_a_and_tensor(param, fft_f4_trace_a);
    struct F4_Trace_Prod *f4_trace_prod = xcalloc(1, sizeof(struct F4_Trace_Prod));
    init_f4_trace_prod(param, f4_trace_prod);

    printf("Benchmarking PCG evaluation\n");
    clock_t start_expand_time = clock();
    run_f4_trace_prod(param, f4_trace_prod, fft_f4_trace_a->fft_a_tensor);
    double end_expand_time = clock();
    double time_taken = ((double)(end_expand_time - start_expand_time)) / (CLOCKS_PER_SEC / 1000.0); // ms
    printf("Eval time (total) %f ms\n", time_taken);
    printf("DONE Benchmarking PCG evaluation\n\n");

    free_f4_trace_prod(param, f4_trace_prod);
    free_fft_f4_trace_a(param, fft_f4_trace_a);
    free(param);

    clock_t end_time = clock();
    pcg_time->pp_time = ((double)(start_expand_time-start_time))/(CLOCKS_PER_SEC/1000.0);
    pcg_time->expand_time = time_taken;
    pcg_time->total_time = ((double)(end_time-start_time))/(CLOCKS_PER_SEC / 1000.0);
}


void free_f4_trace_prod(const struct Param *param, struct F4_Trace_Prod *f4_trace_prod) {
    free_f4_trace_prod_dpf_keys(param, f4_trace_prod->keys);

    free(f4_trace_prod->polys);
    free(f4_trace_prod->shares);
    free(f4_trace_prod->cache);

    for (size_t i = 0; i < param->m; ++i) {
        free(f4_trace_prod->rlt[i]);
    }
    free(f4_trace_prod);
}

void free_fft_f4_trace_a(const struct Param *param, struct FFT_F4_Trace_A *fft_f4_trace_a) {
    free(fft_f4_trace_a->fft_a);
    free(fft_f4_trace_a->fft_a_tensor);
    free(fft_f4_trace_a);
}

void init_fft_f4_trace_a(const struct Param *param, struct FFT_F4_Trace_A *fft_f4_trace_a) {
    size_t poly_size = param->poly_size;
    size_t c = param->c;
    size_t m = param->m;
    uint32_t *fft_a = xcalloc(poly_size, sizeof(uint32_t));
    uint32_t *fft_a_tensor = xcalloc(m*c*poly_size, sizeof(uint32_t));
    fft_f4_trace_a->fft_a = fft_a;
    fft_f4_trace_a->fft_a_tensor = fft_a_tensor;
}

/**
 * Multiplies two elements of F4 only without packing support.
 */
static uint8_t mult_f4_single(uint8_t a, uint8_t b) {

    uint8_t tmp = ((a & 0b10) & (b & 0b10));
    uint8_t res = tmp ^ ((a & 0b10) & ((b & 0b01) << 1) ^ (((a & 0b01) << 1) & (b & 0b10)));
    res |= ((a & 0b01) & (b & 0b01)) ^ (tmp >> 1);
    return res;
}

void sample_f4_trace_a_and_tensor(const struct Param *param, struct FFT_F4_Trace_A * fft_f4_trace_a) {
    uint32_t *fft_a = fft_f4_trace_a->fft_a;
    uint32_t *fft_a_tensor = fft_f4_trace_a->fft_a_tensor;
    const size_t poly_size = param->poly_size;
    const size_t c = param->c;
    const size_t m = param->m;

    RAND_bytes((uint8_t *)fft_a, sizeof(uint32_t) * poly_size);
    // make a_0 the identity polynomial (in FFT space) i.e., all 1s
    for (size_t i = 0; i < poly_size; i++) {
        fft_a[i] = fft_a[i] >> 2;
        fft_a[i] = fft_a[i] << 2;
        fft_a[i] |= 1;
    }
    
    for (size_t l=0; l < m; ++l) {
        for (size_t i = 0; i < c; ++i) {
            for (size_t j = 0; j < c; ++j) {
                for (size_t ii = 0; ii < poly_size; ++ii) {
                    uint8_t ai = (fft_a[ii] >> (2 * i)) & 0b11;
                    uint8_t aj = (fft_a[ii] >> (2 * j)) & 0b11;
                    uint32_t w = mult_f4_single(ai, aj);
                    if (l==1) {
                        // This only works for m=2.
                        w = mult_f4_single(w, aj);
                    }
                    fft_a_tensor[(l*c+i)*poly_size+ii] |= w<<(2*j);
                }
            }
        }
    }
    printf("Done with sampling the public values\n");
}

void init_f4_trace_prod(const struct Param *param, struct F4_Trace_Prod *f4_trace_prod) {

    size_t c = param->c;
    size_t t = param->t;
    size_t m = param->m;
    size_t dpf_block_size = param->dpf_block_size;
    size_t poly_size = param->poly_size;
    size_t packed_dpf_block_size = param->packed_dpf_block_size;
    if (t*t*dpf_block_size != poly_size) {
        printf("t*t*dpf_block_size != poly_size\n");
        exit(-1);
    }
    struct Keys *keys = xcalloc(1, sizeof(struct Keys));
    sample_f4_trace_prod_dpf_keys(param, keys);
    f4_trace_prod->keys = keys;

    uint8_t **rlt = xcalloc(m, sizeof(void *));
    for (size_t i = 0; i < m; ++i) {
        rlt[i] = xcalloc(t*t*dpf_block_size, sizeof(uint8_t));
    }
    f4_trace_prod->rlt = rlt;

    uint32_t *polys = xcalloc(m*c*t*t*dpf_block_size, sizeof(uint32_t));
    uint128_t *shares = xcalloc(packed_dpf_block_size, sizeof(uint128_t));
    uint128_t *cache = xcalloc(packed_dpf_block_size, sizeof(uint128_t));
    
    f4_trace_prod->polys = polys;
    f4_trace_prod->shares = shares;
    f4_trace_prod->cache = cache;
}

// sample c*c*m*t*t DPF keys
void sample_f4_trace_prod_dpf_keys(const struct Param *param, struct Keys *keys) {
    size_t c = param->c;
    size_t t = param->t;
    size_t dpf_block_size = param->dpf_block_size;
    size_t dpf_domain_bits = param->dpf_domain_bits;
    const size_t m = param->m;

    size_t packed_dpf_block_size = param->packed_dpf_block_size;
    size_t packed_dpf_domain_bits = param->packed_dpf_domain_bits;

    struct DPFKey **dpf_keys_A = xmalloc(c*c*m*t*t*sizeof(void *));
    struct DPFKey **dpf_keys_B = xmalloc(c*c*m*t*t*sizeof(void *));
    struct PRFKeys *prf_keys = xmalloc(sizeof(struct PRFKeys));
    PRFKeyGen(prf_keys);
    for (size_t k=0; k < m; ++k) {
        for (size_t i = 0; i < c; ++i) {
            for (size_t j = 0; j < c; ++j) {
                for (size_t l = 0; l < t; ++l) {
                    for (size_t w = 0; w < t; ++w) {
                        size_t index = (((k*c+i)*c+j)*t+l)*t+w;
                        size_t alpha = random_index(packed_dpf_block_size);
                        uint128_t beta = 0;
                        RAND_bytes((unsigned char*)&beta, sizeof(uint128_t));
                        // 27 = base**floor(math.log(128/m, base)) with base=2^m-1
                        beta &= 0b11<<(2*random_index(27));
                        struct DPFKey *kA = xmalloc(sizeof(struct DPFKey));
                        struct DPFKey *kB = xmalloc(sizeof(struct DPFKey));
                        DPFGen(prf_keys, packed_dpf_domain_bits, alpha, &beta, 1, kA, kB);
                        dpf_keys_A[index] = kA;
                        dpf_keys_B[index] = kB;
                    }
                }
            }
        }
    }
    keys->dpf_keys_A = dpf_keys_A;
    keys->dpf_keys_B = dpf_keys_B;
    keys->prf_keys = prf_keys;
}

static void free_f4_trace_prod_dpf_keys(const struct Param *param, struct Keys *keys) {

    const size_t c = param->c;
    const size_t t = param->t;
    const size_t m = param->m;
    for (size_t i = 0; i < c*c*m*t*t; ++i) {
        free(keys->dpf_keys_A[i]);
        free(keys->dpf_keys_B[i]);
    }
    free(keys->dpf_keys_A);
    free(keys->dpf_keys_B);
    DestroyPRFKey(keys->prf_keys);
    free(keys);
}

/**
 * TODO: maybe the parameter zeta_powers is necessary.
 */
void run_f4_trace_prod(const struct Param *param, struct F4_Trace_Prod *f4_trace_prod, uint32_t *fft_a_tensor) {

    struct Keys *keys = f4_trace_prod->keys;
    uint32_t *polys = f4_trace_prod->polys;
    uint128_t *shares = f4_trace_prod->shares;
    uint128_t *cache = f4_trace_prod->cache;
    uint8_t **rlt = f4_trace_prod->rlt;
    evaluate_f4_trace_prod_dpf(param, keys, polys, shares, cache);
    convert_f4_trace_prod_to_fft(param, polys);
    multiply_and_sum_f4_trace_prod(param, fft_a_tensor, polys, rlt);

    compute_f4_trace(f4_trace_prod->rlt[0], param);
    compute_f4_trace(f4_trace_prod->rlt[1], param);
}

void multiply_and_sum_f4_trace_prod(const struct Param *param, uint32_t *fft_a_tensor, uint32_t *polys, uint8_t **rlt) {
    const size_t c = param->c;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;

    for (size_t l = 0; l < m; l++) {
        for (size_t i = 0; i < c; i++) {
            for (size_t j = 0; j < c; j++) {
                for (size_t k = 0; k < poly_size; k++) {
                    uint8_t a_lijk = fft_a_tensor[(l*c+i)*poly_size+k]>>(2*j) & 0b11;
                    uint8_t e_lijk = polys[(l*c+i)*poly_size+k]>>(2*j) & 0b11;

                    uint8_t prod = mult_f4_single(a_lijk, e_lijk);
                    rlt[0][k] ^= prod;

                    if (l == 0) { // zeta^{1+2^l}=zeta^2=zeta+1
                        prod = mult_f4_single(prod, 0b11);
                    }
                    if (l == 1) { // zeta^{1+2^l}=zeta^3=1
                    } else {
                    }
                    rlt[1][k] ^= prod;
                }
            }
        }
    }
}

void convert_f4_trace_prod_to_fft(const struct Param *param, uint32_t *polys) {
    const size_t c = param->c;
    const size_t m = param->m;
    const size_t n = param->n;
    const size_t poly_size = param->poly_size;
    // m*c
    for(size_t i = 0; i < c*m; ++i) {
        uint32_t *poly = &polys[i*poly_size];
        /**
         * FFT for uint32 packs 16 f4 values.
         */
        fft_recursive_uint32(poly, n, poly_size / 3);
    }
}

/**
 * Compute the trace function for the array of F4 elements.
 */
static void compute_f4_trace(uint8_t *rlt, const struct Param *param) {
    const size_t poly_size = param->poly_size;
    for (size_t i = 0; i < poly_size; ++i) {
        // In F4, the trace function a+a^2
        size_t a = rlt[i];
        rlt[i] = a^mult_f4(a, a);
    }
}

/**
 * TODO: maybe change the parameters to struct F4_Trace_Prod to include most of the inputs
 */
void evaluate_f4_trace_prod_dpf(const struct Param *param, const struct Keys *keys, uint32_t *polys, uint128_t *shares, uint128_t *cache) {

    const size_t c = param->c;
    const size_t t = param->t;
    const size_t m = param->m;
    const size_t dpf_block_size = param->dpf_block_size;
    for (size_t k=0; k < m; ++k) {
        for (size_t i = 0; i < c; ++i) {
            for (size_t j = 0; j < c; ++j) {
                for (size_t l = 0; l < t; ++l) {
                    for (size_t w = 0; w < t; ++w) {
                        // key_index is correct
                        size_t key_index = (((k*c+i)*c+j)*t+l)*t+w;
                        struct DPFKey *dpf_key = keys->dpf_keys_A[key_index];
                        size_t poly_index = ((k*c+i)*t+l)*t+w;
                        uint32_t *poly_block = &polys[poly_index*dpf_block_size];
                        DPFFullDomainEval(dpf_key, cache, shares);
                        copy_f4_block(poly_block, dpf_block_size, shares, param->packed_dpf_block_size, j);
                    }
                }
            }
        }
    }
}

static void copy_f4_block(uint32_t *poly_block, const size_t dpf_block_size, uint128_t *shares, const size_t packed_dpf_block_size, size_t loc) {

    size_t unit_size = dpf_block_size/packed_dpf_block_size;
    if (unit_size * packed_dpf_block_size != dpf_block_size) {
        printf("unit_size * packed_dpf_block_size != dpf_block_size\n");
        exit(-1);
    }
    for (size_t l = 0; l < packed_dpf_block_size; ++l) {
        for (size_t w = 0; w < unit_size; ++w) {
            uint8_t rlt = (shares[l]>>(2*w)) & 0b11;
            poly_block[l*unit_size+w] |= rlt<<(2*loc);
        }
    }
}