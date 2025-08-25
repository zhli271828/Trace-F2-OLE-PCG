#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "test.h"
#include "modular_test.h"
#include "modular_bench.h"
#include "trace_bench.h"
#include "trace_f8_bench.h"
#include "dpf.h"
#include "prf.h"
#include "fft.h"
#include "utils.h"
#include "f4ops.h"

/**
 * For q=8, c=3.
 */
void trace_f8_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time) {
    if (c > 3) {
        printf("ERROR: currently only implemented for c <= 16");
        exit(-1);
    }
    clock_t start_time = clock();
    
    struct Param *param = calloc(1, sizeof(struct Param));
    init_f8_trace_bench_params(param, n, c, t);
    const size_t m = 3;
    const size_t base = (1<<m)-1;
    const size_t q = 1<<m; // field size
    uint8_t f8_zeta_powers[base];
    uint8_t f8_tr_tbl[q];
    memset(f8_zeta_powers, 0, sizeof(f8_zeta_powers));
    memset(f8_tr_tbl, 0, sizeof(f8_tr_tbl));
    compute_f8_zeta_powers(f8_zeta_powers, base);
    compute_f8_tr_tbl(f8_tr_tbl, m, f8_zeta_powers);

    struct FFT_F8_Trace_A *fft_f8_trace_a = xcalloc(1, sizeof(struct FFT_F8_Trace_A));
    init_fft_f8_trace_a(param, fft_f8_trace_a);
    sample_f8_trace_a_and_tensor(param, fft_f8_trace_a);
    struct F8_Trace_Prod *f8_trace_prod = xcalloc(1, sizeof(struct F8_Trace_Prod));
    init_f8_trace_prod(param, f8_trace_prod);
    printf("Benchmarking PCG evaluation\n");

    clock_t start_expand_time = clock();
    run_f8_trace_prod(param, f8_trace_prod, fft_f8_trace_a->fft_a_tensor, f8_tr_tbl, f8_zeta_powers);
    double end_expand_time = clock();
    double time_taken = ((double)(end_expand_time - start_expand_time)) / (CLOCKS_PER_SEC / 1000.0); // ms
    printf("Eval time (total) %f ms\n", time_taken);
    printf("DONE Benchmarking PCG evaluation\n\n");

    free_f8_trace_prod(param, f8_trace_prod);
    free_fft_f8_trace_a(param, fft_f8_trace_a);
    free(param);

    clock_t end_time = clock();
    pcg_time->pp_time = ((double)(start_expand_time-start_time))/(CLOCKS_PER_SEC/1000.0);
    pcg_time->expand_time = time_taken;
    pcg_time->total_time = ((double)(end_time-start_time))/(CLOCKS_PER_SEC / 1000.0);
}

static void compute_f8_zeta_powers(uint8_t *f8_zeta_powers, const size_t base) {

    uint8_t zeta = 1<<1;
    f8_zeta_powers[0] = 1;
    for (size_t i = 1; i<base; ++i) {
        f8_zeta_powers[i] = mult_f8_single(zeta, f8_zeta_powers[i-1]);
    }
}

static void compute_f8_tr_tbl(uint8_t *f8_tr_tbl, const size_t m, uint8_t *f8_zeta_powers) {
    f8_tr_tbl[0] = 0;
    const size_t base = (1<<m)-1;
    const size_t q = 1<<m; // field size
    for(size_t i = 1; i < q; ++i) {
        uint8_t v = f8_zeta_powers[i];
        f8_tr_tbl[v] = 0;
        for (size_t j = 0; j < m; ++j) {
            size_t id = (i * (1<<j))%base;
            f8_tr_tbl[v] ^= f8_zeta_powers[id];
        }
    }
}

void free_f8_trace_prod(const struct Param *param, struct F8_Trace_Prod *f8_trace_prod) {
    free_f8_trace_prod_dpf_keys(param, f8_trace_prod->keys);

    free(f8_trace_prod->polys);
    free(f8_trace_prod->shares);
    free(f8_trace_prod->cache);

    for (size_t i = 0; i < param->m; ++i) {
        free(f8_trace_prod->rlt[i]);
    }
    free(f8_trace_prod);
}

void free_fft_f8_trace_a(const struct Param *param, struct FFT_F8_Trace_A *fft_f8_trace_a) {
    free(fft_f8_trace_a->fft_a);
    free(fft_f8_trace_a->fft_a_tensor);
    free(fft_f8_trace_a);
}

void init_fft_f8_trace_a(const struct Param *param, struct FFT_F8_Trace_A *fft_f8_trace_a) {
    size_t poly_size = param->poly_size;
    size_t c = param->c;
    size_t m = param->m;
    uint16_t *fft_a = xcalloc(poly_size, sizeof(uint16_t));
    uint32_t *fft_a_tensor = xcalloc(m*poly_size, sizeof(uint32_t));
    fft_f8_trace_a->fft_a = fft_a;
    fft_f8_trace_a->fft_a_tensor = fft_a_tensor;
}

static uint8_t mult_f8_single(uint8_t a, uint8_t b) {
    // Mask to 3 bits since GF(8) uses 3-bit elements
    a &= 0b111;
    b &= 0b111;
    
    // Extract individual bits (a2 a1 a0) and (b2 b1 b0)
    uint8_t a0 = a & 0b001;
    uint8_t a1 = a & 0b010;
    uint8_t a2 = a & 0b100;
    uint8_t b0 = b & 0b001;
    uint8_t b1 = b & 0b010;
    uint8_t b2 = b & 0b100;
    
    // Calculate intermediate products (without reduction)
    uint8_t p0 = a0 & b0;                          // x⁰ term
    uint8_t p1 = (a0 & b1) | (a1 & b0);            // x¹ term
    uint8_t p2 = (a0 & b2) | (a1 & b1) | (a2 & b0); // x² term
    uint8_t p3 = (a1 & b2) | (a2 & b1);            // x³ term
    uint8_t p4 = a2 & b2;                          // x⁴ term
    
    // Apply reduction using x³ ≡ x + 1 and x⁴ ≡ x² + x
    // x³ term becomes x + 1, x⁴ term becomes x² + x
    uint8_t red1 = (p3 << 1) | p3;    // x³ reduction: x³ → x+1 (shift for x term)
    uint8_t red2 = (p4 << 2) | (p4 << 1); // x⁴ reduction: x⁴ → x²+x
    
    // Combine all terms with XOR (addition in GF(2))
    uint8_t res = p0 ^ p1 ^ p2 ^ red1 ^ red2;
    
    // Keep only the lower 3 bits as result
    return res & 0b111;
}

void sample_f8_trace_a_and_tensor(const struct Param *param, struct FFT_F8_Trace_A * fft_f8_trace_a) {
    uint16_t *fft_a = fft_f8_trace_a->fft_a;
    uint32_t *fft_a_tensor = fft_f8_trace_a->fft_a_tensor;
    const size_t poly_size = param->poly_size;
    const size_t c = param->c;
    const size_t m = param->m;

    RAND_bytes((uint8_t *)fft_a, sizeof(uint16_t) * poly_size);
    // make a_0 the identity polynomial (in FFT space) i.e., all 1s
    for (size_t i = 0; i < poly_size; i++) {
        fft_a[i] = fft_a[i] >> m;
        fft_a[i] = fft_a[i] << m;
        fft_a[i] |= 1;
    }
    // 0b111 can be the value base
    for (size_t l=0; l < m; ++l) {
        for (size_t i = 0; i < c; ++i) {
            for (size_t j = 0; j < c; ++j) {
                for (size_t ii = 0; ii < poly_size; ++ii) {
                    uint8_t ai = (fft_a[ii] >> (m * i)) & 0b111;
                    uint8_t aj = (fft_a[ii] >> (m * j)) & 0b111;
                    uint8_t ai_aj = mult_f8_single(ai, aj);
                    // The computation for aj_square and aj_quartic could be optimized.
                    uint8_t aj_square= mult_f8_single(aj, aj);
                    uint8_t aj_quartic = mult_f8_single(aj_square, aj_square);
                    uint32_t w = 0;
                    if (l == 0) {
                        w = ai_aj;
                    } else if (l == 1) {
                        w = mult_f8_single(ai, aj_square);
                    } else if (l == 2) {
                        w = mult_f8_single(ai, aj_quartic);
                    }
                    // m for the number of bits of F8
                    fft_a_tensor[l*poly_size+ii] |= w<<((i*c+j)*m);
                }
            }
        }
    }
    printf("Done with sampling the public values\n");
}

void init_f8_trace_prod(const struct Param *param, struct F8_Trace_Prod *f8_trace_prod) {

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
    struct KeysHD *keys = xcalloc(1, sizeof(struct KeysHD));
    sample_f8_trace_prod_dpf_keys(param, keys);
    f8_trace_prod->keys = keys;

    uint8_t **rlt = xcalloc(m, sizeof(void *));
    for (size_t i = 0; i < m; ++i) {
        rlt[i] = xcalloc(t*t*dpf_block_size, sizeof(uint8_t));
    }
    f8_trace_prod->rlt = rlt;

    uint32_t *polys = xcalloc(m*t*t*dpf_block_size, sizeof(uint32_t));
    uint128_t *shares = xcalloc(packed_dpf_block_size, sizeof(uint128_t));
    uint128_t *cache = xcalloc(packed_dpf_block_size, sizeof(uint128_t));
    
    f8_trace_prod->polys = polys;
    f8_trace_prod->shares = shares;
    f8_trace_prod->cache = cache;
}

// sample c*c*m*t*t DPF keys
void sample_f8_trace_prod_dpf_keys(const struct Param *param, struct KeysHD *keys) {
    size_t c = param->c;
    size_t t = param->t;
    size_t dpf_block_size = param->dpf_block_size;
    size_t dpf_domain_bits = param->dpf_domain_bits;
    const size_t m = param->m;
    const size_t base = param->base;

    size_t packed_dpf_block_size = param->packed_dpf_block_size;
    size_t packed_dpf_domain_bits = param->packed_dpf_domain_bits;

    struct DPFKeyZ **dpf_keys_A = xmalloc(c*c*m*t*t*sizeof(void *));
    struct DPFKeyZ **dpf_keys_B = xmalloc(c*c*m*t*t*sizeof(void *));
    struct PRFKeysZ *prf_keys = xmalloc(sizeof(struct PRFKeysZ));
    PRFKeyGenZ(prf_keys, base);
    for (size_t k=0; k < m; ++k) {
        for (size_t i = 0; i < c; ++i) {
            for (size_t j = 0; j < c; ++j) {
                for (size_t l = 0; l < t; ++l) {
                    for (size_t w = 0; w < t; ++w) {
                        size_t index = (((k*c+i)*c+j)*t+l)*t+w;
                        size_t alpha = random_index(packed_dpf_block_size);
                        uint128_t beta = 0;
                        RAND_bytes((unsigned char*)&beta, sizeof(uint128_t));
                        // 7 = base**floor(math.log(128/m, base)) with base=2^m-1
                        beta &= 0b111<<(m*random_index(7));
                        struct DPFKeyZ *kA = xmalloc(sizeof(struct DPFKeyZ));
                        struct DPFKeyZ *kB = xmalloc(sizeof(struct DPFKeyZ));
                        DPFGenZ(base, prf_keys, packed_dpf_domain_bits, alpha, &beta, 1, kA, kB);
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

static void free_f8_trace_prod_dpf_keys(const struct Param *param, struct KeysHD *keys) {

    const size_t c = param->c;
    const size_t t = param->t;
    const size_t m = param->m;
    const size_t base = param->base;
    for (size_t i = 0; i < c*c*m*t*t; ++i) {
        free(keys->dpf_keys_A[i]);
        free(keys->dpf_keys_B[i]);
    }
    free(keys->dpf_keys_A);
    free(keys->dpf_keys_B);
    DestroyPRFKeyZ(keys->prf_keys, base);
    free(keys);
}

void run_f8_trace_prod(const struct Param *param, struct F8_Trace_Prod *f8_trace_prod, uint32_t *fft_a_tensor, const uint8_t *f8_tr_tbl, const uint8_t *f8_zeta_powers) {

    struct KeysHD *keys = f8_trace_prod->keys;
    uint32_t *polys = f8_trace_prod->polys;
    uint128_t *shares = f8_trace_prod->shares;
    uint128_t *cache = f8_trace_prod->cache;
    uint8_t **rlt = f8_trace_prod->rlt;
    evaluate_f8_trace_prod_dpf(param, keys, polys, shares, cache);
    convert_f8_trace_prod_to_fft(param, f8_zeta_powers, polys);
    multiply_and_sum_f8_trace_prod(param, fft_a_tensor, polys, rlt, f8_zeta_powers);

    for (size_t i = 0; i < param->m; ++i) {
        compute_f8_trace(f8_trace_prod->rlt[i], param, f8_tr_tbl);
    }
}

void multiply_and_sum_f8_trace_prod(const struct Param *param, uint32_t *fft_a_tensor, uint32_t *polys, uint8_t **rlt, const uint8_t *f8_zeta_powers) {
    const size_t c = param->c;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;

    for (size_t l = 0; l < m; l++) {
        for (size_t i = 0; i < c; i++) {
            for (size_t j = 0; j < c; j++) {
                for (size_t w = 0; w < poly_size; w++) {
                    uint8_t a_lijk = fft_a_tensor[l*poly_size+w] >> (m*(i*c+j)) * 0b111;
                    uint8_t e_lijk = polys[l*poly_size+w] >> (m*(i*c+j)) * 0b111;

                    uint8_t prod = mult_f8_single(a_lijk, e_lijk);
                    for (size_t k = 0; k < m; ++k) {
                        rlt[k][w] ^= mult_f8_single(prod, f8_zeta_powers[k*(1+(1<<l))]);
                    }
                }
            }
        }
    }
}

void convert_f8_trace_prod_to_fft(const struct Param *param, const uint8_t *f8_zeta_powers, uint32_t *polys) {
    const size_t c = param->c;
    const size_t m = param->m;
    const size_t n = param->n;
    const size_t base = param-> base;
    const size_t poly_size = param->poly_size;
    // m
    for(size_t i = 0; i < m; ++i) {
        uint32_t *poly = &polys[i*poly_size];
        /**
         * TODO: change the FFT for this
         * FFT for uint32 packs 9 f8 values.
         */
        fft_recursive_f8_uint32(poly, f8_zeta_powers, n, poly_size/base, param, base);
    }
}

/**
 * Compute the trace function for the array of F8 elements.
 */
static void compute_f8_trace(uint8_t *rlt, const struct Param *param, const uint8_t *f8_tr_tbl) {

    const size_t poly_size = param->poly_size;
    for (size_t i = 0; i < poly_size; ++i) {
        rlt[i] = f8_tr_tbl[rlt[i]];
    }
}

/**
 * TODO: maybe change the parameters to struct F8_Trace_Prod to include most of the inputs
 */
void evaluate_f8_trace_prod_dpf(const struct Param *param, const struct KeysHD *keys, uint32_t *polys, uint128_t *shares, uint128_t *cache) {

    const size_t c = param->c;
    const size_t t = param->t;
    const size_t m = param->m;
    const size_t base = param->base;
    const size_t dpf_block_size = param->dpf_block_size;
    for (size_t k=0; k < m; ++k) {
        for (size_t i = 0; i < c; ++i) {
            for (size_t j = 0; j < c; ++j) {
                for (size_t l = 0; l < t; ++l) {
                    for (size_t w = 0; w < t; ++w) {
                        // key_index is correct
                        size_t key_index = (((k*c+i)*c+j)*t+l)*t+w;
                        struct DPFKeyZ *dpf_key = keys->dpf_keys_A[key_index];
                        size_t poly_index = (k*t+l)*t+w;
                        uint32_t *poly_block = &polys[poly_index*dpf_block_size];
                        DPFFullDomainEvalZ(base, dpf_key, cache, shares);
                        copy_f8_block(param, poly_block, dpf_block_size, shares, param->packed_dpf_block_size, i, j);
                    }
                }
            }
        }
    }
}

static void copy_f8_block(const struct Param *param, uint32_t *poly_block, const size_t dpf_block_size, uint128_t *shares, const size_t packed_dpf_block_size, size_t i, size_t j) {

    size_t m = param->m;
    size_t c = param->c;
    size_t unit_size = dpf_block_size/packed_dpf_block_size;
    if (unit_size * packed_dpf_block_size != dpf_block_size) {
        printf("unit_size * packed_dpf_block_size != dpf_block_size\n");
        exit(-1);
    }
    for (size_t l = 0; l < packed_dpf_block_size; ++l) {
        for (size_t w = 0; w < unit_size; ++w) {
            uint8_t rlt = (shares[l]>>(m*w)) & 0b111;
            poly_block[l*unit_size+w] |= rlt<<(m*(i*c+j));
        }
    }
}

void packed_scalar_mult_f8_trace(const struct Param *param, const uint32_t a, const uint8_t b, uint32_t* t) {
    size_t c = param->c;
    size_t m = param->m;
    *t = 0;
    for (size_t i = 0; i < c*c; ++i) {
        uint32_t w = mult_f8_single(a>>(i*m), b);
        (*t) |= (w<<(i*m));
    }
}