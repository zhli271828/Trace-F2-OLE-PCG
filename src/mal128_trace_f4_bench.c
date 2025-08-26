#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "test.h"
#include "modular_test.h"
#include "modular_bench.h"
#include "trace_bench.h"
#include "mal128_trace_f4_bench.h"
#include "dpf.h"
#include "prf.h"
#include "fft.h"
#include "utils.h"
#include "f4ops.h"

// GF(4) multiplication table
static const uint8_t f4_mul_tbl[4][4] = {
    {0,0,0,0},
    {0,1,2,3},
    {0,2,3,1},
    {0,3,1,2}
};
// Add in GF(4) is just XOR
static inline uint8_t f4_add(uint8_t a, uint8_t b) {
    return a ^ b;
}
// Multiply in GF(4)
static inline uint8_t f4_mul(uint8_t a, uint8_t b) {
    return f4_mul_tbl[a][b];
}

/**
 * For q=4, existing parameter only supports c<=9.
 * The program supports c<=16
 */
void mal128_trace_f4_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time) {
    if (c > 16) {
        printf("ERROR: currently only implemented for c <= 16");
        exit(-1);
    }
    clock_t start_time = clock();
    struct Param *param = calloc(1, sizeof(struct Param));
    /**
     * TODO: double check the parameter function, zeta_powers and trace function.
     */
    init_f4_trace_bench_params(param, n, c, t);
    const size_t m = 2;
    const size_t base = (1<<m)-1;
    const size_t q = 1<<m;
    uint8_t f4_zeta_powers[base];
    uint8_t f4_tr_tbl[q];
    memset(f4_zeta_powers, 0, sizeof(f4_zeta_powers));
    memset(f4_tr_tbl, 0, sizeof(f4_tr_tbl));
    compute_f4_zeta_powers(f4_zeta_powers, base);
    compute_f4_tr_tbl(f4_tr_tbl, m, f4_zeta_powers);

    struct FFT_Mal128_F4_Trace_A *fft_mal128_f4_trace_a = xcalloc(1, sizeof(struct FFT_Mal128_F4_Trace_A));
    init_fft_mal128_f4_trace_a(param, fft_mal128_f4_trace_a);
    sample_mal128_f4_trace_a_and_tensor(param, fft_mal128_f4_trace_a);
    
    struct Mal128_F4_Trace_Prod *mal128_f4_trace_prod = xcalloc(1, sizeof(struct Mal128_F4_Trace_Prod));
    init_mal128_f4_trace_prod(param, mal128_f4_trace_prod);
    printf("Benchmarking PCG evaluation\n");

    clock_t start_expand_time = clock();
    run_mal128_f4_trace_prod(param, mal128_f4_trace_prod, fft_mal128_f4_trace_a->fft_a_tensor, f4_tr_tbl, f4_zeta_powers);
    double end_expand_time = clock();
    double time_taken = ((double)(end_expand_time - start_expand_time)) / (CLOCKS_PER_SEC / 1000.0); // ms
    printf("Eval time (total) %f ms\n", time_taken);
    printf("DONE Benchmarking PCG evaluation\n\n");

    free_mal128_f4_trace_prod(param, mal128_f4_trace_prod);
    free_fft_mal128_f4_trace_a(param, fft_mal128_f4_trace_a);
    free(param);

    clock_t end_time = clock();
    pcg_time->pp_time = ((double)(start_expand_time-start_time))/(CLOCKS_PER_SEC/1000.0);
    pcg_time->expand_time = time_taken;
    pcg_time->total_time = ((double)(end_time-start_time))/(CLOCKS_PER_SEC / 1000.0);
}

static void compute_f4_zeta_powers(uint8_t *f4_zeta_powers, const size_t base) {

    uint8_t zeta = 1<<1;
    f4_zeta_powers[0] = 1;
    for (size_t i = 1; i<base; ++i) {
        f4_zeta_powers[i] = f4_mul(zeta, f4_zeta_powers[i-1]);
    }
}

static void compute_f4_tr_tbl(uint8_t *f4_tr_tbl, const size_t m, uint8_t *f4_zeta_powers) {
    f4_tr_tbl[0] = 0;
    const size_t base = (1<<m)-1;
    const size_t q = 1<<m; // field size
    for(size_t i = 1; i < q; ++i) {
        uint8_t v = f4_zeta_powers[i];
        f4_tr_tbl[v] = 0;
        for (size_t j = 0; j < m; ++j) {
            size_t id = (i * (1<<j))%base;
            f4_tr_tbl[v] ^= f4_zeta_powers[id];
        }
    }
}

void free_mal128_f4_trace_prod(const struct Param *param, struct Mal128_F4_Trace_Prod *mal128_f4_trace_prod) {

    free_mal128_f4_trace_prod_dpf_keys(param, mal128_f4_trace_prod->keys);
    free(mal128_f4_trace_prod->polys);
    free(mal128_f4_trace_prod->shares);
    free(mal128_f4_trace_prod->cache);

    for (size_t i = 0; i < param->m; ++i) {
        free(mal128_f4_trace_prod->rlt[i]);
    }
    free(mal128_f4_trace_prod);
}

void free_fft_mal128_f4_trace_a(const struct Param *param, struct FFT_Mal128_F4_Trace_A *fft_mal128_f4_trace_a) {

    free(fft_mal128_f4_trace_a->fft_a);
    free(fft_mal128_f4_trace_a->fft_a_square);
    free(fft_mal128_f4_trace_a->fft_a_tensor);
    free(fft_mal128_f4_trace_a);
}

void init_fft_mal128_f4_trace_a(const struct Param *param, struct FFT_Mal128_F4_Trace_A *fft_mal128_f4_trace_a) {
    size_t poly_size = param->poly_size;
    size_t c = param->c;
    size_t m = param->m;
    uint32_t *fft_a = xcalloc(poly_size, sizeof(uint32_t));
    uint32_t *fft_a_square = xcalloc(poly_size, sizeof(uint32_t));
    uint32_t *fft_a_tensor = xcalloc(m*m*c*poly_size, sizeof(uint32_t));
    fft_mal128_f4_trace_a->fft_a = fft_a;
    fft_mal128_f4_trace_a->fft_a_square = fft_a_square;
    fft_mal128_f4_trace_a->fft_a_tensor = fft_a_tensor;
}

/**
 * Multiply two F_{2^128} elements over GF(4).
 * The minimal polynomial is x^64 + z*x^3 + z*x + 1, where GF(4)=GF(2)(z).
 * It is converted to F_{2^128} via polynomial (x^64 + z*x^3 + z*x + 1)*x^64 + (z + 1)*x^3 + (z + 1)*x + 1=x^128 + x^67 + x^65 + x^6 + x^3 + x^2 + x + 1
.
 */
uint128_t mult_mal128_f4(uint128_t a, uint128_t b) {
    const uint128_t GF_POLY = 0xA000000000000004F;
    uint128_t result = 0;
    uint128_t carry;

    // Perform polynomial multiplication
    for (int i = 0; i < 128; i++) {
        if (b & 1) { // If the least significant bit of b is 1
            result ^= a;
        }
        // Check if the highest bit of `a` is 1 for reduction
        carry = a & ((uint128_t)1 << 127);
        a <<= 1;

        // If carry exists, reduce with the irreducible polynomial
        if (carry) {
            a ^= (uint128_t)GF_POLY;
        }
        b >>= 1;
    }
    return result;
}

void sample_mal128_f4_trace_a_and_tensor(const struct Param *param, struct FFT_Mal128_F4_Trace_A *fft_mal128_f4_trace_a) {
    uint32_t *fft_a = fft_mal128_f4_trace_a->fft_a;
    uint32_t *fft_a_square = fft_mal128_f4_trace_a->fft_a_square;
    uint32_t *fft_a_tensor = fft_mal128_f4_trace_a->fft_a_tensor;
    const size_t poly_size = param->poly_size;
    const size_t c = param->c;
    const size_t m = param->m;

    RAND_bytes((uint8_t *)fft_a, sizeof(uint32_t) * poly_size);
    // make a_0 the identity polynomial (in FFT space) i.e., all 1s
    for (size_t i = 0; i < poly_size; i++) {
        fft_a[i] = fft_a[i] >> m;
        fft_a[i] = fft_a[i] << m;
        fft_a[i] |= 1;
    }

    // compute a_square
    for (size_t i = 0; i < c; i++) {
        for (size_t ii = 0; ii < poly_size; ii++) {
            uint8_t ai = (fft_a[ii] >> (m * i)) & 0b11;
            uint32_t w = f4_mul(ai, ai);
            fft_a_square[ii] |= w<<(m*i);
        }
    }

    // compute tensor of a
    /**
     * TODO: Merge fft_a and fft_a_square to one array to allow indexing.
     */
    for (size_t k = 0; k < m; k++) {
        for (size_t l = 0; l < m; l++) {
            for (size_t i = 0; i < c; i++) {
                for (size_t j = 0; j < c; j++) {
                    for (size_t ii = 0; ii < poly_size; ++ii) {
                        uint8_t ai = (fft_a[ii] >> (m * i)) & 0b11;
                        uint8_t aj = (fft_a[ii] >> (m * j)) & 0b11;
                        uint8_t ai_square = (fft_a_square[ii] >> (m * i)) & 0b11;
                        uint8_t aj_square = (fft_a_square[ii] >> (m * j)) & 0b11;
                        uint32_t w = 0;
                        if (k == 0 && l == 0) {
                            w = f4_mul(ai, aj);
                        } else if (k == 0 && l == 1) {
                            w = f4_mul(ai, aj_square);
                        } else if (k == 1 && l == 0) {
                            w = f4_mul(ai_square, aj);
                        } else if (k == 1 && l == 1) {
                            w = f4_mul(ai_square, aj_square);
                        } else {
                            printf("Missing items\n");
                        }
                        size_t tensor_idx = ((k*m+l)*c+i)*poly_size+ii;
                        fft_a_tensor[tensor_idx] |= w<<(m*j);
                    }
                }
            }
        }
    }
    printf("Done with sampling the public values\n");
}

void init_mal128_f4_trace_prod(const struct Param *param, struct Mal128_F4_Trace_Prod *mal128_f4_trace_prod) {

    size_t c = param->c;
    size_t t = param->t;
    size_t m = param->m;
    mal128_f4_trace_prod->m = m;
    size_t dpf_block_size = param->dpf_block_size;
    size_t poly_size = param->poly_size;
    if (t*t*dpf_block_size != poly_size) {
        printf("t*t*dpf_block_size != poly_size\n");
        exit(-1);
    }
    struct Keys *keys = xcalloc(1, sizeof(struct Keys));
    sample_mal128_f4_trace_prod_dpf_keys(param, keys);
    mal128_f4_trace_prod->keys = keys;

    uint128_t **rlt = xcalloc(m, sizeof(void *));
    for (size_t i = 0; i < m; ++i) {
        rlt[i] = xcalloc(t*t*dpf_block_size, sizeof(uint128_t));
    }
    mal128_f4_trace_prod->rlt = rlt;

    uint128_t *polys = xcalloc(m*m*c*c*t*t*dpf_block_size, sizeof(uint128_t));
    uint128_t *shares = xcalloc(dpf_block_size, sizeof(uint128_t));
    uint128_t *cache = xcalloc(dpf_block_size, sizeof(uint128_t));
    
    mal128_f4_trace_prod->polys = polys;
    mal128_f4_trace_prod->shares = shares;
    mal128_f4_trace_prod->cache = cache;
}

// sample m*m*c*c*t*t DPF keys
void sample_mal128_f4_trace_prod_dpf_keys(const struct Param *param, struct Keys *keys) {
    size_t c = param->c;
    size_t t = param->t;
    size_t dpf_block_size = param->dpf_block_size;
    size_t dpf_domain_bits = param->dpf_domain_bits;
    const size_t m = param->m;

    struct DPFKey **dpf_keys_A = xmalloc(c*c*m*m*t*t*sizeof(void *));
    struct DPFKey **dpf_keys_B = xmalloc(c*c*m*m*t*t*sizeof(void *));
    struct PRFKeys *prf_keys = xmalloc(sizeof(struct PRFKeys));
    PRFKeyGen(prf_keys);
    for (size_t k=0; k < m; ++k) {
        for (size_t l = 0; l < m; ++l) {
            for (size_t i = 0; i < c; ++i) {
                for (size_t j = 0; j < c; ++j) {
                    for (size_t u = 0; u < t; ++u) {
                        for (size_t v = 0; v < t; ++v) {
                            size_t index = ((((k*m+l)*c+i)*c+j)*t+u)*t+v;
                            size_t alpha = random_index(dpf_block_size);
                            uint128_t beta = 0;
                            RAND_bytes((unsigned char*)&beta, sizeof(uint128_t));
                            struct DPFKey *kA = xmalloc(sizeof(struct DPFKey));
                            struct DPFKey *kB = xmalloc(sizeof(struct DPFKey));
                            DPFGen(prf_keys, dpf_domain_bits, alpha, &beta, 1, kA, kB);
                            dpf_keys_A[index] = kA;
                            dpf_keys_B[index] = kB;
                        }
                    }
                }
            }
        }
    }
    keys->dpf_keys_A = dpf_keys_A;
    keys->dpf_keys_B = dpf_keys_B;
    keys->prf_keys = prf_keys;
}

static void free_mal128_f4_trace_prod_dpf_keys(const struct Param *param, struct Keys *keys) {

    const size_t c = param->c;
    const size_t t = param->t;
    const size_t m = param->m;
    for (size_t i = 0; i < m*m*c*c*t*t; ++i) {
        free(keys->dpf_keys_A[i]);
        free(keys->dpf_keys_B[i]);
    }
    free(keys->dpf_keys_A);
    free(keys->dpf_keys_B);
    DestroyPRFKey(keys->prf_keys);
    free(keys);
}

void run_mal128_f4_trace_prod(const struct Param *param, struct Mal128_F4_Trace_Prod *mal128_f4_trace_prod, uint32_t *fft_a_tensor, const uint8_t *f4_tr_tbl, const uint8_t *f4_zeta_powers) {

    struct Keys *keys = mal128_f4_trace_prod->keys;
    uint128_t *polys = mal128_f4_trace_prod->polys;
    uint128_t *shares = mal128_f4_trace_prod->shares;
    uint128_t *cache = mal128_f4_trace_prod->cache;
    uint128_t **rlt = mal128_f4_trace_prod->rlt;
    evaluate_mal128_f4_trace_prod_dpf(param, keys, polys, shares, cache);
    convert_mal128_f4_trace_prod_to_fft(param, polys, f4_zeta_powers);
    multiply_and_sum_mal128_f4_trace_prod(param, fft_a_tensor, polys, rlt, f4_zeta_powers);
}

void multiply_and_sum_mal128_f4_trace_prod(const struct Param *param, uint32_t *fft_a_tensor, uint128_t *polys, uint128_t **rlt, const uint8_t *f4_zeta_powers) {
    const size_t c = param->c;
    const size_t m = param->m;
    const size_t base = param->base;
    const size_t poly_size = param->poly_size;

    for (size_t k = 0; k < m; ++k) {
        for (size_t l = 0; l < m; ++l) {
            for(size_t i = 0; i < c; ++i) {
                for (size_t j = 0; j < c; ++j) {
                    for (size_t u = 0; u < poly_size; ++u) {
                        uint8_t a = (fft_a_tensor[((k*m+l)*c+i)*poly_size+u] >> (m*j)) & 0b11;
                        uint128_t e = polys[(((k*m+l)*c+i)*c+j)*poly_size+u];
                        uint128_t prod = 0;
                        scalar_mult_mal128_trace_f4(a, e, &prod);
                        rlt[0][u] ^= prod;

                        uint128_t w = 0;
                        scalar_mult_mal128_trace_f4(f4_zeta_powers[((1<<k)+(1<<l))%base], prod, &w);
                        rlt[1][u] ^= w;
                    }
                }
            }
        }
    }
}

/**
 * Multiply F_{2^128} elements by F4 elements.
 */
static void scalar_mult_mal128_trace_f4(uint8_t scalar, uint128_t b, uint128_t* t) {

    for (size_t i = 0; i < 128/2; ++i) {
        uint128_t w = f4_mul(scalar, (b>>(2*i)) & 0b11);
        *t |= (w<<(2*i));
    }
}

void convert_mal128_f4_trace_prod_to_fft(const struct Param *param, uint128_t *polys, const uint8_t *f4_zeta_powers) {
    const size_t c = param->c;
    const size_t m = param->m;
    const size_t n = param->n;
    const size_t base = param->base;
    const size_t poly_size = param->poly_size;
    // m*c
    for(size_t i = 0; i < c*m; ++i) {

        uint128_t *poly = &polys[i*poly_size];
        /**
         * FFT for uint32 packs 16 f4 values.
         */
        fft_recursive_mal128_f4_trace(poly, f4_zeta_powers, n, poly_size / base, param, base);
    }
}

/**
 * Compute the trace function for the array of F4 elements.
 */
static void compute_f4_trace(uint8_t *rlt, const struct Param *param, const uint8_t *f4_tr_tbl) {
    
    const size_t poly_size = param->poly_size;
    for (size_t i = 0; i < poly_size; ++i) {
        rlt[i] = f4_tr_tbl[rlt[i]];
    }
}

/**
 * TODO: maybe change the parameters to struct F4_Trace_Prod to include most of the inputs
 */
void evaluate_mal128_f4_trace_prod_dpf(const struct Param *param, const struct Keys *keys, uint128_t *polys, uint128_t *shares, uint128_t *cache) {

    const size_t c = param->c;
    const size_t t = param->t;
    const size_t m = param->m;
    const size_t dpf_block_size = param->dpf_block_size;
    for (size_t k=0; k < m; ++k) {
        for (size_t l=0; l < m; ++l) {
            for (size_t i = 0; i < c; ++i) {
                for (size_t j = 0; j < c; ++j) {
                    for (size_t u = 0; u < t; ++u) {
                        for (size_t v = 0; v < t; ++v) {
                            size_t key_index = ((((k*m+l)*c+i)*c+j)*t+u)*t+v;
                            struct DPFKey *dpf_key = keys->dpf_keys_A[key_index];
                            size_t poly_index = key_index;
                            uint128_t *poly_block = &polys[poly_index*dpf_block_size];
                            DPFFullDomainEval(dpf_key, cache, shares);
                            copy_mal128_f4_block(poly_block, shares, dpf_block_size);
                        }
                    }
                }
            }
        }
    }
}

static void copy_mal128_f4_block(uint128_t *poly_block, uint128_t *shares, const size_t dpf_block_size) {

    memcpy(poly_block, shares, dpf_block_size*sizeof(uint128_t));
}