#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "test.h"
#include "modular_test.h"
#include "modular_bench.h"
#include "trace_bench.h"
#include "mal128_trace_f16_bench.h"
#include "dpf.h"
#include "prf.h"
#include "fft.h"
#include "utils.h"
#include "f4ops.h"

/**
 * For q=4, existing parameter only supports c<=9.
 * The program supports c<=16
 */
void mal128_trace_f16_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time) {
    if (c > 16) {
        printf("ERROR: currently only implemented for c <= 16");
        exit(-1);
    }
    clock_t start_time = clock();
    struct Param *param = calloc(1, sizeof(struct Param));
    init_f16_trace_bench_params(param, n, c, t);
    const size_t m = 4;
    const size_t base = (1<<m)-1;
    const size_t q = 1<<m;
    uint8_t f16_zeta_powers[base];
    uint8_t f16_tr_tbl[q];
    memset(f16_zeta_powers, 0, sizeof(f16_zeta_powers));
    memset(f16_tr_tbl, 0, sizeof(f16_tr_tbl));
    compute_f16_zeta_powers(f16_zeta_powers, base);
    compute_f16_tr_tbl(f16_tr_tbl, m, f16_zeta_powers);

    struct FFT_Mal128_F16_Trace_A *fft_mal128_f16_trace_a = xcalloc(1, sizeof(struct FFT_Mal128_F16_Trace_A));
    init_fft_mal128_f16_trace_a(param, fft_mal128_f16_trace_a);
    sample_mal128_f16_trace_a_and_tensor(param, fft_mal128_f16_trace_a);
    
    struct Mal128_F16_Trace_Prod *mal128_f16_trace_prod = xcalloc(1, sizeof(struct Mal128_F16_Trace_Prod));
    init_mal128_f16_trace_prod(param, mal128_f16_trace_prod);
    printf("Benchmarking PCG evaluation\n");

    clock_t start_expand_time = clock();
    run_mal128_f16_trace_prod(param, mal128_f16_trace_prod, fft_mal128_f16_trace_a->fft_a_tensor, f16_tr_tbl, f16_zeta_powers);
    double end_expand_time = clock();
    double time_taken = ((double)(end_expand_time - start_expand_time)) / (CLOCKS_PER_SEC / 1000.0); // ms
    printf("Eval time (total) %f ms\n", time_taken);
    printf("DONE Benchmarking PCG evaluation\n\n");

    free_mal128_f16_trace_prod(param, mal128_f16_trace_prod);
    free_fft_mal128_f16_trace_a(param, fft_mal128_f16_trace_a);
    free(param);

    clock_t end_time = clock();
    pcg_time->pp_time = ((double)(start_expand_time-start_time))/(CLOCKS_PER_SEC/1000.0);
    pcg_time->expand_time = time_taken;
    pcg_time->total_time = ((double)(end_time-start_time))/(CLOCKS_PER_SEC / 1000.0);
}

static void compute_f16_zeta_powers(uint8_t *f16_zeta_powers, const size_t base) {

    uint8_t zeta = 1<<1;
    f16_zeta_powers[0] = 1;
    for (size_t i = 1; i<base; ++i) {
        f16_zeta_powers[i] = mult_f16_single(zeta, f16_zeta_powers[i-1]);
    }
}

static void compute_f16_tr_tbl(uint8_t *f16_tr_tbl, const size_t m, uint8_t *f16_zeta_powers) {
    f16_tr_tbl[0] = 0;
    const size_t base = (1<<m)-1;
    const size_t q = 1<<m; // field size
    for(size_t i = 1; i < q; ++i) {
        uint8_t v = f16_zeta_powers[i];
        f16_tr_tbl[v] = 0;
        for (size_t j = 0; j < m; ++j) {
            size_t id = (i * (1<<j))%base;
            f16_tr_tbl[v] ^= f16_zeta_powers[id];
        }
    }
}

void free_mal128_f16_trace_prod(const struct Param *param, struct Mal128_F16_Trace_Prod *mal128_f16_trace_prod) {

    free_mal128_f16_trace_prod_dpf_keys(param, mal128_f16_trace_prod->keys);
    free(mal128_f16_trace_prod->polys);
    free(mal128_f16_trace_prod->shares);
    free(mal128_f16_trace_prod->cache);

    for (size_t i = 0; i < param->m; ++i) {
        free(mal128_f16_trace_prod->rlt[i]);
    }
    free(mal128_f16_trace_prod);
}

void free_fft_mal128_f16_trace_a(const struct Param *param, struct FFT_Mal128_F16_Trace_A *fft_mal128_f16_trace_a) {

    free(fft_mal128_f16_trace_a->fft_a_tensor);
    for (size_t i = 0; i < param->m; ++i) {
        free(fft_mal128_f16_trace_a->fft_a_powers[i]);
    }
    free(fft_mal128_f16_trace_a->fft_a_powers);
    free(fft_mal128_f16_trace_a);
}

void init_fft_mal128_f16_trace_a(const struct Param *param, struct FFT_Mal128_F16_Trace_A *fft_mal128_f16_trace_a) {
    size_t poly_size = param->poly_size;
    size_t c = param->c;
    size_t m = param->m;
    
    uint32_t *fft_a_tensor = xcalloc(m*m*c*poly_size, sizeof(uint32_t));
    uint32_t **fft_a_powers = xcalloc(m, sizeof(void*));
    for (size_t i = 0; i < m; ++i) {
        fft_a_powers[i] = xcalloc(poly_size, sizeof(uint32_t));
    }
    fft_mal128_f16_trace_a->fft_a_powers = fft_a_powers;
    fft_mal128_f16_trace_a->fft_a_tensor = fft_a_tensor;
}

/**
 * Multiplies two elements of F16 only without packing.
 * Mod the polynomial x^4+x+1.
 */
static uint8_t mult_f16_single(uint8_t a, uint8_t b) {
    a &= 0x0F;
    b &= 0x0F;
    
    uint8_t result = 0;
    for (int i = 0; i < 4; i++) {
        if (b & (1 << i)) {
            // If the i-th bit of b is set, add (XOR) a shifted left by i positions
            result ^= (a << i); 
        }
    }
    // Reduction mod the polynomial x^4+x+1.
    // Polynomial x + 1 (binary 0011)
    const uint8_t irreducible = 0b11;
    
    for (int i = 7; i >= 4; i--) {
        if (result & (1 << i)) { 
            // If there's a bit set at position i, reducing it mod x^4+x+1
            result ^= (irreducible << (i - 4));
        }
    }
    return result & 0xF;
}


// Precomputed F_16 multiplication table (β⁴ = β + 1)
static uint8_t f16_mul[16][16];

// Initialize F_16 multiplication table
static void init_f16_table(void) {
    for (int a = 0; a < 16; a++) {
        for (int b = 0; b < 16; b++) {
            // Extract coefficients for a = a0 + a1β + a2β² + a3β³
            int a0 = (a >> 0) & 1, a1 = (a >> 1) & 1;
            int a2 = (a >> 2) & 1, a3 = (a >> 3) & 1;
            
            // Extract coefficients for b = b0 + b1β + b2β² + b3β³
            int b0 = (b >> 0) & 1, b1 = (b >> 1) & 1;
            int b2 = (b >> 2) & 1, b3 = (b >> 3) & 1;
            
            // Compute product coefficients before reduction (up to β⁶)
            int c0 = a0*b0;
            int c1 = a0*b1 + a1*b0;
            int c2 = a0*b2 + a1*b1 + a2*b0;
            int c3 = a0*b3 + a1*b2 + a2*b1 + a3*b0;
            int c4 = a1*b3 + a2*b2 + a3*b1;
            int c5 = a2*b3 + a3*b2;
            int c6 = a3*b3;
            
            // Reduce using β⁴ = β + 1, β⁵ = β² + β, β⁶ = β³ + β²
            int res0 = c0 % 2;
            int res1 = (c1 + c4) % 2;
            int res2 = (c2 + c4 + c5) % 2;
            int res3 = (c3 + c5 + c6) % 2;
            
            f16_mul[a][b] = (res3 << 3) | (res2 << 2) | (res1 << 1) | res0;
        }
    }
}

// Convert uint128_t to 32 F_16 coefficients (polynomial basis)
static void uint128_to_f128(uint128_t val, uint8_t coeffs[32]) {
    for (int i = 0; i < 32; i++) {
        // Each F_16 coefficient is 4 bits, stored little-endian in uint128_t
        coeffs[i] = (val >> (4 * i)) & 0xF;
    }
}

// Convert 32 F_16 coefficients back to uint128_t
static uint128_t f128_to_uint128(const uint8_t coeffs[32]) {
    uint128_t val = 0;
    for (int i = 0; i < 32; i++) {
        val |= (uint128_t)(coeffs[i] & 0xF) << (4 * i);
    }
    return val;
}

// Multiply two F_{2^128} elements stored as uint128_t
uint128_t mult_mal128_f16(const uint128_t a, const uint128_t b) {
    // Initialize F_16 multiplication table (run once)
    static int initialized = 0;
    if (!initialized) {
        init_f16_table();
        initialized = 1;
    }

    // Convert inputs to polynomial basis (32 F_16 coefficients each)
    uint8_t a_coeffs[32], b_coeffs[32];
    uint128_to_f128(a, a_coeffs);
    uint128_to_f128(b, b_coeffs);

    // Temporary array for product (degree up to 62 before reduction)
    uint8_t temp[63] = {0};

    // Step 1: Polynomial multiplication without reduction
    for (int i = 0; i < 32; i++) {
        if (a_coeffs[i] == 0) continue;  // Skip zero coefficients
        for (int j = 0; j < 32; j++) {
            if (b_coeffs[j] == 0) continue;  // Skip zero coefficients
            uint8_t prod = f16_mul[a_coeffs[i]][b_coeffs[j]];
            temp[i + j] ^= prod;  // Addition in F_2 is XOR
        }
    }

    // Step 2: Reduce modulo x^32 + βx + 1 → x^32 = βx + 1
    uint8_t result[32] = {0};
    const uint8_t beta = 0x2;  // β = 0b0010 (β^1 coefficient)

    for (int k = 0; k < 63; k++) {
        if (temp[k] == 0) continue;

        if (k < 32) {
            // No reduction needed for degrees < 32
            result[k] ^= temp[k];
        } else {
            // For degrees ≥32: x^k = x^(k-32) * (βx + 1)
            int exp = k - 32;
            
            // x^k = β·x^(exp+1) + x^exp
            if (exp + 1 < 32) {
                result[exp + 1] ^= f16_mul[temp[k]][beta];
            }
            if (exp < 32) {
                result[exp] ^= temp[k];  // Multiply by 1 (identity)
            }
        }
    }

    // Convert result back to uint128_t
    return f128_to_uint128(result);
}

void sample_mal128_f16_trace_a_and_tensor(const struct Param *param, struct FFT_Mal128_F16_Trace_A *fft_mal128_f16_trace_a) {

    uint32_t *fft_a_tensor = fft_mal128_f16_trace_a->fft_a_tensor;
    uint32_t **fft_a_powers = fft_mal128_f16_trace_a->fft_a_powers;
    const size_t poly_size = param->poly_size;
    const size_t c = param->c;
    const size_t m = param->m;

    RAND_bytes((uint8_t *)&fft_a_powers[0][0], sizeof(uint32_t) * poly_size);
    // make a_0 the identity polynomial (in FFT space) i.e., all 1s
    for (size_t i = 0; i < poly_size; i++) {
        fft_a_powers[0][i] = fft_a_powers[0][i] >> m;
        fft_a_powers[0][i] = fft_a_powers[0][i] << m;
        fft_a_powers[0][i] |= 1;
    }

    for (size_t k = 1; k < m; ++k) {
        for (size_t i = 0; i < c; ++i) {
            for (size_t u = 0; u < poly_size; ++u) {
                uint32_t a_kiu = (fft_a_powers[k-1][u] >> (m*i)) & 0xF;
                uint32_t w = mult_f16_single(a_kiu, a_kiu);
                fft_a_powers[k][u] |= w<<(m*i);
            }
        }
    }
    // compute tensor of a
    for (size_t k = 0; k < m; k++) {
        for (size_t l = 0; l < m; l++) {
            for (size_t i = 0; i < c; i++) {
                for (size_t j = 0; j < c; j++) {
                    for (size_t u = 0; u < poly_size; ++u) {
                        uint8_t a_kiu = (fft_a_powers[k][u] >> (m*i)) & 0xF;
                        uint8_t a_lju = (fft_a_powers[l][u] >> (m*j)) & 0xF;
                        uint32_t w = mult_f16_single(a_kiu, a_lju);
                        size_t tensor_idx = ((k*m+l)*c+i)*poly_size+u;
                        fft_a_tensor[tensor_idx] |= w<<(m*j);
                    }
                }
            }
        }
    }
    printf("Done with sampling the public values\n");
}

void init_mal128_f16_trace_prod(const struct Param *param, struct Mal128_F16_Trace_Prod *mal128_f16_trace_prod) {

    size_t c = param->c;
    size_t t = param->t;
    size_t m = param->m;
    mal128_f16_trace_prod->m = m;
    size_t dpf_block_size = param->dpf_block_size;
    size_t poly_size = param->poly_size;
    if (t*t*dpf_block_size != poly_size) {
        printf("t*t*dpf_block_size != poly_size\n");
        exit(-1);
    }
    struct KeysHD *keys = xcalloc(1, sizeof(struct KeysHD));
    sample_mal128_f16_trace_prod_dpf_keys(param, keys);
    mal128_f16_trace_prod->keys = keys;

    uint128_t **rlt = xcalloc(m, sizeof(void *));
    for (size_t i = 0; i < m; ++i) {
        rlt[i] = xcalloc(t*t*dpf_block_size, sizeof(uint128_t));
    }
    mal128_f16_trace_prod->rlt = rlt;

    uint128_t *polys = xcalloc(m*m*c*c*t*t*dpf_block_size, sizeof(uint128_t));
    uint128_t *shares = xcalloc(dpf_block_size, sizeof(uint128_t));
    uint128_t *cache = xcalloc(dpf_block_size, sizeof(uint128_t));
    
    mal128_f16_trace_prod->polys = polys;
    mal128_f16_trace_prod->shares = shares;
    mal128_f16_trace_prod->cache = cache;
}

// sample m*m*c*c*t*t DPF keys
void sample_mal128_f16_trace_prod_dpf_keys(const struct Param *param, struct KeysHD *keys) {
    size_t c = param->c;
    size_t t = param->t;
    size_t dpf_block_size = param->dpf_block_size;
    size_t dpf_domain_bits = param->dpf_domain_bits;
    const size_t m = param->m;
    const size_t base = param->base;

    struct DPFKeyZ **dpf_keys_A = xmalloc(c*c*m*m*t*t*sizeof(void *));
    struct DPFKeyZ **dpf_keys_B = xmalloc(c*c*m*m*t*t*sizeof(void *));
    struct PRFKeysZ *prf_keys = xmalloc(sizeof(struct PRFKeysZ));
    PRFKeyGenZ(prf_keys, base);
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
                            struct DPFKeyZ *kA = xmalloc(sizeof(struct DPFKeyZ));
                            struct DPFKeyZ *kB = xmalloc(sizeof(struct DPFKeyZ));
                            DPFGenZ(base, prf_keys, dpf_domain_bits, alpha, &beta, 1, kA, kB);
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

static void free_mal128_f16_trace_prod_dpf_keys(const struct Param *param, struct KeysHD *keys) {

    const size_t c = param->c;
    const size_t t = param->t;
    const size_t m = param->m;
    const size_t base = param->base;
    for (size_t i = 0; i < m*m*c*c*t*t; ++i) {
        free(keys->dpf_keys_A[i]);
        free(keys->dpf_keys_B[i]);
    }
    free(keys->dpf_keys_A);
    free(keys->dpf_keys_B);
    DestroyPRFKeyZ(keys->prf_keys, base);
    free(keys);
}

void run_mal128_f16_trace_prod(const struct Param *param, struct Mal128_F16_Trace_Prod *mal128_f16_trace_prod, uint32_t *fft_a_tensor, const uint8_t *f16_tr_tbl, const uint8_t *f16_zeta_powers) {

    struct KeysHD *keys = mal128_f16_trace_prod->keys;
    uint128_t *polys = mal128_f16_trace_prod->polys;
    uint128_t *shares = mal128_f16_trace_prod->shares;
    uint128_t *cache = mal128_f16_trace_prod->cache;
    uint128_t **rlt = mal128_f16_trace_prod->rlt;
    evaluate_mal128_f16_trace_prod_dpf(param, keys, polys, shares, cache);
    convert_mal128_f16_trace_prod_to_fft(param, polys, f16_zeta_powers);
    multiply_and_sum_mal128_f16_trace_prod(param, fft_a_tensor, polys, rlt, f16_zeta_powers);
}

void multiply_and_sum_mal128_f16_trace_prod(const struct Param *param, uint32_t *fft_a_tensor, uint128_t *polys, uint128_t **rlt, const uint8_t *f16_zeta_powers) {
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
                        scalar_mult_mal128_trace_f16(m, a, e, &prod);
                        rlt[0][u] ^= prod;

                        uint128_t w = 0;
                        scalar_mult_mal128_trace_f16(m, f16_zeta_powers[((1<<k)+(1<<l))%base], prod, &w);
                        rlt[1][u] ^= w;
                    }
                }
            }
        }
    }
}

/**
 * Multiply F_{2^128} elements by F16 elements.
 */
static void scalar_mult_mal128_trace_f16(const size_t m, uint8_t scalar, uint128_t b, uint128_t* t) {

    for (size_t i = 0; i < 128/m; ++i) {
        uint128_t w = mult_f16_single(scalar, (b>>(m*i)) & 0xF);
        *t |= (w<<(m*i));
    }
}

void convert_mal128_f16_trace_prod_to_fft(const struct Param *param, uint128_t *polys, const uint8_t *f16_zeta_powers) {
    const size_t c = param->c;
    const size_t m = param->m;
    const size_t n = param->n;
    const size_t base = param-> base;
    const size_t poly_size = param->poly_size;
    // m*c
    for(size_t i = 0; i < c*m; ++i) {

        uint128_t *poly = &polys[i*poly_size];
        /**
         * FFT for uint32 packs 16 f16 values.
         */
        fft_recursive_mal128_f16_trace(poly, f16_zeta_powers, n, poly_size / base, param, base);
    }
}

/**
 * Compute the trace function for the array of F16 elements.
 */
static void compute_f16_trace(uint8_t *rlt, const struct Param *param, const uint8_t *f16_tr_tbl) {
    
    const size_t poly_size = param->poly_size;
    for (size_t i = 0; i < poly_size; ++i) {
        rlt[i] = f16_tr_tbl[rlt[i]];
    }
}

void evaluate_mal128_f16_trace_prod_dpf(const struct Param *param, const struct KeysHD *keys, uint128_t *polys, uint128_t *shares, uint128_t *cache) {

    const size_t c = param->c;
    const size_t t = param->t;
    const size_t m = param->m;
    const size_t base = param->base;
    const size_t dpf_block_size = param->dpf_block_size;
    for (size_t k=0; k < m; ++k) {
        for (size_t l=0; l < m; ++l) {
            for (size_t i = 0; i < c; ++i) {
                for (size_t j = 0; j < c; ++j) {
                    for (size_t u = 0; u < t; ++u) {
                        for (size_t v = 0; v < t; ++v) {
                            size_t key_index = ((((k*m+l)*c+i)*c+j)*t+u)*t+v;
                            struct DPFKeyZ *dpf_key = keys->dpf_keys_A[key_index];
                            size_t poly_index = key_index;
                            uint128_t *poly_block = &polys[poly_index*dpf_block_size];
                            DPFFullDomainEvalZ(base, dpf_key, cache, shares);
                            copy_mal128_f16_block(poly_block, shares, dpf_block_size);
                        }
                    }
                }
            }
        }
    }
}

static void copy_mal128_f16_block(uint128_t *poly_block, uint128_t *shares, const size_t dpf_block_size) {

    memcpy(poly_block, shares, dpf_block_size*sizeof(uint128_t));
}