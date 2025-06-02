#include <stdint.h>
#include <stdlib.h>

#include "common.h"
#include "dpf.h"
#include "fft.h"
#include "modular_bench.h"
#include "gr64_bench.h"
#include "gr64_trace_bench.h"
#include "SPDZ2k_32_bench.h"
#define DPF_MSG_LEN 2
#define DPF_MSG_NUM sizeof(struct GR64)/sizeof(uint64_t)

// Parameters
// k = 32, s = 26, l = k+s=58
// k = 64, s = 57, l = k+s=121
void SPDZ2k_32_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time) {

    clock_t start_time = clock();
    struct Param *param = xcalloc(1, sizeof(struct Param));
    size_t m = 2;
    size_t k = 32;
    size_t s = 26;
    printf("k=%zu, s=%zu\n", k, s);
    init_SPDZ2k_32_bench_params(param, n, c, t, m, k, s);
    size_t dpf_block_size = param->dpf_block_size;
    size_t poly_size = param->poly_size;
    size_t block_size = param->block_size;
    if (t*t*dpf_block_size == poly_size && block_size*t == poly_size) {
        printf("OK\n");
    } else {
        printf("Incorrect\n");
        exit(-1);
    }
    struct FFT_GR64_Trace_A *fft_gr64_trace_a = xcalloc(1, sizeof(struct FFT_GR64_Trace_A));
    init_FFT_GR64_Trace_A(param, fft_gr64_trace_a);
    sample_SPDZ2k_32_a_and_tensor(param, fft_gr64_trace_a);

    struct SPDZ2k_32_b *spdz2k_32_b = xcalloc(1, sizeof(struct SPDZ2k_32_b));
    init_SPDZ2k_32_b(param, spdz2k_32_b);
    // ***************************
    // variables for the product
    struct SPDZ2k_32_Prod *spdz2k_32_prod = xcalloc(1, sizeof(struct SPDZ2k_32_Prod));

    init_SPDZ2k_32_prod(param, spdz2k_32_prod);
    printf("Benchmarking PCG evaluation \n");

    clock_t start_expand_time = clock();
    run_SPDZ2k_32_b(param, spdz2k_32_b, fft_gr64_trace_a->fft_a);
    run_SPDZ2k_32_prod(param, spdz2k_32_prod, fft_gr64_trace_a->fft_a_tensor_maps);

    double end_expand_time = clock();
    double time_taken = ((double)(end_expand_time - start_expand_time)) / (CLOCKS_PER_SEC / 1000.0); // ms
    printf("Eval time (total) %f ms\n", time_taken);
    printf("DONE Benchmarking PCG evaluation\n\n");
    free_SPDZ2k_32_prod(param, spdz2k_32_prod);
    free_FFT_GR64_Trace_A(param, fft_gr64_trace_a);
    free_SPDZ2k_32_b(param, spdz2k_32_b);
    free(param);

    clock_t end_time = clock();
    pcg_time->pp_time = ((double)(start_expand_time-start_time))/(CLOCKS_PER_SEC/1000.0);
    pcg_time->expand_time = time_taken;
    pcg_time->total_time = ((double)(end_time-start_time))/(CLOCKS_PER_SEC / 1000.0);
}

/**
 * Sample a and axa in FFT form.
 */
void sample_SPDZ2k_32_a_and_tensor(const struct Param *param, struct FFT_GR64_Trace_A *fft_gr64_trace_a) {
    // TODO: merge to the same function
    // call sample_gr64_trace_a_and_tensor to compute the relation first and then mod the special modulus
    sample_gr64_trace_a_and_tensor(param, fft_gr64_trace_a);
    uint64_t modulus64 = param->modulus64;
    struct GR64 **fft_a = fft_gr64_trace_a->fft_a;
    struct GR64 **fft_a_maps = fft_gr64_trace_a->fft_a_maps;
    struct GR64 **fft_a_tensor_maps = fft_gr64_trace_a->fft_a_tensor_maps;
    const size_t poly_size = param->poly_size;
    const size_t c = param->c;

    // reduce a
    for (size_t i = 1; i < c; ++i) {
        for (size_t j = 0; j < poly_size; ++j) {
            fft_a[i][j].c0 = fft_a[i][j].c0 % modulus64;
            fft_a[i][j].c1 = fft_a[i][j].c1 % modulus64;
        }
    }
    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < poly_size; j++) {
            struct GR64 *cur_a = &fft_a_maps[i][j];
            cur_a->c0 = cur_a->c0 % modulus64;
            cur_a->c1 = cur_a->c1 % modulus64;
        }
    }
    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t k = 0; k < poly_size; ++k) {
                struct GR64 *cur_a = &fft_a_tensor_maps[i*c+j][k];
                cur_a->c0 = cur_a->c0%modulus64;
                cur_a->c1 = cur_a->c1%modulus64;
                cur_a = &fft_a_tensor_maps[i*c+j][k+poly_size];
                cur_a->c0 = cur_a->c0%modulus64;
                cur_a->c1 = cur_a->c1%modulus64;
            }
        }
    }
}

// Multiply two SPDZ2k_32 elements
void mult_SPDZ2k_32(const struct GR64 *a, const struct GR64 *b, struct GR64 *t, const uint64_t modulus) {
    t->c0 = (a->c0*b->c0 - a->c1*b->c1) % modulus;
    t->c1 = (a->c0*b->c1 + a->c1*b->c0 - a->c1 * b->c1) % modulus;
}

// sample DPF keys for b0 and b1 with authentication
void sample_SPDZ2k_32_b_DPF_keys(const struct Param *param, struct Keys *keys) {
    size_t c = param->c;
    size_t t = param->t;
    size_t block_size = param->block_size;
    size_t block_bits = param->block_bits;
    uint64_t modulus64 = param->modulus64;
    const size_t m = param->m;

    struct DPFKey **dpf_keys_A = xmalloc(c*t*sizeof(void *));
    struct DPFKey **dpf_keys_B = xmalloc(c*t*sizeof(void *));
    struct PRFKeys *prf_keys = xmalloc(sizeof(struct PRFKeys));
    PRFKeyGen(prf_keys);
    for (size_t i=0; i<c; ++i) {
        for (size_t j=0; j<t; ++j) {
            size_t index = i*t+j;
            // Pick a random position for benchmarking purposes
            size_t alpha = random_index(block_size);
            uint128_t beta[DPF_MSG_LEN*DPF_MSG_NUM] ={0};
            RAND_bytes((uint8_t *)beta, sizeof(uint128_t));
            uint128_t c0 = beta[0]%modulus64;
            uint128_t c1 = (beta[0]>>64)%modulus64;
            beta[0] = (c1<<64)+c0;
            uint64_t K = param->K64;
            uint128_t m0 = (K*c0)%modulus64;
            uint128_t m1 = (K*c1)%modulus64;
            beta[1] = (m1<<64)+m0;
            // DPF keys
            struct DPFKey *kA = xmalloc(sizeof(struct DPFKey));
            struct DPFKey *kB = xmalloc(sizeof(struct DPFKey));
            // Now the DPF keys has two elements
            DPFGen(prf_keys, block_bits, alpha, beta, DPF_MSG_LEN*DPF_MSG_NUM, kA, kB);
            dpf_keys_A[index] = kA;
            dpf_keys_B[index] = kB;
        }
    }
    keys->dpf_keys_A = dpf_keys_A;
    keys->dpf_keys_B = dpf_keys_B;
    keys->prf_keys = prf_keys;
}

void free_SPDZ2k_32_b_DPF_keys(const struct Param *param, struct Keys *keys) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t m = param->m;
    for (size_t i=0; i<c; ++i) {
        for (size_t j = 0; j < t; ++j) {
            size_t index = i*t+j;
            free(keys->dpf_keys_A[index]);
            free(keys->dpf_keys_B[index]);
        }
    }
    free(keys->dpf_keys_A);
    free(keys->dpf_keys_B);
    DestroyPRFKey(keys->prf_keys);
    free(keys);
}

/**
 * Evaluate the DPF keys to polynomials containing both the output value and the MAC value: each is of length c*t*block_size=c*poly_size.
 * @param polys is of length c*poly_size*DPF_MSG_LEN=c*t*block_size*DPF_MSG_LEN=c*poly_size*DPF_MSG_LEN.
 * @param shares is of length block_size*DPF_MSG_LEN*DPF_MSG_NUM
 * @param cache is of length block_size*DPF_MSG_LEN*DPF_MSG_NUM
 */
void evaluate_SPDZ2k_32_b_DPF(const struct Param *param, const struct Keys *keys, struct GR64 *polys, uint128_t *shares, uint128_t *cache) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t poly_size = param->poly_size;
    const size_t block_size = param->block_size;
    const size_t m = param->m;
    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < t; ++j) {
            const size_t poly_index = i*t+j;
            struct GR64 *poly_block = &polys[poly_index*block_size*DPF_MSG_LEN];
            const size_t key_index = poly_index;
            struct DPFKey *dpf_key = keys->dpf_keys_A[key_index];
            // DPF evaluates the value and the MAC in consecutive positions and copy_gr64_block parses the DPF evaluation results
            DPFFullDomainEval(dpf_key, cache, shares);
            copy_gr64_block(poly_block, shares, DPF_MSG_LEN*block_size);
        }
    }
}

// Convert each DPF output polynomial to FFT
void convert_SPDZ2k_32_b_to_FFT(const struct Param *param, struct GR64 *polys) {
    const size_t c = param->c;
    const size_t n = param->n;
    const size_t poly_size = param->poly_size;
    const uint64_t modulus64 = param->modulus64;
    
    for (size_t i = 0; i < c*DPF_MSG_LEN; ++i) {

        struct GR64 *poly = &polys[i*poly_size];
        fft_recursive_SPDZ2k_32(poly, n, poly_size / 3, modulus64);
    }
}

/**
 * Multiply the polynomial a and the DPF output polynomial.
 * The polynomial a is reused.
 * @param a_polys
 * @param b_poly 
 * @param res_poly the result polynomial of length c*poly_size*DPF_MSG_LEN
 */
void multiply_SPDZ2k_32_b_FFT(const struct Param *param, struct GR64 **a_polys, const struct GR64 *b_poly, struct GR64 *res_poly) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;
    uint64_t modulus64 = param->modulus64;
    // b_poly and res_poly are of length c*DPF_MSG_LEN*poly_size
    for (size_t i = 0; i < c; ++i) {
        const struct GR64 *a_poly = a_polys[i];
        for (size_t k = 0; k < DPF_MSG_LEN; ++k) {
            for (size_t j = 0; j < poly_size; ++j) {
                mult_SPDZ2k_32(&a_poly[j], &b_poly[(i*DPF_MSG_LEN+k)*poly_size+j], &res_poly[(i*DPF_MSG_LEN+k)*poly_size+j], modulus64);
            }
        }
    }
}

/**
 * Sum up the c polynomials in poly_buf to z_poly
 * @param poly_buf is of length DPF_MSG_LEN*c*poly_size.
 * @param z_poly is of length DPF_MSG_LEN*poly_size.
 */
void sum_SPDZ2k_32_b_FFT_polys(const struct Param *param, struct GR64 *poly_buf, struct GR64 *z_poly) {
    const size_t c = param->c;
    const size_t poly_size = param->poly_size;
    const uint64_t modulus64 = param->modulus64;

    for (size_t i = 0; i < c; ++i) {
        size_t poly_index = i*DPF_MSG_LEN*poly_size;
        for (size_t l = 0; l < DPF_MSG_LEN; ++l) {
            for (size_t k = 0; k < poly_size; ++k) {
                struct GR64 *z = &z_poly[l*poly_size+k];
                z->c0 = (z->c0 + poly_buf[poly_index+k].c0) % modulus64;
                z->c1 = (z->c1 + poly_buf[poly_index+k].c1) % modulus64;
            }
        }
    }
}

/**
 * evaluate for b and K*b
 * 1 Evaluate DPF to poly
 * 2 Convert poly to FFT
 * 3 Multiply with a
 * 4 Sum up to obtain b
 */
// evaluate DPF to poly, convert poly to FFT, multiply with a and sum up
void evaluate_SPDZ2k_32_b_DPF_and_sum(
    const struct Param *param,
    const struct Keys *keys,
    struct GR64 **fft_a,
    struct GR64 *polys,
    struct GR64 *poly_buf,
    struct GR64 *z_poly,
    uint128_t *shares,
    uint128_t *cache) {

    evaluate_SPDZ2k_32_b_DPF(param, keys, polys, shares, cache);
    convert_SPDZ2k_32_b_to_FFT(param, polys);
    multiply_SPDZ2k_32_b_FFT(param, fft_a, polys, poly_buf);
    sum_SPDZ2k_32_b_FFT_polys(param, poly_buf, z_poly);
}

/**
 * Compute (Tr(b), Tr(zeta*b)) and (Tr(K*b), Tr(zeta*K*b))
 * src is for b and K*b and each is of length poly_size
 * b_0 is for Tr(b)
 * b_1 is for Tr(zeta*b)
 * bm_0 is for Tr(K*b)
 * bm_1 is for Tr(K*zeta*b)
 */
void trace_SPDZ2k_32_b_FFT_polys(const struct Param *param, const struct GR64 *src, uint64_t *b_0, uint64_t *b_1, uint64_t *bm_0, uint64_t *bm_1) {
    const size_t poly_size = param->poly_size;
    const uint64_t modulus64 = param->modulus64;
    const struct GR64 *poly = src;
    // Tr(b) and Tr(zeta*b)
    for (size_t i = 0; i < poly_size; ++i) {
        b_0[i] = (2*poly[i].c0-poly[i].c1)%modulus64;
        b_1[i] = (-(poly[i].c0+poly[i].c1))%modulus64;
    }
    // Tr(K*b) and Tr(zeta*K*b)
    poly = &src[poly_size];
    for (size_t i = 0; i < poly_size; ++i) {
        bm_0[i] = (2*poly[i].c0-poly[i].c1)%modulus64;
        bm_1[i] = (-(poly[i].c0 + poly[i].c1))%modulus64;
    }
}

/**
 * init the memory for b and sample the DPF_keys
 */
void init_SPDZ2k_32_b(const struct Param *param, struct SPDZ2k_32_b *spdz2k_32_b) {

    size_t c = param->c;
    size_t t = param->t;
    size_t m = param->m;
    size_t dpf_block_size = param->dpf_block_size;
    size_t poly_size = param->poly_size;
    size_t block_size = param->block_size;
    // generate keys for b0, b1
    struct Keys *keys_b0 = xcalloc(1, sizeof(struct Keys));
    sample_SPDZ2k_32_b_DPF_keys(param, keys_b0);
    struct Keys *keys_b1 = xcalloc(1, sizeof(struct Keys));
    sample_SPDZ2k_32_b_DPF_keys(param, keys_b1);
    // output for (b0,b1) and each has two values for the trace
    uint64_t *b0_0 = xcalloc(poly_size, sizeof(uint64_t));
    uint64_t *b0_1 = xcalloc(poly_size, sizeof(uint64_t));
    uint64_t *b1_0 = xcalloc(poly_size, sizeof(uint64_t));
    uint64_t *b1_1 = xcalloc(poly_size, sizeof(uint64_t));
    uint64_t *bm0_0 = xcalloc(poly_size, sizeof(uint64_t));
    uint64_t *bm0_1 = xcalloc(poly_size, sizeof(uint64_t));
    uint64_t *bm1_0 = xcalloc(poly_size, sizeof(uint64_t));
    uint64_t *bm1_1 = xcalloc(poly_size, sizeof(uint64_t));
    
    struct GR64 *polys = xcalloc(DPF_MSG_LEN*c*poly_size, sizeof(struct GR64));
    struct GR64 *poly_buf = xcalloc(DPF_MSG_LEN*c*poly_size, sizeof(struct GR64));
    struct GR64 *z_poly = xcalloc(DPF_MSG_LEN*poly_size, sizeof(struct GR64));
    uint128_t *shares = xcalloc(DPF_MSG_NUM*DPF_MSG_LEN*block_size, sizeof(uint128_t));
    uint128_t *cache = xcalloc(DPF_MSG_NUM*DPF_MSG_LEN*block_size, sizeof(uint128_t));
    
    spdz2k_32_b->keys_b0 = keys_b0;
    spdz2k_32_b->keys_b1 = keys_b1;
    
    spdz2k_32_b->b0_0 = b0_0;
    spdz2k_32_b->b0_1 = b0_1;
    spdz2k_32_b->b1_0 = b1_0;
    spdz2k_32_b->b1_1 = b1_1;
    spdz2k_32_b->bm0_0 = bm0_0;
    spdz2k_32_b->bm0_1 = bm0_1;
    spdz2k_32_b->bm1_0 = bm1_0;
    spdz2k_32_b->bm1_1 = bm1_1;

    spdz2k_32_b->polys = polys;
    spdz2k_32_b->poly_buf = poly_buf;
    spdz2k_32_b->z_poly = z_poly;
    spdz2k_32_b->shares = shares;
    spdz2k_32_b->cache = cache;
}

void free_SPDZ2k_32_b(const struct Param *param, struct SPDZ2k_32_b *spdz2k_32_b) {
    free_SPDZ2k_32_b_DPF_keys(param, spdz2k_32_b->keys_b0);
    free_SPDZ2k_32_b_DPF_keys(param, spdz2k_32_b->keys_b1);
    free(spdz2k_32_b->b0_0);
    free(spdz2k_32_b->b0_1);
    free(spdz2k_32_b->b1_0);
    free(spdz2k_32_b->b1_1);
    free(spdz2k_32_b->bm0_0);
    free(spdz2k_32_b->bm0_1);
    free(spdz2k_32_b->bm1_0);
    free(spdz2k_32_b->bm1_1);

    free(spdz2k_32_b->polys);
    free(spdz2k_32_b->poly_buf);
    free(spdz2k_32_b->z_poly);
    free(spdz2k_32_b->shares);
    free(spdz2k_32_b->cache);
    free(spdz2k_32_b);
}

void run_SPDZ2k_32_b(const struct Param *param, struct SPDZ2k_32_b *spdz2k_32_b, struct GR64 **fft_a) {
    size_t poly_size = param->poly_size;
    struct Keys *keys_b0 = spdz2k_32_b->keys_b0;
    struct Keys *keys_b1 = spdz2k_32_b->keys_b1;
    uint64_t *b0_0 = spdz2k_32_b->b0_0;
    uint64_t *b0_1 = spdz2k_32_b->b0_1;
    uint64_t *b1_0 = spdz2k_32_b->b1_0;
    uint64_t *b1_1 = spdz2k_32_b->b1_1;

    uint64_t *bm0_0 = spdz2k_32_b->bm0_0;
    uint64_t *bm0_1 = spdz2k_32_b->bm0_1;
    uint64_t *bm1_0 = spdz2k_32_b->bm1_0;
    uint64_t *bm1_1 = spdz2k_32_b->bm1_1;
    
    struct GR64 *polys = spdz2k_32_b->polys;
    struct GR64 *poly_buf = spdz2k_32_b->poly_buf;
    struct GR64 *z_poly = spdz2k_32_b->z_poly;
    uint128_t *shares = spdz2k_32_b->shares;
    uint128_t *cache = spdz2k_32_b->cache;

    /**
     * evaluate for b0 and K*b0
     * 1 Evaluate DPF to poly
     * 2 Convert poly to FFT
     * 3 Multiply with a
     * 4 Sum up to obtain b
     * 5 Compute the trace for zeta^j*b
     */
    // evaluate for b0 and K*b0
    evaluate_SPDZ2k_32_b_DPF_and_sum(param, keys_b0, fft_a, polys, poly_buf, z_poly, shares, cache);
    trace_SPDZ2k_32_b_FFT_polys(param, z_poly, b0_0, b0_1, bm0_0, bm0_1);
    
    // evaluate for b1 and K*b1
    memset(z_poly, 0, poly_size*sizeof(struct GR64));
    evaluate_SPDZ2k_32_b_DPF_and_sum(param, keys_b1, fft_a, polys, poly_buf, z_poly, shares, cache);
    trace_SPDZ2k_32_b_FFT_polys(param, z_poly, b1_0, b1_1, bm1_0, bm1_1);
}

/**
 * Init the memory for products and sample the DPF keys.
 */
void init_SPDZ2k_32_prod(const struct Param *param, struct SPDZ2k_32_Prod *spdz2k_32_prod) {

    size_t c = param->c;
    size_t t = param->t;
    size_t m = param->m;
    size_t dpf_block_size = param->dpf_block_size;
    size_t poly_size = param->poly_size;
    struct Keys *keys = xcalloc(1, sizeof(struct Keys));
    sample_SPDZ2k_32_prod_DPF_keys(param, keys);
    // DPF_MSG_LEN indicates the value z and z*K
    // m indicates the number of automorphisms

    // output for Tr(z) and Tr(z*K)
    uint64_t *rlt0 = xcalloc(DPF_MSG_LEN*t*t*dpf_block_size, sizeof(uint64_t));
    // output for Tr(z*zeta) and Tr(z*zeta*K)
    uint64_t *rlt1 = xcalloc(DPF_MSG_LEN*t*t*dpf_block_size, sizeof(uint64_t));
    // The polynomial for z and z*K
    struct GR64 *z_poly0 = xcalloc(m*t*t*dpf_block_size, sizeof(struct GR64));
    // The polynomial for zeta*z and zeta*z*K
    struct GR64 *z_poly1 = xcalloc(m*t*t*dpf_block_size, sizeof(struct GR64));
    /**
     * @param m is for the number of automorphisms
     * @param DPF_MSG_LEN is for the number of outputs
     */
    struct GR64 *polys = xcalloc(DPF_MSG_LEN*c*c*t*t*m*dpf_block_size, sizeof(struct GR64));
    struct GR64 *poly_buf = xcalloc(DPF_MSG_LEN*c*c*t*t*m*dpf_block_size, sizeof(struct GR64));
    uint128_t *shares = xcalloc(DPF_MSG_NUM*DPF_MSG_LEN*dpf_block_size, sizeof(uint128_t));
    uint128_t *cache = xcalloc(DPF_MSG_NUM*DPF_MSG_LEN*dpf_block_size, sizeof(uint128_t));

    spdz2k_32_prod->keys = keys;
    spdz2k_32_prod->polys = polys;
    spdz2k_32_prod->poly_buf = poly_buf;
    spdz2k_32_prod->z_poly0 = z_poly0;
    spdz2k_32_prod->z_poly1 = z_poly1;
    spdz2k_32_prod->rlt0 = rlt0;
    spdz2k_32_prod->rlt1 = rlt1;
    spdz2k_32_prod->shares = shares;
    spdz2k_32_prod->cache = cache;
}


void free_SPDZ2k_32_prod(const struct Param *param, struct SPDZ2k_32_Prod *spdz2k_32_prod) {
    free_SPDZ2k_32_prod_DPF_keys(param, spdz2k_32_prod->keys);
    free(spdz2k_32_prod->polys);
    free(spdz2k_32_prod->poly_buf);
    free(spdz2k_32_prod->z_poly0);
    free(spdz2k_32_prod->z_poly1);
    free(spdz2k_32_prod->rlt0);
    free(spdz2k_32_prod->rlt1);
    free(spdz2k_32_prod->shares);
    free(spdz2k_32_prod->cache);
    free(spdz2k_32_prod);
}

// sample DPF keys for z and z*K
void sample_SPDZ2k_32_prod_DPF_keys(const struct Param *param, struct Keys *keys) {
    size_t c = param->c;
    size_t t = param->t;
    size_t block_size = param->block_size;
    size_t dpf_block_size = param->dpf_block_size;
    size_t dpf_domain_bits = param->dpf_domain_bits;
    uint64_t modulus64 = param->modulus64;
    const size_t m = param->m;

    struct DPFKey **dpf_keys_A = xmalloc(c*c*m*t*t*sizeof(void *));
    struct DPFKey **dpf_keys_B = xmalloc(c*c*m*t*t*sizeof(void *));
    struct PRFKeys *prf_keys = xmalloc(sizeof(struct PRFKeys));
    PRFKeyGen(prf_keys);
    for (size_t i = 0; i < c; i++) {
        for (size_t j = 0; j < c; j++) {
            for (size_t w = 0; w < m; ++w) {
                for (size_t k = 0; k < t; k++) {
                    for (size_t l = 0; l < t; l++) {
                        size_t index = (((i*c+j)*m+w)*t+k)*t+l;
                        size_t alpha = random_index(dpf_block_size);
                        uint128_t beta[DPF_MSG_LEN*DPF_MSG_NUM] ={0};
                        // only init the first position
                        RAND_bytes((uint8_t *)beta, sizeof(struct GR64));
                        uint128_t c0 = beta[0]%modulus64;
                        uint128_t c1 = (beta[0]>>64)%modulus64;
                        beta[0] = (c1<<64)+c0;
                        uint64_t K = param->K64;
                        uint128_t m0 = (K*c0)%modulus64;
                        uint128_t m1 = (K*c1)%modulus64;
                        beta[1] = (m1<<64)+m0;
                        // DPF keys
                        struct DPFKey *kA = xmalloc(sizeof(struct DPFKey));
                        struct DPFKey *kB = xmalloc(sizeof(struct DPFKey));
                        // Now the DPF keys has two elements
                        DPFGen(prf_keys, dpf_domain_bits, alpha, beta, DPF_MSG_LEN*DPF_MSG_NUM, kA, kB);
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

void run_SPDZ2k_32_prod(const struct Param *param, struct SPDZ2k_32_Prod *spdz2k_32_prod, struct GR64 **fft_a_tensor_maps) {
    size_t poly_size = param->poly_size;
    struct Keys *keys = spdz2k_32_prod->keys;
    
    uint64_t *rlt0 = spdz2k_32_prod->rlt0;
    uint64_t *rlt1 = spdz2k_32_prod->rlt1;
    
    struct GR64 *polys = spdz2k_32_prod->polys;
    struct GR64 *poly_buf = spdz2k_32_prod->poly_buf;
    struct GR64 *z_poly0 = spdz2k_32_prod->z_poly0;
    struct GR64 *z_poly1 = spdz2k_32_prod->z_poly1;
    uint128_t *shares = spdz2k_32_prod->shares;
    uint128_t *cache = spdz2k_32_prod->cache;

    // evaluate for z and K*z
    evaluate_SPDZ2k_32_prod_DPF_and_sum(param, keys, fft_a_tensor_maps, polys, poly_buf, z_poly0, z_poly1, shares, cache);
    trace_SPDZ2k_32_prod_FFT_polys(param, z_poly0, rlt0);
    trace_SPDZ2k_32_prod_FFT_polys(param, z_poly1, rlt1);
}

// evaluate DPF to poly, convert poly to FFT, multiply with a and sum up
void evaluate_SPDZ2k_32_prod_DPF_and_sum(
    const struct Param *param,
    const struct Keys *keys,
    struct GR64 **fft_a_tensor_maps,
    struct GR64 *polys,
    struct GR64 *poly_buf,
    struct GR64 *z_poly0,
    struct GR64 *z_poly1,
    uint128_t *shares,
    uint128_t *cache) {

    evaluate_SPDZ2k_32_prod_DPF(param, keys, polys, shares, cache);
    convert_SPDZ2k_32_prod_to_FFT(param, polys);
    multiply_SPDZ2k_32_prod_FFT(param, fft_a_tensor_maps, polys, poly_buf);
    sum_SPDZ2k_32_prod_FFT_polys(param, poly_buf, z_poly0);
    sum_SPDZ2k_32_prod_FFT_polys_special(param, poly_buf, z_poly1);
}

/**
 * Evaluate the DPF keys to polynomials
 * @param polys is of length c*c*poly_size*DPF_MSG_LEN=c*c*block_size*DPF_MSG_LEN=c*c*t*t*dpf_block_size*DPF_MSG_LEN
 * @param shares is of length dpf_block_size*DPF_MSG_LEN*DPF_MSG_NUM
 * @param cache is of length dpf_block_size*DPF_MSG_LEN*DPF_MSG_NUM
 */
void evaluate_SPDZ2k_32_prod_DPF(const struct Param *param, const struct Keys *keys, struct GR64 *polys, uint128_t *shares, uint128_t *cache) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t m = param->m;
    const size_t dpf_block_size = param->dpf_block_size;

    for (size_t i = 0; i < c; i++) {
        for (size_t j = 0; j < c; j++) {
            for (size_t w = 0; w < m; ++w) {
                for (size_t k = 0; k < t; k++) {
                    for (size_t l = 0; l < t; l++) {
                        const size_t key_index = (((i*c+j)*m+w)*t+k)*t+l;
                        struct DPFKey *dpf_key = keys->dpf_keys_A[key_index];
                        struct GR64 *poly_block = &polys[key_index*DPF_MSG_LEN*dpf_block_size];
                        DPFFullDomainEval(dpf_key, cache, shares);
                        copy_gr64_block(poly_block, shares, DPF_MSG_LEN*dpf_block_size);
                    }
                }
            }
        }
    }
}

// Convert each DPF output polynomial to FFT
void convert_SPDZ2k_32_prod_to_FFT(const struct Param *param, struct GR64 *polys) {
    const size_t c = param->c;
    const size_t n = param->n;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;
    const uint64_t modulus64 = param->modulus64;
    
    for (size_t i = 0; i < c*c*m*DPF_MSG_LEN; ++i) {

        struct GR64 *poly = &polys[i*poly_size];
        fft_recursive_SPDZ2k_32(poly, n, poly_size / 3, modulus64);
    }
}

/**
 * Multiply the polynomial a and the DPF output polynomial.
 * The polynomial a is reused.
 * @param a_polys
 * @param b_poly 
 * @param res_poly the result polynomial of length c*c*m*DPF_MSG_LEN*poly_size
 */
void multiply_SPDZ2k_32_prod_FFT(const struct Param *param, struct GR64 **a_polys, const struct GR64 *b_poly, struct GR64 *res_poly) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;
    uint64_t modulus64 = param->modulus64;
    // b_poly and res_poly are of length DPF_MSG_LEN*c*c*m*poly_size
    for (size_t i = 0; i < c*c; ++i) {
        const struct GR64 *a_poly = a_polys[i];

        size_t poly_index = i*poly_size*DPF_MSG_LEN;
        mult_SPDZ2k_32(a_polys[i], &b_poly[poly_index], &res_poly[poly_index], modulus64);
        mult_SPDZ2k_32(a_polys[i], &b_poly[poly_index+poly_size], &res_poly[poly_index+poly_size], modulus64);
    }

    for (size_t i = 0; i < c*c; ++i) {
        const struct GR64 *a_poly = a_polys[i];
        for (size_t k = 0; k < DPF_MSG_LEN; ++k) {
            // reuse the a here
            size_t poly_index = (i*DPF_MSG_LEN+k)*poly_size*m;
            for (size_t j = 0; j < poly_size*m; ++j) {
               mult_SPDZ2k_32(&a_poly[j], &b_poly[poly_index+j], &res_poly[poly_index+j], modulus64);
            }
        }
    }
}

/**
 * Sum up the c polynomials in poly_buf to z_poly
 * @param poly_buf is of length DPF_MSG_LEN*c*c*m*poly_size.
 * @param z_poly is of length m*poly_size.
 */
void sum_SPDZ2k_32_prod_FFT_polys(const struct Param *param, struct GR64 *poly_buf, struct GR64 *z_poly) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;
    const uint64_t modulus64 = param->modulus64;

    for (size_t i = 0; i < c*c; ++i) {
        for (size_t l = 0; l < DPF_MSG_LEN; ++l) {
            for (size_t j = 0; j < poly_size*m; ++j) {
                size_t index = (i*DPF_MSG_LEN+l)*poly_size*m;
                z_poly[j].c0 = (z_poly[j].c0+poly_buf[index].c0)%modulus64;
                z_poly[j].c1 = (z_poly[j].c1+poly_buf[index].c1)%modulus64;
            }
        }
    }
}

void sum_SPDZ2k_32_prod_FFT_polys_special(const struct Param *param, struct GR64 *poly_buf, struct GR64 *z_poly) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;
    const uint64_t modulus64 = param->modulus64;

    for (size_t i = 0; i < c*c; ++i) {
        for (size_t l = 0; l < DPF_MSG_LEN; ++l) {
            for (size_t w = 0; w < m; ++w) {
                size_t poly_index = ((i*DPF_MSG_LEN+l)*m+w)*poly_size;
                if (w == 0) {
                    for (size_t j = 0; j < poly_size; ++j) {
                        z_poly[w*poly_size+j].c0 = (z_poly[w*poly_size+j].c0+poly_buf[poly_index+j].c1-poly_buf[poly_index+j].c0)%modulus64;
                        z_poly[w*poly_size+j].c1 = (z_poly[w*poly_size+j].c1-poly_buf[poly_index+j].c0)%modulus64;
                    }
                } else {
                    for (size_t j = 0; j < poly_size; ++j) {
                        z_poly[w*poly_size+j].c0 = (z_poly[w*poly_size+j].c0+poly_buf[poly_index+j].c0)%modulus64;
                        z_poly[w*poly_size+j].c1 = (z_poly[w*poly_size+j].c1+poly_buf[poly_index+j].c1)%modulus64;
                    }
                }
            }
        }
    }
}

// Compute (Tr(z), Tr(z*K))
void trace_SPDZ2k_32_prod_FFT_polys(const struct Param *param, struct GR64 *z_poly, uint64_t *rlt) {
    const size_t poly_size = param->poly_size;
    const uint64_t modulus64 = param->modulus64;

    for (size_t i = 0; i < poly_size*DPF_MSG_LEN; ++i) {
        rlt[i] = (2*z_poly[i].c0-z_poly[i].c1)%modulus64;
    }
}

void free_SPDZ2k_32_prod_DPF_keys(const struct Param *param, struct Keys *keys) {
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