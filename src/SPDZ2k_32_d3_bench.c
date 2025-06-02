#include <stdint.h>
#include <stdlib.h>

#include "common.h"
#include "dpf.h"
#include "fft.h"
#include "utils.h"
#include "modular_bench.h"
#include "modular_test.h"
#include "gr64_bench.h"
#include "gr64_trace_bench.h"
#include "SPDZ2k_32_d3_bench.h"


// Indicates the num for value and the MAC
#define MSG_NUM 2
// #define DPF_MSG_LEN 2
// Indicates the num of value and MAC in DPF
#define DPF_MSG_NUM (sizeof(struct GR64_D3)*MSG_NUM/sizeof(uint128_t))

void SPDZ2k_32_D3_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time) {
    clock_t start_time = clock();
    struct Param *param = xcalloc(1, sizeof(struct Param));
    const size_t m = 3;
    const size_t k = 32;
    const size_t s = 32;
    const size_t base = 7;
    printf("k=%zu, s=%zu\n", k, s);
    struct GR64_D3 zeta_powers[base];
    memset(zeta_powers, 0, base*sizeof(struct GR64_D3));
    struct GR64_D3 **frob_powers = xcalloc(m, sizeof(void *));
    struct GR64_D3 **power_scalar = xcalloc(m, sizeof(void *));
    for (size_t i = 0; i < m; ++i) {
        frob_powers[i] = xcalloc(m, sizeof(struct GR64_D3));
        power_scalar[i] = xcalloc(m, sizeof(struct GR64_D3));
    }

    compute_zeta_powers(zeta_powers, base);
    compute_Frob_map_zeta_powers(zeta_powers, frob_powers, m);
    compute_power_scalar(zeta_powers, frob_powers, power_scalar, m);

    init_SPDZ2k_32_HD_bench_params(param, n, c, t, m, k, s);
    size_t dpf_block_size = param->dpf_block_size;
    size_t poly_size = param->poly_size;
    size_t block_size = param->block_size;
    if (t*t*dpf_block_size == poly_size && block_size*t == poly_size) {
        printf("OK\n");
    } else {
        printf("Incorrect\n");
        exit(-1);
    }

    struct FFT_GR64_D3_Trace_A *fft_gr64_d3_trace_a = xcalloc(1, sizeof(struct FFT_GR64_D3_Trace_A));
    init_FFT_GR64_d3_Trace_A(param, fft_gr64_d3_trace_a);
    sample_SPDZ2k_32_d3_a_and_tensor(param, fft_gr64_d3_trace_a);

    /**
     * 1 init the memory and DPF for b
     * 2 init the memory  and DPF for product
     * 3 run the DPF evaluation
     */
    struct SPDZ2k_32_D3_b *spdz2k_32_d3_b = xcalloc(1, sizeof(struct SPDZ2k_32_D3_b));
    init_SPDZ2k_32_d3_b(param, spdz2k_32_d3_b);

    struct SPDZ2k_32_D3_Prod *spdz2k_32_d3_prod = xcalloc(1, sizeof(struct SPDZ2k_32_D3_Prod));
    init_SPDZ2k_32_d3_prod(param, spdz2k_32_d3_prod);

    printf("Benchmarking PCG evaluation\n");
    clock_t start_expand_time = clock();
    run_SPDZ2k_32_d3_b(param, spdz2k_32_d3_b, fft_gr64_d3_trace_a->fft_a, zeta_powers);
    run_SPDZ2k_32_d3_prod(param, spdz2k_32_d3_prod, fft_gr64_d3_trace_a->fft_a_tensor, power_scalar, zeta_powers);

    double end_expand_time = clock();
    double time_taken = ((double)(end_expand_time - start_expand_time)) / (CLOCKS_PER_SEC / 1000.0); // ms
    printf("Eval time (total) %f ms\n", time_taken);
    printf("DONE Benchmarking PCG evaluation\n\n");
    free_SPDZ2k_32_d3_prod(param, spdz2k_32_d3_prod);
    free_SPDZ2k_32_d3_b(param, spdz2k_32_d3_b);
    free_FFT_GR64_d3_Trace_A(param, fft_gr64_d3_trace_a);
    free(param);
    for (size_t i = 0; i < m; ++i) {
        free(frob_powers[i]);
        free(power_scalar[i]);
    }
    free(frob_powers);
    free(power_scalar);

    clock_t end_time = clock();
    pcg_time->pp_time = ((double)(start_expand_time-start_time))/(CLOCKS_PER_SEC/1000.0);
    pcg_time->expand_time = time_taken;
    pcg_time->total_time = ((double)(end_time-start_time))/(CLOCKS_PER_SEC / 1000.0);

}

void free_SPDZ2k_32_d3_b_DPF_keys(const struct Param *param, struct KeysHD *keys) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t m = param->m;
    const size_t base = param->base;
    for (size_t i=0; i<c; ++i) {
        for (size_t j = 0; j < t; ++j) {
            size_t index = i*t+j;
            free(keys->dpf_keys_A[index]);
            free(keys->dpf_keys_B[index]);
        }
    }
    free(keys->dpf_keys_A);
    free(keys->dpf_keys_B);
    DestroyPRFKeyZ(keys->prf_keys, base);
    free(keys);
}

void free_SPDZ2k_32_d3_b(const struct Param *param, struct SPDZ2k_32_D3_b *spdz2k_32_d3_b) {
    free_SPDZ2k_32_d3_b_DPF_keys(param, spdz2k_32_d3_b->keys_b0);
    free_SPDZ2k_32_d3_b_DPF_keys(param, spdz2k_32_d3_b->keys_b1);
    for (size_t i = 0; i < param->m; ++i) {
        free(spdz2k_32_d3_b->b0[i]);
        free(spdz2k_32_d3_b->b1[i]);
        free(spdz2k_32_d3_b->bm0[i]);
        free(spdz2k_32_d3_b->bm1[i]);
    }
    free(spdz2k_32_d3_b->b0);
    free(spdz2k_32_d3_b->b1);
    free(spdz2k_32_d3_b->bm0);
    free(spdz2k_32_d3_b->bm1);
    
    for (size_t i = 0; i < MSG_NUM; ++i) {
        free(spdz2k_32_d3_b->polys[i]);
        free(spdz2k_32_d3_b->poly_buf[i]);
        free(spdz2k_32_d3_b->z_poly[i]);
    }
    free(spdz2k_32_d3_b->polys);
    free(spdz2k_32_d3_b->poly_buf);
    free(spdz2k_32_d3_b->z_poly);

    free(spdz2k_32_d3_b->shares);
    free(spdz2k_32_d3_b->cache);
    free(spdz2k_32_d3_b);
}

void free_SPDZ2k_32_d3_prod_DPF_keys(const struct Param *param, struct KeysHD *keys) {
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

void free_SPDZ2k_32_d3_prod(const struct Param *param, struct SPDZ2k_32_D3_Prod *spdz2k_32_d3_prod) {
    free_SPDZ2k_32_d3_prod_DPF_keys(param, spdz2k_32_d3_prod->keys);

    for (size_t i = 0; i < MSG_NUM; ++i) {
        free(spdz2k_32_d3_prod->polys[i]);
        free(spdz2k_32_d3_prod->poly_buf[i]);
        free(spdz2k_32_d3_prod->z_poly[i]);
    }
    free(spdz2k_32_d3_prod->polys);
    free(spdz2k_32_d3_prod->poly_buf);
    free(spdz2k_32_d3_prod->z_poly);
    for (size_t i = 0; i < param->m; ++i) {
        free(spdz2k_32_d3_prod->rlt[i]);
    }
    free(spdz2k_32_d3_prod->rlt);
    free(spdz2k_32_d3_prod->shares);
    free(spdz2k_32_d3_prod->cache);
    free(spdz2k_32_d3_prod);
}

void run_SPDZ2k_32_d3_prod(const struct Param *param, struct SPDZ2k_32_D3_Prod *spdz2k_32_d3_prod, struct GR64_D3 **fft_a_tensor, struct GR64_D3 **power_scalar, const struct GR64_D3 *zeta_powers) {
    size_t poly_size = param->poly_size;
    struct KeysHD *keys = spdz2k_32_d3_prod->keys;

    uint64_t **rlt = spdz2k_32_d3_prod->rlt;
    
    struct GR64_D3 **polys = spdz2k_32_d3_prod->polys;
    struct GR64_D3 **poly_buf = spdz2k_32_d3_prod->poly_buf;

    struct GR64_D3 **z_poly = spdz2k_32_d3_prod->z_poly;
    uint128_t *shares = spdz2k_32_d3_prod->shares;
    uint128_t *cache = spdz2k_32_d3_prod->cache;

    // evaluate for z and K*z together
    evaluate_SPDZ2k_32_d3_prod_DPF_and_sum(param, keys, fft_a_tensor, polys, poly_buf, z_poly, shares, cache, power_scalar, zeta_powers);
    // map to trace functions
    trace_SPDZ2k_32_d3_prod_FFT_polys(param, z_poly, rlt);
}

void trace_SPDZ2k_32_d3_prod_FFT_polys(const struct Param *param, struct GR64_D3 **z_poly, uint64_t **rlt) {
    const size_t poly_size = param->poly_size;
    const size_t m = param->m;
    const uint64_t modulus64 = param->modulus64;

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < poly_size; ++j) {
            rlt[i][j] = trace_GR64_d3(&z_poly[0][i*poly_size+j], m);
            rlt[i][j+poly_size] = trace_GR64_d3(&z_poly[1][i*poly_size+j], m);
        }
    }
}

/**
 * 1 Evaluate DPF to poly
 * 2 Convert poly to FFT
 * 3 Multiply with a
 * 4 Sum up the result
 */
// evaluate DPF to poly, convert poly to FFT, multiply with a and sum up
void evaluate_SPDZ2k_32_d3_prod_DPF_and_sum(
    const struct Param *param,
    const struct KeysHD *keys,
    struct GR64_D3 **fft_a_tensor_maps,
    struct GR64_D3 **polys,
    struct GR64_D3 **poly_buf,
    struct GR64_D3 **z_poly,
    uint128_t *shares,
    uint128_t *cache,
    struct GR64_D3 **power_scalar,
    const struct GR64_D3 *zeta_powers
) {

    evaluate_SPDZ2k_32_d3_prod_DPF(param, keys, polys, shares, cache);
    // TODO: add the powers
    convert_SPDZ2k_32_d3_prod_to_FFT(param, polys, zeta_powers);
    multiply_SPDZ2k_32_d3_prod_FFT(param, fft_a_tensor_maps, polys, poly_buf);
    sum_SPDZ2k_32_d3_prod_FFT_polys(param, poly_buf, z_poly, power_scalar);
}

void sum_SPDZ2k_32_d3_prod_FFT_polys(const struct Param *param, struct GR64_D3 **poly_buf, struct GR64_D3 **z_poly, struct GR64_D3 **power_scalar) {
    const size_t c = param->c;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;
    const uint64_t modulus64 = param->modulus64;

    // c*c*m
    for (size_t i = 0; i < c; i++) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t k = 0; k < m; ++k) {
                size_t index = (i*c+j)*m+k;
                for (size_t l = 0; l < m; ++l) {

                    add_SPDZ2k_32_d3_list_scalar(&poly_buf[0][index*poly_size], &power_scalar[k][l], &z_poly[0][l*poly_size], &z_poly[0][l*poly_size], poly_size, modulus64);
                    add_SPDZ2k_32_d3_list_scalar(&poly_buf[1][index*poly_size], &power_scalar[k][l], &z_poly[1][l*poly_size], &z_poly[1][l*poly_size], poly_size, modulus64);
                }
            }
        }
    }
}

void compute_zeta_powers(struct GR64_D3 *zeta_powers, const size_t m) {
    struct GR64_D3 zeta = {0};
    zeta.c0 = 0; zeta.c1=1; zeta.c2=0;
    // compute zeta_powers
    // struct GR64_D3 zeta_powers[m];
    zeta_powers[0].c0 = 1; 
    zeta_powers[0].c1 = 0;
    zeta_powers[0].c2 = 0;

    for (size_t i = 1; i<m; ++i) {
        mult_gr64_D3(&zeta, &zeta_powers[i-1], &zeta_powers[i]);
    }
}

void compute_Frob_map_zeta_powers(const struct GR64_D3 *powers, struct GR64_D3 **frob_powers, const size_t m) {

    memcpy(&frob_powers[0][0], powers, sizeof(struct GR64_D3)*m);
    for (size_t i = 1; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            compute_Frob_gr64_d3(&frob_powers[i-1][j], &frob_powers[i][j]);
        }
    }
}

void compute_power_scalar(const struct GR64_D3 *zeta_powers, struct GR64_D3 **frob_powers, struct GR64_D3 **power_scalar, const size_t m) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            mult_gr64_D3(&frob_powers[i][j], &zeta_powers[j], &power_scalar[i][j]);
        }
    }
}

void add_SPDZ2k_32_d3_list_scalar(const struct GR64_D3 *a, const struct GR64_D3 *scalar, const struct GR64_D3 *b, struct GR64_D3 *t, size_t length, const size_t modulus) {
    struct GR64_D3 mult_rlt = {0};
    for (size_t i = 0; i < length; i++) {
        mult_gr64_D3(scalar, &a[i], &mult_rlt);
        add_SPDZ2k_32_d3(&mult_rlt, &b[i], &t[i], modulus);
    }
}

// Convert each DPF output polynomial to FFT
void convert_SPDZ2k_32_d3_prod_to_FFT(const struct Param *param, struct GR64_D3 **polys, const struct GR64_D3 *zeta_powers) {
    const size_t c = param->c;
    const size_t n = param->n;
    const size_t m = param->m;
    const size_t base = param->base;
    const size_t poly_size = param->poly_size;
    const uint64_t modulus64 = param->modulus64;
    
    for (size_t i = 0; i < c*c*m; ++i) {

        struct GR64_D3 *poly0 = &polys[0][i*poly_size];
        struct GR64_D3 *poly1 = &polys[1][i*poly_size];
        fft_recursive_SPDZ2k_32_D3(poly0, zeta_powers, n, poly_size / base, modulus64, base);
        fft_recursive_SPDZ2k_32_D3(poly1, zeta_powers, n, poly_size / base, modulus64, base);
    }
}

/**
 * Multiply the polynomial a and the DPF output polynomial.
 * The polynomial a is reused.
 * @param a_polys
 * @param b_poly 
 * @param res_poly the result polynomial of length c*c*m*DPF_MSG_LEN*poly_size
 */
void multiply_SPDZ2k_32_d3_prod_FFT(const struct Param *param, struct GR64_D3 **a_polys, struct GR64_D3 **b_poly, struct GR64_D3 **res_poly) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;
    uint64_t modulus64 = param->modulus64;
    // b_poly and res_poly are of length DPF_MSG_LEN*c*c*m*poly_size

    // TODO: ensure a_tensor and poly_b are in the same order c*c*m
    for (size_t i = 0; i < c*c*m; ++i) {
        mult_SPDZ2k_32_d3_list(a_polys[i], &b_poly[0][i*poly_size], &res_poly[0][i*poly_size], poly_size, modulus64);
        mult_SPDZ2k_32_d3_list(a_polys[i], &b_poly[1][i*poly_size], &res_poly[1][i*poly_size], poly_size, modulus64);
    }
}

/**
 * Evaluate the DPF keys to polynomials
 * @param polys is of length c*c*poly_size*DPF_MSG_LEN=c*c*block_size*DPF_MSG_LEN=c*c*t*t*dpf_block_size*DPF_MSG_LEN
 * @param shares is of length dpf_block_size*DPF_MSG_LEN*DPF_MSG_NUM
 * @param cache is of length dpf_block_size*DPF_MSG_LEN*DPF_MSG_NUM
 */
void evaluate_SPDZ2k_32_d3_prod_DPF(const struct Param *param, const struct KeysHD *keys, struct GR64_D3 **polys, uint128_t *shares, uint128_t *cache) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t m = param->m;
    const size_t dpf_block_size = param->dpf_block_size;
    const size_t base = param->base;

    for (size_t i = 0; i < c; i++) {
        for (size_t j = 0; j < c; j++) {
            for (size_t w = 0; w < m; ++w) {
                for (size_t k = 0; k < t; k++) {
                    for (size_t l = 0; l < t; l++) {
                        const size_t key_index = (((i*c+j)*m+w)*t+k)*t+l;
                        struct DPFKeyZ *dpf_key = keys->dpf_keys_A[key_index];
                        struct GR64_D3 *poly_block0 = &polys[0][key_index*dpf_block_size];
                        struct GR64_D3 *poly_block1 = &polys[1][key_index*dpf_block_size];

                        // DPF evaluates the value and the MAC in consecutive positions and copy_gr64_d3_block parses the DPF evaluation results
                        DPFFullDomainEvalZ(base, dpf_key, cache, shares);
                        copy_gr64_d3_block(poly_block0, poly_block1, shares, dpf_block_size);
                    }
                }
            }
        }
    }
}

void init_SPDZ2k_32_d3_b(const struct Param *param, struct SPDZ2k_32_D3_b *spdz2k_32_d3_b) {

    size_t c = param->c;
    size_t t = param->t;
    size_t m = param->m;
    size_t dpf_block_size = param->dpf_block_size;
    size_t poly_size = param->poly_size;
    size_t block_size = param->block_size;
    // generate keys for b0, b1
    struct KeysHD *keys_b0 = xcalloc(1, sizeof(struct KeysHD));
    sample_SPDZ2k_32_d3_b_DPF_keys(param, keys_b0);
    struct KeysHD *keys_b1 = xcalloc(1, sizeof(struct KeysHD));
    sample_SPDZ2k_32_d3_b_DPF_keys(param, keys_b1);
    // output for (Tr(zeta^j*b0), Tr(zeta^j*b1)) and each has m values for the trace
    uint64_t **b0 = xcalloc(m, sizeof(void*));
    uint64_t **b1 = xcalloc(m, sizeof(void*));
    // output for (Tr(zeta^j*K*b0), Tr(zeta^j*K*b1)) and each has m values for the trace
    uint64_t **bm0 = xcalloc(m, sizeof(void*));
    uint64_t **bm1 = xcalloc(m, sizeof(void*));
   
    for (size_t i = 0; i < m; ++i) {
        b0[i] = xcalloc(poly_size, sizeof(uint64_t));
        b1[i] = xcalloc(poly_size, sizeof(uint64_t));
        bm0[i] = xcalloc(poly_size, sizeof(uint64_t));
        bm1[i] = xcalloc(poly_size, sizeof(uint64_t));
    }

    // 2*c*poly_size for the secret of (b,K*b) and each has c parts.
    // For (b, K*b) with dimension MSG_NUM*(c*poly_size)
    struct GR64_D3 **polys = xcalloc(MSG_NUM, sizeof(void*));
    struct GR64_D3 **poly_buf = xcalloc(MSG_NUM, sizeof(void*));
    // For b and K*b, with dimension MSG_NUM x poly_size
    struct GR64_D3 **z_poly = xcalloc(MSG_NUM, sizeof(void*));
    
    for (size_t i = 0; i < MSG_NUM; ++i) {
        polys[i] = xcalloc(c*poly_size, sizeof(struct GR64_D3));
        poly_buf[i] = xcalloc(c*poly_size, sizeof(struct GR64_D3));
        z_poly[i] = xcalloc(poly_size, sizeof(struct GR64_D3));
    }

    uint128_t *shares = xcalloc(DPF_MSG_NUM*block_size, sizeof(uint128_t));
    uint128_t *cache = xcalloc(DPF_MSG_NUM*block_size, sizeof(uint128_t));

    spdz2k_32_d3_b->keys_b0 = keys_b0;
    spdz2k_32_d3_b->keys_b1 = keys_b1;
    
    spdz2k_32_d3_b->b0 = b0;
    spdz2k_32_d3_b->b1 = b1;
    spdz2k_32_d3_b->bm0 = bm0;
    spdz2k_32_d3_b->bm1 = bm1;

    spdz2k_32_d3_b->polys = polys;
    spdz2k_32_d3_b->poly_buf = poly_buf;
    spdz2k_32_d3_b->z_poly = z_poly;

    spdz2k_32_d3_b->shares = shares;
    spdz2k_32_d3_b->cache = cache;
}

// sample DPF keys for b0 and b1 with authentication
void sample_SPDZ2k_32_d3_b_DPF_keys(const struct Param *param, struct KeysHD *keys) {
    size_t c = param->c;
    size_t t = param->t;
    size_t block_size = param->block_size;
    size_t block_bits = param->block_bits;
    uint64_t modulus64 = param->modulus64;
    const size_t m = param->m;
    const size_t base = param->base;

    struct DPFKeyZ **dpf_keys_A = xmalloc(c*t*sizeof(void *));
    struct DPFKeyZ **dpf_keys_B = xmalloc(c*t*sizeof(void *));
    struct PRFKeysZ *prf_keys = xmalloc(sizeof(struct PRFKeysZ));
    PRFKeyGenZ(prf_keys, base);
    for (size_t i=0; i<c; ++i) {
        for (size_t j=0; j<t; ++j) {
            size_t index = i*t+j;
            // Pick a random position for benchmarking purposes
            size_t alpha = random_index(block_size);

            struct GR64_D3 beta[MSG_NUM];
            // randomize the value
            RAND_bytes((uint8_t *)beta, sizeof(struct GR64_D3));
            // compute the MAC
            uint64_t K = param->K64;
            beta[1].c0 = beta[0].c0*K;
            beta[1].c1 = beta[0].c1*K;
            beta[1].c2 = beta[0].c2*K;

            // DPF keys
            struct DPFKeyZ *kA = xmalloc(sizeof(struct DPFKeyZ));
            struct DPFKeyZ *kB = xmalloc(sizeof(struct DPFKeyZ));
            // Now the DPF keys has two elements
            DPFGenZ(base, prf_keys, block_bits, alpha, (uint128_t *)beta, DPF_MSG_NUM, kA, kB);
            dpf_keys_A[index] = kA;
            dpf_keys_B[index] = kB;
        }
    }
    keys->dpf_keys_A = dpf_keys_A;
    keys->dpf_keys_B = dpf_keys_B;
    keys->prf_keys = prf_keys;
}

/**
 * Init the memory for products and sample the DPF keys.
 */
void init_SPDZ2k_32_d3_prod(const struct Param *param, struct SPDZ2k_32_D3_Prod *spdz2k_32_d3_prod) {

    size_t c = param->c;
    size_t t = param->t;
    size_t m = param->m;
    size_t dpf_block_size = param->dpf_block_size;
    size_t poly_size = param->poly_size;
    struct KeysHD *keys = xcalloc(1, sizeof(struct KeysHD));
    sample_SPDZ2k_32_d3_prod_DPF_keys(param, keys);

    // MSG_NUM indicates the value z and z*K
    // m indicates the number of automorphisms
    uint64_t **rlt = xcalloc(m, sizeof(void *));
    for (size_t i = 0; i < m; ++i) {
        // for Tr(z) and Tr(z*K)
        rlt[i] = xcalloc(MSG_NUM*t*t*dpf_block_size, sizeof(uint64_t));
    }
    spdz2k_32_d3_prod->rlt = rlt;

    struct GR64_D3 **z_poly = xcalloc(MSG_NUM, sizeof(void*));
    for (size_t i = 0; i < MSG_NUM; ++i) {
        z_poly[i] = xcalloc(m*t*t*dpf_block_size, sizeof(struct GR64_D3));
    }
    /**
     * TODO: check for the number of polynomials
     * @param m is for the number of automorphisms
     * @param MSG_NUM is for the number of outputs
     */
    struct GR64_D3 **polys = xcalloc(MSG_NUM, sizeof(void*));
    struct GR64_D3 **poly_buf = xcalloc(MSG_NUM, sizeof(void*));
    for (size_t i = 0; i < MSG_NUM; ++i) {
        polys[i] = xcalloc(c*c*t*t*m*dpf_block_size, sizeof(struct GR64_D3));
        poly_buf[i] = xcalloc(c*c*t*t*m*dpf_block_size, sizeof(struct GR64_D3));
    }

    uint128_t *shares = xcalloc(DPF_MSG_NUM*dpf_block_size, sizeof(uint128_t));
    uint128_t *cache = xcalloc(DPF_MSG_NUM*dpf_block_size, sizeof(uint128_t));

    spdz2k_32_d3_prod->keys = keys;
    spdz2k_32_d3_prod->polys = polys;
    spdz2k_32_d3_prod->poly_buf = poly_buf;
    spdz2k_32_d3_prod->z_poly = z_poly;
    spdz2k_32_d3_prod->rlt = rlt;
    spdz2k_32_d3_prod->shares = shares;
    spdz2k_32_d3_prod->cache = cache;
}

// sample DPF keys for z and z*K
void sample_SPDZ2k_32_d3_prod_DPF_keys(const struct Param *param, struct KeysHD *keys) {
    size_t c = param->c;
    size_t t = param->t;
    // size_t block_size = param->block_size;
    size_t dpf_block_size = param->dpf_block_size;
    size_t dpf_domain_bits = param->dpf_domain_bits;
    uint64_t modulus64 = param->modulus64;
    const size_t m = param->m;
    const size_t base = param->base;

    struct DPFKeyZ **dpf_keys_A = xmalloc(c*c*m*t*t*sizeof(void *));
    struct DPFKeyZ **dpf_keys_B = xmalloc(c*c*m*t*t*sizeof(void *));
    struct PRFKeysZ *prf_keys = xmalloc(sizeof(struct PRFKeysZ));
    PRFKeyGenZ(prf_keys, base);
    // TODO: generate a_tensor in the same order
    for (size_t k=0; k < m; ++k) {
    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < c; ++j) {
            // for (size_t k=0; k < m; ++k) {
                for (size_t l = 0; l < t; ++l) {
                    for (size_t w = 0; w < t; ++w) {
                        size_t index = (((k*c+i)*c+j)*t+l)*t+w;
                        // size_t index = (((i*c+j)*m+k)*t+l)*t+w;
                        size_t alpha = random_index(dpf_block_size);

                        struct GR64_D3 beta[MSG_NUM];
                        // randomize the first value
                        RAND_bytes((uint8_t *)beta, sizeof(struct GR64_D3));
                        uint64_t K = param->K64;
                        beta[1].c0 = beta[0].c0*K;
                        beta[1].c1 = beta[0].c1*K;
                        beta[1].c2 = beta[0].c2*K;

                        // DPF keys
                        struct DPFKeyZ *kA = xmalloc(sizeof(struct DPFKeyZ));
                        struct DPFKeyZ *kB = xmalloc(sizeof(struct DPFKeyZ));
                        DPFGenZ(base, prf_keys, dpf_domain_bits, alpha, (uint128_t *)beta, DPF_MSG_NUM, kA, kB);
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

void run_SPDZ2k_32_d3_b(const struct Param *param, struct SPDZ2k_32_D3_b *spdz2k_32_d3_b, struct GR64_D3 **fft_a, const struct GR64_D3 *zeta_powers) {
    size_t poly_size = param->poly_size;
    struct KeysHD *keys_b0 = spdz2k_32_d3_b->keys_b0;
    struct KeysHD *keys_b1 = spdz2k_32_d3_b->keys_b1;
    uint64_t **b0 = spdz2k_32_d3_b->b0;
    uint64_t **b1 = spdz2k_32_d3_b->b1;
    
    uint64_t **bm0 = spdz2k_32_d3_b->bm0;
    uint64_t **bm1 = spdz2k_32_d3_b->bm1;
    
    struct GR64_D3 **polys = spdz2k_32_d3_b->polys;
    struct GR64_D3 **poly_buf = spdz2k_32_d3_b->poly_buf;
    struct GR64_D3 **z_poly = spdz2k_32_d3_b->z_poly;
    uint128_t *shares = spdz2k_32_d3_b->shares;
    uint128_t *cache = spdz2k_32_d3_b->cache;

    /**
     * evaluate for b0 and K*b0
     * 1 Evaluate DPF to poly
     * 2 Convert poly to FFT
     * 3 Multiply with a
     * 4 Sum up to obtain b
     * 5 Compute the trace for zeta^j*b
     */
    evaluate_SPDZ2k_32_d3_b_DPF_and_sum(param, keys_b0, fft_a, polys, poly_buf, z_poly, shares, cache, zeta_powers);
    trace_SPDZ2k_32_d3_b_FFT_polys(param, z_poly, b0, bm0, zeta_powers);

    // evaluate for b1 and K*b1
    memset(z_poly[0], 0, poly_size*sizeof(struct GR64_D3));
    memset(z_poly[1], 0, poly_size*sizeof(struct GR64_D3));

    evaluate_SPDZ2k_32_d3_b_DPF_and_sum(param, keys_b1, fft_a, polys, poly_buf, z_poly, shares, cache, zeta_powers);
    trace_SPDZ2k_32_d3_b_FFT_polys(param, z_poly, b1, bm1, zeta_powers);
}


/**
 * evaluate for b and K*b
 * 1 Evaluate DPF to poly
 * 2 Convert poly to FFT
 * 3 Multiply with a
 * 4 Sum up to obtain b
 */
// evaluate DPF to poly, convert poly to FFT, multiply with a and sum up
void evaluate_SPDZ2k_32_d3_b_DPF_and_sum(
    const struct Param *param,
    const struct KeysHD *keys,
    struct GR64_D3 **fft_a,
    struct GR64_D3 **polys,
    struct GR64_D3 **poly_buf,
    struct GR64_D3 **z_poly,
    uint128_t *shares,
    uint128_t *cache,
    const struct GR64_D3 *zeta_powers
) {

    evaluate_SPDZ2k_32_d3_b_DPF(param, keys, polys, shares, cache);
    convert_SPDZ2k_32_d3_b_to_FFT(param, polys, zeta_powers);
    multiply_SPDZ2k_32_d3_b_FFT(param, fft_a, polys, poly_buf);
    sum_SPDZ2k_32_d3_b_FFT_polys(param, poly_buf, z_poly);
}

/**
 * Evaluate the DPF keys to polynomials
 * @param polys is of length c*poly_size*2=c*t*block_size*2
 * @param shares should be of length 2*sizeof(struct GR64_D3)/sizeof(uint128_t)*block_size
 * @param cache should be of length 2*sizeof(struct GR64_D3)/sizeof(uint128_t)*block_size
 */
void evaluate_SPDZ2k_32_d3_b_DPF(const struct Param *param, const struct KeysHD *keys, struct GR64_D3 **polys, uint128_t *shares, uint128_t *cache) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t poly_size = param->poly_size;
    const size_t block_size = param->block_size;
    const size_t m = param->m;
    const size_t base = param->base;
    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < t; ++j) {
            const size_t poly_index = i*t+j;

            struct GR64_D3 *poly_block0 = &polys[0][poly_index*block_size];
            struct GR64_D3 *poly_block1 = &polys[1][poly_index*block_size];
            
            const size_t key_index = poly_index;
            struct DPFKeyZ *dpf_key = keys->dpf_keys_A[key_index];
            // DPF evaluates the value and the MAC in consecutive positions and copy_gr64_d3_block parses the DPF evaluation results
            DPFFullDomainEvalZ(base, dpf_key, cache, shares);
            // printf("After DPFFullDomainEvalZ polys[0]=%p, polys[1]=%p\n", polys[0], polys[1]);
            // printf("poly_block0=%p, poly_block1=%p\n", poly_block0, poly_block1);
            copy_gr64_d3_block(poly_block0, poly_block1, shares, block_size);
            // printf("After copy_gr64_d3_block\n");
        }
    }
}

// Convert each DPF output polynomial to FFT
void convert_SPDZ2k_32_d3_b_to_FFT(const struct Param *param, struct GR64_D3 **polys, const struct GR64_D3 *zeta_powers) {
    const size_t c = param->c;
    const size_t n = param->n;
    const size_t base = param->base;
    const size_t poly_size = param->poly_size;
    const uint64_t modulus64 = param->modulus64;

    for (size_t i = 0; i < c; ++i) {
        struct GR64_D3 *poly0 = &polys[0][i*poly_size];
        struct GR64_D3 *poly1 = &polys[1][i*poly_size];
        fft_recursive_SPDZ2k_32_D3(poly0, zeta_powers, n, poly_size / base, modulus64, base);
        fft_recursive_SPDZ2k_32_D3(poly1, zeta_powers, n, poly_size / base, modulus64, base);
    }
}

/**
 * Multiply the polynomial a and the DPF output polynomial.
 * The polynomial a is reused.
 * @param a_polys
 * @param b_poly 
 * @param res_poly the result polynomial of length c*poly_size*MSG_LEN
 */
void multiply_SPDZ2k_32_d3_b_FFT(const struct Param *param, struct GR64_D3 **a_polys, struct GR64_D3 **b_poly, struct GR64_D3 **res_poly) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;
    uint64_t modulus64 = param->modulus64;
    for (size_t i = 0; i < c; ++i) {
        mult_SPDZ2k_32_d3_list(a_polys[i], &b_poly[0][i*poly_size], &res_poly[0][i*poly_size], poly_size, modulus64);
        mult_SPDZ2k_32_d3_list(a_polys[i], &b_poly[1][i*poly_size], &res_poly[1][i*poly_size], poly_size, modulus64);
    }
}

void mult_SPDZ2k_32_d3_list(const struct GR64_D3 *a, const struct GR64_D3 *b, struct GR64_D3 *t, size_t length, const size_t modulus) {
    for (size_t i = 0; i < length; i++) {
        mult_SPDZ2k_32_d3(&a[i], &b[i], &t[i], modulus);
    }
}

/**
 * Sum up the c polynomials in poly_buf to z_poly
 * @param poly_buf is of dimension MSG_LEN*(c*poly_size)
 * @param z_poly is of dimension MSG_LEN*poly_size.
 */
void sum_SPDZ2k_32_d3_b_FFT_polys(const struct Param *param, struct GR64_D3 **poly_buf, struct GR64_D3 **z_poly) {
    const size_t c = param->c;
    const size_t poly_size = param->poly_size;
    const uint64_t modulus64 = param->modulus64;

    for (size_t i = 0; i < c; i++) {
        add_SPDZ2k_32_d3_list(&poly_buf[0][i*poly_size], z_poly[0], z_poly[0], poly_size, modulus64);

        add_SPDZ2k_32_d3_list(&poly_buf[1][i*poly_size], z_poly[1], z_poly[1], poly_size, modulus64);
    }
}

void add_SPDZ2k_32_d3_list(const struct GR64_D3 *a, const struct GR64_D3 *b, struct GR64_D3 *t, size_t length, const size_t modulus) {
    for (size_t i = 0; i < length; i++) {
        add_SPDZ2k_32_d3(&a[i], &b[i], &t[i], modulus);
    }
}

/**
 * Compute Tr(zeta^j*b) and Tr(zeta^j*K*b)
 * src is for b and K*b with dimension MSG_LEN*(c*poly_size) and each is of length poly_size.
 * b is for Tr(zeta^j*b) with total length m*poly_size
 * bm is for Tr(K*zeta^j*b) with total length m*poly_size
 */
void trace_SPDZ2k_32_d3_b_FFT_polys(const struct Param *param, struct GR64_D3 **src, uint64_t **b, uint64_t **bm, const struct GR64_D3 *powers) {
    const size_t poly_size = param->poly_size;
    const uint64_t modulus64 = param->modulus64;
    const size_t m = param->m;

    struct GR64_D3 mult_rlt = {0};
    // Tr(zeta^j*b)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < poly_size; ++j) {
            mult_gr64_D3(&powers[i], &src[0][j], &mult_rlt);
            b[i][j] = trace_GR64_d3(&mult_rlt, m);
        }
    }
    // Tr(zeta^j*K*b)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < poly_size; ++j) {
            mult_gr64_D3(&powers[i], &src[1][j], &mult_rlt);
            bm[i][j] = trace_GR64_d3(&mult_rlt, m);
        }
    }
}


void mult_SPDZ2k_32_d3(const struct GR64_D3 *a, const struct GR64_D3 *b, struct GR64_D3 *t, const uint64_t modulus) {
    mult_gr64_D3(a, b, t);
    if (modulus != 0) {
        t->c0 = t->c0 % modulus;
        t->c1 = t->c1 % modulus;
        t->c2 = t->c2 % modulus;
    }
}

/**
 * Parse each block of type uint128_t to GR64_D3. In fact, it converts each uint128_t to GR64_D3 for both the value and MACs.
 * @param poly_block is of length block_size
 * @param shares is of length DPF_MSG_NUM*block_size
 */
void copy_gr64_d3_block(struct GR64_D3 *poly_block0, struct GR64_D3 *poly_block1, uint128_t *shares, const size_t dpf_block_size) {
    for (size_t w = 0; w < dpf_block_size; ++w) {
        struct GR64_D3 *v = (struct GR64_D3*)&shares[w*DPF_MSG_NUM];
        memcpy(&poly_block0[w], &v[0], sizeof(struct GR64_D3));
        memcpy(&poly_block1[w], &v[1], sizeof(struct GR64_D3));
    }
}

// TODO: check the logic here
void sample_SPDZ2k_32_d3_a_and_tensor(const struct Param *param, struct FFT_GR64_D3_Trace_A *fft_gr64_d3_trace_a) {
    // call sample_gr64_trace_a_and_tensor to compute the relation first and then mod the special modulus
    uint64_t modulus64 = param->modulus64;
    struct GR64_D3 **fft_a = fft_gr64_d3_trace_a->fft_a;
    struct GR64_D3 **fft_a_tensor = fft_gr64_d3_trace_a->fft_a_tensor;
    const size_t poly_size = param->poly_size;
    const size_t c = param->c;
    const size_t m = param->m;

    for (size_t i = 1; i < c; ++i) {
        RAND_bytes((uint8_t *)fft_a[i], sizeof(struct GR64_D3) * poly_size);
    }
    // make first a the identity polynomial (in FFT space) i.e., all 1s
    for (size_t i = 0; i < poly_size; ++i) {
        fft_a[0][i].c0 = 1;
        fft_a[0][i].c1 = 0;
        fft_a[0][i].c2 = 0;
    }
    
    for (size_t i = 1; i < m; ++i) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t k = 0; k < poly_size; ++k) {
                struct GR64_D3 *prev = &fft_a[(i-1)*c+j][k];
                struct GR64_D3 *cur = &fft_a[i*c+j][k];
                compute_Frob_gr64_d3(prev, cur);
            }
        }
    }
    // TODO: revisit this: fft_a_tensor depends on the order of it.
    // compute the tensor of each
    for (size_t k = 0; k < m; ++k) {
        for (size_t i = 0; i < c; ++i) {
            for (size_t j = 0; j < c; ++j) {
                mult_gr64_D3_list(fft_a[i], fft_a[k*c+j], fft_a_tensor[(k*c+i)*c+j], poly_size);
            }
        }
    }
}

void compute_Frob_gr64_d3(const struct GR64_D3 *prev, struct GR64_D3 *cur) {
    // use primitive polynomial: X^3 + 17520588382079786918*X^2 + 17520588382079786917*X - 1
    const static uint64_t a = 17520588382079786918UL;
    cur->c0 = prev->c0-prev->c2*a;
    cur->c1 = -prev->c2;
    cur->c2 = prev->c1-prev->c2;
}