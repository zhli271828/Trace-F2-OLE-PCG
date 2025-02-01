#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "test.h"
#include "modular_test.h"
#include "trace_bench.h"
#include "dpf.h"
#include "prf.h"
#include "fft.h"
#include "utils.h"
#include "f4ops.h"

#define DPF_MSG_SIZE 8
#define RLT_NUM 2

double bench_trace_pcg(size_t n, size_t c, size_t t) {
   if (c > 4) {
        printf("ERROR: currently only implemented for c <= 4");
        exit(0);
    }
    struct Param *param = calloc(1, sizeof(struct Param));
    init_bench_params(param, n, c, t);
    struct FFT_Trace_A *fft_trace_a = calloc(1, sizeof(struct FFT_Trace_A));
    init_FFT_Trace_A(param, fft_trace_a);
    sample_a_and_tensor(param, fft_trace_a);

    struct Keys *keys1 = calloc(1, sizeof(struct Keys));
    struct Keys *keys2 = calloc(1, sizeof(struct Keys));
    // Step 1: Sample DPF keys for the cross product.
    sample_trace_DPF_keys(param, keys1, keys2);

    struct DPF_Output *dpf_output1 = calloc(1, sizeof(struct DPF_Output));
    struct DPF_Output *dpf_output2 = calloc(1, sizeof(struct DPF_Output));
    init_DPF_output(param, dpf_output1);
    init_DPF_output(param, dpf_output2);

    size_t poly_size = param->poly_size;
    uint32_t *poly_buf = calloc(poly_size, sizeof(uint32_t));

    // store the DPF evaluation result
    uint32_t *fft_u1 = calloc(poly_size, sizeof(uint32_t));
    uint32_t *fft_u2 = calloc(poly_size, sizeof(uint32_t));

    // trace multiplication results
    uint8_t **trace_mult_rlts = calloc(RLT_NUM, sizeof(uint8_t*));
    init_trace_mult_rlts(param, trace_mult_rlts, RLT_NUM);
    // final results
    uint8_t **final_mult_rlts = calloc(RLT_NUM, sizeof(uint8_t*));
    init_trace_mult_rlts(param, final_mult_rlts, RLT_NUM);

    clock_t time = clock();
    // Step 2: Evaluate all the DPFs to recover shares of the 2*c*c polynomials.
    // DPF is evaluated to fft_u1, fft_u2
    evaluate_DPF_to_FFT(param, keys1, dpf_output1, fft_u1);
    evaluate_DPF_to_FFT(param, keys2, dpf_output2, fft_u2);

    FFT_multiply_then_sum(param, fft_trace_a->fft_a2a, fft_u1, poly_buf, trace_mult_rlts[0]);
    FFT_multiply_then_sum(param, fft_trace_a->fft_a2, fft_u2, poly_buf, trace_mult_rlts[1]);

    // (1, zeta^2)=(1, 1+zeta)
    scalar_multiply_fft_f4(trace_mult_rlts[1], 0b11, final_mult_rlts[0], poly_size);
    // (zeta^2, zeta)=(1, zeta)
    scalar_multiply_fft_f4(trace_mult_rlts[1], 0b01, final_mult_rlts[1], poly_size);

    sum_poly(trace_mult_rlts[0], final_mult_rlts[0], final_mult_rlts[0], poly_size);
    trace_poly(final_mult_rlts[0], poly_size);
    
    sum_poly(trace_mult_rlts[1], final_mult_rlts[1], final_mult_rlts[1], poly_size);
    trace_poly(final_mult_rlts[1], poly_size);

    time = clock()-time;
    double time_taken = ((double)time) / (CLOCKS_PER_SEC / 1000.0); // ms
    printf("Eval time (total) %f ms\n", time_taken);
    printf("DONE\n\n");
    printf("Benchmarking PCG evaluation \n");

    free_DPF_output(dpf_output1);
    free_DPF_output(dpf_output2);
    free(poly_buf);
    free(fft_u1);
    free(fft_u2);
    free_trace_mult_rlts(trace_mult_rlts, RLT_NUM);
    free_trace_mult_rlts(final_mult_rlts, RLT_NUM);
    free_fft_trace_a(fft_trace_a);
    free_DPF_keys(param, keys1);
    free_DPF_keys(param, keys2);
    free(param);
    return time_taken;
}

// Step 1: Sample DPF keys for the cross product.
// For benchmarking purposes, we sample random DPF functions for a
// sufficiently large domain size to express a block of coefficients.
void sample_trace_DPF_keys(const struct Param* param, struct Keys *keys1, struct Keys *keys2) {

    // TODO: optimize DPF KeyGen as PRF key can be reused.
    sample_DPF_keys(param, keys1);
    sample_DPF_keys(param, keys2);
}

void free_DPF_keys(const struct Param* param, struct Keys *keys) {
    const size_t c = param->c;
    const size_t t = param->t;
    for (size_t i=0; i<c; ++i) {
        for (size_t j=0; j<c; ++j) {
            for (size_t k=0; k<t; ++k) {
                for (size_t l=0; l<t; ++l) {
                    size_t index = i*c*t*t+j*t*t+k*t+l;
                    free(keys->dpf_keys_A[index]);
                    free(keys->dpf_keys_B[index]);
                }
            }
        }
    }
    free(keys->dpf_keys_A);
    free(keys->dpf_keys_B);
    DestroyPRFKey(keys->prf_keys);
    free(keys);
}

void init_DPF_output(const struct Param *param, struct DPF_Output *dpf_output) {
    const size_t c = param->c;
    const size_t packed_poly_size = param->packed_poly_size;
    const size_t dpf_block_size = param->dpf_block_size;

    // Allocate memory for the concatenated DPF outputs
    uint128_t *packed_polys = calloc(c*c*packed_poly_size, sizeof(uint128_t));
    uint128_t *shares = calloc(dpf_block_size, sizeof(uint128_t));
    uint128_t *cache = calloc(dpf_block_size, sizeof(uint128_t));

    dpf_output->packed_polys = packed_polys;
    dpf_output->shares = shares;
    dpf_output->cache = cache;
}

void free_DPF_output(struct DPF_Output *dpf_output) {
    free(dpf_output->packed_polys);
    free(dpf_output->shares);
    free(dpf_output->cache);
    free(dpf_output);
}

// void init_output_FFT(const struct Param *param, struct Output_FFT *output_fft) {
//     const size_t poly_size = param->poly_size;
//     // Allocate memory for the final inner product
//     uint8_t *z_poly = calloc(poly_size, sizeof(uint8_t));
//     uint32_t *res_poly_mat = calloc(poly_size, sizeof(uint32_t));

//     // output_fft->fft_u = fft_u;
//     output_fft->z_poly = z_poly;
//     output_fft->res_poly_mat = res_poly_mat;
// }

// void free_output_FFT(struct Output_FFT *output_fft) {

//     // free(output_fft->fft_u);
//     free(output_fft->z_poly);
//     free(output_fft->res_poly_mat);
//     free(output_fft);
// }

// Given the DPF keys, output the FFT form evaluation result.
void evaluate_DPF_to_FFT(const struct Param *param, const struct Keys *keys, struct DPF_Output *dpf_output, uint32_t *fft_u) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t poly_size = param->poly_size;
    const size_t block_size = param->block_size;
    const size_t packed_poly_size = param->packed_poly_size;
    const size_t packed_block_size = param->packed_block_size;
    
    uint128_t *packed_polys = dpf_output->packed_polys;
    uint128_t *shares = dpf_output->shares;
    uint128_t *cache = dpf_output->cache;
    // uint32_t *fft_u = output_fft->fft_u;
    
    // memset(packed_polys, 0, c*c*packed_poly_size*16);
    for (size_t i=0; i<c; ++i) {
        for (size_t j=0; j<c; ++j) {
            const size_t poly_index=i*c+j;
            uint128_t *packed_poly=&packed_polys[poly_index*packed_poly_size];

            for(size_t k=0; k<t; ++k) {
                uint128_t *poly_block = &packed_poly[k*packed_block_size];

                for (size_t l=0; l<t; ++l) {
                    size_t key_index = poly_index*t*t+k*t+l;
                    struct DPFKey *dpf_key = keys->dpf_keys_A[key_index];
                    DPFFullDomainEval(dpf_key, cache, shares);
                    
                    for (size_t w=0; w<packed_block_size; ++w) {
                        poly_block[w] ^= shares[w];
                    }
                }
            }
        }
    }
    // size_t block_idx = 0;
    // size_t coeff_idx = 0;
    // size_t packed_coeff_idx = 0;
    // uint128_t packed_coeff = 0;
    for (size_t i=0; i<c*c; ++i) {
        size_t poly_index = i*packed_poly_size;
        const uint128_t *poly = &packed_polys[poly_index];
        __builtin_prefetch(&poly[0], 0, 3);
        size_t block_idx = 0;
        size_t coeff_idx = 0;
        size_t packed_coeff_idx = 0;
        uint128_t packed_coeff = 0;

        for (size_t k=0; k<poly_size-64; k=k+64) {
            packed_coeff = poly[block_idx*packed_block_size+packed_coeff_idx];
            __builtin_prefetch(&fft_u[k], 0, 0);
            __builtin_prefetch(&fft_u[k], 1, 0);

            for (size_t l=0; l<64; ++l) {
                packed_coeff = packed_coeff >> 2;
                fft_u[k + l] |= packed_coeff & 0b11;
                fft_u[k + l] = fft_u[k + l] << 2;
            }
            ++packed_coeff_idx;
            coeff_idx=coeff_idx+64;
            if (coeff_idx>block_size) {
                coeff_idx = 0;
                ++block_idx;
                packed_coeff_idx = 0;
                __builtin_prefetch(&poly[block_idx * packed_block_size], 0, 2);
            }
        }
        packed_coeff = poly[block_idx*packed_block_size+packed_coeff_idx];
        for (size_t k=poly_size-64+1; k<poly_size; ++k) {
            packed_coeff = packed_coeff>>2;
            fft_u[k] |= packed_coeff & 0b11;
            fft_u[k] = fft_u[k] << 2;
        }
    }
    fft_recursive_uint32(fft_u, param->n, poly_size / 3);
}

uint32_t pack_to_uint32(const size_t size, const uint32_t a) {
    uint32_t rlt = 0;
    for (size_t j = 0; j < size; j++) {
        rlt ^= ((a&0b11)<<(2*j));
    }
    return rlt;
}

void FFT_multiply_then_sum(const struct Param *param, const uint32_t *fft_a, const uint32_t *fft_u, uint32_t *res_poly_mat, uint8_t *z_poly) {
    const size_t c = param->c;
    const size_t poly_size = param->poly_size;

    multiply_fft_32(fft_a, fft_u, res_poly_mat, poly_size);
    for (size_t j = 0; j < c * c; j++) {
        for (size_t i = 0; i < poly_size; i++) {
            z_poly[i] ^= (res_poly_mat[i] >> (2 * j)) & 0b11;
        }
    }
}

void init_FFT_Trace_A(const struct Param *param, struct FFT_Trace_A *fft_trace_a) {
    size_t poly_size = param->poly_size;

    uint8_t *fft_a = calloc(poly_size, sizeof(uint8_t));;
    // ai^2 x aj
    uint32_t *fft_a2a = calloc(poly_size, sizeof(uint32_t));
    // ai x aj
    uint32_t *fft_a2 = calloc(poly_size, sizeof(uint32_t));
    
    fft_trace_a->fft_a = fft_a;
    fft_trace_a->fft_a2a = fft_a2a;
    fft_trace_a->fft_a2 = fft_a2;
}

void init_FFT_Trace_basis(const struct Param *param, struct FFT_Trace_Basis *fft_trace_basis) {
    size_t poly_size = param->poly_size;

    // ai^2 x aj
    fft_trace_basis->fft_a2a = calloc(poly_size, sizeof(uint32_t));
    // ai x aj
    fft_trace_basis->fft_a2 = calloc(poly_size, sizeof(uint32_t));
}

// samples the a polynomials and tensor axa polynomials
void sample_a_and_tensor(const struct Param *param, struct FFT_Trace_A *fft_trace_a) {

    const size_t poly_size = param->poly_size;
    const size_t c = param->c;

    uint8_t *fft_a = fft_trace_a->fft_a;
    uint32_t *fft_a2a = fft_trace_a->fft_a2a;
    uint32_t *fft_a2 = fft_trace_a->fft_a2;

    RAND_bytes((uint8_t *)fft_a, sizeof(uint8_t) * poly_size);

    // make a_0 the identity polynomial (in FFT space) i.e., all 1s
    for (size_t i = 0; i < poly_size; i++) {
        fft_a[i] = fft_a[i] >> 2;
        fft_a[i] = fft_a[i] << 2;
        fft_a[i] |= 1;
    }
    // FOR DEBUGGING: set fft_a to the identity
    // for (size_t i = 0; i < poly_size; i++)
    // {
    //     fft_a[i] = (0xaaaa >> 1);
    // }
    for (size_t j = 0; j < c; j++) {
        for (size_t k = 0; k < c; k++) {
            for (size_t i = 0; i < poly_size; i++) {
                uint8_t u = (fft_a[i] >> (2 * j)) & 0b11;
                uint8_t v = (fft_a[i] >> (2 * k)) & 0b11;
                uint32_t w = mult_f4(u,v);
                uint32_t prod = mult_f4(u, w);
                size_t slot = j * c + k;
                fft_a2[i] |= w<<(2*slot);
                fft_a2a[i] |= prod <<(2*slot);
            }
        }
    }
    printf("Done with sampling the public values\n");
}

void free_fft_trace_a(struct FFT_Trace_A *fft_trace_a) {
    free(fft_trace_a->fft_a);
    free(fft_trace_a->fft_a2a);
    free(fft_trace_a->fft_a2);
    free(fft_trace_a);
}

void free_fft_trace_basis(struct FFT_Trace_Basis *fft_trace_basis) {
    free(fft_trace_basis->fft_a2a);
    free(fft_trace_basis->fft_a2);
}

void init_trace_mult_rlts(const struct Param *param, uint8_t **trace_mult_rlts, size_t size) {

    size_t poly_size = param->poly_size;
    for (size_t i = 0; i < size; i++) {
        trace_mult_rlts[i] = calloc(poly_size, sizeof(uint8_t));
    }
}

void free_trace_mult_rlts(uint8_t **trace_mult_rlts, size_t size) {
    for (size_t i = 0; i < size; i++) {
        free(trace_mult_rlts[i]);
        trace_mult_rlts[i] = NULL;
    }
    free(trace_mult_rlts);
}