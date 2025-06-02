// This is a modular design of bench.c
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "test.h"
#include "modular_bench.h"
#include "dpf.h"
#include "prf.h"
#include "fft.h"
#include "utils.h"
#include "f4ops.h"

#define DPF_MSG_SIZE 8

void init_SPDZ2k_32_bench_params(struct Param* param, const size_t n, const size_t c, const size_t t, const size_t m, const size_t k, const size_t s) {
    param->k = k;
    param->s = s;
    // It is incorrect for 2^64
    param->modulus64 = 1<<(k+s);
    RAND_bytes((uint8_t *)&param->K64, sizeof(param->K64));
    param->K64 = param->K64%param->modulus64;
    init_gr64_bench_params(param, n, c, t, m);
}

void init_SPDZ2k_32_HD_bench_params(struct Param* param, const size_t n, const size_t c, const size_t t, const size_t m, const size_t k, const size_t s) {
    param->k = k;
    param->s = s;
    param->modulus64 = 1<<(k+s);
    RAND_bytes((uint8_t *)&param->K64, sizeof(param->K64));
    if (param->modulus64 != 0) {
        param->K64 = param->K64%param->modulus64;
    }
    init_gr_HD_bench_params(param, n, c, t, m);
}

void init_SPDZ2k_64_bench_params(struct Param* param, const size_t n, const size_t c, const size_t t, const size_t m, const size_t k, const size_t s) {
    param->k = k;
    param->s = s;
    param->modulus128 = 1<<(k+s);
    RAND_bytes((uint8_t *)&param->K128, sizeof(param->K128));
    param->K128 = param->K128%param->modulus128;
    init_gr128_bench_params(param, n, c, t, m);
}

void init_gr64_bench_params(struct Param* param, const size_t n, const size_t c, const size_t t, const size_t m) {
    init_gr_bench_params(param, n, c, t, m);
}
void init_gr128_bench_params(struct Param *param, const size_t n, const size_t c, const size_t t, const size_t m) {
    init_gr_bench_params(param, n, c, t, m);
}
// for GR64 and GR128
void init_gr_bench_params(struct Param *param, const size_t n, const size_t c, const size_t t, const size_t m) {

    param->n=n;
    param->c=c;
    param->t=t;
    param->m=m;
    size_t poly_size = ipow(3, n);
    size_t block_size = ceil(poly_size / t);
    printf("block_size = %zu \n", block_size);
    // size_t dpf_domain_bits = ceil(log_base(poly_size / (t*t), 3));
    size_t dpf_domain_bits = (size_t)(log_base(poly_size / (t*t), 3));
    printf("DPF domain bits %zu \n", dpf_domain_bits);
    size_t dpf_block_size = ipow(3, dpf_domain_bits);
    printf("dpf_block_size = %zu\n", dpf_block_size);
    size_t block_bits = (size_t)(log_base(poly_size/t, 3));
    printf("block_bits = %zu\n", block_bits);
    param->poly_size = poly_size;
    param->block_size = block_size;
    param->dpf_block_size = dpf_block_size;
    param->dpf_domain_bits = dpf_domain_bits;
    param->block_bits = block_bits;
}

// for GR64 and GR128 and higher degree
void init_gr_HD_bench_params(struct Param *param, const size_t n, const size_t c, const size_t t, const size_t m) {

    param->n=n;
    param->c=c;
    param->t=t;
    param->m=m;
    size_t base = (1<<m)-1;
    param->base = base;
    size_t poly_size = ipow(base, n);

    // For single DPF, each is t-regular.
    size_t block_size = ceil(poly_size / t);
    printf("block_size = %zu \n", block_size);
    size_t block_bits = (size_t)(log_base(poly_size/t, base));
    printf("block_bits = %zu\n", block_bits);

    // For product DPF, each is t^2-regular.
    size_t dpf_domain_bits = (size_t)(log_base(poly_size / (t*t), base));
    printf("DPF domain bits %zu \n", dpf_domain_bits);
    size_t dpf_block_size = ipow(base, dpf_domain_bits);
    printf("dpf_block_size = %zu\n", dpf_block_size);

    param->poly_size = poly_size;
    param->block_size = block_size;
    param->dpf_block_size = dpf_block_size;
    param->dpf_domain_bits = dpf_domain_bits;
    param->block_bits = block_bits;
}

void init_bench_params(struct Param* param, const size_t n, const size_t c, const size_t t) {
    param->n=n;
    param->c=c;
    param->t=t;
    size_t poly_size = ipow(3, n);

    size_t dpf_domain_bits = ceil(log_base(poly_size / (t*DPF_MSG_SIZE*64), 3));
    printf("DPF domain bits %zu \n", dpf_domain_bits);

    // size_t seed_size_bits = (128 * (dpf_domain_bits * 3 + 1) + DPF_MSG_SIZE * 128) * c * c * t * t;
    // printf("PCG seed size: %.2f MB\n", seed_size_bits / 8000000.0);

    size_t dpf_block_size = DPF_MSG_SIZE * ipow(3, dpf_domain_bits);
    printf("dpf_block_size = %zu\n", dpf_block_size);
    // Note: We assume that t is a power of 3 and so it divides poly_size
    size_t block_size = ceil(poly_size / t);
    printf("block_size = %zu \n", block_size);
    
    size_t packed_block_size=ceil(block_size/64.0);
    size_t packed_poly_size=t*packed_block_size;
    
    param->poly_size = poly_size;
    param->block_size = block_size;
    param->dpf_block_size = dpf_block_size;
    param->dpf_domain_bits = dpf_domain_bits;
    param->packed_poly_size = packed_poly_size;
    param->packed_block_size = packed_block_size;

    printf("packed_block_size = %zu\n", packed_block_size);
    printf("packed_poly_size = %zu\n", packed_poly_size);

    printf("Done with initializing parameters.\n");
}

// Step 1: Sample DPF keys for the cross product.
// For benchmarking purposes, we sample random DPF functions for a
// sufficiently large domain size to express a block of coefficients.
void sample_DPF_keys(const struct Param* param, struct Keys *keys) {
    
    size_t c = param->c;
    size_t t = param->t;
    size_t dpf_block_size = param->dpf_block_size;
    size_t dpf_domain_bits = param->dpf_domain_bits;

    struct DPFKey **dpf_keys_A = malloc(c * c * t * t * sizeof(void *));
    struct DPFKey **dpf_keys_B = malloc(c * c * t * t * sizeof(void *));
    // Sample PRF keys for the DPFs
    struct PRFKeys *prf_keys = malloc(sizeof(struct PRFKeys));
    PRFKeyGen(prf_keys);
    // There are c*c*t*t keys for the PCG expanding.
    for (size_t i=0; i<c; ++i) {
        for (size_t j=0; j<c; ++j) {
            for (size_t k=0; k<t; ++k) {
                for (size_t l=0; l<t; ++l) {
                    size_t index = i*c*t*t+j*t*t+k*t+l;
                    // Pick a random index for benchmarking purposes
                    size_t alpha = random_index(dpf_block_size);
                    uint128_t beta[DPF_MSG_SIZE]={0};
                    RAND_bytes((uint8_t *)beta, DPF_MSG_SIZE*sizeof(uint128_t));
                    // DPF keys
                    struct DPFKey *kA = malloc(sizeof(struct DPFKey));
                    struct DPFKey *kB = malloc(sizeof(struct DPFKey));
                    // Message (beta) is of size 8 blocks of 128 bits
                    DPFGen(prf_keys, dpf_domain_bits, alpha, beta, DPF_MSG_SIZE, kA, kB);
                    dpf_keys_A[index] = kA;
                    dpf_keys_B[index] = kB;
                }
            }
        }
    }
    keys->dpf_keys_A = dpf_keys_A;
    keys->dpf_keys_B = dpf_keys_B;
    keys->prf_keys = prf_keys;
}

// Step 2: Evaluate all the DPFs to recover shares of the c*c polynomials.
double evaluate_DPF(const struct Param *param, const struct FFT_A *fft, const struct Keys *keys, clock_t *start_expand_time) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t poly_size = param->poly_size;
    const size_t block_size = param->block_size;
    const size_t packed_poly_size = param->packed_poly_size;
    const size_t packed_block_size = param->packed_block_size;
    const size_t dpf_block_size = param->dpf_block_size;

    // Allocate memory for the concatenated DPF outputs
    uint128_t *packed_polys = calloc(c*c*packed_poly_size, sizeof(uint128_t));
    uint128_t *shares = calloc(dpf_block_size, sizeof(uint128_t));
    uint128_t *cache = calloc(dpf_block_size, sizeof(uint128_t));

    // Allocate memory for the output FFT
    uint32_t *fft_u = calloc(poly_size, sizeof(uint32_t));
    // Allocate memory for the final inner product
    uint8_t *z_poly = calloc(poly_size, sizeof(uint8_t));
    uint32_t *res_poly_mat = calloc(poly_size, sizeof(uint32_t));

    *start_expand_time = clock();
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
    fft_recursive_uint32(fft_u, n, poly_size / 3);
    multiply_fft_32(fft->fft_a2, fft_u, res_poly_mat, poly_size);

    for (size_t j = 0; j < c * c; j++) {
        for (size_t i = 0; i < poly_size; i++) {
            z_poly[i] ^= (res_poly_mat[i] >> (2 * j)) & 0b11;
        }
    }

    double end_expand_time = clock();
    double time_taken = ((double)(end_expand_time - *start_expand_time)) / (CLOCKS_PER_SEC / 1000.0); // ms
    printf("Eval time (total) %f ms\n", time_taken);
    printf("DONE\n\n");

    free(shares);
    free(cache);
    free(fft_u);
    free(packed_polys);
    free(res_poly_mat);
    free(z_poly);
    return time_taken;
}

double evaluate_DPF2(const struct Param *param, const struct FFT_A *fft, const struct Keys *keys) {

    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t poly_size = param->poly_size;
    const size_t block_size = param->block_size;
    const size_t packed_poly_size = param->packed_poly_size;
    const size_t packed_block_size = param->packed_block_size;
    const size_t dpf_block_size = param->dpf_block_size;

    // Allocate memory for the DPF outputs (this is reused for each evaluation)
    uint128_t *shares_A=malloc(sizeof(uint128_t)*dpf_block_size);
    uint128_t *cache=malloc(sizeof(uint128_t)*dpf_block_size);

    // Allocate memory for the concatenated DPF outputs
    uint128_t *packed_polys_A=calloc(c*c*packed_poly_size, sizeof(uint128_t));
    uint32_t *res_poly_mat_A=calloc(poly_size, sizeof(uint32_t));
    uint8_t *z_poly_A=calloc(poly_size, sizeof(uint8_t));
    uint32_t *fft_uA=calloc(poly_size, sizeof(uint32_t));
    
    clock_t time = clock();
    for (int i = 0; i<c; ++i) {
        for (int j = 0; j<c; ++j) {
            size_t poly_index=i*c+j;
            // each entry is of length packed_poly_size
            uint128_t* packed_polyA=&packed_polys_A[poly_index*packed_poly_size];
            
            for (size_t k=0; k<t; ++k) {
                // each entry is of length packed_block_size
                uint128_t *poly_blockA=&packed_polyA[k*packed_block_size];
            
                for (size_t l=0; l<t; ++l) {
                    size_t index=i*c*t*t+j*t*t+k*t+l;
                    struct DPFKey *dpf_keyA=keys->dpf_keys_A[index];
                    DPFFullDomainEval(dpf_keyA, cache, shares_A);
                    for (size_t w=0; w<packed_block_size; ++w) {
                        poly_blockA[w] ^=shares_A[w];
                    }
                }
            }
        }
    } // output packed_polys_A

    for (size_t j=0; j<c; ++j) {
        for (size_t k=0; k<c; ++k) {
            size_t poly_index=(j*c+k)*packed_poly_size;
            uint128_t *polyA=&packed_polys_A[poly_index];
            size_t block_idx=0;
            size_t bit_idx=0;
            for (size_t i=0; i<poly_size; ++i) {
                if (i%block_size==0 && i!=0) {
                    ++block_idx;
                    bit_idx=0;
                }
                size_t packed_idx=block_idx*packed_block_size+floor(bit_idx/64.0);
                size_t packed_bit=(63-bit_idx%64);

                uint128_t packedA=polyA[packed_idx];

                uint32_t coefA=(packedA>>(2*packed_bit))&0b11;

                size_t idx=j*c+k;
                fft_uA[i] |= coefA<<(2*idx);
                ++bit_idx;
            }
        }
    }

    fft_recursive_uint32(fft_uA, n, poly_size/3);
    multiply_fft_32(fft->fft_a2, fft_uA, res_poly_mat_A, poly_size);
    // XOR the (packed) columns into the accumulator.
    // Specifically, we perform column-wise XORs to get the result.
    for (size_t j=0; j<c*c; ++j) {
        for (size_t i=0; i<poly_size; ++i) {
            z_poly_A[i] ^= (res_poly_mat_A[i]>>(2*j))&0b11;
        }
    }
    time = clock()-time;
    double time_taken = ((double)time) / (CLOCKS_PER_SEC / 1000.0); // ms

    printf("Eval time (total) %f ms\n", time_taken);
    printf("DONE\n\n");
    free(shares_A);
    free(cache);
    free(fft_uA);
    free(packed_polys_A);
    free(res_poly_mat_A);
    free(z_poly_A);
    return time_taken;
}

void modular_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time) {
   if (c > 4) {
        printf("ERROR: currently only implemented for c <= 4");
        exit(0);
    }

    clock_t start_time = clock();
    struct Param *param = calloc(1, sizeof(struct Param));
    init_bench_params(param, n, c, t);

    struct FFT_A *fft = calloc(1, sizeof(struct FFT_A));
    step0(param, fft);
    struct Keys *keys = calloc(1, sizeof(struct Keys));
    sample_DPF_keys(param, keys);
    printf("Benchmarking PCG evaluation \n");
    clock_t start_expand_time = 0;
    double time_taken = evaluate_DPF(param, fft, keys, &start_expand_time);
    step0_free(fft);
    step4_free(param, keys);
    free(param);

    clock_t end_time = clock();
    pcg_time->pp_time = (start_expand_time-start_time)/(CLOCKS_PER_SEC/1000.0);
    pcg_time->expand_time = time_taken;
    pcg_time->total_time = (end_time-start_time)/(CLOCKS_PER_SEC / 1000.0);
}
