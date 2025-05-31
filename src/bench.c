#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "test.h"
#include "dpf.h"
#include "prf.h"
#include "fft.h"
#include "utils.h"
#include "f4ops.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Benchmarks are less documented compared to test.c; see test.c to
// better understand what is done here for timing purposes.

#define DPF_MSG_SIZE 8

void bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time)
{
    if (c > 4)
    {
        printf("ERROR: currently only implemented for c <= 4");
        exit(0);
    }

    clock_t start_time = clock();
    const size_t poly_size = ipow(3, n);

    //************************************************************************
    // Step 0: Sample the global (1, a1 ... a_c-1) polynomials
    //************************************************************************
    uint8_t *fft_a = calloc(poly_size, sizeof(uint8_t));
    uint32_t *fft_a2 = calloc(poly_size, sizeof(uint32_t));
    sample_a_and_a2(fft_a, fft_a2, poly_size, c);

    //************************************************************************
    // Step 1: Sample DPF keys for the cross product.
    // For benchmarking purposes, we sample random DPF functions for a
    // sufficiently large domain size to express a block of coefficients.
    //************************************************************************
    size_t dpf_domain_bits = ceil(log_base(poly_size / (t * DPF_MSG_SIZE * 64), 3));
    printf("dpf_domain_bits = %zu \n", dpf_domain_bits);

    size_t seed_size_bits = (128 * (dpf_domain_bits * 3 + 1) + DPF_MSG_SIZE * 128) * c * c * t * t;
    printf("PCG seed size: %.2f MB\n", seed_size_bits / 8000000.0);

    size_t dpf_block_size = DPF_MSG_SIZE * ipow(3, dpf_domain_bits);
    size_t block_size = ceil(poly_size / t);

    printf("block_size = %zu \n", block_size);

    struct DPFKey **dpf_keys_A = malloc(c * c * t * t * sizeof(void *));
    struct DPFKey **dpf_keys_B = malloc(c * c * t * t * sizeof(void *));

    // Sample PRF keys for the DPFs
    struct PRFKeys *prf_keys = malloc(sizeof(struct PRFKeys));
    PRFKeyGen(prf_keys);

    // Sample DPF keys for each of the t errors in the t blocks
    for (size_t i = 0; i < c; i++)
    {
        for (size_t j = 0; j < c; j++)
        {
            for (size_t k = 0; k < t; k++)
            {
                for (size_t l = 0; l < t; l++)
                {
                    size_t index = i * c * t * t + j * t * t + k * t + l;

                    // Pick a random index for benchmarking purposes
                    size_t alpha = random_index(block_size);

                    // Pick a random output message for benchmarking purposes
                    uint128_t beta[DPF_MSG_SIZE] = {0};
                    RAND_bytes((uint8_t *)beta, DPF_MSG_SIZE * sizeof(uint128_t));

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

    //************************************************
    printf("Benchmarking PCG evaluation \n");
    //************************************************

    // Allocate memory for the DPF outputs (this is reused for each evaluation)
    uint128_t *shares = malloc(sizeof(uint128_t) * dpf_block_size);
    uint128_t *cache = malloc(sizeof(uint128_t) * dpf_block_size);

    // Allocate memory for the concatenated DPF outputs
    const size_t packed_block_size = ceil(block_size / 64.0);
    const size_t packed_poly_size = t * packed_block_size;
    uint128_t *packed_polys = calloc(c * c * packed_poly_size, sizeof(uint128_t));

    // Allocate memory for the output FFT
    uint32_t *fft_u = calloc(poly_size, sizeof(uint32_t));

    // Allocate memory for the final inner product
    uint8_t *z_poly = calloc(poly_size, sizeof(uint8_t));
    uint32_t *res_poly_mat = malloc(sizeof(uint32_t) * poly_size);

    //************************************************************************
    // Step 3: Evaluate all the DPFs to recover shares of the c*c polynomials.
    //************************************************************************

    clock_t start_expand_time = clock();

    struct DPFKey *dpf_key;
    size_t key_index;
    uint128_t *poly_block;
    size_t i, j, k, l, w;
    for (i = 0; i < c; i++)
    {
        for (j = 0; j < c; j++)
        {
            const size_t poly_index = i * c + j;
            uint128_t *packed_poly = &packed_polys[poly_index * packed_poly_size];

            for (k = 0; k < t; k++)
            {
                poly_block = &packed_poly[k * packed_block_size];

                for (l = 0; l < t; l++)
                {
                    key_index = i * c * t * t + j * t * t + k * t + l;
                    dpf_key = dpf_keys_A[key_index];

                    DPFFullDomainEval(dpf_key, cache, shares);

                    for (w = 0; w < packed_block_size; w++)
                        poly_block[w] ^= shares[w];
                }
            }
        }
    }

    //************************************************************************
    // Step 3: Compute the transpose of the polynomials to pack them into
    // the parallel FFT format.
    //
    // TODO: this is the bottleneck of the computation and can be improved
    // using SIMD operations for performing matrix transposes (see TODO in test.c).
    //************************************************************************
    for (size_t i = 0; i < c * c; i++)
    {
        size_t poly_index = i * packed_poly_size;
        const uint128_t *poly = &packed_polys[poly_index];
        __builtin_prefetch(&poly[0], 0, 3);

        size_t block_idx, packed_coeff_idx, coeff_idx;
        uint8_t packed_bit_idx;
        uint128_t packed_coeff;

        block_idx = 0;
        packed_coeff_idx = 0;
        coeff_idx = 0;

        for (size_t k = 0; k < poly_size - 64; k += 64)
        {
            packed_coeff = poly[block_idx * packed_block_size + packed_coeff_idx];
            __builtin_prefetch(&fft_u[k], 0, 0);
            __builtin_prefetch(&fft_u[k], 1, 0);

            for (size_t l = 0; l < 64; l++)
            {
                packed_coeff = packed_coeff >> 2;
                fft_u[k + l] |= packed_coeff & 0b11;
                fft_u[k + l] = fft_u[k + l] << 2;
            }

            packed_coeff_idx++;
            coeff_idx += 64;

            if (coeff_idx > block_size)
            {
                coeff_idx = 0;
                block_idx++;
                packed_coeff_idx = 0;
                __builtin_prefetch(&poly[block_idx * packed_block_size], 0, 2);
            }
        }

        packed_coeff = poly[block_idx * packed_block_size + packed_coeff_idx];
        for (size_t k = poly_size - 64 + 1; k < poly_size; k++)
        {
            packed_coeff = packed_coeff >> 2;
            fft_u[k] |= packed_coeff & 0b11;
            fft_u[k] = fft_u[k] << 2;
        }
    }

    fft_recursive_uint32(fft_u, n, poly_size / 3);
    multiply_fft_32(fft_a2, fft_u, res_poly_mat, poly_size);

    // Perform column-wise XORs to get the result
    for (size_t i = 0; i < poly_size; i++)
    {
        // XOR the (packed) columns into the accumulator
        for (size_t j = 0; j < c * c; j++)
        {
            z_poly[i] ^= res_poly_mat[i] & 0b11;
            res_poly_mat[i] = res_poly_mat[i] >> 2;
        }
    }

    double end_expand_time = clock();
    double time_taken = ((double)(end_expand_time - start_expand_time)) / (CLOCKS_PER_SEC / 1000.0); // ms
    printf("Eval time (total) %f ms\n", time_taken);
    printf("DONE\n\n");

    DestroyPRFKey(prf_keys);
    free(fft_a);
    free(fft_a2);
    free(dpf_keys_A);
    free(dpf_keys_B);
    free(shares);
    free(cache);
    free(fft_u);
    free(packed_polys);
    free(res_poly_mat);
    free(z_poly);
    clock_t end_time = clock();

    pcg_time->pp_time = ((double)(start_expand_time-start_time))/(CLOCKS_PER_SEC/1000.0);
    pcg_time->expand_time = time_taken;
    pcg_time->total_time = ((double)(end_time-start_time))/(CLOCKS_PER_SEC / 1000.0);
}
