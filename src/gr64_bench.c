#include <stdint.h>
#include <stdlib.h>

#include "gr64_bench.h"
#include "common.h"
#include "modular_bench.h"
#include "modular_test.h"
#include "dpf.h"
#include "fft.h"
#define DPF_MSG_LEN 1

void gr64_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time) {
    clock_t start_time = clock();

    struct Param *param = xcalloc(1, sizeof(struct Param));
    // In this Galois ring based parameters, m stands for the number of extraction of PCG OLE. For instance, the OLE over Galois ring sets m=1. while in trace-based OLE over Z_{p^k} and authenticated multiplication triples, m is set to 2.
    size_t m = 1;
    init_gr64_bench_params(param, n, c, t, m);
    struct FFT_GR64_A *fft_gr64_a = xcalloc(1, sizeof(struct FFT_GR64_A));
    init_FFT_GR64_A(param, fft_gr64_a);
    sample_gr64_a_and_tensor(param, fft_gr64_a);
    struct Keys *keys = xcalloc(1, sizeof(struct Keys));
    sample_gr64_DPF_keys(param, keys);
    size_t dpf_block_size = param->dpf_block_size;
    size_t poly_size = param->poly_size;
    if (t*t*dpf_block_size == poly_size) {
        printf("OK\n");
    } else {
        printf("Incorrect\n");
        exit(-1);
    }
    printf("Benchmarking PCG evaluation \n");
    struct GR64 *polys = xcalloc(c*c*m*t*t*dpf_block_size, sizeof(struct GR64));
    struct GR64 *poly_buf = xcalloc(c*c*m*t*t*dpf_block_size, sizeof(struct GR64));
    struct GR64 *z_poly = xcalloc(t*t*dpf_block_size, sizeof(struct GR64));

    uint128_t *shares = xcalloc(dpf_block_size, sizeof(uint128_t));
    uint128_t *cache = xcalloc(dpf_block_size, sizeof(uint128_t));

    clock_t start_expand_time = clock();
    evaluate_gr64_DPF(param, keys, polys, shares, cache);
    convert_gr64_to_FFT(param, polys);
    multiply_gr64_FFT(param, fft_gr64_a->fft_a_tensor, polys, poly_buf);
    sum_gr64_FFT_polys(param, poly_buf, z_poly);

    double end_expand_time = clock();
    double time_taken = ((double)(end_expand_time - start_expand_time)) / (CLOCKS_PER_SEC / 1000.0); // ms
    printf("Eval time (total) %f ms\n", time_taken);
    printf("DONE\n\n");
    printf("Benchmarking PCG evaluation \n");

    free_FFT_GR64_A(param, fft_gr64_a);
    free_gr64_DPF_keys(param, keys);
    free(param);
    free(polys);
    free(poly_buf);
    free(z_poly);
    free(shares);
    free(cache);

    clock_t end_time = clock();
    pcg_time->pp_time = ((double)(start_expand_time-start_time))/(CLOCKS_PER_SEC/1000.0);
    pcg_time->expand_time = time_taken;
    pcg_time->total_time = ((double)(end_time-start_time))/(CLOCKS_PER_SEC / 1000.0);
}

/**
 * Init the space for a and a x a.
 * The parameter c matters for memory allocation.
 */
void init_FFT_GR64_A(const struct Param *param, struct FFT_GR64_A *fft_gr64_a) {

    size_t poly_size = param->poly_size;
    size_t c = param->c;
    struct GR64 **fft_a = xcalloc(c, sizeof(void*));
    for (size_t i = 0; i < c; ++i) {
        fft_a[i] = xcalloc(poly_size, sizeof(struct GR64));
    }
    struct GR64 **fft_a_tensor = xcalloc(c*c, sizeof(void*));
    for(size_t i = 0; i < c*c; ++i) {
        fft_a_tensor[i] = xcalloc(poly_size, sizeof(struct GR64));
    }
    fft_gr64_a->fft_a = fft_a;
    fft_gr64_a->fft_a_tensor = fft_a_tensor;
}

void free_FFT_GR64_A(const struct Param *param, struct FFT_GR64_A *fft_gr64_a) {
    size_t c = param->c;

    for (size_t i = 0; i < c; ++i) {
        free(fft_gr64_a->fft_a[i]);
    }
    // Free fft_a
    free(fft_gr64_a->fft_a);
    // Free the nested arrays in fft_a_tensor
    for (size_t i = 0; i < c * c; ++i) {
        free(fft_gr64_a->fft_a_tensor[i]);
    }
    free(fft_gr64_a->fft_a_tensor);
    free(fft_gr64_a);
}

/**
 * Randomize a in FFT space first and then compute a x a in FFT space.
 */
void sample_gr64_a_and_tensor(const struct Param *param, struct FFT_GR64_A *fft_gr64_a) {

    const size_t poly_size = param->poly_size;
    const size_t c = param->c;
    struct GR64 **fft_a = fft_gr64_a->fft_a;
    struct GR64 **fft_a_tensor = fft_gr64_a->fft_a_tensor;
    for (size_t i = 1; i < c; ++i) {
        RAND_bytes((uint8_t *)fft_a[i], sizeof(struct GR64) * poly_size);
    }
    // make first a the identity polynomial (in FFT space) i.e., all 1s
    for (size_t i = 0; i < poly_size; ++i) {
        fft_a[0][i].c0 = 1;
        fft_a[0][i].c1 = 0;
    }

    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t k = 0; k < poly_size; ++k) {
                mult_gr64(&fft_a[i][k], &fft_a[j][k], &fft_a_tensor[i*c+j][k]);
            }
        }
    }
}

// Multiply two GR64 elements
void mult_gr64(const struct GR64 *a, const struct GR64 *b, struct GR64 *t) {
    t->c0 = a->c0*b->c0 - a->c1*b->c1;
    t->c1 = a->c0*b->c1 + a->c1*b->c0 - a->c1 * b->c1;
}

void add_gr64_D3(const struct GR64_D3 *a, const struct GR64_D3 *b, struct GR64_D3 *t) {
    t->c0 = a->c0 + b->c0;
    t->c1 = a->c1 + b->c1;
    t->c2 = a->c2 + b->c2;
}

// Multiply two degree 3 GR64 elements mod X^3 + 17520588382079786918*X^2 + 17520588382079786917*X - 1
void mult_gr64_D3(const struct GR64_D3 *a, const struct GR64_D3 *b, struct GR64_D3 *t) {
// Modulus: X^3 + aX^2 + bX - 1
    const static uint64_t A = 17520588382079786918ULL;
    const static uint64_t B = 17520588382079786917ULL;

    // Compute negative coefficients modulo 2^64
    const static uint64_t A_ = ~A + 1; // -A mod 2^64
    const static uint64_t B_ = ~B + 1; // -B mod 2^64

    // Intermediate products (up to degree 4)
    uint64_t c0 = a->c0 * b->c0;
    uint64_t c1 = a->c0 * b->c1 + a->c1 * b->c0;
    uint64_t c2 = a->c0 * b->c2 + a->c1 * b->c1 + a->c2 * b->c0;
    uint64_t c3 = a->c1 * b->c2 + a->c2 * b->c1;
    uint64_t c4 = a->c2 * b->c2;

    // Reduce c4 (X^4)
    // X^4 ≡ -X^2 - X + A_ mod f
    uint64_t r4_c2 = -1;
    uint64_t r4_c1 = -1;
    uint64_t r4_c0 = A_;

    c0 += c4 * r4_c0;
    c1 += c4 * r4_c1;
    c2 += c4 * r4_c2;

    // Reduce c3 (X^3 ≡ A_*X^2 + B_*X + 1)
    c0 += c3;
    c1 += c3 * B_;
    c2 += c3 * A_;

    t->c0 = c0;
    t->c1 = c1;
    t->c2 = c2;
}

// Multiply two degree 3 GR64 arrays
void mult_gr64_D3_list(const struct GR64_D3 *a, const struct GR64_D3 *b, struct GR64_D3 *t, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        mult_gr64_D3(&a[i], &b[i], &t[i]);
    }
}

void mult_gr64_D4(const struct GR64_D4 *a, const struct GR64_D3 *b, struct GR64_D3 *t) {
    printf("Not implemented function mult_gr64_D4");
}
// sample c*c*m*t*t DPF keys
void sample_gr64_DPF_keys(const struct Param *param, struct Keys *keys) {
    size_t c = param->c;
    size_t t = param->t;
    size_t dpf_block_size = param->dpf_block_size;
    size_t dpf_domain_bits = param->dpf_domain_bits;
    const size_t m = param->m;

    struct DPFKey **dpf_keys_A = xmalloc(c*c*m*t*t*sizeof(void *));
    struct DPFKey **dpf_keys_B = xmalloc(c*c*m*t*t*sizeof(void *));
    struct PRFKeys *prf_keys = xmalloc(sizeof(struct PRFKeys));
    PRFKeyGen(prf_keys);
    for (size_t i=0; i<c; ++i) {
        for (size_t j=0; j<c; ++j) {
            for (size_t w=0; w<m; ++w) {
                for (size_t k=0; k<t; ++k) {
                    for (size_t l=0; l<t; ++l) {
                        size_t index = (((i*c+j)*m+w)*t+k)*t+l;
                        // Pick a random position for benchmarking purposes
                        size_t alpha = random_index(dpf_block_size);
                        uint128_t beta[DPF_MSG_LEN] ={0};
                        RAND_bytes((uint8_t *)beta, DPF_MSG_LEN*sizeof(uint128_t));
                        // DPF keys
                        struct DPFKey *kA = xmalloc(sizeof(struct DPFKey));
                        struct DPFKey *kB = xmalloc(sizeof(struct DPFKey));
                        DPFGen(prf_keys, dpf_domain_bits, alpha, beta, DPF_MSG_LEN, kA, kB);
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

void free_gr64_DPF_keys(const struct Param *param, struct Keys *keys) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t m = param->m;
    for (size_t i=0; i<c; ++i) {
        for (size_t j=0; j<c; ++j) {
            for (size_t w=0; w<m; ++w) {
                for (size_t k=0; k<t; ++k) {
                    for (size_t l=0; l<t; ++l) {
                        size_t index = (((i*c+j)*m+w)*t+k)*t+l;
                        free(keys->dpf_keys_A[index]);
                        free(keys->dpf_keys_B[index]);
                    }
                }
            }
        }
    }
    free(keys->dpf_keys_A);
    free(keys->dpf_keys_B);
    DestroyPRFKey(keys->prf_keys);
    free(keys);
}

// evaluate DPF to c*c*m polynomials
void evaluate_gr64_DPF(const struct Param *param, const struct Keys *keys, struct GR64 *polys, uint128_t *shares, uint128_t *cache) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t dpf_block_size = param->dpf_block_size;
    const size_t m = param->m;
    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t v=0; v<m; ++v) {
                const size_t poly_index = (i*c+j)*m+v;
                struct GR64 *poly = &polys[poly_index*t*t*dpf_block_size];
                for (size_t k = 0; k < t; ++k) {
                    for (size_t l = 0; l < t; ++l) {
                        const size_t key_index = poly_index*t*t+k*t+l;
                        struct GR64 *poly_block = &poly[(k*t+l)*dpf_block_size];
                        struct DPFKey *dpf_key = keys->dpf_keys_A[key_index];
                        // TODO: test the DPF evaluation results
                        DPFFullDomainEval(dpf_key, cache, shares);
                        copy_gr64_block(poly_block, shares, dpf_block_size);
                    }
                }
            }
        }
    }
}

// convert c*c*m polynomials to FFT forms
void convert_gr64_to_FFT(const struct Param *param, struct GR64 *polys) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;
    
    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t w = 0; w < m; ++w) {
                struct GR64 *poly = &polys[((i*c+j)*m+w)*poly_size];
                fft_recursive_gr64(poly, n, poly_size / 3);
            }
        }
    }
}

// multiply polynomial a and DPF evaluation results
void multiply_gr64_FFT(const struct Param *param, struct GR64 **a_polys, const struct GR64 *b_poly, struct GR64 *res_poly) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;
    
    for (size_t i = 0; i < c*c; ++i) {
        for (size_t w = 0; w < m; ++w) {
            const struct GR64 *a_poly = &a_polys[i][w*poly_size];
            for (size_t j = 0; j < poly_size; ++j) {
                mult_gr64(&a_poly[j], &b_poly[(i*m+w)*poly_size+j], &res_poly[(i*m+w)*poly_size+j]);
            }
        }
    }
}

// sum up the c*c*m polynomials to z_poly
void sum_gr64_FFT_polys(const struct Param *param, struct GR64 *poly_buf, struct GR64 *z_poly) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;

    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t w = 0; w < m; ++w) {
                struct GR64 *poly = &poly_buf[((i*c+j)*m+w)*poly_size];
                for (size_t k = 0; k < poly_size; ++k) {
                    z_poly[k].c0 += poly[k].c0;
                    z_poly[k].c1 += poly[k].c1;
                }
            }
        }
    }
}

// multiply the first by zeta^2 and second by 1
void sum_gr64_FFT_polys_special(const struct Param *param, struct GR64 *poly_buf, struct GR64 *z_poly) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;

    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t w = 0; w < m; ++w) {
                struct GR64 *poly = &poly_buf[((i*c+j)*m+w)*poly_size];
                if (w == 0) {
                    for (size_t k = 0; k < poly_size; ++k) {
                        z_poly[k].c0 += poly[k].c1-poly[k].c0;
                        z_poly[k].c1 += (-poly[k].c0);
                    }
                } else {
                    for (size_t k = 0; k < poly_size; ++k) {
                        z_poly[k].c0 += poly[k].c0;
                        z_poly[k].c1 += poly[k].c1;
                    }
                }
            }
        }
    }
}

/**
 * Parse each block of type uint128_t to GR64
 * @param poly_block is of length DPF_MSG_LEN*block_size
 * @param shares is of length DPF_MSG_LEN*block_size
 */
void copy_gr64_block(struct GR64 *poly_block, uint128_t *shares, const size_t dpf_block_size) {

    for (size_t w = 0; w < dpf_block_size; ++w) {
        poly_block[w].c0 = (uint64_t)(shares[w]);
        poly_block[w].c1 = (uint64_t)(shares[w]>>64);
    }
}
