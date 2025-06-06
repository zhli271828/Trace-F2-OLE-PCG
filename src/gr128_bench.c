#include <stdint.h>
#include <stdlib.h>

#include "gr128_bench.h"
#include "common.h"
#include "modular_bench.h"
#include "modular_test.h"
#include "dpf.h"
#include "fft.h"

void gr128_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time) {
    clock_t start_time = clock();

    struct Param *param = calloc(1, sizeof(struct Param));
    size_t m = 1;
    init_gr128_bench_params(param, n, c, t, m);
    struct FFT_GR128_A *fft_gr128_a = calloc(1, sizeof(struct FFT_GR128_A));
    init_FFT_GR128_A(param, fft_gr128_a);
    sample_gr128_a_and_tensor(param, fft_gr128_a);
    struct Keys *keys = calloc(1, sizeof(struct Keys));
    sample_gr128_DPF_keys(param, keys);
    size_t dpf_block_size = param->dpf_block_size;
    size_t poly_size = param->poly_size;
    if (t*t*dpf_block_size == poly_size) {
        printf("OK\n");
    } else {
        printf("Incorrect\n");
        exit(-1);
    }
    printf("Benchmarking PCG evaluation \n");

    struct GR128 *polys = calloc(c*c*t*t*dpf_block_size, sizeof(struct GR128));
    struct GR128 *poly_buf = calloc(c*c*t*t*dpf_block_size, sizeof(struct GR128));
    struct GR128 *z_poly = calloc(t*t*dpf_block_size, sizeof(struct GR128));

    uint128_t *shares = calloc(dpf_block_size*2, sizeof(uint128_t));
    uint128_t *cache = calloc(dpf_block_size*2, sizeof(uint128_t));
    
    clock_t start_expand_time = clock();
    
    evaluate_gr128_DPF(param, keys, polys, shares, cache);
    convert_gr128_to_FFT(param, polys);
    multiply_gr128_FFT(param, fft_gr128_a->fft_a_tensor, polys, poly_buf);
    sum_gr128_FFT_polys(param, poly_buf, z_poly);

    double end_expand_time = clock();
    double time_taken = ((double)(end_expand_time - start_expand_time)) / (CLOCKS_PER_SEC / 1000.0); // ms
    printf("Eval time (total) %f ms\n", time_taken);
    printf("DONE Benchmarking PCG evaluation\n\n");

    free_FFT_GR128_A(param, fft_gr128_a);
    free_gr128_DPF_keys(param, keys);
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

void init_FFT_GR128_A(const struct Param *param, struct FFT_GR128_A *fft_gr128_a) {

    size_t poly_size = param->poly_size;
    size_t c = param->c;
    struct GR128 **fft_a = calloc(c, sizeof(void*));
    for (size_t i = 0; i < c; ++i) {
        fft_a[i] = calloc(poly_size, sizeof(struct GR128));
    }
    struct GR128 **fft_a_tensor = calloc(c*c, sizeof(void*));
    for(size_t i = 0; i < c*c; ++i) {
        fft_a_tensor[i] = calloc(poly_size, sizeof(struct GR128));
    }
    fft_gr128_a->fft_a = fft_a;
    fft_gr128_a->fft_a_tensor = fft_a_tensor;
}

void free_FFT_GR128_A(const struct Param *param, struct FFT_GR128_A *fft_gr128_a) {
    size_t c = param->c;

    for (size_t i = 0; i < c; ++i) {
        free(fft_gr128_a->fft_a[i]);
    }
    // Free fft_a
    free(fft_gr128_a->fft_a);
    // Free the nested arrays in fft_a_tensor
    for (size_t i = 0; i < c * c; ++i) {
        free(fft_gr128_a->fft_a_tensor[i]);
    }
    free(fft_gr128_a->fft_a_tensor);
    free(fft_gr128_a);
}

void sample_gr128_a_and_tensor(const struct Param *param, struct FFT_GR128_A *fft_gr128_a) {

    const size_t poly_size = param->poly_size;
    const size_t c = param->c;
    struct GR128 **fft_a = fft_gr128_a->fft_a;
    struct GR128 **fft_a_tensor = fft_gr128_a->fft_a_tensor;
    for (size_t i = 1; i < c; ++i) {
        RAND_bytes((uint8_t *)fft_a[i], sizeof(struct GR128) * poly_size);
    }
    // make first a the identity polynomial (in FFT space) i.e., all 1s
    for (size_t i = 0; i < poly_size; ++i) {
        fft_a[0][i].c0 = 1;
        fft_a[0][i].c1 = 0;
    }

    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t k = 0; k < poly_size; ++k) {
                mult_gr128(&fft_a[i][k], &fft_a[j][k], &fft_a_tensor[i*c+j][k]);
            }
        }
    }
}

// Multiply two GR128 elements
void mult_gr128(const struct GR128 *a, const struct GR128 *b, struct GR128 *t) {
    t->c0 = a->c0*b->c0 - a->c1*b->c1;
    t->c1 = a->c0*b->c1 + a->c1*b->c0 - a->c1 * b->c1;
}

// sample c*c*m*t*t DPF keys
void sample_gr128_DPF_keys(const struct Param *param, struct Keys *keys) {
    size_t c = param->c;
    size_t t = param->t;
    size_t dpf_block_size = param->dpf_block_size;
    size_t dpf_domain_bits = param->dpf_domain_bits;
    const size_t m = param->m;

    struct DPFKey **dpf_keys_A = malloc(c*c*m*t*t*sizeof(void *));
    struct DPFKey **dpf_keys_B = malloc(c*c*m*t*t*sizeof(void *));
    struct PRFKeys *prf_keys = malloc(sizeof(struct PRFKeys));
    PRFKeyGen(prf_keys);
    for (size_t i=0; i<c; ++i) {
        for (size_t j=0; j<c; ++j) {
            for (size_t w=0; w<m; ++w) {
                for (size_t k=0; k<t; ++k) {
                    for (size_t l=0; l<t; ++l) {
                        size_t index = (((i*c+j)*m+w)*t+k)*t+l;
                        // Pick a random position for benchmarking purposes
                        size_t alpha = random_index(dpf_block_size);
                        uint128_t beta[2] ={0};
                        RAND_bytes((uint8_t *)beta, 2*sizeof(uint128_t));
                        // DPF keys
                        struct DPFKey *kA = malloc(sizeof(struct DPFKey));
                        struct DPFKey *kB = malloc(sizeof(struct DPFKey));
                        DPFGen(prf_keys, dpf_domain_bits, alpha, beta, 2, kA, kB);
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

void free_gr128_DPF_keys(const struct Param *param, struct Keys *keys) {
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
void evaluate_gr128_DPF(const struct Param *param, const struct Keys *keys, struct GR128 *polys, uint128_t *shares, uint128_t *cache) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t dpf_block_size = param->dpf_block_size;
    const size_t m = param->m;
    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t v=0; v<m; ++v) {
                const size_t poly_index = (i*c+j)*m+v;
                struct GR128 *poly = &polys[poly_index*t*t*dpf_block_size];
                for (size_t k = 0; k < t; ++k) {
                    for (size_t l=0; l < t; ++l) {
                        const size_t key_index = poly_index*t*t+k*t+l;
                        struct GR128 *poly_block = &poly[(k*t+l)*dpf_block_size];
                        struct DPFKey *dpf_key = keys->dpf_keys_A[key_index];
                        // TODO: test the DPF evaluation results
                        DPFFullDomainEval(dpf_key, cache, shares);
                        copy_gr128_block(poly_block, shares, dpf_block_size);
                    }
                }
            }
        }
    }
}

// convert c*c*m polynomials to FFT forms
void convert_gr128_to_FFT(const struct Param *param, struct GR128 *polys) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;
    
    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t w = 0; w < m; ++w) {
                struct GR128 *poly = &polys[((i*c+j)*m+w)*poly_size];
                fft_recursive_gr128(poly, n, poly_size / 3);
            }
        }
    }
}

// multiply polynomial a and DPF evaluation results
void multiply_gr128_FFT(const struct Param *param, struct GR128 **a_polys, const struct GR128 *b_poly, struct GR128 *res_poly) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;
    
    for (size_t i = 0; i < c*c; ++i) {
        for (size_t w = 0; w < m; ++w) {
            const struct GR128 *a_poly = &a_polys[i][w*poly_size];
            for (size_t j = 0; j < poly_size; ++j) {
                mult_gr128(&a_poly[j], &b_poly[(i*m+w)*poly_size+j], &res_poly[(i*m+w)*poly_size+j]);
            }
        }
    }
}

// multiply the first by zeta^2 and second by 1
void sum_gr128_FFT_polys_special(const struct Param *param, struct GR128 *poly_buf, struct GR128 *z_poly) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;

    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t w = 0; w < m; ++w) {
                struct GR128 *poly = &poly_buf[((i*c+j)*m+w)*poly_size];
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

// sum up the c*c*m polynomials to z_poly
void sum_gr128_FFT_polys(const struct Param *param, struct GR128 *poly_buf, struct GR128 *z_poly) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;

    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t w = 0; w < m; ++w) {
                struct GR128 *poly = &poly_buf[((i*c+j)*m+w)*poly_size];
                for (size_t k = 0; k < poly_size; ++k) {
                    z_poly[k].c0 += poly[k].c0;
                    z_poly[k].c1 += poly[k].c1;
                }
            }
        }
    }
}

// copy each block of GR128 output polynomials
void copy_gr128_block(struct GR128 *poly_block, uint128_t *shares, const size_t dpf_block_size) {

    for (size_t w = 0; w < dpf_block_size; ++w) {
        // perhaps a direct assignment suffices
        poly_block[w].c0 = shares[w*2];
        poly_block[w].c1 = shares[w*2+1];
    }
}


void add_gr128_D3(const struct GR128_D3 *a, const struct GR128_D3 *b, struct GR128_D3 *t) {
    t->c0 = a->c0 + b->c0;
    t->c1 = a->c1 + b->c1;
    t->c2 = a->c2 + b->c2;
}

// Multiply two degree 3 GR128 elements mod X^3 + 184612259339076664511367859069250674598*X^2 + 184612259339076664511367859069250674597*X - 1
void mult_gr128_D3(const struct GR128_D3 *a, const struct GR128_D3 *b, struct GR128_D3 *t) {
// Modulus: X^3 + AX^2 + BX - 1
    const static uint128_t A = 184612259339076664511367859069250674598ULL;
    const static uint128_t B = 184612259339076664511367859069250674597ULL;

    // Compute negative coefficients modulo 2^128
    const static uint128_t A_ = ~A + 1; // -A mod 2^128
    const static uint128_t B_ = ~B + 1; // -B mod 2^128

    // Intermediate products (up to degree 4)
    uint128_t c0 = a->c0 * b->c0;
    uint128_t c1 = a->c0 * b->c1 + a->c1 * b->c0;
    uint128_t c2 = a->c0 * b->c2 + a->c1 * b->c1 + a->c2 * b->c0;
    uint128_t c3 = a->c1 * b->c2 + a->c2 * b->c1;
    uint128_t c4 = a->c2 * b->c2;

    // Reduce c4 (X^4)
    // X^4 ≡ -X^2 - X + A_ mod f
    uint128_t r4_c2 = -1;
    uint128_t r4_c1 = -1;
    uint128_t r4_c0 = A_;

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

// Multiply two degree 3 GR128 arrays
void mult_gr128_D3_list(const struct GR128_D3 *a, const struct GR128_D3 *b, struct GR128_D3 *t, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        mult_gr128_D3(&a[i], &b[i], &t[i]);
    }
}

// Multiply two degree 4 GR128 elements mod X^4 + 23254378673668297602512925802836206188*X^3 - 2*X^2 - 23254378673668297602512925802836206189*X + 1
void mult_gr128_D4(const struct GR128_D4 *a, const struct GR128_D4 *b, struct GR128_D4 *t) {
    // Modulus: X^4 + A·X^3 - 2·X^2 - B·X + 1
    const uint128_t A = 23254378673668297602512925802836206188ULL;
    const uint128_t B = A+1;
    const uint128_t A_ = ~A + 1; // -A mod 2^64
    const uint128_t B_ = ~B + 1; // -B mod 2^64
    const uint128_t M1 = ~0UL; // -1 mod 2^64

    // Schoolbook multiplication (up to degree 6)
    uint128_t c0 = a->c0 * b->c0;
    uint128_t c1 = a->c0 * b->c1 + a->c1 * b->c0;
    uint128_t c2 = a->c0 * b->c2 + a->c1 * b->c1 + a->c2 * b->c0;
    uint128_t c3 = a->c0 * b->c3 + a->c1 * b->c2 + a->c2 * b->c1 + a->c3 * b->c0;
    uint128_t c4 = a->c1 * b->c3 + a->c2 * b->c2 + a->c3 * b->c1;
    uint128_t c5 = a->c2 * b->c3 + a->c3 * b->c2;
    uint128_t c6 = a->c3 * b->c3;

    // === Reduce c6 · X^6 ===
    // X^6 = X^2 · X^4 ≡ X^2( -A·X^3 + 2·X^2 + B·X - 1 )
    //     = -A·X^5 + 2·X^4 + B·X^3 - X^2
    c2 += c6 * M1;
    c3 += c6 * B;
    c4 += c6 * 2;
    c5 += c6 * A_;

    // === Reduce c5 · X^5 ===
    // X^5 = X · X^4 ≡ X( -A·X^3 + 2·X^2 + B·X - 1 )
    //     = -A·X^4 + 2·X^3 + B·X^2 - X
    c1 += c5 * M1;
    c2 += c5 * B;
    c3 += c5 * 2;
    c4 += c5 * A_;

    // === Reduce c4 · X^4 ===
    // X^4 ≡ -A·X^3 + 2·X^2 + B·X - 1
    c0 += c4 * M1;
    c1 += c4 * B;
    c2 += c4 * 2;
    c3 += c4 * A_;

    t->c0 = c0;
    t->c1 = c1;
    t->c2 = c2;
    t->c3 = c3;
}

// Multiply two degree 4 GR64 list
void mult_gr128_D4_list(const struct GR128_D4 *a, const struct GR128_D4 *b, struct GR128_D4 *t, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        mult_gr128_D4(&a[i], &b[i], &t[i]);
    }
}

void add_gr128_D4(const struct GR128_D4 *a, const struct GR128_D4 *b, struct GR128_D4 *t) {
    t->c0 = a->c0 + b->c0;
    t->c1 = a->c1 + b->c1;
    t->c2 = a->c2 + b->c2;
    t->c3 = a->c3 + b->c3;
}
