#ifndef __SPDZ2K_64_BENCH
#define __SPDZ2K_64_BENCH

#include "common.h"
#include "gr128_bench.h"

struct SPDZ2k_64_FFT_A {
    // length c*poly_size for a
    struct GR128 **fft_a;
};

// product structure
struct SPDZ2k_64_b {
    // generate keys for b0, b1
    struct Keys *keys_b0;
    struct Keys *keys_b1;
    // output for (b0,b1) and each has two values for the trace
    uint128_t *b0_0;
    uint128_t *b0_1;
    uint128_t *b1_0;
    uint128_t *b1_1;
    // output for authenticated (b0,b1) and each has two values for the trace
    uint128_t *bm0_0;
    uint128_t *bm0_1;
    uint128_t *bm1_0;
    uint128_t *bm1_1;
    
    struct GR128 *polys;
    struct GR128 *poly_buf;
    struct GR128 *z_poly;
    // Shares and cache for each dpf_block
    uint128_t *shares;
    uint128_t *cache;
};

// product structure
struct SPDZ2k_64_Prod {
    // keys for z
    struct Keys *keys;
    // The output for Tr(z) and Tr(z*K)
    uint128_t *rlt0;
    // The ouput for Tr(zeta*z) and Tr(zeta*z*K)
    uint128_t *rlt1;

    // DPF evaluation polynomials
    struct GR128 *polys;
    struct GR128 *poly_buf;
    // The polynomial for z and z*K
    struct GR128 *z_poly0;
    // The polynomial for zeta*z and zeta*z*K
    struct GR128 *z_poly1;
    // Shares and cache for each dpf_block
    uint128_t *shares;
    uint128_t *cache;
};


void SPDZ2k_64_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time);

void sample_SPDZ2k_64_a_and_tensor(const struct Param *param, struct FFT_GR128_Trace_A *fft_gr128_trace_a);

void mult_SPDZ2k_64(const struct GR128 *a, const struct GR128 *b, struct GR128 *t, const uint128_t modulus);

void sample_SPDZ2k_64_b_DPF_keys(const struct Param *param, struct Keys *keys);
void free_SPDZ2k_64_b_DPF_keys(const struct Param *param, struct Keys *keys);

void evaluate_SPDZ2k_64_b_DPF(const struct Param *param, const struct Keys *keys, struct GR128 *polys, uint128_t *shares, uint128_t *cache);
void convert_SPDZ2k_64_b_to_FFT(const struct Param *param, struct GR128 *polys);
void multiply_SPDZ2k_64_b_FFT(const struct Param *param, struct GR128 **a_polys, const struct GR128 *b_poly, struct GR128 *res_poly);
void sum_SPDZ2k_64_b_FFT_polys(const struct Param *param, struct GR128 *poly_buf, struct GR128 *z_poly);

void evaluate_SPDZ_64_b_DPF_and_sum(
    const struct Param *param,
    const struct Keys *keys,
    struct GR128 **fft_a,
    struct GR128 *polys,
    struct GR128 *poly_buf,
    struct GR128 *z_poly,
    uint128_t *shares,
    uint128_t *cache);

void trace_SPDZ2k_64_b_FFT_polys(const struct Param *param, const struct GR128 *src, uint128_t *b_0, uint128_t *b_1, uint128_t *bm_0, uint128_t *bm_1);

void init_SPDZ2k_64_b(const struct Param *param, struct SPDZ2k_64_b *spdz2k_64_b);
void free_SPDZ2k_64_b(const struct Param *param, struct SPDZ2k_64_b *spdz2k_64_b);
void run_SPDZ2k_64_b(const struct Param *param, struct SPDZ2k_64_b *spdz2k_64_b, struct GR128 **fft_a);

void init_SPDZ2k_64_prod(const struct Param *param, struct SPDZ2k_64_Prod *spdz2k_64_prod);
void free_SPDZ2k_64_prod(const struct Param *param, struct SPDZ2k_64_Prod *spdz2k_64_prod);
void run_SPDZ2k_64_prod(const struct Param *param, struct SPDZ2k_64_Prod *spdz2k_64_prod, struct GR128 **fft_a_tensor_maps);

void sample_SPDZ2k_64_prod_DPF_keys(const struct Param *param, struct Keys *keys);
void free_SPDZ2k_64_prod_DPF_keys(const struct Param *param, struct Keys *keys);

void evaluate_SPDZ2k_64_prod_DPF_and_sum(
    const struct Param *param,
    const struct Keys *keys,
    struct GR128 **fft_a_tensor_maps,
    struct GR128 *polys,
    struct GR128 *poly_buf,
    struct GR128 *z_poly0,
    struct GR128 *z_poly1,
    uint128_t *shares,
    uint128_t *cache);
void evaluate_SPDZ2k_64_prod_DPF(const struct Param *param, const struct Keys *keys, struct GR128 *polys, uint128_t *shares, uint128_t *cache);
void convert_SPDZ2k_64_prod_to_FFT(const struct Param *param, struct GR128 *polys);
void multiply_SPDZ2k_64_prod_FFT(const struct Param *param, struct GR128 **a_polys, const struct GR128 *b_poly, struct GR128 *res_poly);
void sum_SPDZ2k_64_prod_FFT_polys(const struct Param *param, struct GR128 *poly_buf, struct GR128 *z_poly);
void sum_SPDZ2k_64_prod_FFT_polys_special(const struct Param *param, struct GR128 *poly_buf, struct GR128 *z_poly);
void trace_SPDZ2k_64_prod_FFT_polys(const struct Param *param, struct GR128 *z_poly, uint128_t *rlt);

#endif