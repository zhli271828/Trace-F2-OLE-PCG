#ifndef __SPDZ2K_32_BENCH
#define __SPDZ2K_32_BENCH

#include "common.h"
#include "gr64_bench.h"

struct SPDZ2k_32_FFT_A {
    // length c*poly_size for a
    struct GR64 **fft_a;
};

struct SPDZ2k_32_b {
    // generate keys for b0, b1
    struct Keys *keys_b0;
    struct Keys *keys_b1;

    // output for (b0,b1) and each has two values for the trace
    uint64_t *b0_0;
    uint64_t *b0_1;
    uint64_t *b1_0;
    uint64_t *b1_1;
    // output for authenticated (b0,b1) and each has two values for the trace
    uint64_t *bm0_0;
    uint64_t *bm0_1;
    uint64_t *bm1_0;
    uint64_t *bm1_1;
    
    // all of the DPF evaluation results for b
    // each is of length 2*c*poly_size
    struct GR64 *polys;
    struct GR64 *poly_buf;

    // for b and K*b with total length 2*poly_size
    struct GR64 *z_poly;
    // Shares and cache for each dpf_block
    // Each is of length 4*block_size
    uint128_t *shares;
    uint128_t *cache;
};

// product structure
struct SPDZ2k_32_Prod {
    // keys for z
    struct Keys *keys;
    // The output for Tr(z) and Tr(z*K) with length 2*t*t*dpf_block_size=2*poly_size
    uint64_t *rlt0;
    // The output for Tr(zeta*z) and Tr(zeta*z*K) with length 2*t*t*dpf_block_size=2*poly_size
    uint64_t *rlt1;

    // The polynomial for z and z*K with length 2*t*t*dpf_block_size=2*poly_size
    struct GR64 *z_poly0;
    // The polynomial for zeta*z and zeta*z*K with length 2*t*t*dpf_block_size=2*poly_size
    struct GR64 *z_poly1;

    // DPF evaluation polynomials with length 2*2*c*c*t*t*dpf_block_size=2*2*c*c*poly_size
    struct GR64 *polys;
    struct GR64 *poly_buf;
    
    // Shares and cache for each dpf_block with length 2*2*dpf_block_size
    uint128_t *shares;
    uint128_t *cache;
};

void SPDZ2k_32_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time);

void sample_SPDZ2k_32_a_and_tensor(const struct Param *param, struct FFT_GR64_Trace_A *fft_gr64_trace_a);

void mult_SPDZ2k_32(const struct GR64 *a, const struct GR64 *b, struct GR64 *t, const uint64_t modulus);

void sample_SPDZ2k_32_b_DPF_keys(const struct Param *param, struct Keys *keys);
void free_SPDZ2k_32_b_DPF_keys(const struct Param *param, struct Keys *keys);

void evaluate_SPDZ2k_32_b_DPF(const struct Param *param, const struct Keys *keys, struct GR64 *polys, uint128_t *shares, uint128_t *cache);
void convert_SPDZ2k_32_b_to_FFT(const struct Param *param, struct GR64 *polys);
void multiply_SPDZ2k_32_b_FFT(const struct Param *param, struct GR64 **a_polys, const struct GR64 *b_poly, struct GR64 *res_poly);
void sum_SPDZ2k_32_b_FFT_polys(const struct Param *param, struct GR64 *poly_buf, struct GR64 *z_poly);

void evaluate_SPDZ2k_32_b_DPF_and_sum(
    const struct Param *param,
    const struct Keys *keys,
    struct GR64 **fft_a,
    struct GR64 *polys,
    struct GR64 *poly_buf,
    struct GR64 *z_poly,
    uint128_t *shares,
    uint128_t *cache);

void trace_SPDZ2k_32_b_FFT_polys(const struct Param *param, const struct GR64 *src, uint64_t *b_0, uint64_t *b_1, uint64_t *bm_0, uint64_t *bm_1);

void init_SPDZ2k_32_b(const struct Param *param, struct SPDZ2k_32_b *spdz2k_32_b);
void free_SPDZ2k_32_b(const struct Param *param, struct SPDZ2k_32_b *spdz2k_32_b);
void run_SPDZ2k_32_b(const struct Param *param, struct SPDZ2k_32_b *spdz2k_32_b, struct GR64 **fft_a);

void init_SPDZ2k_32_prod(const struct Param *param, struct SPDZ2k_32_Prod *spdz2k_32_prod);
void free_SPDZ2k_32_prod(const struct Param *param, struct SPDZ2k_32_Prod *spdz2k_32_prod);
void run_SPDZ2k_32_prod(const struct Param *param, struct SPDZ2k_32_Prod *spdz2k_32_prod, struct GR64 **fft_a_tensor_maps);

void sample_SPDZ2k_32_prod_DPF_keys(const struct Param *param, struct Keys *keys);
void free_SPDZ2k_32_prod_DPF_keys(const struct Param *param, struct Keys *keys);

void evaluate_SPDZ2k_32_prod_DPF_and_sum(
    const struct Param *param,
    const struct Keys *keys,
    struct GR64 **fft_a_tensor_maps,
    struct GR64 *polys,
    struct GR64 *poly_buf,
    struct GR64 *z_poly0,
    struct GR64 *z_poly1,
    uint128_t *shares,
    uint128_t *cache);
void evaluate_SPDZ2k_32_prod_DPF(const struct Param *param, const struct Keys *keys, struct GR64 *polys, uint128_t *shares, uint128_t *cache);
void convert_SPDZ2k_32_prod_to_FFT(const struct Param *param, struct GR64 *polys);
void multiply_SPDZ2k_32_prod_FFT(const struct Param *param, struct GR64 **a_polys, const struct GR64 *b_poly, struct GR64 *res_poly);
void sum_SPDZ2k_32_prod_FFT_polys(const struct Param *param, struct GR64 *poly_buf, struct GR64 *z_poly);
void sum_SPDZ2k_32_prod_FFT_polys_special(const struct Param *param, struct GR64 *poly_buf, struct GR64 *z_poly);
void trace_SPDZ2k_32_prod_FFT_polys(const struct Param *param, struct GR64 *z_poly, uint64_t *rlt);

#endif