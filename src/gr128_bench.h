#ifndef __GR128_BENCH
#define __GR128_BENCH

#include "common.h"
#include "modular_test.h"

struct GR128 {
    uint128_t c0;
    uint128_t c1;
};
struct FFT_GR128_A {
    // length c*poly_size for a
    struct GR128 **fft_a;
    // length c*c*poly_size for a\tensor a
    struct GR128 **fft_a_tensor;
};

void gr128_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time);

// init the FFT_GR128_A structure
void init_FFT_GR128_A(const struct Param *param, struct FFT_GR128_A *fft_gr128_a);
void free_FFT_GR128_A(const struct Param *param, struct FFT_GR128_A *fft_gr128_a);

// samples the a polynomials and tensor axa polynomials
void sample_gr128_a_and_tensor(const struct Param *param, struct FFT_GR128_A *fft_gr128_a);
void mult_gr128(const struct GR128 *a, const struct GR128 *b, struct GR128 *t);

void sample_gr128_DPF_keys(const struct Param* param, struct Keys *keys);
void free_gr128_DPF_keys(const struct Param *param, struct Keys *keys);

void evaluate_gr128_DPF(const struct Param *param, const struct Keys *keys, struct GR128 *polys, uint128_t *shares, uint128_t *cache);

void copy_gr128_block(struct GR128 *poly_block, uint128_t *shares, const size_t dpf_block_size);

void convert_gr128_to_FFT(const struct Param *param, struct GR128 *polys);

void multiply_gr128_FFT(const struct Param *param, struct GR128 **a_polys, const struct GR128 *b_poly, struct GR128 *res_poly);

void sum_gr128_FFT_polys(const struct Param *param, struct GR128 *poly_buf, struct GR128 *z_poly);
void sum_gr128_FFT_polys_special(const struct Param *param, struct GR128 *poly_buf, struct GR128 *z_poly);

#endif