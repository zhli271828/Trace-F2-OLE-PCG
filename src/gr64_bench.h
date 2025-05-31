#ifndef __GR64_BENCH
#define __GR64_BENCH

#include "common.h"
#include "modular_test.h"

/**
 * @brief The structure for GR64.
 */
struct GR64 {
    uint64_t c0;
    uint64_t c1;
};
struct FFT_GR64_A {
    // length c*poly_size for a
    struct GR64 **fft_a;
    // length c*c*poly_size for a\tensor a
    struct GR64 **fft_a_tensor;
};

void gr64_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time);
// init the FFT_GR64_A structure
void init_FFT_GR64_A(const struct Param *param, struct FFT_GR64_A *fft_gr64_a);
void free_FFT_GR64_A(const struct Param *param, struct FFT_GR64_A *fft_gr64_a);

// samples the a polynomials and tensor axa polynomials
void sample_gr64_a_and_tensor(const struct Param *param, struct FFT_GR64_A *fft_gr64_a);
void mult_gr64(const struct GR64 *a, const struct GR64 *b, struct GR64 *t);

void sample_gr64_DPF_keys(const struct Param* param, struct Keys *keys);
void free_gr64_DPF_keys(const struct Param *param, struct Keys *keys);

void evaluate_gr64_DPF(const struct Param *param, const struct Keys *keys, struct GR64 *polys, uint128_t *shares, uint128_t *cache);

void copy_gr64_block(struct GR64 *poly_block, uint128_t *shares, const size_t dpf_block_size);

void convert_gr64_to_FFT(const struct Param *param, struct GR64 *polys);

void multiply_gr64_FFT(const struct Param *param, struct GR64 **a_polys, const struct GR64 *b_poly, struct GR64 *res_poly);

void sum_gr64_FFT_polys(const struct Param *param, struct GR64 *poly_buf, struct GR64 *z_poly);
void sum_gr64_FFT_polys_special(const struct Param *param, struct GR64 *poly_buf, struct GR64 *z_poly);

#endif