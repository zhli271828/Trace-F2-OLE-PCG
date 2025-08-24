#ifndef __MAL_128_TRACE_F4_BENCH
#define __MAL_128_TRACE_F4_BENCH

#include <openssl/rand.h>

#include "f4ops.h"

// Packed for F4 elements, at most 16 F4 elements
struct FFT_Mal128_F4_Trace_A {
    // each uint32_t contains c elements
    // fft_a contains c*poly_size elements
    uint32_t *fft_a;
    // fft_a contains c*poly_size elements
    uint32_t *fft_a_square;
    // fft_a_tensor contains c*c*m*m*poly_size elements
    // For a^{2^k}_i a^{2^l}_j
    uint32_t *fft_a_tensor;
};

struct Mal128_F4_Trace_Prod {
    struct Keys *keys;
    size_t m;

    // The output for Delta*Tr(zeta^j*z) with dimension m*poly_size
    uint128_t **rlt;
    
    // The DPF evaluation results with size m*m*c*c*poly_size
    uint128_t *polys;

    // Shares and cache for each dpf_block
    // used by DPF evaluation
    uint128_t *shares;
    uint128_t *cache;
};

void init_fft_mal128_f4_trace_a(const struct Param *param, struct FFT_Mal128_F4_Trace_A *fft_mal128_f4_trace_a);
void sample_mal128_f4_trace_a_and_tensor(const struct Param *param, struct FFT_Mal128_F4_Trace_A *fft_mal128_f4_trace_a);
void free_fft_mal128_f4_trace_a(const struct Param *param, struct FFT_Mal128_F4_Trace_A *fft_mal128_f4_trace_a);

void init_mal128_f4_trace_prod(const struct Param *param, struct Mal128_F4_Trace_Prod *mal128_f4_trace_prod);
void run_mal128_f4_trace_prod(const struct Param *param, struct Mal128_F4_Trace_Prod *mal128_f4_trace_prod, uint32_t *fft_a_tensor, const uint8_t *f4_tr_tbl, const uint8_t *f4_zeta_powers);
void free_mal128_f4_trace_prod(const struct Param *param, struct Mal128_F4_Trace_Prod *mal128_f4_trace_prod);

void sample_mal128_f4_trace_prod_dpf_keys(const struct Param *param, struct Keys *keys);
static void free_mal128_f4_trace_prod_dpf_keys(const struct Param *param, struct Keys *keys);

void mal128_trace_f4_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time);

void evaluate_mal128_f4_trace_prod_dpf(const struct Param *param, const struct Keys *keys, uint128_t *polys, uint128_t *shares, uint128_t *cache);
void convert_mal128_f4_trace_prod_to_fft(const struct Param *param, uint128_t *polys, const uint8_t *f4_zeta_powers);
void multiply_and_sum_mal128_f4_trace_prod(const struct Param *param, uint32_t *fft_a_tensor, uint128_t *polys, uint128_t **rlt, const uint8_t *f4_zeta_powers);

static void compute_f4_trace(uint8_t *rlt, const struct Param *param, const uint8_t *f4_tr_tbl);

static void copy_mal128_f4_block(uint128_t *poly_block, uint128_t *shares, const size_t dpf_block_size);

static void compute_f4_zeta_powers(uint8_t *f4_zeta_powers, const size_t base);
static void compute_f4_tr_tbl(uint8_t *f4_tr_tbl, const size_t m, uint8_t *f4_zeta_powers);

static uint8_t mult_f4_single(uint8_t a, uint8_t b);
uint128_t mult_mal128_f4(const uint128_t a, const uint128_t b);
static void scalar_mult_mal128_trace_f4(uint8_t scalar, uint128_t b, uint128_t* t);

#endif