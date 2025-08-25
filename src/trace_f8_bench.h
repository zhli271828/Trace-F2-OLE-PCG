#ifndef __TRACE_F8_BENCH
#define __TRACE_F8_BENCH

#include <openssl/rand.h>

#include "f4ops.h"

// Packed for F8 elements, uint16_t contains at most 5 F8 elements
struct FFT_F8_Trace_A {
    // each uint32_t contains c elements
    // fft_a contains c*poly_size elements
    uint16_t *fft_a;
    // fft_a_tensor contains c*c*m*poly_size elements and c*c elements are packed
    // For a_i a^{2^l}_j
    uint32_t *fft_a_tensor;
};

struct F8_Trace_Prod {
    struct KeysHD *keys;
    size_t m;

    // The output for Tr(zeta^j*z) with dimension m*poly_size
    uint8_t **rlt;
    
    // The DPF evaluation results
    uint32_t *polys;

    // Shares and cache for each dpf_block
    // used by DPF evaluation
    uint128_t *shares;
    uint128_t *cache;
};

void init_fft_f8_trace_a(const struct Param *param, struct FFT_F8_Trace_A *fft_f8_trace_a);
void sample_f8_trace_a_and_tensor(const struct Param *param, struct FFT_F8_Trace_A * fft_f8_trace_a);
void free_fft_f8_trace_a(const struct Param *param, struct FFT_F8_Trace_A *fft_f8_trace_a);

void init_f8_trace_prod(const struct Param *param, struct F8_Trace_Prod *prod);
void run_f8_trace_prod(const struct Param *param, struct F8_Trace_Prod *f8_trace_prod, uint32_t *fft_a_tensor, const uint8_t *f8_tr_tbl, const uint8_t *f8_zeta_powers);
void free_f8_trace_prod(const struct Param *param, struct F8_Trace_Prod *f8_trace_prod);

void sample_f8_trace_prod_dpf_keys(const struct Param *param, struct KeysHD *keys);
static void free_f8_trace_prod_dpf_keys(const struct Param *param, struct KeysHD *keys);

void trace_f8_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time);

void evaluate_f8_trace_prod_dpf(const struct Param *param, const struct KeysHD *keys, uint32_t *polys, uint128_t *shares, uint128_t *cache);
void convert_f8_trace_prod_to_fft(const struct Param *param, const uint8_t *f8_zeta_powers, uint32_t *polys);
void multiply_and_sum_f8_trace_prod(const struct Param *param, uint32_t *fft_a_tensor, uint32_t *polys, uint8_t **rlt, const uint8_t *f8_zeta_powers);

static void compute_f8_trace(uint8_t *rlt, const struct Param *param, const uint8_t *f8_tr_tbl);

static void copy_f8_block(const struct Param *param, uint32_t *poly_block, const size_t dpf_block_size, uint128_t *shares, const size_t packed_dpf_block_size, size_t i, size_t j);

static void compute_f8_zeta_powers(uint8_t *f8_zeta_powers, const size_t base);
static void compute_f8_tr_tbl(uint8_t *f8_tr_tbl, const size_t m, uint8_t *f8_zeta_powers);

static uint8_t mult_f8_single(uint8_t a, uint8_t b);
void packed_scalar_mult_f8_trace(const struct Param *param, const uint32_t a, const uint8_t b, uint32_t* t);

#endif
