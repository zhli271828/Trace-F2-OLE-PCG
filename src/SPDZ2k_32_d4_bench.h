#ifndef __SPDZ2K_32_D4_BENCH
#define __SPDZ2K_32_D4_BENCH

#include "common.h"
#include "gr64_bench.h"

struct SDPZ2k_32_D4_FFT_A {
    // length c*poly_size for a
    struct GR64_D4 **fft_a;
};

// struct SDPZ2k_32_HD_FFT_A {
//     // length c*poly_size for a
//     struct GR64_HD **fft_a;
// };

/**
 * The structure for (b0, b1) and authenticated (b0, b1).
 */
struct SPDZ2k_32_D4_b {
    // generate keys for (b0, b1)
    struct KeysHD *keys_b0;
    struct KeysHD *keys_b1;

    size_t m;
    // output for (Tr(zeta^j*b0), Tr(zeta^j*b1)) and each has m values for the trace
    // total length m*poly_size
    uint64_t **b0;
    uint64_t **b1;

    // output for (Tr(zeta^j*K*b0), Tr(zeta^j*K*b1)) and each has m values for the trace
    // total length m*poly_size
    uint64_t **bm0;
    uint64_t **bm1;

    // all of the DPF evaluation results for (b, K*b)
    // each is of dimension 2x(c*poly_size)
    struct GR64_D4 **polys; // for DPF evaluation results
    struct GR64_D4 **poly_buf; // for multiplication results

    // for b and K*b with dimension 2xpoly_size
    // the summed up results
    struct GR64_D4 **z_poly;

    // Shares and cache for each dpf_block and length should be DPF_MSG_NUM*block_size
    // used by DPF evaluation
    uint128_t *shares;
    uint128_t *cache;
};

// product structure
struct SPDZ2k_32_D4_Prod {
    // keys for z
    struct KeysHD *keys;

    size_t m;
    // The output for (Tr(zeta^j*z), Tr(zeta^j*z*K)) with dimension mx(2*poly_size)
    uint64_t **rlt;

    // The polynomial for (zeta^i*z, zeta^i*z*K) with dimension mx(2*poly_size)
    struct GR64_D4 **z_poly;

    // all of the DPF evaluation results for (b, K*b)
    // each is of dimension 2x(c*c*m*poly_size)
    struct GR64_D4 **polys;
    struct GR64_D4 **poly_buf;

    // Shares and cache for each dpf_block
    // used by DPF evaluation
    uint128_t *shares;
    uint128_t *cache;
};

void SPDZ2k_32_D4_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time);

void init_SPDZ2k_32_d4_b(const struct Param *param, struct SPDZ2k_32_D4_b *spdz2k_32_d4_b);

void sample_SPDZ2k_32_d4_b_DPF_keys(const struct Param *param, struct KeysHD *keys);

void init_SPDZ2k_32_d4_prod(const struct Param *param, struct SPDZ2k_32_D4_Prod *spdz2k_32_d4_prod);
void sample_SPDZ2k_32_d4_prod_DPF_keys(const struct Param *param, struct KeysHD *keys);

void evaluate_SPDZ2k_32_d4_b_DPF(const struct Param *param, const struct KeysHD *keys, struct GR64_D4 **polys, uint128_t *shares, uint128_t *cache);
void evaluate_SPDZ2k_32_d4_prod_DPF_and_sum(
    const struct Param *param,
    const struct KeysHD *keys,
    struct GR64_D4 **fft_a_tensor_maps,
    struct GR64_D4 **polys,
    struct GR64_D4 **poly_buf,
    struct GR64_D4 **z_poly,
    uint128_t *shares,
    uint128_t *cache,
    struct GR64_D4 **power_scalar,
    const struct GR64_D4 *zeta_powers
);
void evaluate_SPDZ2k_32_d4_b_DPF_and_sum(
    const struct Param *param,
    const struct KeysHD *keys,
    struct GR64_D4 **fft_a,
    struct GR64_D4 **polys,
    struct GR64_D4 **poly_buf,
    struct GR64_D4 **z_poly,
    uint128_t *shares,
    uint128_t *cache,
    const struct GR64_D4 *zeta_powers
);

void mult_SPDZ2k_32_d4_list(const struct GR64_D4 *a, const struct GR64_D4 *b, struct GR64_D4 *t, size_t length, const size_t modulus);
void mult_SPDZ2k_32_d4(const struct GR64_D4 *a, const struct GR64_D4 *b, struct GR64_D4 *t, const uint64_t modulus);

void multiply_SPDZ2k_32_d4_b_FFT(const struct Param *param, struct GR64_D4 **a_polys, struct GR64_D4 **b_poly, struct GR64_D4 **res_poly);

void multiply_SPDZ2k_32_d4_prod_FFT(const struct Param *param, struct GR64_D4 **a_polys, struct GR64_D4 **b_poly, struct GR64_D4 **res_poly);

void evaluate_SPDZ2k_32_d4_prod_DPF(const struct Param *param, const struct KeysHD *keys, struct GR64_D4 **polys, uint128_t *shares, uint128_t *cache);

void free_SPDZ2k_32_d4_b(const struct Param *param, struct SPDZ2k_32_D4_b *spdz2k_32_d4_b);
void free_SPDZ2k_32_d4_b_DPF_keys(const struct Param *param, struct KeysHD *keys);
void free_SPDZ2k_32_d4_prod_DPF_keys(const struct Param *param, struct KeysHD *keys);
void free_SPDZ2k_32_d4_prod(const struct Param *param, struct SPDZ2k_32_D4_Prod *spdz2k_32_d4_prod);


void add_SPDZ2k_32_d4_list_scalar(const struct GR64_D4 *a, const struct GR64_D4 *scalar, const struct GR64_D4 *b, struct GR64_D4 *t, size_t length, const size_t modulus);
void add_SPDZ2k_32_d4_list(const struct GR64_D4 *a, const struct GR64_D4 *b, struct GR64_D4 *t, size_t length, const size_t modulus);
void add_SPDZ2k_32_d4_list(const struct GR64_D4 *a, const struct GR64_D4 *b, struct GR64_D4 *t, size_t length, const size_t modulus);

void compute_d4_Frob_map_zeta_powers(const struct GR64_D4 *powers, struct GR64_D4 **frob_powers, const size_t m);
void compute_d4_power_scalar(const struct GR64_D4 *zeta_powers, struct GR64_D4 **frob_powers, struct GR64_D4 **power_scalar, const size_t m);
void compute_d4_zeta_powers(struct GR64_D4 *zeta_powers, const size_t m);

void convert_SPDZ2k_32_d4_b_to_FFT(const struct Param *param, struct GR64_D4 **polys, const struct GR64_D4 *zeta_powers);
void convert_SPDZ2k_32_d4_prod_to_FFT(const struct Param *param, struct GR64_D4 **polys, const struct GR64_D4 *zeta_powers);

void copy_gr64_d4_block(struct GR64_D4 *poly_block0, struct GR64_D4 *poly_block1, uint128_t *shares, const size_t dpf_block_size);

void run_SPDZ2k_32_d4_b(const struct Param *param, struct SPDZ2k_32_D4_b *spdz2k_32_d4_b, struct GR64_D4 **fft_a, const struct GR64_D4 *zeta_powers);
void run_SPDZ2k_32_d4_prod(const struct Param *param, struct SPDZ2k_32_D4_Prod *spdz2k_32_d4_prod, struct GR64_D4 **fft_a_tensor, struct GR64_D4 **power_scalar, const struct GR64_D4 *zeta_powers);
void sample_SPDZ2k_32_d4_a_and_tensor(const struct Param *param, struct FFT_GR64_D4_Trace_A *fft_gr64_d4_trace_a);

void sample_SPDZ2k_32_d4_b_DPF_keys(const struct Param *param, struct KeysHD *keys);
void sample_SPDZ2k_32_d4_prod_DPF_keys(const struct Param *param, struct KeysHD *keys);

void SPDZ2k_32_D4_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time);

void sum_SPDZ2k_32_d4_b_FFT_polys(const struct Param *param, struct GR64_D4 **poly_buf, struct GR64_D4 **z_poly);

void sum_SPDZ2k_32_d4_prod_FFT_polys(const struct Param *param, struct GR64_D4 **poly_buf, struct GR64_D4 **z_poly, struct GR64_D4 **power_scalar);

void trace_SPDZ2k_32_d4_b_FFT_polys(const struct Param *param, struct GR64_D4 **src, uint64_t **b, uint64_t **bm, const struct GR64_D4 *powers);

void trace_SPDZ2k_32_d4_prod_FFT_polys(const struct Param *param, struct GR64_D4 **z_poly, uint64_t **rlt);

void compute_Frob_gr64_d4(const struct GR64_D4 *prev, struct GR64_D4 *cur);

static inline uint64_t trace_GR64_d4(const struct GR64_D4* z, const size_t m) {
    // use primitive polynomial: X^4 + 4004063733259641452*X^3 - 2*X^2 - 4004063733259641453*X + 1
    const static uint64_t a = 4004063733259641452UL;
    return m*z->c0 - a * (z->c1+z->c2) - z->c3;
}
static inline void add_SPDZ2k_32_d4(const struct GR64_D4 *a, const struct GR64_D4 *b, struct GR64_D4 *t, const uint64_t modulus) {
    t->c0 = a->c0+b->c0;
    t->c1 = a->c1+b->c1;
    t->c2 = a->c2+b->c2;
    t->c3 = a->c3 + b->c3;
    if (modulus != 0) {
        t->c0 = t->c0%modulus;
        t->c1 = t->c1%modulus;
        t->c2 = t->c2%modulus;
        t->c3 = t->c3%modulus;
    }
}

#endif