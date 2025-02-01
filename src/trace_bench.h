#ifndef __TRACE_BENCH
#define __TRACE_BENCH

#include <openssl/rand.h>

#include "f4ops.h"

// DPF evaluation result and the caches
struct DPF_Output {
    uint128_t *packed_polys;
    uint128_t *shares;
    uint128_t *cache;
};

// DPF output FFT result
// struct Output_FFT {
//     // Allocate memory for the output FFT
//     // uint32_t *fft_u;
//     uint32_t *res_poly_mat;
//     // Allocate memory for the final inner product
//     uint8_t *z_poly;
// };


// Contain fft_a, ai^2 x aj, ai x aj
struct FFT_Trace_A {
    uint8_t *fft_a;
    // ai^2 x aj
    uint32_t *fft_a2a;
    // ai x aj
    uint32_t *fft_a2;
};

// FFT trace basis polynomials
struct FFT_Trace_Basis {
    uint32_t *fft_a2a;
    uint32_t *fft_a2;
};

static void sum_poly(const uint8_t *z1, const uint8_t *z2, uint8_t *z, const size_t size) {
    for (size_t i = 0; i < size; ++i) {
        z[i] = z1[i]^z2[i];
    }
}

// F4 trace: x -> x*(x+1)
static void trace_poly(uint8_t *z, const size_t size) {
    for (size_t i = 0; i < size; ++i) {
        uint8_t v = z[i];
        z[i] = mult_f4(v, v^0b01);
    }
}

// init the FFT_Trace_A structure
void init_FFT_Trace_A(const struct Param *param, struct FFT_Trace_A *fft_trace_a);
void sample_a_and_tensor(const struct Param *param, struct FFT_Trace_A *fft_trace_a);
void free_fft_trace_a(struct FFT_Trace_A *fft_trace_a);

void sample_trace_DPF_keys(const struct Param* param, struct Keys *keys1, struct Keys *keys2);
void free_DPF_keys(const struct Param* param, struct Keys *keys);

void init_trace_mult_rlts(const struct Param *param, uint8_t **trace_mult_rlts, size_t size);
void free_trace_mult_rlts(uint8_t **trace_mult_rlts, size_t size);

void init_DPF_output(const struct Param *param, struct DPF_Output *dpf_output);
void free_DPF_output(struct DPF_Output *dpf_output);

// Multiply a x u and then sum up all of the polynomials
// FFT evaluate the shared polynomial first
void FFT_multiply_then_sum(const struct Param *param, const uint32_t *fft_a, const uint32_t *fft_u, uint32_t *res_poly_mat, uint8_t *z_poly);

void evaluate_DPF_to_FFT(const struct Param *param, const struct Keys *keys, struct DPF_Output *dpf_output, uint32_t *fft_u);
uint32_t pack_to_uint32(const size_t size, const uint32_t a);


double bench_trace_pcg(size_t n, size_t c, size_t t);

#endif
