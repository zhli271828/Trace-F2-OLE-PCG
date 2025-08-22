#ifndef __GR128_TRACE_BENCH
#define __GR128_TRACE_BENCH

#include "common.h"
#include "gr128_bench.h"

struct FFT_GR128_Trace_A {
    struct GR128 **fft_a;
    struct GR128 **fft_a_maps;
    struct GR128 **fft_a_tensor_maps;
};

struct FFT_GR128_D3_Trace_A {
    // length m*c*poly_size for \sigma^i(a_j)
    struct GR128_D3 **fft_a;
    // length m*m*c*c*poly_size for \sigma^i(a_k)*\sigma^j(a_l)
    struct GR128_D3 **fft_a_tensor;
};

struct FFT_GR128_D4_Trace_A {
    // length c*poly_size for a=\sigma^0(a)
    struct GR128_D4 **fft_a;
    // length m*m*c*c*poly_size for \sigma^i(a_k)*\sigma^j(a_l)
    struct GR128_D4 **fft_a_tensor;
};

void init_FFT_GR128_d3_Trace_A(const struct Param *param, struct FFT_GR128_D3_Trace_A *fft_gr128_d3_trace_a);
void free_FFT_GR128_d3_Trace_A(const struct Param *param, struct FFT_GR128_D3_Trace_A *fft_gr128_d3_trace_a);
void init_FFT_GR128_d4_Trace_A(const struct Param *param, struct FFT_GR128_D4_Trace_A *fft_gr128_d4_trace_a);
void free_FFT_GR128_d4_Trace_A(const struct Param *param, struct FFT_GR128_D4_Trace_A *fft_gr128_d3_trace_a);


void gr128_trace_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time);

void init_FFT_GR128_Trace_A(const struct Param *param, struct FFT_GR128_Trace_A *fft_gr128_trace_a);
void free_FFT_GR128_Trace_A(const struct Param *param, struct FFT_GR128_Trace_A *fft_gr128_trace_a);

void sample_gr128_trace_a_and_tensor(const struct Param *param, struct FFT_GR128_Trace_A *fft_gr128_trace_a);

void trace_gr128_FFT_polys(const struct Param *param, const struct GR128 *src, uint128_t *rlt);

#endif