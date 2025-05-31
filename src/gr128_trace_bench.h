#ifndef __GR128_TRACE_BENCH
#define __GR128_TRACE_BENCH

#include "common.h"
#include "gr128_bench.h"

struct FFT_GR128_Trace_A {
    struct GR128 **fft_a;
    struct GR128 **fft_a_maps;
    struct GR128 **fft_a_tensor_maps;
};

void gr128_trace_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time);

void init_FFT_GR128_Trace_A(const struct Param *param, struct FFT_GR128_Trace_A *fft_gr128_trace_a);
void free_FFT_GR128_Trace_A(const struct Param *param, struct FFT_GR128_Trace_A *fft_gr128_trace_a);

void sample_gr128_trace_a_and_tensor(const struct Param *param, struct FFT_GR128_Trace_A *fft_gr128_trace_a);

void trace_gr128_FFT_polys(const struct Param *param, const struct GR128 *src, uint128_t *rlt);

#endif