#ifndef __GR64_TRACE_BENCH
#define __GR64_TRACE_BENCH

#include "common.h"
#include "gr64_bench.h"

struct FFT_GR64_Trace_A {
    // length c*poly_size for a=\sigma^0(a)
    struct GR64 **fft_a;
    // length c*poly_size for \sigma^1(a)
    struct GR64 **fft_a_maps;
    // length c^2*(2*poly_size) for ai*\sigma^0(aj) and ai*\sigma^1(aj)
    struct GR64 **fft_a_tensor_maps;
};


struct FFT_GR64_D3_Trace_A {
    // length m*c*poly_size for \sigma^i(a_j)
    struct GR64_D3 **fft_a;
    // length m*m*c*c*poly_size for \sigma^i(a_k)*\sigma^j(a_l)
    struct GR64_D3 **fft_a_tensor;
};

struct FFT_GR64_D4_Trace_A {
    // length c*poly_size for a=\sigma^0(a)
    struct GR64_D4 **fft_a;
    // length m*m*c*c*poly_size for \sigma^i(a_k)*\sigma^j(a_l)
    struct GR64_D4 **fft_a_tensor;
};

// product structure
struct GR64_Trace_Prod {
    struct Keys *keys;
    struct GR64 *polys;
    struct GR64 *poly_buf;
    struct GR64 *z_poly0;
    struct GR64 *z_poly1;
    uint64_t *rlt0;
    uint64_t *rlt1;
    uint128_t *shares;
    uint128_t *cache;
};

void gr64_trace_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time);

void init_FFT_GR64_Trace_A(const struct Param *param, struct FFT_GR64_Trace_A *fft_gr64_trace_a);
void free_FFT_GR64_Trace_A(const struct Param *param, struct FFT_GR64_Trace_A *fft_gr64_trace_a);

void init_FFT_GR64_d3_Trace_A(const struct Param *param, struct FFT_GR64_D3_Trace_A *fft_gr64_d3_trace_a);
void free_FFT_GR64_d3_Trace_A(const struct Param *param, struct FFT_GR64_D3_Trace_A *fft_gr64_d3_trace_a);
void init_FFT_GR64_d4_Trace_A(const struct Param *param, struct FFT_GR64_D4_Trace_A *fft_gr64_d4_trace_a);
void free_FFT_GR64_d4_Trace_A(const struct Param *param, struct FFT_GR64_D4_Trace_A *fft_gr64_d3_trace_a);

void sample_gr64_trace_a_and_tensor(const struct Param *param, struct FFT_GR64_Trace_A *fft_gr64_trace_a);

void trace_gr64_FFT_polys(const struct Param *param, const struct GR64 *src, uint64_t *rlt);

void init_gr64_trace_prod(const struct Param *param, struct GR64_Trace_Prod *prod);
void free_gr64_trace_prod(const struct Param *param, struct GR64_Trace_Prod *prod);

void run_gr64_trace_prod(const struct Param *param, struct GR64_Trace_Prod *prod, struct GR64 **fft_a_tensor_maps);

#endif