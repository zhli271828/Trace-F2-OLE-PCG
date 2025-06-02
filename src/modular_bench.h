#ifndef _MODULAR_BENCH
#define _MODULAR_BENCH

#include <openssl/rand.h>
#include <math.h>

#include "modular_test.h"
#include "prf.h"

void init_gr64_bench_params(struct Param *param, const size_t n, const size_t c, const size_t t, const size_t m);
void init_gr128_bench_params(struct Param *param, const size_t n, const size_t c, const size_t t, const size_t m);
void init_bench_params(struct Param* param, const size_t n, const size_t c, const size_t t);
void init_gr_bench_params(struct Param *param, const size_t n, const size_t c, const size_t t, const size_t m);
void init_SPDZ2k_32_bench_params(struct Param* param, const size_t n, const size_t c, const size_t t, const size_t m, const size_t k, const size_t s);
void init_SPDZ2k_64_bench_params(struct Param* param, const size_t n, const size_t c, const size_t t, const size_t m, const size_t k, const size_t s);

void init_SPDZ2k_32_HD_bench_params(struct Param* param, const size_t n, const size_t c, const size_t t, const size_t m, const size_t k, const size_t s);
void init_gr_HD_bench_params(struct Param *param, const size_t n, const size_t c, const size_t t, const size_t m);

void sample_DPF_keys(const struct Param* param, struct Keys *keys);
void modular_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time);

size_t find_index(size_t a, size_t base);
#endif
