#ifndef _MODULAR_BENCH
#define _MODULAR_BENCH

#include <openssl/rand.h>
#include <math.h>

#include "modular_test.h"
#include "prf.h"

void init_bench_params(struct Param* param, const size_t n, const size_t c, const size_t t);

void sample_DPF_keys(const struct Param* param, struct Keys *keys);
double modular_bench_pcg(size_t n, size_t c, size_t t);
#endif
