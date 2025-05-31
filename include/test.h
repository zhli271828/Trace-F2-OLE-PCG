#ifndef _TEST
#define _TEST

#include <openssl/rand.h>
#include <math.h>
#include "utils.h"
#include "common.h"

void test_pcg();
void bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time);
void sample_a_and_a2(uint8_t *fft_a, uint32_t *fft_a2, size_t poly_size, size_t c);

#endif
