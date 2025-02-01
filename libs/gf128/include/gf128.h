#ifndef __GF128
#define __GF128

#include <stdint.h>
#include <time.h>

typedef __int128 int128_t;
typedef unsigned __int128 uint128_t;

// Note that we can store 128-bit integers, but we cannot explicitly write 128-bit integers.
void print_uint128_t(char *str, uint128_t x);
void print_uint128_t_bits(char *str, uint128_t num);

// Function to multiply two elements in GF(2^128)
// Refer to Algorithm 1 in [MV05] David A. and McGrew John Viega: The Galois/Counter Mode of Operation (GCM). 2005.
// The field is defined by the polynomial f(x) = x^128 + x^7 + x^2 + x + 1.
// The leftmost bit is x_127 and the right most bit is x_0.
uint128_t gf128_multiply(uint128_t a, uint128_t b);
uint128_t gf128_power(uint128_t base, uint128_t exp);

// Iterative FFT does not overwrite the input array
void fft_iterative_f4(
    const uint8_t *a, // FFT input data
    uint8_t *cache, // cache for the FFT operation
    uint8_t *rlt,  // result for the FFT
    size_t n);

// FFT overwrites the input array
void fft_iterative_f4_no_data(
    uint8_t *a, // both FFT input data and cache
    uint8_t *rlt, // result for the FFT
    size_t n);

// this function has no explicit data input
void fft_iterative_gf128_no_data(
    uint128_t *a,
    uint128_t *rlt,
    const size_t num_vars,
    const size_t num_coeffs,
    const uint128_t zeta);

// TODO: optimize the code to avoid the copy of data
void fft_iterative_gf128(
    const uint128_t *a,
    uint128_t *cache,
    uint128_t *rlt,
    const size_t num_vars,
    const size_t num_coeffs,
    uint128_t zeta);

void multiply_fft_gf128(const uint128_t *a_poly, const uint128_t *b_poly, uint128_t *res_poly, size_t size);
void scalar_multiply_fft_gf128(const uint128_t *a_poly, const uint128_t scalar, uint128_t *res_poly, size_t size);

uint128_t gen_gf128_zeta();

#endif