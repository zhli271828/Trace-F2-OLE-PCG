#ifndef __GF64
#define __GF64

#include <stdint.h>
#include <time.h>


void print_uint64_t(char *str, uint64_t x);


// Function to multiply two elements in GF(2^64)
// The field is defined by the polynomial f(x) = x^64 + x^4 + x^3 + x + 1
// The leftmost bit is x_63 and the right most bit is x_0.
uint64_t gf64_multiply(uint64_t a, uint64_t b);
uint64_t gf64_power(uint64_t base, uint64_t exp);

// this function has no explicit data input
void fft_iterative_gf64_no_data(
    uint64_t *a,
    uint64_t *rlt,
    const size_t num_vars,
    const size_t num_coeffs,
    const uint64_t zeta);

// TODO: optimize the code to avoid the copy of data
void fft_iterative_gf64(
    const uint64_t *a,
    uint64_t *cache,
    uint64_t *rlt,
    const size_t num_vars,
    const size_t num_coeffs,
    uint64_t zeta);

void multiply_fft_gf64(const uint64_t *a_poly, const uint64_t *b_poly, uint64_t *res_poly, size_t size);
void scalar_multiply_fft_gf64(const uint64_t *a_poly, const uint64_t scalar, uint64_t *res_poly, size_t size);

uint64_t gen_gf64_zeta();

#endif