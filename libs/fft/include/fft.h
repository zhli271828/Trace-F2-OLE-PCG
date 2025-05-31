#ifndef _FFT
#define _FFT

#include <string.h>
#include <stdint.h>
#include "gr64_bench.h"
#include "gr128_bench.h"

typedef __int128 int128_t;
typedef unsigned __int128 uint128_t;

// FFT for (up to) 32 polynomials over F4
void fft_recursive_uint64(
    uint64_t *coeffs,
    const size_t num_vars,
    const size_t num_coeffs);

// FFT for (up to) 16 polynomials over F4
void fft_recursive_uint32(
    uint32_t *coeffs,
    const size_t num_vars,
    const size_t num_coeffs);

// FFT for (up to) 8 polynomials over F4
void fft_recursive_uint16(
    uint16_t *coeffs,
    const size_t num_vars,
    const size_t num_coeffs);

// FFT for (up to) 4 polynomials over F4
void fft_recursive_uint8(
    uint8_t *coeffs,
    const size_t num_vars,
    const size_t num_coeffs);


// Iterative FFT does not overwrite the input array
void fft_iterative_uint32(
    const uint32_t *a, // FFT input data
    uint32_t *cache, // cache for the FFT operation
    uint32_t *rlt, // result for the FFT
    const size_t num_vars,
    const size_t num_coeffs
    );

// FFT overwrites the input array
void fft_iterative_uint32_no_data(
    uint32_t *a, // both FFT input data and cache
    uint32_t *rlt, // result for the FFT
    const size_t num_vars,
    const size_t num_coeffs
    );

void fft_iterative_uint64_no_data(
    uint64_t *a,
    uint64_t *rlt,
    const size_t num_vars,
    const size_t num_coeffs
    );
void fft_iterative_gr64_no_data(
    uint64_t *a0,
    uint64_t *a1,
    uint64_t *rlt0,
    uint64_t *rlt1,
    const size_t num_vars,
    const size_t num_coeffs);
void fft_iterative_gr64(
    const uint64_t *a0,
    const uint64_t *a1,
    uint64_t *cache0,
    uint64_t *cache1,
    uint64_t *rlt0,
    uint64_t *rlt1,
    const size_t num_vars,
    const size_t num_coeffs
    );


void fft_recursive_gr64(
    struct GR64 *coeffs,
    const size_t num_vars,
    const size_t num_coeffs
    );

void fft_recursive_SPDZ2k_32(
    struct GR64 *coeffs,
    const size_t num_vars,
    const size_t num_coeffs,
    const uint64_t modulus64
    );

void fft_recursive_SPDZ2k_64(
    struct GR128 *coeffs,
    const size_t num_vars,
    const size_t num_coeffs,
    const uint128_t modulus128
    );

void fft_recursive_gr128(
    struct GR128 *coeffs,
    const size_t num_vars,
    const size_t num_coeffs
    );

#endif
