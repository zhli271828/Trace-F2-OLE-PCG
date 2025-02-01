#ifndef _FFT
#define _FFT

#include <string.h>
#include <stdint.h>

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

#endif
