#ifndef __COMMON
#define __COMMON

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <openssl/rand.h>

typedef __int128 int128_t;
typedef unsigned __int128 uint128_t;

// Contain fft_a, ai^2 x aj^2, ai x aj^2, ai^2 x aj, ai x aj
// Each is of length poly_size
struct FFT_Mal_Tensor_A {
    uint8_t *fft_a;
    uint32_t **fft_tensor_a;
    // uint32_t *fft_ai2aj2;
    // uint32_t *fft_aiaj2;
    // uint32_t *fft_ai2aj;
    // uint32_t *fft_aiaj;
};

struct Param {
    size_t n;
    size_t c;
    size_t t;
    size_t poly_size;
    size_t block_size;
    size_t dpf_block_size;
    size_t dpf_domain_bits;
    size_t packed_poly_size;
    size_t packed_block_size;
    uint128_t zeta128;
    uint64_t zeta64;
};

// init the FFT_Mal_Tensor_A structure
void init_fft_mal_tensor_a(const struct Param *param, struct FFT_Mal_Tensor_A *fft_mal_tensor_a, const size_t size);
void free_fft_mal_tensor_a(struct FFT_Mal_Tensor_A *fft_mal_tensor_a, const size_t size);
// samples the a polynomials and ai^2 x aj^2, ai^2 x aj, ai x aj^2, ai x aj polynomials for malicious tensor
void sample_a_and_tensor_mal(const struct Param *param, struct FFT_Mal_Tensor_A *fft_mal_tensor_a);


#endif