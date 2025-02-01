#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "gf128.h"
#include "utils.h"

// %llX only specifies 64-bit integer and a 128-bit integer
// is splitted into two 64-bit integers.
void print_uint128_t(char *str, uint128_t x) {
    uint64_t l = x&0xFFFFFFFFFFFFFFFF;
    uint64_t h = x>>64;
    printf("%s0x%016llX%016llX\n", str, h, l);
}

// Sometimes uint128_t is stored in distinct orders in different machines or OSs.
// Bit by bit print function outputs the unique binary reprentation.
void print_uint128_t_bits(char *str, uint128_t num) {
    int bits = sizeof(num) * 8; // Number of bits in an integer
    printf("%s", str);
    for (int i = bits - 1; i >= 0; i--) {
        // Extract the i-th bit using bitwise AND and shifting
        uint8_t cur = (uint8_t)((num >> i) & 1);
        printf("%d", cur);
    }
    printf("\n");
}

#define GF_POLY 0x87
// Function to multiply two elements in GF(2^128)
// Refer to Algorithm 1 in [MV05] David A. and McGrew John Viega: The Galois/Counter Mode of Operation (GCM). 2005.
// The field is defined by the polynomial f(x) = x^128 + x^7 + x^2 + x + 1.
// The leftmost bit is x_127 and the right most bit is x_0.
uint128_t gf128_multiply(uint128_t a, uint128_t b) {
    uint128_t result = 0;
    uint128_t carry;

    // Perform polynomial multiplication
    for (int i = 0; i < 128; i++) {
        if (b & 1) { // If the least significant bit of b is 1
            result ^= a;
        }
        // Check if the highest bit of `a` is 1 for reduction
        carry = a & ((uint128_t)1 << 127);
        a <<= 1;

        // If carry exists, reduce with the irreducible polynomial
        if (carry) {
            a ^= (uint128_t)GF_POLY;
        }
        b >>= 1;
    }
    return result;
}

uint128_t gf128_power(uint128_t base, uint128_t exp) {
    uint128_t rlt = 1;

    while (exp > 0) {
        if (exp & 1 == 1) {
            rlt = gf128_multiply(rlt, base);
        }
        base = gf128_multiply(base, base);
        exp = exp >> 1;
    }
    return rlt;
}

// TODO: move the F4 related function to common or utils
void fft_iterative_f4_no_data(uint8_t *pre, uint8_t *cur, size_t n) {
    uint8_t *cache = pre;
    uint8_t *rlt = cur;
    uint8_t zeta=0b10;
    size_t pow_3_n = ipow(3,n);

    for (size_t i = 0; i < n; ++i) {
        size_t blk_num = ipow(3, n-(i+1));
        size_t pow_3_i = ipow(3, i);
        for (size_t blk_id = 0; blk_id < blk_num; blk_id++) {
            for (size_t j=blk_id*3*pow_3_i; j < blk_id*3*pow_3_i+pow_3_i; ++j) {
                uint8_t f0 = pre[j];
                uint8_t f1 = pre[j+pow_3_i];
                uint8_t f2 = pre[j+2*pow_3_i];
                uint8_t tmp_mult = mult_f4(f1^f2, zeta);
                cur[j] = f0^f1^f2;
                cur[j+pow_3_i] = f0^f2^tmp_mult;
                cur[j+2*pow_3_i] = f0^f1^tmp_mult;
            }
        }
        uint8_t *tmp = pre;
        pre = cur;
        cur = tmp;
    }
    if (n%2==0) {
        memcpy(rlt, cache, sizeof(uint8_t)*pow_3_n);
    }
}

void fft_iterative_f4(const uint8_t *a, uint8_t *cache, uint8_t *rlt, size_t n) {

    size_t pow_3_n = ipow(3,n);
    memcpy(cache, a, sizeof(uint8_t)*pow_3_n);
    fft_iterative_f4_no_data(cache, rlt, n);
}

// TODO: optimize the code to avoid the copy of data
void fft_iterative_gf128(
    const uint128_t *a,
    uint128_t *cache,
    uint128_t *rlt,
    const size_t num_vars,
    const size_t num_coeffs,
    uint128_t zeta) {

    memcpy(cache, a, sizeof(uint128_t)*num_coeffs);
    fft_iterative_gf128_no_data(cache, rlt, num_vars, num_coeffs, zeta);
}

void fft_iterative_gf128_no_data(
    uint128_t *a,
    uint128_t *rlt,
    const size_t num_vars,
    const size_t num_coeffs,
    const uint128_t zeta) {

    uint128_t *pre = a;
    uint128_t *cur = rlt;

    for (size_t i = 0; i < num_vars; ++i) {
        size_t blk_num = ipow(3, num_vars-(i+1));
        size_t pow_3_i = ipow(3, i);
        for (size_t blk_id = 0; blk_id < blk_num; blk_id++) {
            for (size_t j=blk_id*3*pow_3_i; j < blk_id*3*pow_3_i+pow_3_i; ++j) {
                uint128_t f0 = pre[j];
                uint128_t f1 = pre[j+pow_3_i];
                uint128_t f2 = pre[j+2*pow_3_i];
                uint128_t tmp_mult = gf128_multiply(f1^f2, zeta);
                cur[j] = f0^f1^f2;
                cur[j+pow_3_i] = f0^f2^tmp_mult;
                cur[j+2*pow_3_i] = f0^f1^tmp_mult;
            }
        }
        uint128_t *tmp = pre;
        pre = cur;
        cur = tmp;
    }
    if (num_vars%2==0) {
        memcpy(rlt, a, sizeof(uint128_t)*num_coeffs);
    }
}


uint128_t gen_gf128_zeta() {
    uint128_t prim = 0b10;
    uint128_t max_uint128_t = (uint128_t)-1;
    return gf128_power(prim, max_uint128_t/3);
}

// TODO: optimize the code to avoid the copy of data
void multiply_fft_gf128(const uint128_t *a_poly, const uint128_t *b_poly, uint128_t *res_poly, size_t size) {
    for (size_t i = 0; i < size; i++) {
        res_poly[i] = gf128_multiply(a_poly[i], b_poly[i]);
    }
}

void scalar_multiply_fft_gf128(const uint128_t *a_poly, const uint128_t scalar, uint128_t *res_poly, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        res_poly[i] = gf128_multiply(a_poly[i], scalar);
    }
}
