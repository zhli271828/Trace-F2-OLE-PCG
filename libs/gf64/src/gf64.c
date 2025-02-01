#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include "gf64.h"
#include "utils.h"


void print_uint64_t(char *str, uint64_t x) {
    printf("%s"PRIu64"\n", str, x);
}

#define IRREDUCIBLE_POLY 0x1B

uint64_t gf64_multiply(uint64_t a, uint64_t b) {
    uint64_t result = 0;    // Result of the multiplication
    uint64_t mask = 1ULL << 63; // To check overflow

    while (b) {
        if (b & 1) {
            result ^= a;    // Add (XOR in GF(2))
        }
        b >>= 1;            // Shift b to process the next bit
        if (a & mask) {     // Check if the high bit is set
            a = (a << 1) ^ IRREDUCIBLE_POLY; // Reduce modulo p(x)
        } else {
            a <<= 1;        // Just shift if no reduction needed
        }
    }
    return result;
}

uint64_t gf64_power(uint64_t base, uint64_t exp) {
    uint64_t rlt = 1;

    while (exp > 0) {
        if (exp & 1 == 1) {
            rlt = gf64_multiply(rlt, base);
        }
        base = gf64_multiply(base, base);
        exp = exp >> 1;
    }
    return rlt;
}

void fft_iterative_gf64_no_data(
    uint64_t *a,
    uint64_t *rlt,
    const size_t num_vars,
    const size_t num_coeffs,
    const uint64_t zeta) {

    uint64_t *pre = a;
    uint64_t *cur = rlt;

    for (size_t i = 0; i < num_vars; ++i) {
        size_t blk_num = ipow(3, num_vars-(i+1));
        size_t pow_3_i = ipow(3, i);
        for (size_t blk_id = 0; blk_id < blk_num; blk_id++) {
            for (size_t j=blk_id*3*pow_3_i; j < blk_id*3*pow_3_i+pow_3_i; ++j) {
                uint64_t f0 = pre[j];
                uint64_t f1 = pre[j+pow_3_i];
                uint64_t f2 = pre[j+2*pow_3_i];
                uint64_t tmp_mult = gf64_multiply(f1^f2, zeta);
                cur[j] = f0^f1^f2;
                cur[j+pow_3_i] = f0^f2^tmp_mult;
                cur[j+2*pow_3_i] = f0^f1^tmp_mult;
            }
        }
        uint64_t *tmp = pre;
        pre = cur;
        cur = tmp;
    }
    if (num_vars%2==0) {
        memcpy(rlt, a, sizeof(uint64_t)*num_coeffs);
    }
}

// TODO: optimize the code to avoid the copy of data
void fft_iterative_gf64(
    const uint64_t *a,
    uint64_t *cache,
    uint64_t *rlt,
    const size_t num_vars,
    const size_t num_coeffs,
    uint64_t zeta) {

    memcpy(cache, a, sizeof(uint64_t)*num_coeffs);
    fft_iterative_gf64_no_data(cache, rlt, num_vars, num_coeffs, zeta);
}

uint64_t gen_gf64_zeta() {
    uint64_t prim = 0b10;
    uint64_t max_uint64_t = (uint64_t)-1;
    return gf64_power(prim, max_uint64_t/3);
}

void multiply_fft_gf64(const uint64_t *a_poly, const uint64_t *b_poly, uint64_t *res_poly, size_t size) {
    for (size_t i = 0; i < size; i++) {
        res_poly[i] = gf64_multiply(a_poly[i], b_poly[i]);
    }
}

void scalar_multiply_fft_gf64(const uint64_t *a_poly, const uint64_t scalar, uint64_t *res_poly, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        res_poly[i] = gf64_multiply(a_poly[i], scalar);
    }
}