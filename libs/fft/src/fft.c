#include <stdlib.h>
#include <stdio.h>
#include "fft.h"
#include "utils.h"
// #include "gf128.h"
#include "gr64_bench.h"
#include "gr128_bench.h"
#include "trace_f8_bench.h"

void fft_recursive_uint64(
    uint64_t *coeffs,
    const size_t num_vars,
    const size_t num_coeffs)
{
    // coeffs (coeffs_h, coeffs_l) are parsed as L(left)|M(middle)|R(right)
    if (num_vars > 1) {
        // apply FFT on all left coefficients
        fft_recursive_uint64(
            &coeffs[0],
            num_vars - 1,
            num_coeffs / 3);

        // apply FFT on all middle coefficients
        fft_recursive_uint64(
            &coeffs[num_coeffs],
            num_vars - 1,
            num_coeffs / 3);

        // apply FFT on all right coefficients
        fft_recursive_uint64(
            &coeffs[2 * num_coeffs],
            num_vars - 1,
            num_coeffs / 3);
    }

    // temp variables to store intermediate values
    uint64_t tL, tM;
    uint64_t mult, xor_h, xor_l;

    uint64_t *coeffsL = &coeffs[0];
    uint64_t *coeffsM = &coeffs[num_coeffs];
    uint64_t *coeffsR = &coeffs[2 * num_coeffs];

    const uint64_t pattern = 0xaaaaaaaaaaaaaaaa;
    const uint64_t mask_h = pattern;     // 0b101010101010101001010
    const uint64_t mask_l = mask_h >> 1; // 0b010101010101010100101

    for (size_t j = 0; j < num_coeffs; j++)
    {
        xor_h = (coeffsM[j] ^ coeffsR[j]) & mask_h;
        xor_l = (coeffsM[j] ^ coeffsR[j]) & mask_l;

        // pre compute: \alpha * (cM[j] ^ cR[j])
        // computed as: mult_l = (h ^ l) and mult_h = l
        // mult_l = (xor&mask_h>>1) ^ (xor & mask_l) [align h and l then xor]
        // mult_h = (xor&mask_l) shifted left by 1 to put in h place [shift and OR into place]
        mult = (xor_h >> 1) ^ (xor_l) | (xor_l << 1);

        // tL coefficient obtained by evaluating on X_i=1
        tL = coeffsL[j] ^ coeffsM[j] ^ coeffsR[j];

        // tM coefficient obtained by evaluating on X_i=\alpha
        tM = coeffsL[j] ^ coeffsR[j] ^ mult;

        // Explanation:
        // cL + cM*\alpha + cR*\alpha^2
        // = cL + cM*\alpha + cR*\alpha + cR
        // = cL + cR + \alpha*(cM + cR)

        // tR: coefficient obtained by evaluating on X_i=\alpha^2=\alpha + 1
        coeffsR[j] = coeffsL[j] ^ coeffsM[j] ^ mult;

        // Explanation:
        // cL + cM*(\alpha+1) + cR(\alpha+1)^2
        // = cL + cM + cM*\alpha + cR*(3\alpha + 2)
        // = cL + cM + \alpha*(cM + cR)
        // Note: we're in the F_2 field extension so 3\alpha+2 = \alpha+0.

        coeffsL[j] = tL;
        coeffsM[j] = tM;
    }
}

void fft_iterative_uint32_no_data(
    uint32_t *a,
    uint32_t *rlt,
    const size_t num_vars,
    const size_t num_coeffs) {

    uint32_t *pre = a;
    uint32_t *cur = rlt;

    static uint32_t zeta_32 = 0xAAAAAAAA;

    for (size_t i = 0; i < num_vars; ++i) {
        size_t blk_num = ipow(3, num_vars-(i+1));
        size_t pow_3_i = ipow(3, i);
        for (size_t blk_id = 0; blk_id < blk_num; blk_id++) {
            for (size_t j=blk_id*3*pow_3_i; j < blk_id*3*pow_3_i+pow_3_i; ++j) {
                uint32_t f0 = pre[j];
                uint32_t f1 = pre[j+pow_3_i];
                uint32_t f2 = pre[j+2*pow_3_i];
                uint32_t tmp_mult = multiply_32(f1^f2, zeta_32);
                cur[j] = f0^f1^f2;
                cur[j+pow_3_i] = f0^f2^tmp_mult;
                cur[j+2*pow_3_i] = f0^f1^tmp_mult;
            }
        }
        uint32_t *tmp = pre;
        pre = cur;
        cur = tmp;
    }
    if (num_vars%2==0) {
        memcpy(rlt, a, sizeof(uint32_t)*num_coeffs);
    }
}

void fft_iterative_uint64_no_data(
    uint64_t *a,
    uint64_t *rlt,
    const size_t num_vars,
    const size_t num_coeffs)
{

    uint64_t *pre = a;
    uint64_t *cur = rlt;

    static uint64_t zeta_64 = 0xAAAAAAAAAAAAAAAA;

    for (size_t i = 0; i < num_vars; ++i)
    {
        size_t blk_num = ipow(3, num_vars - (i + 1));
        size_t pow_3_i = ipow(3, i);
        for (size_t blk_id = 0; blk_id < blk_num; blk_id++)
        {
            for (size_t j = blk_id * 3 * pow_3_i; j < blk_id * 3 * pow_3_i + pow_3_i; ++j)
            {
                uint64_t f0 = pre[j];
                uint64_t f1 = pre[j + pow_3_i];
                uint64_t f2 = pre[j + 2 * pow_3_i];
                uint64_t tmp_mult = multiply_64(f1 ^ f2, zeta_64);
                cur[j] = f0 ^ f1 ^ f2;
                cur[j + pow_3_i] = f0 ^ f2 ^ tmp_mult;
                cur[j + 2 * pow_3_i] = f0 ^ f1 ^ tmp_mult;
            }
        }
        uint64_t *tmp = pre;
        pre = cur;
        cur = tmp;
    }
    if (num_vars % 2 == 0)
    {
        memcpy(rlt, a, sizeof(uint64_t) * num_coeffs);
    }
}

void fft_iterative_uint32(
    const uint32_t *a,
    uint32_t *cache,
    uint32_t *rlt,
    const size_t num_vars,
    const size_t num_coeffs
    ) {

    memcpy(cache, a, sizeof(uint32_t)*num_coeffs);
    fft_iterative_uint32_no_data(cache, rlt, num_vars, num_coeffs);
}

void fft_recursive_uint32(
    uint32_t *coeffs,
    const size_t num_vars,
    const size_t num_coeffs)
{
    // coeffs (coeffs_h, coeffs_l) are parsed as L(left)|M(middle)|R(right)

    if (num_vars > 1)
    {
        // apply FFT on all left coefficients
        fft_recursive_uint32(
            &coeffs[0],
            num_vars - 1,
            num_coeffs / 3);

        // apply FFT on all middle coefficients
        fft_recursive_uint32(
            &coeffs[num_coeffs],
            num_vars - 1,
            num_coeffs / 3);

        // apply FFT on all right coefficients
        fft_recursive_uint32(
            &coeffs[2 * num_coeffs],
            num_vars - 1,
            num_coeffs / 3);
    }

    // temp variables to store intermediate values
    uint32_t tL, tM;
    uint32_t mult, xor_h, xor_l;

    uint32_t *coeffsL = &coeffs[0];
    uint32_t *coeffsM = &coeffs[num_coeffs];
    uint32_t *coeffsR = &coeffs[2 * num_coeffs];

    const uint32_t pattern = 0xaaaaaaaa;
    const uint32_t mask_h = pattern;     // 0b101010101010101001010
    const uint32_t mask_l = mask_h >> 1; // 0b010101010101010100101

    for (size_t j = 0; j < num_coeffs; j++)
    {
        xor_h = (coeffsM[j] ^ coeffsR[j]) & mask_h;
        xor_l = (coeffsM[j] ^ coeffsR[j]) & mask_l;

        // pre compute: \alpha * (cM[j] ^ cR[j])
        // computed as: mult_l = (h ^ l) and mult_h = l
        // mult_l = (xor&mask_h>>1) ^ (xor & mask_l) [align h and l then xor]
        // mult_h = (xor&mask_l) shifted left by 1 to put in h place [shift and OR into place]
        mult = (xor_h >> 1) ^ (xor_l) | (xor_l << 1);

        // tL coefficient obtained by evaluating on X_i=1
        tL = coeffsL[j] ^ coeffsM[j] ^ coeffsR[j];

        // tM coefficient obtained by evaluating on X_i=\alpha
        tM = coeffsL[j] ^ coeffsR[j] ^ mult;

        // Explanation:
        // cL + cM*\alpha + cR*\alpha^2
        // = cL + cM*\alpha + cR*\alpha + cR
        // = cL + cR + \alpha*(cM + cR)

        // tR: coefficient obtained by evaluating on X_i=\alpha^2=\alpha + 1
        coeffsR[j] = coeffsL[j] ^ coeffsM[j] ^ mult;

        // Explanation:
        // cL + cM*(\alpha+1) + cR(\alpha+1)^2
        // = cL + cM + cM*\alpha + cR*(3\alpha + 2)
        // = cL + cM + \alpha*(cM + cR)
        // Note: we're in the F_2 field extension so 3\alpha+2 = \alpha+0.

        coeffsL[j] = tL;
        coeffsM[j] = tM;
    }
}

void fft_iterative_gr64_no_data(
    uint64_t *a0,
    uint64_t *a1,
    uint64_t *rlt0,
    uint64_t *rlt1,
    const size_t num_vars,
    const size_t num_coeffs) {

    uint64_t *pre0 = a0;
    uint64_t *pre1 = a1;
    uint64_t *cur0 = rlt0;
    uint64_t *cur1 = rlt1;

    for (size_t i = 0; i < num_vars; ++i) {
        size_t blk_num = ipow(3, num_vars-(i+1));
        size_t pow_3_i = ipow(3, i);
        for (size_t blk_id = 0; blk_id < blk_num; blk_id++) {
            for (size_t j=blk_id*3*pow_3_i; j < blk_id*3*pow_3_i+pow_3_i; ++j) {
                uint64_t f00 = pre0[j];
                uint64_t f10 = pre0[j+pow_3_i];
                uint64_t f20 = pre0[j+2*pow_3_i];
                uint64_t f01 = pre1[j];
                uint64_t f11 = pre1[j+pow_3_i];
                uint64_t f21 = pre1[j+2*pow_3_i];

                uint64_t tmp_mult0 = -(f11-f21);
                uint64_t tmp_mult1 = (f10-f20) - (f11-f21);

                cur0[j] = f00+f10+f20;
                cur1[j] = f01+f11+f21;

                cur0[j+pow_3_i] = f00-f20+tmp_mult0;
                cur1[j+pow_3_i] = f01-f21+tmp_mult1;

                cur0[j+2*pow_3_i] = f00-f10-tmp_mult0;
                cur1[j+2*pow_3_i] = f01-f11-tmp_mult1;
            }
        }
        uint64_t *tmp0 = pre0;
        uint64_t *tmp1 = pre1;
        pre0 = cur0; pre1 = cur1;
        cur0 = tmp0; cur1 = tmp1;
    }
    if (num_vars%2==0) {
        memcpy(rlt0, a0, sizeof(uint64_t)*num_coeffs);
        memcpy(rlt1, a1, sizeof(uint64_t)*num_coeffs);
    }
}

void fft_iterative_gr64(
    const uint64_t *a0,
    const uint64_t *a1,
    uint64_t *cache0,
    uint64_t *cache1,
    uint64_t *rlt0,
    uint64_t *rlt1,
    const size_t num_vars,
    const size_t num_coeffs
    ) {

    memcpy(cache0, a0, sizeof(uint64_t)*num_coeffs);
    memcpy(cache1, a1, sizeof(uint64_t)*num_coeffs);
    fft_iterative_gr64_no_data(cache0, cache1, rlt0, rlt1, num_vars, num_coeffs);
}

void fft_recursive_gr64(
    struct GR64 *coeffs,
    const size_t num_vars,
    const size_t num_coeffs
    ) {
    // coeffs (coeffs_h, coeffs_l) are parsed as L(left)|M(middle)|R(right)
    if (num_vars > 1) {
        // apply FFT on all left coefficients
        fft_recursive_gr64(
            &coeffs[0],
            num_vars - 1,
            num_coeffs / 3);
        // apply FFT on all middle coefficients
        fft_recursive_gr64(
            &coeffs[num_coeffs],
            num_vars - 1,
            num_coeffs / 3);
        // apply FFT on all right coefficients
        fft_recursive_gr64(
            &coeffs[2 * num_coeffs],
            num_vars - 1,
            num_coeffs / 3);
    }

    // temp variables to store intermediate values
    struct GR64 tL, tM;
    struct GR64 mult;

    struct GR64 *coeffsL = &coeffs[0];
    struct GR64 *coeffsM = &coeffs[num_coeffs];
    struct GR64 *coeffsR = &coeffs[2 * num_coeffs];

    for (size_t j = 0; j < num_coeffs; j++) {
        
        // pre compute: \alpha * (cM[j] ^ cR[j])
        // TODO: optimize this code via direct multiplication
        mult.c0 = coeffsM[j].c1 - coeffsR[j].c1;
        mult.c1 = (coeffsM[j].c0 - coeffsR[j].c0)-(coeffsM[j].c1-coeffsR[j].c1);

        // TODO: add three points
        // tL coefficient obtained by evaluating on X_i=1
        tL.c0 = coeffsL[j].c0 + coeffsM[j].c0 + coeffsR[j].c0;
        tL.c1 = coeffsL[j].c1 + coeffsM[j].c1 + coeffsR[j].c1;

        // tM coefficient obtained by evaluating on X_i=\alpha
        tM.c0 = coeffsL[j].c0 - coeffsR[j].c0 + mult.c0;
        tM.c1 = coeffsL[j].c1 - coeffsR[j].c1 + mult.c1;

        // tR: coefficient obtained by evaluating on X_i=\alpha^2=-\alpha - 1
        coeffsR[j].c0 = coeffsL[j].c0 - coeffsM[j].c0 - mult.c0;
        coeffsR[j].c1 = coeffsL[j].c1 - coeffsM[j].c1 - mult.c1;

        memcpy(&coeffsL[j], &tL, sizeof(struct GR64));
        memcpy(&coeffsM[j], &tM, sizeof(struct GR64));
    }
}

void fft_recursive_SPDZ2k_32(
    struct GR64 *coeffs,
    const size_t num_vars,
    const size_t num_coeffs,
    const uint64_t modulus64
    ) {
    // coeffs (coeffs_h, coeffs_l) are parsed as L(left)|M(middle)|R(right)
    if (num_vars > 1) {
        // apply FFT on all left coefficients
        fft_recursive_gr64(
            &coeffs[0],
            num_vars - 1,
            num_coeffs / 3);
        // apply FFT on all middle coefficients
        fft_recursive_gr64(
            &coeffs[num_coeffs],
            num_vars - 1,
            num_coeffs / 3);
        // apply FFT on all right coefficients
        fft_recursive_gr64(
            &coeffs[2 * num_coeffs],
            num_vars - 1,
            num_coeffs / 3);
    }

    // temp variables to store intermediate values
    struct GR64 tL, tM;
    struct GR64 mult;

    struct GR64 *coeffsL = &coeffs[0];
    struct GR64 *coeffsM = &coeffs[num_coeffs];
    struct GR64 *coeffsR = &coeffs[2 * num_coeffs];

    for (size_t j = 0; j < num_coeffs; j++) {
        
        // pre compute: \alpha * (cM[j] ^ cR[j])
        // TODO: optimize this code via direct multiplication
        mult.c0 = (coeffsM[j].c1 - coeffsR[j].c1)%modulus64;
        mult.c1 = ((coeffsM[j].c0 - coeffsR[j].c0)-(coeffsM[j].c1-coeffsR[j].c1))%modulus64;

        // TODO: add three points
        // tL coefficient obtained by evaluating on X_i=1
        tL.c0 = (coeffsL[j].c0 + coeffsM[j].c0 + coeffsR[j].c0)%modulus64;
        tL.c1 = (coeffsL[j].c1 + coeffsM[j].c1 + coeffsR[j].c1)%modulus64;

        // tM coefficient obtained by evaluating on X_i=\alpha
        tM.c0 = (coeffsL[j].c0 - coeffsR[j].c0 + mult.c0)%modulus64;
        tM.c1 = (coeffsL[j].c1 - coeffsR[j].c1 + mult.c1)%modulus64;

        // tR: coefficient obtained by evaluating on X_i=\alpha^2=-\alpha - 1
        coeffsR[j].c0 = (coeffsL[j].c0 - coeffsM[j].c0 - mult.c0)%modulus64;
        coeffsR[j].c1 = (coeffsL[j].c1 - coeffsM[j].c1 - mult.c1)%modulus64;

        memcpy(&coeffsL[j], &tL, sizeof(struct GR64));
        memcpy(&coeffsM[j], &tM, sizeof(struct GR64));
    }
}

void fft_recursive_f8_uint32(
    uint32_t *coeffs,
    const uint8_t *zeta_powers,
    const size_t num_vars,
    const size_t num_coeffs,
    const struct Param *param,
    const size_t base
    ) {
    if (num_vars > 1) {
        for (size_t i = 0; i < base; ++i) {
            // apply FFT on each branch
            fft_recursive_f8_uint32(&coeffs[i*num_coeffs], zeta_powers, num_vars-1, num_coeffs/base, param, base);
        }
    }
    // temp variables to store intermediate values
    uint32_t mult;
    uint32_t *coeffs_pos[base];
    for (size_t i = 0; i < base; ++i) {
        coeffs_pos[i] = &coeffs[i*num_coeffs];
    }

    for (size_t j = 0; j < num_coeffs; j++) {
        uint32_t t_coeffs[base];
        memset(t_coeffs, 0, base*sizeof(uint32_t));
        // compute the first base-1 evaluations and store to t_coeffs
        // coeffs_pos[0...base-1][j] is the coefficients
        for (size_t k = 0; k < base; ++k) {
            for (size_t i = 0; i < base; ++i) {
                packed_scalar_mult_f8_trace(param, coeffs_pos[i][j], zeta_powers[k*i%base], &mult);
                t_coeffs[k] = mult ^ t_coeffs[k];
            }
        }
        // copy temp to the value back
        for (size_t i = 0; i < base; ++i) {
            coeffs_pos[i][j] = t_coeffs[i];
        }
    }
}

void fft_recursive_mal128_f4_trace(
    uint128_t *coeffs,
    const uint8_t *zeta_powers,
    const size_t num_vars,
    const size_t num_coeffs,
    const struct Param *param,
    const size_t base
    ) {
    if (num_vars > 1) {
        for (size_t i = 0; i < base; ++i) {
            // apply FFT on each branch
            fft_recursive_mal128_f4_trace(&coeffs[i*num_coeffs], zeta_powers, num_vars-1, num_coeffs/base, param, base);
        }
    }
    // temp variables to store intermediate values
    uint128_t mult;
    uint128_t *coeffs_pos[base];
    for (size_t i = 0; i < base; ++i) {
        coeffs_pos[i] = &coeffs[i*num_coeffs];
    }

    for (size_t j = 0; j < num_coeffs; j++) {
        uint128_t t_coeffs[base];
        memset(t_coeffs, 0, base*sizeof(uint128_t));
        // compute the first base-1 evaluations and store to t_coeffs
        // coeffs_pos[0...base-1][j] is the coefficients
        for (size_t k = 0; k < base; ++k) {
            for (size_t i = 0; i < base; ++i) {
                /**
                 * TODO: zeta_powers should be GF(2^128) elements.
                 */
                mult = mult_mal128_f4(coeffs_pos[i][j], zeta_powers[k*i%base]);
                t_coeffs[k] ^= mult;
            }
        }
        // copy temp to the value back
        for (size_t i = 0; i < base; ++i) {
            coeffs_pos[i][j] = t_coeffs[i];
        }
    }
}

void fft_recursive_SPDZ2k_32_D3(
    struct GR64_D3 *coeffs,
    const struct GR64_D3 *zeta_powers,
    const size_t num_vars,
    const size_t num_coeffs,
    const uint64_t modulus64,
    const size_t base
    ) {
    if (num_vars > 1) {
        for (size_t i = 0; i < base; ++i) {
            // apply FFT on each branch
            fft_recursive_SPDZ2k_32_D3(&coeffs[i * num_coeffs], zeta_powers, num_vars-1, num_coeffs/base, modulus64, base);
        }
    }
    // temp variables to store intermediate values
    struct GR64_D3 mult;
    struct GR64_D3 *coeffs_pos[base];
    for (size_t i = 0; i < base; ++i) {
        coeffs_pos[i] = &coeffs[i*num_coeffs];
    }

    for (size_t j = 0; j < num_coeffs; j++) {
        struct GR64_D3 t_coeffs[base];
        memset(t_coeffs, 0, base*sizeof(struct GR64_D3));
        // compute the first base-1 evaluations and store to t_coeffs
        // coeffs_pos[0...base-1][j] is the coefficients
        for (size_t k = 0; k < base; ++k) {
            for (size_t i = 0; i < base; ++i) {
                mult_gr64_D3(&coeffs_pos[i][j], &zeta_powers[k*i%base], &mult);
                add_gr64_D3(&mult, &t_coeffs[k], &t_coeffs[k]);
            }
        }
        // copy temp to the value back
        for (size_t i = 0; i < base; ++i) {
            memcpy(&coeffs_pos[i][j], &t_coeffs[i], sizeof(struct GR64_D3));
        }
    }
}


void fft_recursive_SPDZ2k_64_D3(
    struct GR128_D3 *coeffs,
    const struct GR128_D3 *zeta_powers,
    const size_t num_vars,
    const size_t num_coeffs,
    const uint128_t modulus128,
    const size_t base
    ) {
    if (num_vars > 1) {
        for (size_t i = 0; i < base; ++i) {
            // apply FFT on each branch
            fft_recursive_SPDZ2k_64_D3(&coeffs[i * num_coeffs], zeta_powers, num_vars-1, num_coeffs/base, modulus128, base);
        }
    }
    // temp variables to store intermediate values
    struct GR128_D3 mult;
    struct GR128_D3 *coeffs_pos[base];
    for (size_t i = 0; i < base; ++i) {
        coeffs_pos[i] = &coeffs[i*num_coeffs];
    }

    for (size_t j = 0; j < num_coeffs; j++) {
        struct GR128_D3 t_coeffs[base];
        memset(t_coeffs, 0, base*sizeof(struct GR128_D3));
        // compute the first base-1 evaluations and store to t_coeffs
        // coeffs_pos[0...base-1][j] is the coefficients
        for (size_t k = 0; k < base; ++k) {
            for (size_t i = 0; i < base; ++i) {
                mult_gr128_D3(&coeffs_pos[i][j], &zeta_powers[k*i%base], &mult);
                add_gr128_D3(&mult, &t_coeffs[k], &t_coeffs[k]);
            }
        }
        // copy temp to the value back
        for (size_t i = 0; i < base; ++i) {
            memcpy(&coeffs_pos[i][j], &t_coeffs[i], sizeof(struct GR128_D3));
        }
    }
}


void fft_recursive_SPDZ2k_32_D4(
    struct GR64_D4 *coeffs,
    const struct GR64_D4 *zeta_powers,
    const size_t num_vars,
    const size_t num_coeffs,
    const uint64_t modulus64,
    const size_t base
    ) {
    if (num_vars > 1) {
        for (size_t i = 0; i < base; ++i) {
            // apply FFT on each branch
            fft_recursive_SPDZ2k_32_D4(&coeffs[i * num_coeffs], zeta_powers, num_vars-1, num_coeffs/base, modulus64, base);
        }
    }
    // temp variables to store intermediate values
    struct GR64_D4 mult;
    struct GR64_D4 *coeffs_pos[base];
    for (size_t i = 0; i < base; ++i) {
        coeffs_pos[i] = &coeffs[i*num_coeffs];
    }

    for (size_t j = 0; j < num_coeffs; j++) {
        struct GR64_D4 t_coeffs[base];
        memset(t_coeffs, 0, base*sizeof(struct GR64_D4));
        // compute the first base-1 evaluations and store to t_coeffs
        // coeffs_pos[0...base-1][j] is the coefficients
        for (size_t k = 0; k < base; ++k) {
            for (size_t i = 0; i < base; ++i) {
                mult_gr64_D4(&coeffs_pos[i][j], &zeta_powers[k*i%base], &mult);
                add_gr64_D4(&mult, &t_coeffs[k], &t_coeffs[k]);
            }
        }
        // copy temp to the value back
        for (size_t i = 0; i < base; ++i) {
            memcpy(&coeffs_pos[i][j], &t_coeffs[i], sizeof(struct GR64_D4));
        }
    }
}

void fft_recursive_SPDZ2k_64_D4(
    struct GR128_D4 *coeffs,
    const struct GR128_D4 *zeta_powers,
    const size_t num_vars,
    const size_t num_coeffs,
    const uint64_t modulus64,
    const size_t base
    ) {
    if (num_vars > 1) {
        for (size_t i = 0; i < base; ++i) {
            // apply FFT on each branch
            fft_recursive_SPDZ2k_64_D4(&coeffs[i * num_coeffs], zeta_powers, num_vars-1, num_coeffs/base, modulus64, base);
        }
    }
    // temp variables to store intermediate values
    struct GR128_D4 mult;
    struct GR128_D4 *coeffs_pos[base];
    for (size_t i = 0; i < base; ++i) {
        coeffs_pos[i] = &coeffs[i*num_coeffs];
    }

    for (size_t j = 0; j < num_coeffs; j++) {
        struct GR128_D4 t_coeffs[base];
        memset(t_coeffs, 0, base*sizeof(struct GR128_D4));
        // compute the first base-1 evaluations and store to t_coeffs
        // coeffs_pos[0...base-1][j] is the coefficients
        for (size_t k = 0; k < base; ++k) {
            for (size_t i = 0; i < base; ++i) {
                mult_gr128_D4(&coeffs_pos[i][j], &zeta_powers[k*i%base], &mult);
                add_gr128_D4(&mult, &t_coeffs[k], &t_coeffs[k]);
            }
        }
        // copy temp to the value back
        for (size_t i = 0; i < base; ++i) {
            memcpy(&coeffs_pos[i][j], &t_coeffs[i], sizeof(struct GR64_D4));
        }
    }
}

void fft_recursive_SPDZ2k_64(
    struct GR128 *coeffs,
    const size_t num_vars,
    const size_t num_coeffs,
    const uint128_t modulus128
    ) {
    // coeffs (coeffs_h, coeffs_l) are parsed as L(left)|M(middle)|R(right)
    if (num_vars > 1) {
        // apply FFT on all left coefficients
        fft_recursive_gr128(
            &coeffs[0],
            num_vars - 1,
            num_coeffs / 3);
        // apply FFT on all middle coefficients
        fft_recursive_gr128(
            &coeffs[num_coeffs],
            num_vars - 1,
            num_coeffs / 3);
        // apply FFT on all right coefficients
        fft_recursive_gr128(
            &coeffs[2 * num_coeffs],
            num_vars - 1,
            num_coeffs / 3);
    }

    // temp variables to store intermediate values
    struct GR128 tL, tM;
    struct GR128 mult;

    struct GR128 *coeffsL = &coeffs[0];
    struct GR128 *coeffsM = &coeffs[num_coeffs];
    struct GR128 *coeffsR = &coeffs[2 * num_coeffs];

    for (size_t j = 0; j < num_coeffs; j++) {
        
        // pre compute: \alpha * (cM[j] ^ cR[j])
        // TODO: optimize this code via direct multiplication
        mult.c0 = (coeffsM[j].c1 - coeffsR[j].c1)%modulus128;
        mult.c1 = ((coeffsM[j].c0 - coeffsR[j].c0)-(coeffsM[j].c1-coeffsR[j].c1))%modulus128;

        // TODO: add three points
        // tL coefficient obtained by evaluating on X_i=1
        tL.c0 = (coeffsL[j].c0 + coeffsM[j].c0 + coeffsR[j].c0)%modulus128;
        tL.c1 = (coeffsL[j].c1 + coeffsM[j].c1 + coeffsR[j].c1)%modulus128;

        // tM coefficient obtained by evaluating on X_i=\alpha
        tM.c0 = (coeffsL[j].c0 - coeffsR[j].c0 + mult.c0)%modulus128;
        tM.c1 = (coeffsL[j].c1 - coeffsR[j].c1 + mult.c1)%modulus128;

        // tR: coefficient obtained by evaluating on X_i=\alpha^2=-\alpha - 1
        coeffsR[j].c0 = (coeffsL[j].c0 - coeffsM[j].c0 - mult.c0)%modulus128;
        coeffsR[j].c1 = (coeffsL[j].c1 - coeffsM[j].c1 - mult.c1)%modulus128;

        memcpy(&coeffsL[j], &tL, sizeof(struct GR128));
        memcpy(&coeffsM[j], &tM, sizeof(struct GR128));
    }
}

void fft_recursive_gr128(
    struct GR128 *coeffs,
    const size_t num_vars,
    const size_t num_coeffs
    ) {
    // coeffs (coeffs_h, coeffs_l) are parsed as L(left)|M(middle)|R(right)
    if (num_vars > 1) {
        // apply FFT on all left coefficients
        fft_recursive_gr128(
            &coeffs[0],
            num_vars - 1,
            num_coeffs / 3);
        // apply FFT on all middle coefficients
        fft_recursive_gr128(
            &coeffs[num_coeffs],
            num_vars - 1,
            num_coeffs / 3);
        // apply FFT on all right coefficients
        fft_recursive_gr128(
            &coeffs[2 * num_coeffs],
            num_vars - 1,
            num_coeffs / 3);
    }

    // temp variables to store intermediate values
    struct GR128 tL, tM;
    struct GR128 mult;

    struct GR128 *coeffsL = &coeffs[0];
    struct GR128 *coeffsM = &coeffs[num_coeffs];
    struct GR128 *coeffsR = &coeffs[2 * num_coeffs];

    for (size_t j = 0; j < num_coeffs; j++) {
        
        // pre compute: \alpha * (cM[j] ^ cR[j])
        // TODO: optimize this code via direct multiplication
        mult.c0 = coeffsM[j].c1 - coeffsR[j].c1;
        mult.c1 = (coeffsM[j].c0 - coeffsR[j].c0)-(coeffsM[j].c1-coeffsR[j].c1);

        // TODO: add three points
        // tL coefficient obtained by evaluating on X_i=1
        tL.c0 = coeffsL[j].c0 + coeffsM[j].c0 + coeffsR[j].c0;
        tL.c1 = coeffsL[j].c1 + coeffsM[j].c1 + coeffsR[j].c1;

        // tM coefficient obtained by evaluating on X_i=\alpha
        tM.c0 = coeffsL[j].c0 - coeffsR[j].c0 + mult.c0;
        tM.c1 = coeffsL[j].c1 - coeffsR[j].c1 + mult.c1;

        // tR: coefficient obtained by evaluating on X_i=\alpha^2=-\alpha - 1
        coeffsR[j].c0 = coeffsL[j].c0 - coeffsM[j].c0 - mult.c0;
        coeffsR[j].c1 = coeffsL[j].c1 - coeffsM[j].c1 - mult.c1;

        memcpy(&coeffsL[j], &tL, sizeof(struct GR128));
        memcpy(&coeffsM[j], &tM, sizeof(struct GR128));
    }
}

void fft_recursive_uint16(
    uint16_t *coeffs,
    const size_t num_vars,
    const size_t num_coeffs)
{
    // coeffs (coeffs_h, coeffs_l) are parsed as L(left)|M(middle)|R(right)

    if (num_vars > 1)
    {
        // apply FFT on all left coefficients
        fft_recursive_uint16(
            &coeffs[0],
            num_vars - 1,
            num_coeffs / 3);

        // apply FFT on all middle coefficients
        fft_recursive_uint16(
            &coeffs[num_coeffs],
            num_vars - 1,
            num_coeffs / 3);

        // apply FFT on all right coefficients
        fft_recursive_uint16(
            &coeffs[2 * num_coeffs],
            num_vars - 1,
            num_coeffs / 3);
    }

    // temp variables to store intermediate values
    uint16_t tL, tM;
    uint16_t mult, xor_h, xor_l;

    uint16_t *coeffsL = &coeffs[0];
    uint16_t *coeffsM = &coeffs[num_coeffs];
    uint16_t *coeffsR = &coeffs[2 * num_coeffs];

    const uint16_t pattern = 0xaaaa;
    const uint16_t mask_h = pattern;     // 0b101010101010101001010
    const uint16_t mask_l = mask_h >> 1; // 0b010101010101010100101

    for (size_t j = 0; j < num_coeffs; j++)
    {
        xor_h = (coeffsM[j] ^ coeffsR[j]) & mask_h;
        xor_l = (coeffsM[j] ^ coeffsR[j]) & mask_l;

        // pre compute: \alpha * (cM[j] ^ cR[j])
        // computed as: mult_l = (h ^ l) and mult_h = l
        // mult_l = (xor&mask_h>>1) ^ (xor & mask_l) [align h and l then xor]
        // mult_h = (xor&mask_l) shifted left by 1 to put in h place [shift and OR into place]
        mult = (xor_h >> 1) ^ (xor_l) | (xor_l << 1);

        // tL coefficient obtained by evaluating on X_i=1
        tL = coeffsL[j] ^ coeffsM[j] ^ coeffsR[j];

        // tM coefficient obtained by evaluating on X_i=\alpha
        tM = coeffsL[j] ^ coeffsR[j] ^ mult;

        // Explanation:
        // cL + cM*\alpha + cR*\alpha^2
        // = cL + cM*\alpha + cR*\alpha + cR
        // = cL + cR + \alpha*(cM + cR)

        // tR: coefficient obtained by evaluating on X_i=\alpha^2=\alpha + 1
        coeffsR[j] = coeffsL[j] ^ coeffsM[j] ^ mult;

        // Explanation:
        // cL + cM*(\alpha+1) + cR(\alpha+1)^2
        // = cL + cM + cM*\alpha + cR*(3\alpha + 2)
        // = cL + cM + \alpha*(cM + cR)
        // Note: we're in the F_2 field extension so 3\alpha+2 = \alpha+0.

        coeffsL[j] = tL;
        coeffsM[j] = tM;
    }
}

void fft_recursive_uint8(
    uint8_t *coeffs,
    const size_t num_vars,
    const size_t num_coeffs)
{
    // coeffs (coeffs_h, coeffs_l) are parsed as L(left)|M(middle)|R(right)

    if (num_vars > 1)
    {
        // apply FFT on all left coefficients
        fft_recursive_uint8(
            &coeffs[0],
            num_vars - 1,
            num_coeffs / 3);

        // apply FFT on all middle coefficients
        fft_recursive_uint8(
            &coeffs[num_coeffs],
            num_vars - 1,
            num_coeffs / 3);

        // apply FFT on all right coefficients
        fft_recursive_uint8(
            &coeffs[2 * num_coeffs],
            num_vars - 1,
            num_coeffs / 3);
    }

    // temp variables to store intermediate values
    uint8_t tL, tM;
    uint8_t mult, xor_h, xor_l;

    uint8_t *coeffsL = &coeffs[0];
    uint8_t *coeffsM = &coeffs[num_coeffs];
    uint8_t *coeffsR = &coeffs[2 * num_coeffs];

    const uint8_t pattern = 0xaa;
    const uint8_t mask_h = pattern;     // 0b101010101010101001010
    const uint8_t mask_l = mask_h >> 1; // 0b010101010101010100101

    for (size_t j = 0; j < num_coeffs; j++)
    {
        xor_h = (coeffsM[j] ^ coeffsR[j]) & mask_h;
        xor_l = (coeffsM[j] ^ coeffsR[j]) & mask_l;

        // pre compute: \alpha * (cM[j] ^ cR[j])
        // computed as: mult_l = (h ^ l) and mult_h = l
        // mult_l = (xor&mask_h>>1) ^ (xor & mask_l) [align h and l then xor]
        // mult_h = (xor&mask_l) shifted left by 1 to put in h place [shift and OR into place]
        mult = (xor_h >> 1) ^ (xor_l) | (xor_l << 1);

        // tL coefficient obtained by evaluating on X_i=1
        tL = coeffsL[j] ^ coeffsM[j] ^ coeffsR[j];

        // tM coefficient obtained by evaluating on X_i=\alpha
        tM = coeffsL[j] ^ coeffsR[j] ^ mult;

        // Explanation:
        // cL + cM*\alpha + cR*\alpha^2
        // = cL + cM*\alpha + cR*\alpha + cR
        // = cL + cR + \alpha*(cM + cR)

        // tR: coefficient obtained by evaluating on X_i=\alpha^2=\alpha + 1
        coeffsR[j] = coeffsL[j] ^ coeffsM[j] ^ mult;

        // Explanation:
        // cL + cM*(\alpha+1) + cR(\alpha+1)^2
        // = cL + cM + cM*\alpha + cR*(3\alpha + 2)
        // = cL + cM + \alpha*(cM + cR)
        // Note: we're in the F_2 field extension so 3\alpha+2 = \alpha+0.

        coeffsL[j] = tL;
        coeffsM[j] = tM;
    }
}
