#ifndef _MODULAR_TEST
#define _MODULAR_TEST

#include <openssl/rand.h>
#include <math.h>

#include "prf.h"
#include "utils.h"
#include "common.h"

struct FFT_A {
    uint8_t* fft_a;
    uint32_t* fft_a2;
};

struct Err_Poly {
    // the expanded polynomials
    uint8_t *err_polys_A;
    uint8_t *err_polys_B;
    // error coefficients
    uint8_t *err_poly_coeffs_A;
    uint8_t *err_poly_coeffs_B;
    // error positions
    size_t *err_poly_pos_A;
    size_t *err_poly_pos_B;
};

// FFT of eA and eB in packed form.
// Because c=4, 4 FFTs are packed into a uint8_t.
struct FFT_E {
    uint8_t *fft_eA;
    uint8_t *fft_eB;
};

struct Inner_Prod {
    uint8_t *x_poly_A;
    uint8_t *x_poly_B;
};

struct Keys {
    struct DPFKey** dpf_keys_A;
    struct DPFKey** dpf_keys_B;
    struct PRFKeys* prf_keys;
};

struct Err_Cross_Poly {
    uint8_t *err_poly_cross_coef;
    size_t *err_poly_cross_pos;
    uint8_t *err_polys_cross;
};

struct Packed_Polys {
    uint128_t *packed_polys_A;
    uint128_t *packed_polys_B;
};

struct FFT_U {
    uint32_t *fft_uA;
    uint32_t *fft_uB;
};

struct Z_Polys {
    uint8_t *z_poly_A;
    uint8_t *z_poly_B;
};

static void add_trits(const uint8_t *a, const uint8_t *b, uint8_t *res, const size_t size) {
    for (size_t i = 0; i < size; i++) {
        res[i]=(a[i]+b[i])%3;
    }
}

void step0(const struct Param *param, struct FFT_A *fft);
void step1(const struct Param *param, struct Err_Poly *err_poly, struct FFT_E  *fft_e);
void step2(const struct Param *param, struct FFT_A *fft, struct FFT_E *fft_e, struct Inner_Prod *inner_prod);
void step3(const struct Param *param, const struct Err_Poly *err_poly, struct Err_Cross_Poly *err_cross_poly);
void step4(const struct Param *param, const struct Err_Cross_Poly *err_cross_poly, struct Keys *keys);
void step5(const struct Param *param, const struct Keys *keys, struct Packed_Polys *packed_polys);
void step6(const struct Param *param, const struct Packed_Polys *packed_polys, struct FFT_U *fft_u);
void step7(const struct Param *param, const struct FFT_A *fft, const struct FFT_U *fft_u, const struct Inner_Prod *inner_prod, struct Z_Polys *z_polys);

void init_params(struct Param* param, const size_t n, const size_t c, const size_t t);
void verify_cross_prod_error_numbers(const struct Param *param, const struct Packed_Polys *packed_polys);
void verify_shares(const struct Param *param, const struct Inner_Prod *inner_prod, const struct Z_Polys *z_polys);

void step0_free(struct FFT_A *fft);
void step1_free(struct Err_Poly *err_poly, struct FFT_E *fft_e);
void step2_free(struct Inner_Prod *inner_prod);
void step3_free(struct Err_Cross_Poly *err_cross_poly);
void step4_free(const struct Param *param, struct Keys *keys);
void step5_free(struct Packed_Polys *packed_polys);
void step6_free(struct FFT_U *fft_u);
void step7_free(struct Z_Polys *z_polys);

void init_bench_params(struct Param* param, const size_t n, const size_t c, const size_t t);
void sample_DPF_keys(const struct Param* param, struct Keys *keys);
void modular_test_pcg();
void modular_bench_pcg(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time);

#endif
