#include <stdlib.h>
#include <stdio.h>
#include "fft.h"
#include "modular_test.h"
#include "trace_bench.h"
#include "mal_gf128_trace_bench.h"
#include "gf128.h"
#include "utils.h"
#include "common.h"
#include "f4ops.h"

// TODO: what is KEY_NUM or why it is KEY_NUM
#define KEY_NUM 4
#define TENSOR_A_NUM 4
#define RLT_NUM 2

double mal_gf128_trace_bench_pcg(size_t n, size_t c, size_t t) {

   if (c > 4) {
        printf("ERROR: currently only implemented for c <= 4");
        exit(0);
    }
    struct Param *param = calloc(1, sizeof(struct Param));
    uint128_t zeta = gen_gf128_zeta();
    init_mal_gf128_bench_params(param, n, c, t, zeta);
    struct FFT_Mal_Tensor_A *fft_mal_tensor_a = calloc(1, sizeof(struct FFT_Mal_Tensor_A));
    init_fft_mal_tensor_a(param, fft_mal_tensor_a, TENSOR_A_NUM);
    sample_a_and_tensor_mal(param, fft_mal_tensor_a);

    struct Mock_Mal_GF128_DPF_Key **mal_DPF_keys = calloc(KEY_NUM, sizeof(struct Mock_Mal_GF128_DPF_Key*));
    init_mal_gf128_DPF_keys(param, mal_DPF_keys, KEY_NUM);
    mock_mal_gf128_DPF_output(param, mal_DPF_keys, KEY_NUM);

    struct Mal_GF128_Mult_Buf *mal_gf128_mult_buf = calloc(1, sizeof(struct Mal_GF128_Mult_Buf));
    struct Mal_GF128_Mult_Rlt **mal_gf128_mult_rlts = calloc(KEY_NUM, sizeof(struct Mal_GF128_Mult_Rlt*));
    struct Mal_GF128_Mult_Rlt **mal_gf128_final_rlts = calloc(RLT_NUM, sizeof(struct Mal_GF128_Mult_Rlt*));
    init_mal_gf128_mult_buf(param, mal_gf128_mult_buf);
    init_mal_gf128_mult_rlts(param, mal_gf128_mult_rlts, KEY_NUM);
    init_mal_gf128_mult_rlts(param, mal_gf128_final_rlts, RLT_NUM);

    struct FFT_Mal_A_GF128_Tensor *fft_mal_a_gf128_tensor = calloc(1, sizeof(struct FFT_Mal_A_GF128_Tensor));
    init_fft_mal_a_gf128_tensor(param, fft_mal_a_gf128_tensor, TENSOR_A_NUM);

    convert_uint32_tensor_a_to_gf128(param, fft_mal_tensor_a, fft_mal_a_gf128_tensor, TENSOR_A_NUM);
    
    clock_t time = clock();
    // b0^2 x b1^2
    // b0 x b1^2
    // b0^2 x b1
    // b0 x b1
    uint32_t **fft_tensor_a = fft_mal_tensor_a->fft_tensor_a;
    uint128_t **fft_gf128_tensor_a = fft_mal_a_gf128_tensor->fft_gf128_tensor_a;
    for (size_t i = 0; i < TENSOR_A_NUM; i++) {

        mal_gf128_multiply_then_sum(param, fft_tensor_a[i], fft_gf128_tensor_a[i], mal_DPF_keys[i], mal_gf128_mult_buf, mal_gf128_mult_rlts[i]);
    }

    // (1, zeta+1)
    uint8_t scalar = 0b11;
    mal_gf128_scalar_multiply_fft(param, mal_gf128_mult_rlts[3], scalar);
    mal_gf128_sum_polys(param, mal_gf128_mult_rlts, KEY_NUM, mal_gf128_final_rlts[0]);

    // (1, zeta). 
    // Multiply by (zeta+1)^-1 before zeta.
    // It is (zeta+1)^(-1) x zeta = zeta+1
    scalar = 0b11;
    mal_gf128_scalar_multiply_fft(param, mal_gf128_mult_rlts[3], scalar);
    mal_gf128_sum_polys(param, mal_gf128_mult_rlts, KEY_NUM, mal_gf128_final_rlts[1]);

    time = clock()-time;
    double time_taken = ((double)time) / (CLOCKS_PER_SEC / 1000.0); // ms
    printf("Eval time (total) %f ms\n", time_taken);
    printf("DONE\n\n");
    free_fft_mal_a_gf128_tensor(fft_mal_a_gf128_tensor, TENSOR_A_NUM);
    free_fft_mal_tensor_a(fft_mal_tensor_a, TENSOR_A_NUM);
    free_mal_gf128_DPF_keys(mal_DPF_keys, KEY_NUM);
    free_mal_gf128_mult_buf(mal_gf128_mult_buf);
    free_mal_gf128_mult_rlts(mal_gf128_mult_rlts, KEY_NUM);
    free_mal_gf128_mult_rlts(mal_gf128_final_rlts, RLT_NUM);
    
    free(param);
    return time_taken;
}

void mal_gf128_sum_polys(const struct Param *param, struct Mal_GF128_Mult_Rlt **mal_gf128_mult_rlts, size_t size, struct Mal_GF128_Mult_Rlt *final_rlt) {

    const size_t poly_size = param->poly_size;
    for (size_t j = 0; j < size; j++) {
        for (size_t i = 0; i < poly_size; i++) {
            final_rlt->vlu_rlt[i] ^= mal_gf128_mult_rlts[j]->vlu_rlt[i];
            final_rlt->MAC_rlt[i] ^= mal_gf128_mult_rlts[j]->MAC_rlt[i];
        }
    }
}

void mal_gf128_scalar_multiply_fft(const struct Param *param, const struct Mal_GF128_Mult_Rlt *mal_gf128_mult_rlt, uint8_t scalar) {

    const size_t poly_size = param->poly_size;
    uint8_t *vlu_rlt = mal_gf128_mult_rlt->vlu_rlt;
    scalar_multiply_fft_f4(vlu_rlt, scalar, vlu_rlt, poly_size);
    uint128_t scalar_uint128 = 0;
    if (scalar == 0b10) {
        scalar_uint128 = param->zeta128;
    } else if (scalar == 0b11) {
        scalar_uint128 = param->zeta128 ^ 0b01;
    } { // 0 or 1
        scalar_uint128 = scalar;
    }
    uint128_t *MAC_rlt = mal_gf128_mult_rlt->MAC_rlt;
    scalar_multiply_fft_gf128(MAC_rlt, scalar_uint128, MAC_rlt, poly_size);
}

void mal_gf128_multiply_then_sum(
    const struct Param *param,
    const uint32_t *fft_a,
    const uint128_t *fft_a_tensor,
    struct Mock_Mal_GF128_DPF_Key *mal_DPF_key,
    struct Mal_GF128_Mult_Buf *mal_gf128_mult_buf,
    struct Mal_GF128_Mult_Rlt *mal_gf128_mult_rlt) {

    const size_t c = param->c;
    const size_t poly_size = param->poly_size;
    const uint128_t zeta = param->zeta128;

    uint32_t *vlu_buf = mal_gf128_mult_buf->vlu_buf;
    uint8_t *vlu_rlt = mal_gf128_mult_rlt->vlu_rlt;
    FFT_multiply_then_sum(param, fft_a, mal_DPF_key->u, vlu_buf, vlu_rlt);

    uint128_t *MAC_buf = mal_gf128_mult_buf->MAC_buf;
    uint128_t *MAC_rlt = mal_gf128_mult_rlt->MAC_rlt;
    
    multiply_fft_gf128(fft_a_tensor, mal_DPF_key->MAC, MAC_buf, c*c*poly_size);
    for (size_t j = 0; j < c*c; ++j) {
        for (size_t i = 0; i < poly_size; ++i) {
            MAC_rlt[i] = MAC_rlt[i] + MAC_buf[i+j*poly_size];
        }
    }
}

void mock_mal_gf128_DPF_output(const struct Param *param, struct Mock_Mal_GF128_DPF_Key **mal_DPF_keys, size_t size) {

    size_t c = param->c;
    size_t n = param->n;
    size_t poly_size = param->poly_size;
    uint128_t zeta = param->zeta128;
    
    // length poly_size
    uint32_t *coeff_u = calloc(poly_size, sizeof(uint32_t));
    uint128_t *coeff_MAC = calloc(poly_size, sizeof(uint128_t));

    for (size_t i = 0; i < size; ++i) {
        struct Mock_Mal_GF128_DPF_Key *key = mal_DPF_keys[i];
        RAND_bytes((unsigned char *)coeff_u, poly_size*sizeof(uint32_t));
        fft_iterative_uint32_no_data(coeff_u, key->u, n, poly_size);

        uint128_t *MAC = key->MAC;
        for (size_t j = 0; j < c*c; ++j) {
            RAND_bytes((unsigned char *)coeff_MAC, poly_size*sizeof(uint128_t));
            uint128_t *cur_MAC = &(MAC[j*poly_size]);
            fft_iterative_gf128_no_data(coeff_MAC, cur_MAC, n, poly_size, zeta);
        }
    }
    free(coeff_MAC);
    free(coeff_u);
}

void init_mal_gf128_DPF_key(const struct Param *param, struct Mock_Mal_GF128_DPF_Key *mal_DPF_key) {
    size_t c = param->c;
    size_t n = param->n;
    size_t poly_size = param->poly_size;
    // u is in packed form.
    mal_DPF_key->u = calloc(poly_size, sizeof(uint32_t));
    // MAC is in seperated form.
    mal_DPF_key->MAC = calloc(c*c*poly_size, sizeof(uint128_t));
}

void init_mal_gf128_DPF_keys(const struct Param *param, struct Mock_Mal_GF128_DPF_Key **mal_DPF_keys, size_t size) {
    for (size_t i = 0; i < size; i++) {
        mal_DPF_keys[i] = calloc(1, sizeof(struct Mock_Mal_GF128_DPF_Key));
        init_mal_gf128_DPF_key(param, mal_DPF_keys[i]);
    }
}

void init_mal_gf128_mult_buf(const struct Param *param, struct Mal_GF128_Mult_Buf *mal_gf128_mult_buf) {

    size_t c = param->c;
    size_t poly_size = param->poly_size;
    mal_gf128_mult_buf->vlu_buf = calloc(poly_size, sizeof(uint32_t));
    mal_gf128_mult_buf->MAC_buf = calloc(c*c*poly_size, sizeof(uint128_t));
}

void init_mal_gf128_mult_rlt(const struct Param *param, struct Mal_GF128_Mult_Rlt *mal_gf128_mult_rlt) {
    size_t c = param->c;
    size_t poly_size = param->poly_size;
    mal_gf128_mult_rlt->vlu_rlt = calloc(poly_size, sizeof(uint8_t));
    mal_gf128_mult_rlt->MAC_rlt = calloc(poly_size, sizeof(uint128_t));
}

void init_mal_gf128_mult_rlts(const struct Param *param, struct Mal_GF128_Mult_Rlt **mal_gf128_mult_rlts, size_t size) {
    for (size_t i = 0; i < size; i++) {
        mal_gf128_mult_rlts[i] = calloc(1, sizeof(struct Mal_GF128_Mult_Rlt));
        init_mal_gf128_mult_rlt(param, mal_gf128_mult_rlts[i]);
    }
}

void init_fft_mal_a_gf128_tensor(
    const struct Param *param,
    struct FFT_Mal_A_GF128_Tensor *fft_mal_a_gf128_tensor,
    size_t size
    ) {

    size_t c = param->c;
    size_t poly_size = param->poly_size;
    uint128_t **fft_gf128_tensor_a = fft_mal_a_gf128_tensor->fft_gf128_tensor_a = calloc(size, sizeof(uint128_t*));

    for (size_t i = 0; i < size; i++) {
        fft_gf128_tensor_a[i] = calloc(c*c*poly_size, sizeof(uint128_t));
    }
}

void convert_uint32_tensor_a_to_gf128(
    const struct Param *param,
    struct FFT_Mal_Tensor_A *fft_mal_tensor_a,
    struct FFT_Mal_A_GF128_Tensor *fft_mal_a_gf128_tensor,
    const size_t size
    ) {
    
    const size_t c = param->c;
    const size_t poly_size = param->poly_size;
    const size_t zeta = param->zeta128;

    uint32_t **fft_tensor_a = fft_mal_tensor_a->fft_tensor_a;
    uint128_t **fft_gf128_tensor_a = fft_mal_a_gf128_tensor->fft_gf128_tensor_a;

    // TODO: change this to KEY_NUM
    for (size_t i = 0; i < size; ++i) {
        convert_uint32_to_gf128(fft_tensor_a[i], fft_gf128_tensor_a[i], poly_size, c, zeta);
    }
}

void free_fft_mal_a_gf128_tensor(
    struct FFT_Mal_A_GF128_Tensor *fft_mal_a_gf128_tensor,
    const size_t size
    ) {

    uint128_t **fft_gf128_tensor_a = fft_mal_a_gf128_tensor->fft_gf128_tensor_a;
    for (size_t i = 0; i < size; i++) {
        free(fft_gf128_tensor_a[i]);
    }
    free(fft_gf128_tensor_a);
    free(fft_mal_a_gf128_tensor);
}

void free_mal_gf128_mult_buf(struct Mal_GF128_Mult_Buf *mal_gf128_mult_buf) {
    if (mal_gf128_mult_buf != NULL) {
        // Free individual buf components
        free(mal_gf128_mult_buf->vlu_buf);
        free(mal_gf128_mult_buf->MAC_buf);

        // Set pointers to NULL for safety
        mal_gf128_mult_buf->vlu_buf = NULL;
        mal_gf128_mult_buf->MAC_buf = NULL;

        // Optionally free the structure itself (if dynamically allocated)
        free(mal_gf128_mult_buf);
    }
}

// Function to free Mal_GF128_Mult_Rlt
void free_mal_gf128_mult_rlts(struct Mal_GF128_Mult_Rlt **mal_gf128_mult_rlts, size_t size) {
    for (size_t i = 0; i < size; i++) {
        struct Mal_GF128_Mult_Rlt* rlt = mal_gf128_mult_rlts[i];
        free(rlt->vlu_rlt);
        free(rlt->MAC_rlt);
        rlt->vlu_rlt = NULL;
        rlt->MAC_rlt = NULL;
        free(rlt);
        mal_gf128_mult_rlts[i] = NULL;
    }
    free(mal_gf128_mult_rlts);
}

void free_mal_gf128_DPF_keys(struct Mock_Mal_GF128_DPF_Key **mal_DPF_keys, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        struct Mock_Mal_GF128_DPF_Key *key = mal_DPF_keys[i];
        free(key->MAC);
        free(key->u);
        free(key);
    }
    free(mal_DPF_keys);
}

void init_mal_gf128_bench_params(struct Param* param, const size_t n, const size_t c, const size_t t, const uint128_t zeta128) {

    param->n=n;
    param->c=c;
    param->t=t;
    param->zeta128 = zeta128;
    size_t poly_size = ipow(3, n);
    // Note: We assume that t is a power of 3 and so it divides poly_size
    size_t block_size = ceil(poly_size / t);
    printf("block_size = %zu \n", block_size);
    
    param->poly_size = poly_size;
    param->block_size = block_size;
    // param->dpf_block_size = dpf_block_size;
    // param->dpf_domain_bits = dpf_domain_bits;
    // param->packed_poly_size = packed_poly_size;
    // param->packed_block_size = packed_block_size;
    printf("Done with initializing parameters.\n");
}