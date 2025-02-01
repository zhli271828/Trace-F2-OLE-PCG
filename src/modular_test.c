// This is a modular design of test.c
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "test.h"
#include "dpf.h"
#include "prf.h"
#include "fft.h"
#include "utils.h"
#include "f4ops.h"
#include "common.h"
#include "modular_test.h"

#define N 16 // 3^N number of OLEs generated in total
#define C 4  // compression factor
#define T 27 // noise weight


// Step 0: Sample the global (1, a1 ... a_c-1) polynomials
void step0(const struct Param* param, struct FFT_A* fft) {
    // input: param
    // output: fft_a, fft_a2
    const size_t poly_size = param->poly_size;
    const size_t c = param->c;

    uint8_t* fft_a = calloc(poly_size, sizeof(uint8_t));
    uint32_t* fft_a2 = calloc(poly_size, sizeof(uint32_t));
    // Each is of poly_size
    sample_a_and_a2(fft_a,fft_a2,poly_size,c);
    fft->fft_a=fft_a;
    fft->fft_a2=fft_a2;
    printf("[       ]Done with Step 0 (sampling the public values)\n");
}

void step0_free(struct FFT_A *fft) {
    free(fft->fft_a);
    free(fft->fft_a2);
    free(fft);
}

// Step 1: Sample error polynomials eA and eB (c polynomials in total)
// each polynomial is t-sparse and has degree (t * block_size) = poly_size.
void step1(const struct Param* param, struct Err_Poly *err_poly, struct FFT_E * fft_e) {

    const size_t poly_size=param->poly_size;
    const size_t block_size = param->block_size;
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t n = param->n;
    // the expanded polynomials
    uint8_t *err_polys_A = calloc(c*poly_size, sizeof(uint8_t));
    uint8_t *err_polys_B = calloc(c*poly_size, sizeof(uint8_t));
    // error coefficients
    uint8_t *err_poly_coeffs_A = calloc(c*t, sizeof(uint8_t));
    uint8_t *err_poly_coeffs_B = calloc(c*t, sizeof(uint8_t));
    // error positions
    size_t *err_poly_pos_A=calloc(c*t, sizeof(size_t));
    size_t *err_poly_pos_B=calloc(c*t, sizeof(size_t));

    for(size_t i=0; i<c; ++i) {
        for (size_t j=0; j<t; ++j) {
            size_t offset = i*t+j;
            // rand_f4x() samples a nonzero F4 element
            uint8_t a = rand_f4x();
            uint8_t b = rand_f4x();
            err_poly_coeffs_A[offset]=a;
            err_poly_coeffs_B[offset]=b;

            // random index within the block
            size_t pos_A=random_index(block_size-1);
            size_t pos_B=random_index(block_size-1);

            err_poly_pos_A[offset]=pos_A;
            err_poly_pos_B[offset]=pos_B;

            err_polys_A[i*poly_size+j*block_size+pos_A]=a;
            err_polys_B[i*poly_size+j*block_size+pos_B]=b;
        }
    } // output err_polys_A, err_polys_B, err_poly_coeffs_A, err_poly_coeffs_B, err_poly_pos_A, err_poly_pos_B
    err_poly->err_polys_A = err_polys_A;
    err_poly->err_polys_B = err_polys_B;
    err_poly->err_poly_coeffs_A = err_poly_coeffs_A;
    err_poly->err_poly_coeffs_B = err_poly_coeffs_B;
    err_poly->err_poly_pos_A = err_poly_pos_A;
    err_poly->err_poly_pos_B = err_poly_pos_B;

    // compute FFT of eA and eB in packed form
    // Note that because c = 4, we can pack 4 FFTs into a uint8_t
    uint8_t *fft_eA=calloc(poly_size, sizeof(uint8_t));
    uint8_t *fft_eB=calloc(poly_size, sizeof(uint8_t));
    uint8_t coef_A,coef_B;
    for (size_t j = 0; j < c; j++) {
        for (size_t i = 0; i < poly_size; i++) {
            coef_A=err_polys_A[j*poly_size+i];
            coef_B=err_polys_B[j*poly_size+i];

            fft_eA[i] |= coef_A<<(2*j);
            fft_eB[i] |= coef_B<<(2*j);

        }
    }
    fft_recursive_uint8(fft_eA, n, poly_size/3);
    fft_recursive_uint8(fft_eB, n, poly_size/3);
    // output fft_eA, fft_eB
    fft_e->fft_eA = fft_eA;
    fft_e->fft_eB = fft_eB;
    printf("[.      ]Done with Step 1 (sampling error vectors)\n");
}

void step1_free(struct Err_Poly *err_poly, struct FFT_E *fft_e) {
    free(err_poly->err_polys_A);
    free(err_poly->err_polys_B);
    free(err_poly->err_poly_pos_A);
    free(err_poly->err_poly_pos_B);
    free(err_poly->err_poly_coeffs_A);
    free(err_poly->err_poly_coeffs_B);
    free(err_poly);
    free(fft_e->fft_eA);
    free(fft_e->fft_eB);
    free(fft_e);
}

// Step 2: compute the inner product xA = <a, eA> and xB = <a, eB>
void step2(const struct Param* param, struct FFT_A* fft, struct FFT_E *fft_e, struct Inner_Prod *inner_prod) {
    const size_t poly_size = param->poly_size;
    const size_t c = param->c;

    uint8_t *res_poly_A=calloc(poly_size, sizeof(uint8_t));
    uint8_t *res_poly_B=calloc(poly_size, sizeof(uint8_t));
    
    uint8_t *fft_a = fft->fft_a;
    uint8_t *fft_eA = fft_e->fft_eA;
    uint8_t *fft_eB = fft_e->fft_eB;
    // compute the multiplication in the packed form
    multiply_fft_8(fft_a, fft_eA, res_poly_A, poly_size);
    multiply_fft_8(fft_a, fft_eB, res_poly_B, poly_size);

    uint8_t *x_poly_A=calloc(poly_size, sizeof(uint8_t));
    uint8_t *x_poly_B=calloc(poly_size, sizeof(uint8_t));

    // compute the inner product
    for (size_t i = 0; i < poly_size; i++) {
        for (size_t j = 0; j < c; j++) {
            x_poly_A[i] ^= (res_poly_A[i] >> (2*j))&0b11;
            x_poly_B[i] ^= (res_poly_B[i]>>(2*j))&0b11;
        }
    }
    // output x_poly_A, x_poly_B
    inner_prod->x_poly_A=x_poly_A;
    inner_prod->x_poly_B=x_poly_B;

    // add to a system call
    free(res_poly_A);
    free(res_poly_B);
    printf("[..     ]Done with Step 2 (computing the local vectors)\n");
}

void step2_free(struct Inner_Prod *inner_prod) {
    free(inner_prod->x_poly_A);
    free(inner_prod->x_poly_B);
    free(inner_prod);
}

// Step 3: Compute cross product (eA x eB) using the position vectors
void step3(const struct Param *param, const struct Err_Poly *err_poly, struct Err_Cross_Poly *err_cross_poly) {
    // required input: c,t,block_size, 
    // err_poly_coeffs_A, err_poly_coeffs_B
    // err_poly_pos_A, err_poly_pos_B
    size_t c = param->c;
    size_t t = param->t;
    size_t n = param->n;

    size_t block_size = param->block_size;
    size_t poly_size = param->poly_size;

    uint8_t *err_poly_cross_coef=calloc(c*c*t*t, sizeof(uint8_t));
    size_t *err_poly_cross_pos=calloc(c*c*t*t, sizeof(size_t));
    uint8_t *err_polys_cross=calloc(c*c*poly_size, sizeof(uint8_t));

    uint8_t *trit_decomp_A=calloc(n,sizeof(uint8_t));
    uint8_t *trit_decomp_B=calloc(n,sizeof(uint8_t));
    uint8_t *trit_decomp=calloc(n,sizeof(uint8_t));

    // TODO: exchange the for loop order to speedup the runtime
    for (size_t iA = 0; iA < c; iA++) {
        for (size_t iB = 0; iB < c; iB++) {
            size_t poly_index = iA*c*t*t+iB*t*t;
            uint8_t *next_idx=calloc(t, sizeof(uint8_t));
            for (size_t jA = 0; jA < t; jA++) {
                // jA-th coefficient value of the iA-th polynomial
                size_t vA=err_poly->err_poly_coeffs_A[iA*t+jA];
                for (size_t jB = 0; jB < t; jB++) {
                    // jB-th coefficient value of the iB-th polynomial
                    size_t vB=err_poly->err_poly_coeffs_B[iB*t+jB];
                    // Resulting cross-product coefficient
                    uint8_t v=mult_f4(vA,vB);

                    // Compute the position (in the full polynomial)
                    size_t posA=jA*block_size+err_poly->err_poly_pos_A[iA*t+jA];
                    size_t posB=jB*block_size+err_poly->err_poly_pos_B[iB*t+jB];

                    if (err_poly->err_polys_A[iA*poly_size+posA]==0 || err_poly->err_polys_B[iB*poly_size+posB]==0) {
                        printf("FAIL: Incorrect position recovered\n");
                        exit(0);
                    }
                    // Decompose the position into the ternary basis
                    int_to_trits(posA, trit_decomp_A, n);
                    int_to_trits(posB, trit_decomp_B, n);

                    // Sum ternary decomposition coordinate-wise to
                    // get the new position (in ternary).
                    add_trits(trit_decomp_A, trit_decomp_B, trit_decomp, n);

                    size_t pos=trits_to_int(trit_decomp, n);
                    size_t block_idx=floor(pos/block_size);
                    size_t in_block_idx=pos%block_size;

                    err_polys_cross[(iA*c+iB)*poly_size+pos] ^= v;

                    size_t idx=next_idx[block_idx];
                    next_idx[block_idx]++;
                    
                    err_poly_cross_coef[poly_index+block_idx*t+idx]=v;
                    err_poly_cross_pos[poly_index+block_idx*t+idx]=in_block_idx;
                    
                }
            }

            for (size_t k = 0; k < t; k++) {
                if (next_idx[k] > t) {
                    printf("FAIL: next_idx > t at the end: %hhu\n", next_idx[k]);
                    exit(0);
                }
            }
            free(next_idx);
        }
    }
    // cleanup temporary values
    free(trit_decomp);
    free(trit_decomp_A);
    free(trit_decomp_B);
    // output err_polys_cross, err_poly_cross_pos, err_poly_cross_coef
    err_cross_poly->err_poly_cross_coef = err_poly_cross_coef;
    err_cross_poly->err_poly_cross_pos = err_poly_cross_pos;
    err_cross_poly->err_polys_cross = err_polys_cross;
    printf("[...    ]Done with Step 3 (computing the cross product)\n");
}

void step3_free(struct Err_Cross_Poly *err_cross_poly) {

    free(err_cross_poly->err_polys_cross);
    free(err_cross_poly->err_poly_cross_pos);
    free(err_cross_poly->err_poly_cross_coef);
    free(err_cross_poly);
}

// Step 4: Sample the DPF keys for the cross product (eA x eB)
void step4(const struct Param *param, const struct Err_Cross_Poly *err_cross_poly, struct Keys *keys) {
    // input: err_poly_cross_pos, err_poly_cross_coef
    const size_t c = param->c;
    const size_t t = param->t;
    size_t block_size = param->block_size;
    size_t dpf_domain_bits = param->dpf_domain_bits;
    // arrays for the DFA keys
    struct DPFKey** dpf_keys_A=malloc(c*c*t*t*sizeof(void*));
    struct DPFKey** dpf_keys_B=malloc(c*c*t*t*sizeof(void*));
    // Sample PRF keys for the DPFs
    struct PRFKeys* prf_keys=malloc(sizeof(struct PRFKeys));
    PRFKeyGen(prf_keys);

    // Sample DPF keys for each of the t errors in the t blocks
    for (size_t i=0; i<c; ++i) {
        for (size_t j=0; j<c; ++j) {
            for (size_t k=0; k<t; ++k) {
                for (size_t l=0; l<t; ++l) {
                    size_t index = i*c*t*t+j*t*t+k*t+l;

                    // Set the DPF message to the coefficient
                    uint128_t coef=err_cross_poly->err_poly_cross_coef[index];

                    // Parse the index into the right format
                    size_t alpha=err_cross_poly->err_poly_cross_pos[index];
                    // Output message index in the DPF output space
                    // which consists of 256 F4 elements
                    size_t alpha_0=floor(alpha/256.0);
                    // Coeff index in the block of 256 coefficients
                    size_t alpha_1=alpha%256;
                    // Coeff index in the uint128_t output (64 elements of F4)
                    size_t packed_idx=floor(alpha_1/64.0);
                    // Bit index in the uint128_t ouput
                    size_t bit_idx=alpha_1%64;

                    // Position coefficient into the block
                    uint128_t beta[4]={0};
                    beta[packed_idx]=coef <<(2*(63-bit_idx));

                    struct DPFKey *kA=malloc(sizeof(struct DPFKey));
                    struct DPFKey *kB=malloc(sizeof(struct DPFKey));
                    DPFGen(prf_keys, dpf_domain_bits, alpha_0, beta, 4, kA, kB);
                    dpf_keys_A[index]=kA;
                    dpf_keys_B[index]=kB;
                }
            }
        }
    }
    keys->dpf_keys_A = dpf_keys_A;
    keys->dpf_keys_B = dpf_keys_B;
    keys->prf_keys = prf_keys;
    printf("[....   ]Done with Step 4 (sampling DPF keys)\n");
}
void step4_free(const struct Param *param, struct Keys *keys) {
    const size_t c = param->c;
    const size_t t = param->t;
    for (size_t i=0; i<c; ++i) {
        for (size_t j=0; j<c; ++j) {
            for (size_t k=0; k<t; ++k) {
                for (size_t l=0; l<t; ++l) {
                    size_t index = i*c*t*t+j*t*t+k*t+l;
                    free(keys->dpf_keys_A[index]);
                    free(keys->dpf_keys_B[index]);
                }
            }
        }
    }
    free(keys->dpf_keys_A);
    free(keys->dpf_keys_B);
    DestroyPRFKey(keys->prf_keys);
    free(keys);
}

// Here, we test to make sure all polynomials have at most t^2 errors
// and fail the test otherwise.
void verify_cross_prod_error_numbers(const struct Param *param, const struct Packed_Polys *packed_polys) {
    size_t c = param->c;
    size_t t = param->t;
    size_t packed_poly_size = param->packed_poly_size;

    uint128_t *packed_polys_A = packed_polys->packed_polys_A;
    uint128_t *packed_polys_B = packed_polys->packed_polys_B;

    for (size_t i = 0; i < c; i++) {
        for (size_t j = 0; j < c; j++) {
            size_t err_count = 0;
            size_t poly_index = i * c + j;
            uint128_t *poly_A = &packed_polys_A[poly_index * packed_poly_size];
            uint128_t *poly_B = &packed_polys_B[poly_index * packed_poly_size];

            for (size_t p = 0; p < packed_poly_size; p++) {
                uint128_t res = poly_A[p] ^ poly_B[p];
                for (size_t l = 0; l < 64; l++) {
                    if (((res >> (2 * (63 - l))) & 0b11) != 0)
                        err_count++;
                }
            }
            // printf("[DEBUG]: Number of non-zero coefficients in poly (%zu,%zu) is %zu\n", i, j, err_count);
            if (err_count > t * t) {
                printf("FAIL: Number of non-zero coefficients is %zu > t*t\n", err_count);
                exit(0);
            } else if (err_count == 0) {
                printf("FAIL: Number of non-zero coefficients in poly (%zu,%zu) is %zu\n", i, j, err_count);
                exit(0);
            }
        }
    }
}

// Step 5: Evaluate the DPFs to compute shares of (eA x eB)
void step5(const struct Param *param, const struct Keys *keys, struct Packed_Polys *packed_polys) {
    const size_t c = param->c;
    const size_t t = param->t;

    // input: param, dpf_keys_A, dpf_keys_B
    size_t block_size = param->block_size;
    size_t dpf_block_size = param->dpf_block_size;
    size_t packed_block_size = param->packed_block_size;
    size_t packed_poly_size = param->packed_poly_size;

    // Allocate memory for the DPF outputs (this is reused for each evaluation)
    uint128_t *shares_A=malloc(sizeof(uint128_t)*dpf_block_size);
    uint128_t *shares_B=malloc(sizeof(uint128_t)*dpf_block_size);
    uint128_t *cache=malloc(sizeof(uint128_t)*dpf_block_size);

    // Allocate memory for the concatenated DPF outputs
    uint128_t *packed_polys_A=calloc(c*c*packed_poly_size, sizeof(uint128_t));
    uint128_t *packed_polys_B=calloc(c*c*packed_poly_size, sizeof(uint128_t));
    
    for (int i = 0; i<c; ++i) {
        for (int j = 0; j<c; ++j) {
            size_t poly_index=i*c+j;
            // each entry is of length packed_poly_size
            uint128_t* packed_polyA=&packed_polys_A[poly_index*packed_poly_size];
            uint128_t* packed_polyB=&packed_polys_B[poly_index*packed_poly_size];

            for (size_t k=0; k<t; ++k) {
                // each entry is of length packed_block_size
                uint128_t *poly_blockA=&packed_polyA[k*packed_block_size];
                uint128_t *poly_blockB=&packed_polyB[k*packed_block_size];

                for (size_t l=0; l<t; ++l) {
                    size_t index=i*c*t*t+j*t*t+k*t+l;
                    struct DPFKey *dpf_keyA=keys->dpf_keys_A[index];
                    struct DPFKey *dpf_keyB=keys->dpf_keys_B[index];
                    DPFFullDomainEval(dpf_keyA, cache, shares_A);
                    DPFFullDomainEval(dpf_keyB, cache, shares_B);
                    for (size_t w=0; w<packed_block_size; ++w) {
                        poly_blockA[w] ^=shares_A[w];
                        poly_blockB[w] ^=shares_B[w]; 
                    }
                }
            }
        }
    } // output packed_polys_A, packed_polys_B
    packed_polys->packed_polys_A = packed_polys_A;
    packed_polys->packed_polys_B = packed_polys_B;
    verify_cross_prod_error_numbers(param, packed_polys);
    free(shares_A);
    free(shares_B);
    free(cache);
    printf("[.....  ]Done with Step 5 (evaluating all DPFs)\n");
}
void step5_free(struct Packed_Polys *packed_polys) {
    free(packed_polys->packed_polys_A);
    free(packed_polys->packed_polys_B);
    free(packed_polys);
}

// Step 6: Compute an FFT over the shares of (eA x eB)
void step6(const struct Param *param, const struct Packed_Polys *packed_polys, struct FFT_U *fft_u) {
    // TODO: check step6 to find a test for packed_polys

    // Input: packed_polys_A, packed_polys_B
    size_t packed_poly_size = param->packed_poly_size;
    size_t packed_block_size = param->packed_block_size;
    size_t block_size = param->block_size;
    size_t poly_size = param->poly_size;
    size_t n = param->n;
    size_t c = param->c;

    uint32_t *fft_uA=calloc(poly_size, sizeof(uint32_t));
    uint32_t *fft_uB=calloc(poly_size, sizeof(uint32_t));

    uint128_t *packed_polys_A = packed_polys->packed_polys_A;
    uint128_t *packed_polys_B = packed_polys->packed_polys_B;
    
    for (size_t j=0; j<c; ++j) {
        for (size_t k=0; k<c; ++k) {
            size_t poly_index=(j*c+k)*packed_poly_size;
            uint128_t *polyA=&packed_polys_A[poly_index];
            uint128_t *polyB=&packed_polys_B[poly_index];
            size_t block_idx=0;
            size_t bit_idx=0;
            for (size_t i=0; i<poly_size; ++i) {
                if (i%block_size==0 && i!=0) {
                    ++block_idx;
                    bit_idx=0;
                }
                size_t packed_idx=block_idx*packed_block_size+floor(bit_idx/64.0);
                size_t packed_bit=(63-bit_idx%64);

                uint128_t packedA=polyA[packed_idx];
                uint128_t packedB=polyB[packed_idx];

                uint32_t coefA=(packedA>>(2*packed_bit))&0b11;
                uint32_t coefB=(packedB>>(2*packed_bit))&0b11;

                size_t idx=j*c+k;
                fft_uA[i] |= coefA<<(2*idx);
                fft_uB[i] |= coefB<<(2*idx);
                ++bit_idx;
            }
        }
    }
    fft_recursive_uint32(fft_uA, n, poly_size/3);
    fft_recursive_uint32(fft_uB, n, poly_size/3);

    fft_u->fft_uA = fft_uA;
    fft_u->fft_uB = fft_uB;
    printf("[...... ]Done with Step 6 (computing FFTs)\n");

} // output fft_uA, fft_uB
void step6_free(struct FFT_U *fft_u) {
    free(fft_u->fft_uA);
    free(fft_u->fft_uB);
    free(fft_u);
}

// Step 7: Compute shares of z = <axa, u>
void step7(const struct Param *param, const struct FFT_A *fft, const struct FFT_U *fft_u, const struct Inner_Prod *inner_prod, struct Z_Polys *z_polys) {
    // input: fft_a2, fft_uA, fft_uB
    // input: x_poly_A, x_poly_B for comparison
    size_t c = param->c;
    size_t poly_size = param->poly_size;

    uint32_t *res_poly_mat_A=calloc(poly_size, sizeof(uint32_t));
    uint32_t *res_poly_mat_B=calloc(poly_size, sizeof(uint32_t));
    uint8_t *z_poly_A=calloc(poly_size, sizeof(uint8_t));
    uint8_t *z_poly_B=calloc(poly_size, sizeof(uint8_t));

    multiply_fft_32(fft->fft_a2, fft_u->fft_uA, res_poly_mat_A, poly_size);
    multiply_fft_32(fft->fft_a2, fft_u->fft_uB, res_poly_mat_B, poly_size);

    // XOR the (packed) columns into the accumulator.
    // Specifically, we perform column-wise XORs to get the result.
    size_t num_ffts = c*c;
    for (size_t j=0; j<c*c; ++j) {
        for (size_t i=0; i<poly_size; ++i) {
            z_poly_A[i] ^= (res_poly_mat_A[i]>>(2*j))&0b11;
            z_poly_B[i] ^= (res_poly_mat_B[i]>>(2*j))&0b11;
        }
    }

    // Now we check that we got the correct OLE correlations and fail
    // the test otherwise.
    for (size_t i=0; i<poly_size; ++i) {
        uint8_t res=z_poly_A[i]^z_poly_B[i];
        uint8_t exp=mult_f4(inner_prod->x_poly_A[i], inner_prod->x_poly_B[i]);
        if (res != exp) {
            printf("FAIL: Incorrect correlation output at index %zu\n", i);
            printf("Got: (%i,%i), Expected: (%i, %i)\n",
                   (res >> 1) & 1, res & 1, (exp >> 1) & 1, exp & 1);
            exit(0);
        }
    }
    // output z_poly_A, z_poly_B
    z_polys->z_poly_A = z_poly_A;
    z_polys->z_poly_B = z_poly_B;
    printf("[.......]Done with Step 7 (recovering shares)\n\n");
}

void step7_free(struct Z_Polys *z_polys) {
    free(z_polys->z_poly_A);
    free(z_polys->z_poly_B);
    free(z_polys);
}

void verify_shares(const struct Param *param, const struct Inner_Prod *inner_prod, const struct Z_Polys *z_polys) {
    size_t poly_size = param->poly_size;
    for (size_t i=0; i<poly_size; ++i) {
        uint8_t res=z_polys->z_poly_A[i]^z_polys->z_poly_B[i];
        uint8_t exp=mult_f4(inner_prod->x_poly_A[i], inner_prod->x_poly_B[i]);
        if (res != exp) {
            printf("FAIL: Incorrect correlation output at index %zu\n", i);
            printf("Got: (%i,%i), Expected: (%i, %i)\n",
                   (res >> 1) & 1, res & 1, (exp >> 1) & 1, exp & 1);
            exit(0);
        }
    }
}

void init_params(struct Param* param, const size_t n, const size_t c, const size_t t) {
    param->n=n;
    param->c=c;
    param->t=t;
    size_t poly_size = ipow(3, n);
    // Here, we figure out a good block size for the error vectors such that
    // t*block_size = 3^n and block_size/L*128 is close to a power of 3.
    // We pack L=256 coefficients of F4 into each DPF output (note that larger
    // packing values are also okay, but they will do increase key size).
    //
    size_t dpf_domain_bits = ceil(log_base(ceil(poly_size/(t*256.0)), 3));
    if (dpf_domain_bits == 0) dpf_domain_bits = 1;
    printf("DPF domain bits %zu \n", dpf_domain_bits);
    // 4*128 ==> 256 coefficients in F4
    size_t dpf_block_size = 4 * ipow(3, dpf_domain_bits);
    printf("dpf_block_size = %zu\n", dpf_block_size);
    // Note: We assume that t is a power of 3 and so it divides poly_size
    size_t block_size = poly_size / t;
    printf("block_size = %zu \n", block_size);
    
    size_t packed_block_size=ceil(block_size/64.0);
    size_t packed_poly_size=t*packed_block_size;
    
    param->poly_size = poly_size;
    param->block_size = block_size;
    param->dpf_block_size = dpf_block_size;
    param->dpf_domain_bits = dpf_domain_bits;
    param->packed_poly_size = packed_poly_size;
    param->packed_block_size = packed_block_size;

    printf("packed_block_size = %zu\n", packed_block_size);
    printf("packed_poly_size = %zu\n", packed_poly_size);

    printf("Done with initializing parameters.\n");
}

// This test evaluates the full PCG.Expand for both parties and
// checks correctness of the resulting OLE correlation.
void modular_test_pcg() {
    clock_t start_time=clock();
    struct Param* param=calloc(1, sizeof(struct Param));
    const size_t n = N;
    const size_t c = C;
    const size_t t = T;
    init_params(param, n, c, t);
    
    struct FFT_A *fft=calloc(1, sizeof(struct FFT_A));
    step0(param, fft);
    struct Err_Poly *err_poly=calloc(1, sizeof(struct Err_Poly));
    struct FFT_E *fft_e=calloc(1, sizeof(struct FFT_E));
    step1(param, err_poly, fft_e);
    struct Inner_Prod *inner_prod=calloc(1, sizeof(struct Inner_Prod));
    step2(param, fft, fft_e, inner_prod);
    struct Err_Cross_Poly *err_cross_poly = calloc(1, sizeof(struct Err_Cross_Poly));
    step3(param, err_poly, err_cross_poly);
    struct Keys *keys = calloc(1, sizeof(struct Keys));
    step4(param, err_cross_poly, keys);
    struct Packed_Polys *packed_polys = calloc(1, sizeof(struct Packed_Polys));
    step5(param, keys, packed_polys);
    struct FFT_U *fft_u = calloc(1, sizeof(struct FFT_U));
    step6(param, packed_polys, fft_u);
    struct Z_Polys *z_polys = calloc(1, sizeof(struct Z_Polys));
    step7(param, fft, fft_u, inner_prod, z_polys);
    verify_shares(param, inner_prod, z_polys);
    
    clock_t elapsed_time=clock()-start_time;
    double time_taken = ((double)elapsed_time) / (CLOCKS_PER_SEC / 1000.0); // ms
    printf("Time elapsed %f ms\n", time_taken);
    
    step0_free(fft);
    step1_free(err_poly, fft_e);
    step2_free(inner_prod);
    step3_free(err_cross_poly);
    step4_free(param, keys);
    step5_free(packed_polys);
    step6_free(fft_u);
    step7_free(z_polys);
    free(param);
}