#include "utils.h"
#include "common.h"
#include "f4ops.h"

void init_fft_mal_tensor_a(const struct Param *param, struct FFT_Mal_Tensor_A *fft_mal_tensor_a, const size_t size) {

    size_t poly_size = param->poly_size;
    uint8_t *fft_a = calloc(poly_size, sizeof(uint8_t));;
    uint32_t **fft_tensor_a = calloc(size, sizeof(uint32_t*));
    for (size_t i = 0; i < size; i++) {
        fft_tensor_a[i] = calloc(poly_size, sizeof(uint32_t));
    }
    fft_mal_tensor_a->fft_a = fft_a;
    fft_mal_tensor_a->fft_tensor_a = fft_tensor_a;
}

void free_fft_mal_tensor_a(struct FFT_Mal_Tensor_A *fft_mal_tensor_a, const size_t size) {

    free(fft_mal_tensor_a->fft_a);
    uint32_t **fft_tensor_a = fft_mal_tensor_a->fft_tensor_a;
    for (size_t i = 0; i < size; i++) {
        free(fft_tensor_a[i]);
    }
    free(fft_tensor_a);
    free(fft_mal_tensor_a);
}

// samples the a polynomials and ai^2 x aj^2, ai^2 x aj, ai x aj^2, ai x aj polynomials for malicious tensor
void sample_a_and_tensor_mal(const struct Param *param, struct FFT_Mal_Tensor_A *fft_mal_tensor_a) {

    const size_t poly_size = param->poly_size;
    const size_t c = param->c;

    uint8_t *fft_a = fft_mal_tensor_a->fft_a;
    uint32_t **fft_tensor_a = fft_mal_tensor_a->fft_tensor_a;

    RAND_bytes((uint8_t *)fft_a, sizeof(uint8_t) * poly_size);

    // make a_0 the identity polynomial (in FFT space) i.e., all 1s
    for (size_t i = 0; i < poly_size; i++) {
        fft_a[i] = fft_a[i] >> 2;
        fft_a[i] = fft_a[i] << 2;
        fft_a[i] |= 1;
    }
    // FOR DEBUGGING: set fft_a to the identity
    // for (size_t i = 0; i < poly_size; i++)
    // {
    //     fft_a[i] = (0xaaaa >> 1);
    // }
    for (size_t j = 0; j < c; j++) {
        for (size_t k = 0; k < c; k++) {
            for (size_t i = 0; i < poly_size; i++) {
                uint8_t u = (fft_a[i] >> (2 * j)) & 0b11;
                uint8_t v = (fft_a[i] >> (2 * k)) & 0b11;
                uint32_t uv = mult_f4(u, v);
                uint32_t u2v = mult_f4(uv, u);
                uint32_t uv2 = mult_f4(uv, v);
                uint32_t u2v2 = mult_f4(uv, uv);
                if (u2v2 != mult_f4(u2v, v) || u2v2 != mult_f4(uv2, u)) {
                    printf("ERROR: sample_a_and_tensor_mal");
                    exit(0);
                }
                size_t slot = j * c + k;
                fft_tensor_a[0][i] |= u2v2 << (2*slot);
                fft_tensor_a[1][i] |= uv2 << (2*slot);
                fft_tensor_a[2][i] |= u2v << (2*slot);
                fft_tensor_a[3][i] |= uv << (2*slot);
            }
        }
    }
    printf("Done with sampling the public values\n");
}
