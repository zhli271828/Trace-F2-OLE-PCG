#ifndef __MAL_GF64_TRACE_BENCH
#define __MAL_GF64_TRACE_BENCH


struct FFT_Mal_A_GF64_Tensor {
    // length c*c*poly_size
    uint64_t **fft_gf64_tensor_a;
};

// malicious DPF keys
// u is the value share and MAC the MAC share
struct Mock_Mal_GF64_DPF_Key {
    // length c*c*poly_size
    uint64_t *MAC;
    // c^2 values are packed in u and thus length poly_size
    uint32_t *u;
};

// buffer for expand multiplications
struct Mal_GF64_Mult_Buf {
    // length poly_size
    uint32_t *vlu_buf;
    // length c*c*poly_size
    uint64_t *MAC_buf;
};

// expand multiplication results
struct Mal_GF64_Mult_Rlt {
    // length poly_size
    uint8_t *vlu_rlt;
    // length poly_size
    uint64_t *MAC_rlt;
};

// only part of the parameters are initilizated.
void init_mal_gf64_bench_params(struct Param* param, const size_t n, const size_t c, const size_t t, const uint64_t zeta);

// init one DPF key
void init_mal_gf64_DPF_key(const struct Param *param, struct Mock_Mal_GF64_DPF_Key *mal_DPF_key);

// init all of the DPF keys in the system
void init_mal_gf64_DPF_keys(const struct Param *param, struct Mock_Mal_GF64_DPF_Key **mal_DPF_keys, size_t size);
void free_mal_gf64_DPF_keys(struct Mock_Mal_GF64_DPF_Key **mal_DPF_keys, size_t size);

// generate malicious DPF ouput
void mock_mal_gf64_DPF_output(const struct Param *param, struct Mock_Mal_GF64_DPF_Key **mal_DPF_keys, size_t size);

void init_mal_gf64_mult_buf(const struct Param *param, struct Mal_GF64_Mult_Buf *mal_gf64_mult_buf);
void init_mal_gf64_mult_rlt(const struct Param *param, struct Mal_GF64_Mult_Rlt *mal_gf64_mult_rlt);
void init_mal_gf64_mult_rlts(const struct Param *param, struct Mal_GF64_Mult_Rlt **mal_gf64_mult_rlts, size_t size);

void init_fft_mal_a_gf64_tensor(
    const struct Param *param,
    struct FFT_Mal_A_GF64_Tensor *fft_mal_a_gf64_tensor,
    const size_t size
    );
void free_fft_mal_a_gf64_tensor(
    struct FFT_Mal_A_GF64_Tensor *fft_mal_a_gf64_tensor,
    const size_t size
    );

void convert_uint32_tensor_a_to_gf64(
    const struct Param *param,
    struct FFT_Mal_Tensor_A *fft_mal_tensor_a,
    struct FFT_Mal_A_GF64_Tensor *fft_mal_a_gf64_tensor,
    const size_t size
    );

void free_mal_gf64_mult_buf(struct Mal_GF64_Mult_Buf *mal_gf64_mult_buf);
void free_mal_gf64_mult_rlts(struct Mal_GF64_Mult_Rlt **mal_gf64_mult_rlt, size_t size);

void mal_gf64_scalar_multiply_fft(const struct Param *param, const struct Mal_GF64_Mult_Rlt *mal_gf64_mult_rlt, uint8_t scalar);
void mal_gf64_sum_polys(const struct Param *param, struct Mal_GF64_Mult_Rlt **mal_gf64_mult_rlts, size_t size, struct Mal_GF64_Mult_Rlt *final_rlt);

// multiply fft_a and the DPF value and MACs
void mal_gf64_multiply_then_sum(
    const struct Param *param,
    const uint32_t *fft_a, // packed a
    const uint64_t *fft_a_tensor,
    struct Mock_Mal_GF64_DPF_Key *mal_DPF_key, // DPF value and MACs
    struct Mal_GF64_Mult_Buf *mal_gf64_mult_buf, // value and MAC buffer
    struct Mal_GF64_Mult_Rlt *mal_gf64_mult_rlt // value and MAC results
);

double mal_gf64_trace_bench_pcg(size_t n, size_t c, size_t t);

#endif
