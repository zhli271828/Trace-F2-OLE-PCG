#include <openssl/rand.h>
#include <openssl/conf.h>
#include <openssl/evp.h>
#include <openssl/err.h>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "fft.h"
#include "utils.h"

#define NUMVARS 16

typedef double (*p_func)();

double testFFT_uint64()
{
    size_t num_vars = NUMVARS;
    size_t num_coeffs = ipow(3, num_vars);

    uint64_t *coeffs = malloc(sizeof(uint64_t) * num_coeffs);
    RAND_bytes((uint8_t *)coeffs, sizeof(uint64_t) * num_coeffs);

    //************************************************
    printf("Benchmarking FFT evaluation with uint64_t packing \n");
    //************************************************

    clock_t t;
    t = clock();
    fft_recursive_uint64(coeffs, num_vars, num_coeffs / 3);
    t = clock() - t;
    double time_taken = ((double)t) / (CLOCKS_PER_SEC / 1000.0); // ms

    printf("FFT (uint64) eval time (total) %f ms\n", time_taken);

    free(coeffs);

    return time_taken;
}

double testFFT_uint32()
{
    size_t num_vars = NUMVARS;
    size_t num_coeffs = ipow(3, num_vars);

    uint32_t *coeffs = malloc(sizeof(uint32_t) * num_coeffs);
    RAND_bytes((uint8_t *)coeffs, sizeof(uint32_t) * num_coeffs);

    //************************************************
    printf("Benchmarking FFT evaluation with uint32_t packing \n");
    //************************************************

    clock_t t;
    t = clock();
    fft_recursive_uint32(coeffs, num_vars, num_coeffs / 3);
    t = clock() - t;
    double time_taken = ((double)t) / (CLOCKS_PER_SEC / 1000.0); // ms

    printf("FFT (uint32) eval time (total) %f ms\n", time_taken);

    free(coeffs);

    return time_taken;
}

double test_recursive_gr64_FFT() {
    size_t num_vars = NUMVARS;
    size_t num_coeffs = ipow(3, num_vars);
    uint64_t *coeffs0 = malloc(sizeof(uint64_t) * num_coeffs);
    uint64_t *coeffs1 = malloc(sizeof(uint64_t) * num_coeffs);
    RAND_bytes((uint8_t *)coeffs0, sizeof(uint64_t) * num_coeffs);
    RAND_bytes((uint8_t *)coeffs1, sizeof(uint64_t) * num_coeffs);
    //************************************************
    printf("Benchmarking iterative FFT evaluation with GR64 \n");
    //************************************************
    clock_t t = clock();
    fft_recursive_gr64_split(coeffs0, coeffs1, num_vars, num_coeffs/3);
    t = clock() - t;
    double time_taken = ((double)t) / (CLOCKS_PER_SEC / 1000.0); // ms
    printf("Iterative FFT (GR64) eval time (total) %f ms\n", time_taken);

    free(coeffs0);
    free(coeffs1);
    return time_taken;
}

double test_iterative_gr64_FFT() {
    size_t num_vars = NUMVARS;
    size_t num_coeffs = ipow(3, num_vars);

    uint64_t *coeffs0 = malloc(sizeof(uint64_t) * num_coeffs);
    uint64_t *coeffs1 = malloc(sizeof(uint64_t) * num_coeffs);
    uint64_t *rlt0 = calloc(num_coeffs, sizeof(uint64_t));
    uint64_t *rlt1 = calloc(num_coeffs, sizeof(uint64_t));
    RAND_bytes((uint8_t *)coeffs0, sizeof(uint64_t) * num_coeffs);
    RAND_bytes((uint8_t *)coeffs1, sizeof(uint64_t) * num_coeffs);
    //************************************************
    printf("Benchmarking iterative FFT evaluation with GR64\n");
    //************************************************
    clock_t t = clock();
    fft_iterative_gr64_no_data(coeffs0, coeffs1, rlt0, rlt1, num_vars, num_coeffs);
    double time_taken = ((double)t) / (CLOCKS_PER_SEC / 1000.0); // ms
    printf("FFT (GR64) eval time (total) %f ms\n", time_taken);
    free(coeffs0);
    free(coeffs1);
    free(rlt0);
    free(rlt1);
    return time_taken;
}

double test_iterative_FFT_uint32() {
    size_t num_vars = NUMVARS;
    size_t num_coeffs = ipow(3, num_vars);

    uint32_t *coeffs = malloc(sizeof(uint32_t) * num_coeffs);
    uint32_t *rlt = calloc(num_coeffs, sizeof(uint32_t));
    RAND_bytes((uint8_t *)coeffs, sizeof(uint32_t) * num_coeffs);

    //************************************************
    printf("Benchmarking iterative FFT evaluation with uint32_t packing \n");
    //************************************************

    clock_t t;
    t = clock();
    fft_iterative_uint32_no_data(coeffs, rlt, num_vars, num_coeffs);
    t = clock() - t;
    double time_taken = ((double)t) / (CLOCKS_PER_SEC / 1000.0); // ms

    printf("FFT (uint32) eval time (total) %f ms\n", time_taken);

    free(coeffs);
    // free(cache);
    free(rlt);
    return time_taken;
}

double test_iterative_FFT_uint64()
{
    size_t num_vars = NUMVARS;
    size_t num_coeffs = ipow(3, num_vars);

    uint64_t *coeffs = malloc(sizeof(uint64_t) * num_coeffs);
    // uint32_t *cache = calloc(num_coeffs, sizeof(uint32_t));
    uint64_t *rlt = calloc(num_coeffs, sizeof(uint64_t));
    RAND_bytes((uint8_t *)coeffs, sizeof(uint64_t) * num_coeffs);

    //************************************************
    printf("Benchmarking iterative FFT evaluation with uint32_t packing \n");
    //************************************************

    clock_t t;
    t = clock();
    fft_iterative_uint64_no_data(coeffs, rlt, num_vars, num_coeffs);
    t = clock() - t;
    double time_taken = ((double)t) / (CLOCKS_PER_SEC / 1000.0); // ms

    printf("FFT (uint64) eval time (total) %f ms\n", time_taken);

    free(coeffs);
    // free(cache);
    free(rlt);
    return time_taken;
}
double testFFT_uint8()
{
    size_t num_vars = NUMVARS;
    size_t num_coeffs = ipow(3, num_vars);
    uint8_t *coeffs = malloc(sizeof(uint8_t) * num_coeffs);
    RAND_bytes((uint8_t *)coeffs, sizeof(uint8_t) * num_coeffs);

    //************************************************
    printf("Benchmarking FFT evaluation without packing \n");
    //************************************************

    clock_t t;
    t = clock();
    fft_recursive_uint8(coeffs, num_vars, num_coeffs / 3);
    t = clock() - t;
    double time_taken = ((double)t) / (CLOCKS_PER_SEC / 1000.0); // ms

    printf("FFT (uint8) eval time (total) %f ms\n", time_taken);

    free(coeffs);

    return time_taken;
}

void test_framework(char* str, p_func func) {
    double time = 0;
    int testTrials = 5;
    printf("******************************************\n");
    printf("%s\n", str);
    for (int i = 0; i < testTrials; i++)
    {
        time += func();
        printf("Done with trial %i of %i\n", i + 1, testTrials);
    }
    printf("******************************************\n");
    printf("DONE\n");
    printf("Avg time: %0.2f\n", time / testTrials);
    printf("******************************************\n\n");
}

int main(int argc, char **argv) {

    test_framework("Testing Recursive GR64 FFT", test_recursive_gr64_FFT);
    test_framework("Testing Iterative GR64 FFT", test_iterative_gr64_FFT);
    test_framework("Testing FFT (uint8 packing)", testFFT_uint8);
    test_framework("Testing FFT (uint32 packing)", testFFT_uint32);
    test_framework("Testing Iterative FFT (uint32 packing)", test_iterative_FFT_uint32);
    test_framework("Testing FFT (uint64 packing)", testFFT_uint64);
    test_framework("Testing Iterative FFT (uint64 packing)", testFFT_uint64);
}