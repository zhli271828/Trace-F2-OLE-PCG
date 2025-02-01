#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openssl/rand.h>
#include "gf128.h"
#include "utils.h"

#define NUM_COEFFS 9
void test_fft_iterative_f4() {
    size_t num_vars = 2;
    size_t num_coeffs = ipow(3, num_vars);
    // uint8_t aa[NUM_COEFFS] = {0b1,0b1,0b1,0b1,0b1,0b1,0b1,0b1,0b1};
    uint8_t aa[NUM_COEFFS] = {0};
    uint8_t cache[NUM_COEFFS] = {0};
    uint8_t rlt[NUM_COEFFS] = {0};
    
    size_t trials = NUM_COEFFS;
    for (size_t i = 0; i < trials; ++i) {
        aa[i] = random()%4;
        printf("Set a[%u]=%d:\n", i, aa[i]);
        fft_iterative_f4(aa, cache, rlt, num_vars);
        printf("a=[");
        for(size_t i=0; i<NUM_COEFFS; ++i) {
            printf("%d\t", aa[i]);
        }
        printf("]\ncache=[");
        for(size_t i=0; i<NUM_COEFFS; ++i) {
            printf("%d\t", cache[i]);
        }

        printf("]\nrlt=[");
        for(size_t i=0; i<NUM_COEFFS; ++i) {
            printf("%d\t", rlt[i]);
        }
        printf("]\n");
        aa[i] = 0;
    }
}

#define NUMVARS 16
double test_fft_iterative_gf128(uint128_t *coeffs, uint128_t *cache, uint128_t *rlt, size_t num_vars, uint128_t zeta) {

    //************************************************
    printf("Benchmarking FFT evaluation with uint128_t\n");
    //************************************************
    clock_t t;
    t = clock();
    fft_iterative_gf128(coeffs, cache, rlt, num_vars, zeta);
    t = clock() - t;
    double time_taken = ((double)t) / (CLOCKS_PER_SEC / 1000.0); // ms
    printf("FFT (uint128) eval time (total) %f ms\n", time_taken);
    return time_taken;
}


int main(int argc, char **argv) {
    printf("***************************************\n");
    printf("Testing function void print_uint128_t(char *str, uint128_t x)...\n");
    uint128_t max_uint128_t = (uint128_t)-1;
    print_uint128_t("max_uint128_t=", max_uint128_t);
    printf("***************************************\n");
    printf("Testing function  uint128_t gf128_multiply(uint128_t a, uint128_t b)...\n");
    uint128_t a = 0b1;
    uint128_t b = 0b10;
    print_uint128_t("a=", a);
    print_uint128_t("b=", b);
    print_uint128_t("gf128_multiply(a,b)=", gf128_multiply(a,b));
    printf("***************************************\n");
    printf("Testing function uint128_t gf128_power(uint128_t base, uint128_t exp)...\n");
    uint128_t prim = 0b10;
    print_uint128_t("prim=", prim);
    print_uint128_t("prim^{2^128-1}=", gf128_power(prim, max_uint128_t));
    print_uint128_t("gf128_multiply(prim^{2^128-1}, prim)=", gf128_multiply(gf128_power(prim, max_uint128_t), prim));
    uint128_t zeta = gf128_power(prim, max_uint128_t/3);
    // This output should be tested in distinct machines.
    print_uint128_t_bits("GF(4): zeta=", zeta);
    print_uint128_t("GF(4): zeta=", zeta);
    print_uint128_t("GF(4): zeta^3=", gf128_power(zeta, 3));
    print_uint128_t("GF(4): zeta*zeta*zeta=", gf128_multiply(gf128_multiply(zeta, zeta), zeta));
    print_uint128_t("GF(4): zeta*zeta=", gf128_multiply(zeta, zeta));
    print_uint128_t("GF(4): zeta^2=", gf128_power(zeta, 2));
    printf("***************************************\n");
    printf("Testing function fft_iterative_f4(const uint8_t *a, uint8_t *cache, uint8_t *rlt, size_t n)\n");
    test_fft_iterative_f4();
    printf("***************************************\n");
    printf("Testing function test_fft_iterative_gf128(const uint8_t *a, uint8_t *cache, uint8_t *rlt, size_t n)\n");

    double time = 0;
    size_t testTrials = 5;
    size_t num_vars = NUMVARS;
    size_t num_coeffs = ipow(3, num_vars);

    uint128_t *coeffs = malloc(sizeof(uint128_t) * num_coeffs);
    uint128_t *cache = calloc(sizeof(uint128_t), num_coeffs);
    uint128_t *rlt = calloc(sizeof(uint128_t), num_coeffs);
    for (size_t i = 0; i < testTrials; i++) {
        RAND_bytes(coeffs, sizeof(uint128_t) * num_coeffs);
        time += test_fft_iterative_gf128(coeffs, cache, rlt, num_vars, zeta);
        printf("Done with trial %u of %u\n", i + 1, testTrials);
    }
    free(coeffs);
    free(cache);
    free(rlt);
    printf("******************************************\n");
    printf("DONE\n");
    printf("Avg time: %0.2f\n", time / testTrials);
    printf("******************************************\n\n");

}