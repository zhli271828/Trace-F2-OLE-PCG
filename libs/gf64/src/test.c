#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openssl/rand.h>
#include "gf64.h"
#include "utils.h"

#define NUM_COEFFS 9
#define NUMVARS 16
double test_fft_iterative_gf64(uint64_t *coeffs, uint64_t *cache, uint64_t *rlt, size_t num_vars, uint64_t zeta) {

    //************************************************
    printf("Benchmarking FFT evaluation with uint64_t\n");
    //************************************************
    clock_t t;
    t = clock();
    fft_iterative_gf64(coeffs, cache, rlt, num_vars, zeta);
    t = clock() - t;
    double time_taken = ((double)t) / (CLOCKS_PER_SEC / 1000.0); // ms
    printf("FFT (uint64) eval time (total) %f ms\n", time_taken);
    return time_taken;
}

int main(int argc, char **argv) {
    printf("***************************************\n");
    printf("Testing function void print_uint64_t(char *str, uint64_t x)...\n");
    uint64_t max_uint64_t = (uint64_t)-1;
    print_uint64_t("max_uint64_t=", max_uint64_t);
    printf("***************************************\n");
    printf("Testing function uint64_t gf64_multiply(uint64_t a, uint64_t b)...\n");
    uint64_t a = 0b1;
    uint64_t b = 0b10;
    print_uint64_t("a=", a);
    print_uint64_t("b=", b);
    print_uint64_t("gf64_multiply(a,b)=", gf64_multiply(a,b));
    printf("***************************************\n");
    printf("Testing function uint64_t gf64_power(uint64_t base, uint64_t exp)...\n");
    uint64_t prim = 0b10;
    print_uint64_t("prim=", prim);
    print_uint64_t("prim^{2^64-1}=", gf64_power(prim, max_uint64_t));
    print_uint64_t("gf64_multiply(prim^{2^64-1}, prim)=", gf64_multiply(gf64_power(prim, max_uint64_t), prim));
    uint64_t zeta = gf64_power(prim, max_uint64_t/3);
    // This output should be tested in distinct machines.
    // print_uint64_t_bits("GF(4): zeta=", zeta);
    print_uint64_t("GF(4): zeta=", zeta);
    print_uint64_t("GF(4): zeta^3=", gf64_power(zeta, 3));
    print_uint64_t("GF(4): zeta*zeta*zeta=", gf64_multiply(gf64_multiply(zeta, zeta), zeta));
    print_uint64_t("GF(4): zeta*zeta=", gf64_multiply(zeta, zeta));
    print_uint64_t("GF(4): zeta^2=", gf64_power(zeta, 2));
    printf("***************************************\n");
    printf("Testing function test_fft_iterative_gf64(const uint8_t *a, uint8_t *cache, uint8_t *rlt, size_t n)\n");

    double time = 0;
    size_t testTrials = 5;
    size_t num_vars = NUMVARS;
    size_t num_coeffs = ipow(3, num_vars);

    uint64_t *coeffs = malloc(sizeof(uint64_t) * num_coeffs);
    uint64_t *cache = calloc(sizeof(uint64_t), num_coeffs);
    uint64_t *rlt = calloc(sizeof(uint64_t), num_coeffs);
    for (size_t i = 0; i < testTrials; i++) {
        RAND_bytes(coeffs, sizeof(uint64_t) * num_coeffs);
        time += test_fft_iterative_gf64(coeffs, cache, rlt, num_vars, zeta);
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