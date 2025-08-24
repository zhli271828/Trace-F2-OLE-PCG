#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "test.h"
#include "modular_test.h"
#include "trace_bench.h"
#include "mal_gf128_trace_bench.h"
#include "mal_gf64_trace_bench.h"
#include "gr64_bench.h"
#include "gr128_bench.h"
#include "gr64_trace_bench.h"
#include "gr128_trace_bench.h"
#include "SPDZ2k_32_bench.h"
#include "SPDZ2k_32_d3_bench.h"
#include "SPDZ2k_32_d4_bench.h"
#include "SPDZ2k_64_d3_bench.h"
#include "SPDZ2k_64_d4_bench.h"
#include "SPDZ2k_64_bench.h"
#include "trace_f4_bench.h"
#include "trace_f8_bench.h"
#include "mal128_trace_f4_bench.h"
#include "common.h"

void printUsage() {
    printf("Usage: ./pcg [OPTIONS]\n");
    printf("Options:\n");
    printf("  --test\tTests correctness of the PCG.\n");
    printf("  --modular_test\tModularTests correctness of the PCG.\n");
    printf("  --bench\tBenchmarks the PCG on conservative and aggressive parameters.\n");
    printf("  --modular_bench\tModularBenchmarks the PCG on conservative and aggressive parameters.\n");
    printf("  --trace_bench\tTraceBenchmarks the PCG on conservative and aggressive parameters.\n");
    printf("  --gr64_bench\tTraceBenchmarks the PCG on conservative and aggressive parameters.\n");
    printf("  --gr128_bench\tTraceBenchmarks the PCG on conservative and aggressive parameters.\n");
    printf("  --gr64_trace_bench\tTraceBenchmarks the PCG on conservative and aggressive parameters.\n");
    printf("  --gr128_trace_bench\tTraceBenchmarks the PCG on conservative and aggressive parameters.\n");
    printf("  --SPDZ2k_32_bench\tTraceBenchmarks the PCG on conservative and aggressive parameters.\n");
    printf("  --SPDZ2k_64_bench\tTraceBenchmarks the PCG on conservative and aggressive parameters.\n");
    printf("  --SPDZ2k_32_D3_bench\tTraceBenchmarks the SPDZ2k_32 PCG on degree 3 Galois rings.\n");
    printf("  --SPDZ2k_32_D4_bench\tTraceBenchmarks the SPDZ2k_32 PCG on degree 4 Galois rings.\n");
    printf("  --SPDZ2k_64_D3_bench\tTraceBenchmarks the SPDZ2k_64 PCG on degree 3 Galois rings.\n");
    printf("  --SPDZ2k_64_D4_bench\tTraceBenchmarks the SPDZ2k_64 PCG on degree 4 Galois rings.\n");
    printf("  --mal_64_trace_bench\tMaliciousTraceBenchmarks the PCG on conservative and aggressive parameters.\n");
    printf("  --mal_128_trace_bench\tMaliciousTraceBenchmarks the PCG on conservative and aggressive parameters.\n");
    
}

typedef void (*bench_func)(size_t n, size_t c, size_t t, struct PCG_Time *pcg_time);
typedef void (*test_pcg_func)();

void run_pcg_benchmarks(size_t n, size_t c, size_t t, int num_trials, bench_func bf) {
    struct PCG_Time pcg_time;
    double pp_time = 0;
    double expand_time = 0;
    double total_time = 0;
    printf("Run: N=3^%zu, c=%zu, t=%zu\n", n, c, t);
    for (int i = 0; i < num_trials; i++) {
        bf(n, c, t, &pcg_time);
        pp_time += pcg_time.pp_time;
        expand_time += pcg_time.expand_time;
        total_time += pcg_time.total_time;
        printf("Done with trial %i of %i\n", i + 1, num_trials);
    }
    printf("******************************************\n");
    // printf("Avg time (N=3^%zu, c=%zu, t=%zu): %0.4f ms\n", n, c, t, time / num_trials);
    printf("N=3^%zu, c=%zu, t=%zu: Avg PP time %0.4f ms, expand time %0.4f ms, total time %0.4f ms\n", n, c, t, pp_time / num_trials, expand_time / num_trials, total_time / num_trials);
    printf("******************************************\n\n");
}

void run_hd_pcg_benchmarks(size_t base, size_t n, size_t c, size_t t, int num_trials, bench_func bf) {
    struct PCG_Time pcg_time;
    double pp_time = 0;
    double expand_time = 0;
    double total_time = 0;
    printf("Run: N=%zu^%zu, c=%zu, t=%zu\n", base, n, c, t);
    for (int i = 0; i < num_trials; i++) {
        bf(n, c, t, &pcg_time);
        pp_time += pcg_time.pp_time;
        expand_time += pcg_time.expand_time;
        total_time += pcg_time.total_time;
        printf("Done with trial %i of %i\n", i + 1, num_trials);
    }
    printf("******************************************\n");
    printf("N=%zu^%zu, c=%zu, t=%zu: Avg PP time %0.4f ms, expand time %0.4f ms, total time %0.4f ms\n", base, n, c, t, pp_time / num_trials, expand_time / num_trials, total_time / num_trials);
    printf("******************************************\n\n");
}

void pcg_bm_f4_trace_with_param(int num_trials, bench_func bf) {
    printf("******************************************\n");
    size_t c = 3;
    size_t t = 27;
    size_t n = 15;

    printf("Benchmarking PCG with aggressive parameters (c=%zu, t=%zu)\n", c, t);
    run_pcg_benchmarks(n, c, t, num_trials, bf);
    printf("******************************************\n");
}
void pcg_bm_mal128_f4_trace_with_param(int num_trials, bench_func bf) {
    printf("******************************************\n");
    size_t c = 3;
    size_t t = 27;
    size_t n = 15;

    printf("Benchmarking PCG with aggressive parameters (c=%zu, t=%zu)\n", c, t);
    run_pcg_benchmarks(n, c, t, num_trials, bf);
    printf("******************************************\n");
}

void pcg_bm_f8_trace_with_param(int num_trials, bench_func bf) {
    printf("******************************************\n");
    size_t c = 3;
    size_t t = 49;
    size_t n = 8;

    printf("Benchmarking PCG with aggressive parameters (c=%zu, t=%zu)\n", c, t);
    run_pcg_benchmarks(n, c, t, num_trials, bf);
    printf("******************************************\n");
}


void pcg_bm_with_param(int num_trials, bench_func bf) {

    printf("******************************************\n");
    size_t c = 3;
    size_t t = 27;

    printf("Benchmarking PCG with aggressive parameters (c=%zu, t=%zu)\n", c, t);
    run_pcg_benchmarks(15, c, t, num_trials, bf);
    printf("******************************************\n");
    // printf("Benchmarking PCG with conservative parameters (c=4, t=27)\n");
    // run_pcg_benchmarks(14, 4, 27, num_trials, bf);
    // run_pcg_benchmarks(15, 4, 27, num_trials, bf);
    // run_pcg_benchmarks(16, 4, 27, num_trials, bf);
    // run_pcg_benchmarks(18, 4, 27, num_trials, bf);
}

void pcg_bm_32_d3_with_param(int num_trials, bench_func bf) {

    printf("******************************************\n");
    size_t base = 7;
    size_t c = 2;
    size_t t = 49;
    size_t n = 7;

    printf("Benchmarking PCG with parameters (c=%zu, t=%zu)\n", c, t);
    run_hd_pcg_benchmarks(base, n, c, t, num_trials, bf);
    n = 8;
    run_hd_pcg_benchmarks(base, n, c, t, num_trials, bf);
    printf("******************************************\n");

    c = 3;
    printf("Benchmarking PCG with parameters (c=%zu, t=%zu)\n", c, t);
    n = 7;
    run_hd_pcg_benchmarks(base, n, c, t, num_trials, bf);
    n = 8;
    run_hd_pcg_benchmarks(base, n, c, t, num_trials, bf);
    n = 9;
    run_hd_pcg_benchmarks(base, n, c, t, num_trials, bf);
    printf("******************************************\n");
}

// TODO: change the parameters
void pcg_bm_64_d3_with_param(int num_trials, bench_func bf) {

    printf("******************************************\n");
    size_t base = 7;
    size_t c = 3;
    size_t t = 49;
    size_t n = 7;

    printf("Benchmarking PCG with parameters (c=%zu, t=%zu)\n", c, t);
    n = 7;
    run_hd_pcg_benchmarks(base, n, c, t, num_trials, bf);
    n = 8;
    run_hd_pcg_benchmarks(base, n, c, t, num_trials, bf);
    n = 9;
    run_hd_pcg_benchmarks(base, n, c, t, num_trials, bf);
    printf("******************************************\n");
}


void pcg_bm_32_d4_with_param(int num_trials, bench_func bf) {

    printf("******************************************\n");
    size_t base = 15;
    size_t c = 6;
    size_t t = 15;

    size_t n = 4;
    printf("Benchmarking PCG with parameters (c=%zu, t=%zu)\n", c, t);
    run_hd_pcg_benchmarks(base, n, c, t, num_trials, bf);
    n = 5;
    run_hd_pcg_benchmarks(base, n, c, t, num_trials, bf);
    n = 6;
    run_hd_pcg_benchmarks(base, n, c, t, num_trials, bf);
    n = 7;
    run_hd_pcg_benchmarks(base, n, c, t, num_trials, bf);
    printf("******************************************\n");
}

// TODO: chang the parameters
void pcg_bm_64_d4_with_param(int num_trials, bench_func bf) {

    printf("******************************************\n");
    size_t base = 15;
    size_t c = 9;
    size_t t = 15;

    size_t n = 4;
    printf("Benchmarking PCG with parameters (c=%zu, t=%zu)\n", c, t);
    run_hd_pcg_benchmarks(base, n, c, t, num_trials, bf);
    n = 5;
    run_hd_pcg_benchmarks(base, n, c, t, num_trials, bf);
    n = 6;
    run_hd_pcg_benchmarks(base, n, c, t, num_trials, bf);
    n = 7;
    run_hd_pcg_benchmarks(base, n, c, t, num_trials, bf);
    printf("******************************************\n");
}

void run_test_pcg(test_pcg_func tpf) {
    printf("******************************************\n");
    printf("Testing PCG\n");
    tpf();
    printf("******************************************\n");
    printf("PASS\n");
    printf("******************************************\n\n");
}

int main(int argc, char **argv)
{
    int num_trials = 10;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0) {
            pcg_bm_with_param(num_trials, bench_pcg);
        } else if (strcmp(argv[i], "--modular_bench") == 0) {
            pcg_bm_with_param(num_trials, modular_bench_pcg);
        } else if (strcmp(argv[i], "--trace_bench")==0) {
            pcg_bm_with_param(num_trials, trace_bench_pcg);
        } else if (strcmp(argv[i], "--trace_f4_bench") == 0) {
            pcg_bm_f4_trace_with_param(num_trials, trace_f4_bench_pcg);
        } else if (strcmp(argv[i], "--trace_f8_bench") == 0) {
            pcg_bm_f8_trace_with_param(num_trials, trace_f8_bench_pcg);
        } else if (strcmp(argv[i], "--mal_128_trace_f4_bench") ==0) {
            pcg_bm_mal128_f4_trace_with_param(num_trials, mal128_trace_f4_bench_pcg);
        } else if (strcmp(argv[i], "--gr64_bench") == 0) {
            pcg_bm_with_param(num_trials, gr64_bench_pcg);
        }
        else if (strcmp(argv[i], "--gr128_bench") == 0) {
            pcg_bm_with_param(num_trials, gr128_bench_pcg);
        }
        else if (strcmp(argv[i], "--gr64_trace_bench") == 0) {
            pcg_bm_with_param(num_trials, gr64_trace_bench_pcg);
        } else if (strcmp(argv[i], "--gr128_trace_bench") == 0) {
            pcg_bm_with_param(num_trials, gr128_trace_bench_pcg);
        } else if (strcmp(argv[i], "--SPDZ2k_32_bench") == 0) {
            pcg_bm_with_param(num_trials, SPDZ2k_32_bench_pcg);
        } else if (strcmp(argv[i], "--SPDZ2k_64_bench") == 0) {
            pcg_bm_with_param(num_trials, SPDZ2k_64_bench_pcg);
        } else if (strcmp(argv[i], "--SPDZ2k_32_D3_bench") == 0) {
            pcg_bm_32_d3_with_param(num_trials, SPDZ2k_32_D3_bench_pcg);
        } else if (strcmp(argv[i], "--SPDZ2k_64_D3_bench") == 0) {
            pcg_bm_64_d3_with_param(num_trials, SPDZ2k_64_D3_bench_pcg);
        } else if (strcmp(argv[i], "--SPDZ2k_32_D4_bench") == 0) {
            pcg_bm_32_d4_with_param(num_trials, SPDZ2k_32_D4_bench_pcg);
        } else if (strcmp(argv[i], "--SPDZ2k_64_D4_bench") == 0) {
            pcg_bm_64_d4_with_param(num_trials, SPDZ2k_64_D4_bench_pcg);
        }
        else if (strcmp(argv[i], "--mal_128_trace_bench") == 0) {
            pcg_bm_with_param(num_trials, mal_gf128_trace_bench_pcg);
        }
        else if (strcmp(argv[i], "--mal_64_trace_bench") == 0) {
            pcg_bm_with_param(num_trials, mal_gf64_trace_bench_pcg);
        }
        else if (strcmp(argv[i], "--test") == 0)
        {
            run_test_pcg(test_pcg);
        }
        else if (strcmp(argv[i], "--modular_test") == 0)
        {
            run_test_pcg(modular_test_pcg);
        }
        else
        {
            printUsage();
        }
    }
    if (argc == 1)
        printUsage();
}
