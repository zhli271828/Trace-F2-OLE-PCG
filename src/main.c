#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "test.h"
#include "modular_test.h"
#include "trace_bench.h"
#include "mal_gf128_trace_bench.h"
#include "mal_gf64_trace_bench.h"

void printUsage() {
    printf("Usage: ./pcg [OPTIONS]\n");
    printf("Options:\n");
    printf("  --test\tTests correctness of the PCG.\n");
    printf("  --modular_test\tModularTests correctness of the PCG.\n");
    printf("  --bench\tBenchmarks the PCG on conservative and aggressive parameters.\n");
    printf("  --modular_bench\tModularBenchmarks the PCG on conservative and aggressive parameters.\n");
    printf("  --trace_bench\tTraceBenchmarks the PCG on conservative and aggressive parameters.\n");
    printf("  --mal_64_trace_bench\tMaliciousTraceBenchmarks the PCG on conservative and aggressive parameters.\n");
    printf("  --mal_128_trace_bench\tMaliciousTraceBenchmarks the PCG on conservative and aggressive parameters.\n");

}

typedef double (*bench_func)(size_t n, size_t c, size_t t);
typedef void (*test_pcg_func)();

void run_pcg_benchmarks(size_t n, size_t c, size_t t, int num_trials, bench_func bf) {
    double time = 0;

    for (int i = 0; i < num_trials; i++) {
        time += bf(n, c, t);
        printf("Done with trial %i of %i\n", i + 1, num_trials);
    }
    printf("******************************************\n");
    printf("Avg time (N=3^%zu, c=%zu, t=%zu): %0.4f ms\n", n, c, t, time / num_trials);
    printf("******************************************\n\n");
}

void pcg_bm_with_param(int num_trials, bench_func bf) {

    printf("******************************************\n");
    printf("Benchmarking PCG with aggressive parameters (c=3, t=27)\n");
    run_pcg_benchmarks(14, 3, 27, num_trials, bf);
    // run_pcg_benchmarks(15, 3, 27, num_trials, bf);
    run_pcg_benchmarks(16, 3, 27, num_trials, bf);
    // run_pcg_benchmarks(18, 3, 27, num_trials, bf);

    printf("******************************************\n");
    printf("Benchmarking PCG with conservative parameters (c=4, t=27)\n");
    // tmp parameters
    run_pcg_benchmarks(14, 4, 27, num_trials, bf);
    // run_pcg_benchmarks(15, 4, 27, num_trials, bf);
    run_pcg_benchmarks(16, 4, 27, num_trials, bf);
    // run_pcg_benchmarks(18, 4, 27, num_trials, bf);
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
            pcg_bm_with_param(num_trials, bench_trace_pcg);
        } else if (strcmp(argv[i], "--mal_128_trace_bench")==0) {
            pcg_bm_with_param(num_trials, mal_gf128_trace_bench_pcg);
        } else if (strcmp(argv[i], "--mal_64_trace_bench")==0) {
            pcg_bm_with_param(num_trials, mal_gf64_trace_bench_pcg);
        } else if (strcmp(argv[i], "--test") == 0) {
            run_test_pcg(test_pcg);
        } else if (strcmp(argv[i], "--modular_test")==0) {
            run_test_pcg(modular_test_pcg);
        } else{
            printUsage();
        }
    }

    if (argc == 1)
        printUsage();
}
