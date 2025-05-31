# OLE PCG over Field and Ring via Trace

A prototype implementation of the Pseudorandom Correlation Generator (PCG) for OLE and authenticated multiplication triples over $\mathbb{F}_2$ and ${Z}_{2^k}$ in C following the [paper](https://eprint.iacr.org/2025/169). The paper for ring will come soon on eprint.
It is a secondary development based on the implementation of [$\mathbb{F}_4$ OLEAGE](https://github.com/sachaservan/FOLEAGE-PCG/).

## Organization
The [libs/gf64](libs/gf64) and [libs/gf128](libs/gf128) folders contains the implementations of operations over $\mathbb{F}_{2^{64}}$ and $\mathbb{F}_{2^{128}}$, respectively.
The [src/](src/) folder contains the code for benchmark tests.
- [src/modular_bench.c](src/modular_bench.c) is a modular implemetation of [src/bench.c](src/bench.c) with a minor optimization.
- [src/trace_bench.c](src/trace_bench.c) implements benchmarks for the semi-honest multiplication triples over $\mathbb{F}_2$ based on the trace functions.
- [src/mal_gf64_trace_bench.c](src/mal_gf64_trace_bench.c) implements benchmarks for the authenticated multiplication triples over $\mathbb{F}_{2^{64}}/\mathbb{F}_2$ based on the trace functions.
- [src/mal_gf128_trace_bench.c](src/mal_gf128_trace_bench.c) implements benchmarks for the authenticated multiplication triples over $\mathbb{F}_{2^{128}}/\mathbb{F}_2$ based on the trace functions.
- [src/gr64_bench.c](src/gr64_bench.c) implements benchmarks for the semi-honest multiplication triples over $\mathbb{GR}({2^{32}},2)$.
- [src/gr128_bench.c](src/gr128_bench.c) implements benchmarks for the semi-honest multiplication triples over $\mathbb{GR}({2^{64}},2)$.
- [src/gr64_trace_bench.c](src/gr64_trace_bench.c) implements benchmarks for the semi-honest multiplication triples over $\mathbb{Z}_{2^{32}}$ based on the generalized trace functions.
- [src/gr128_trace_bench.c](src/gr128_trace_bench.c) implements benchmarks for the semi-honest multiplication triples over $\mathbb{Z}_{2^{64}}$ based on the generalized trace functions.
- [src/SPDZ2k_32_bench.c](src/SPDZ2k_32_bench.c) implements benchmarks for the authenticated multiplication triples over $\mathbb{Z}_{2^{32}}$.
- [src/SPDZ2k_64_bench.c](src/SPDZ2k_64_bench.c) implements benchmarks for the authenticated multiplication triples over $\mathbb{Z}_{2^{64}}$.

## Dependencies
These dependencies are required by the [ternary DPF](https://github.com/sachaservan/tri-dpf) submodule.
- OpenSSL
- GNU Make
- Cmake
- Clang

## Getting everything to run (tested on Ubuntu, CentOS, and MacOS)

| Install dependencies (Ubuntu):         | Install dependencies (CentOS):              |
| -------------------------------------- | ------------------------------------------- |
| `sudo apt-get install build-essential` | `sudo yum groupinstall 'Development Tools'` |
| `sudo apt-get install cmake`           | `sudo yum install cmake`                    |
| `sudo apt install libssl-dev`          | `sudo yum install openssl-devel`            |
| `sudo apt install clang`               | `sudo yum install clang`                    |

## Running benchmarks

Benchmarks:

```
git submodule update --init --recursive
make
./bin/pcg --modular_bench
./bin/pcg --trace_bench
./bin/pcg --mal_64_trace_bench
./bin/pcg --mal_128_trace_bench

./bin/pcg --gr64_bench
./bin/pcg --gr128_bench
./bin/pcg --gr64_trace_bench
./bin/pcg --gr128_trace_bench
./bin/pcg --SPDZ2k_32_bench
./bin/pcg --SPDZ2k_64_bench
```

<!-- ## Parameter Selection

The parameters `c` and `t` can be computed using the [SageMath parameter selection script](https://github.com/mbombar/estimator_folding) (also available as a submodule in `scripts/parameters_selection`).
We provide reasonable choices of `c` and `t` in Table 2 of [the paper](https://eprint.iacr.org/2024/429.pdf).
In particular, our benchmarks use `(c=4, t=27)` as a conservative parameter choice and `(c=3,t=27)` as an aggressive parameter choice, when targeting at least $\lambda=128$ bits of security. -->

## Future development

The current prototype implementation can be extended in several ways.
TODOs are left in-line, however, the broad strokes include:

- [ ] Add the implementation for OLE over $\mathbb{F}_2$ for larger $c$ or larger fields instead of $\mathbb{F}_4$.
- [ ] ~~Change iterative FFT to recursive as the recursive FFT is more efficient.~~ Our tests show that recursive FFT outperforms iterative FFT for large datasets, likely due to the latter requiring at least one additional data copy.
- [ ] Merge the codes of [src/mal_gf64_trace_bench.c](src/mal_gf64_trace_bench.c) and [src/mal_gf128_trace_bench.c](src/mal_gf128_trace_bench.c) as the code structures are very similar.

- [ ] Merge the codes of [src/gr64_bench.c](src/gr64_bench.c) and [src/gr128_bench.c](src/gr128_bench.c) as the code structures are very similar.
- [ ] Merge the codes of [src/gr64_trace_bench.c](src/gr64_trace_bench.c) and [src/gr6128_trace_bench.c](src/gr6128_trace_bench.c).
- [ ] Merge the codes of [src/SPDZ2k_32_bench.c](src/SPDZ2k_32_bench.c) and [src/SPDZ2k_64_bench.c](src/SPDZ2k_32_bench.c).

## ⚠️ Important Warning

Very recently, a [paper](https://eprint.iacr.org/2025/892) showed that the QA-SD problem with parameters $(c=3, t=27, q=4, n=16)$ is insecure. To make the attack infeasible, the paper proposed to use parameter satisfying $c\ge 1+(n-1)/(q-1)$. Here are some example parameters $(c=9,t=9,q=4,n<=25)$, $(c=3, t=49, q=8, n<=15)$ or $(c=6, t=15, q=16, n<=76)$.

<b>This implementation is intended for _research purposes only_. The code has NOT been reviewed by security experts.
As such, no portion of the code should be used in any real-world or production setting!</b>
