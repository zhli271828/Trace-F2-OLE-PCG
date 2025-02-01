# Trace-OLE PCG

A prototype implementation of the binary OLE Pseudorandom Correlation Generator (PCG) in C.
The paper will come soon on eprint.
It is a secondary development based on the implementation of [$\mathbb{F}_4$ OLEAGE](https://github.com/sachaservan/FOLEAGE-PCG/).

## Organization
The [libs/gf64](libs/gf64) and [libs/gf128](libs/gf128) folders contains the implementations of operations over $\mathbb{F}_{2^{64}}$ and $\mathbb{F}_{2^{128}}$, respectively.

The [src/](src/) folder contains the code for benchmark tests.
- [src/modular_bench.c](src/modular_bench.c) is a modular implemetation of [src/bench.c](src/bench.c) with a minor optimization.
- [src/trace_bench.c](src/trace_bench.c) implements benchmarks for the semi-honest multiplication triples over $\mathbb{F}_2$ based on the trace functions.
- [src/mal_gf64_trace_bench.c](src/mal_gf64_trace_bench.c) implements benchmarks for the authenticated multiplication triples over $\mathbb{F}_{2^{64}}/\mathbb{F}_2$ based on the trace functions.
- [src/mal_gf128_trace_bench.c](src/mal_gf128_trace_bench.c) implements benchmarks for the authenticated multiplication triples over $\mathbb{F}_{2^{128}}/\mathbb{F}_2$ based on the trace functions.

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
```

<!-- ## Parameter Selection

The parameters `c` and `t` can be computed using the [SageMath parameter selection script](https://github.com/mbombar/estimator_folding) (also available as a submodule in `scripts/parameters_selection`).
We provide reasonable choices of `c` and `t` in Table 2 of [the paper](https://eprint.iacr.org/2024/429.pdf).
In particular, our benchmarks use `(c=4, t=27)` as a conservative parameter choice and `(c=3,t=27)` as an aggressive parameter choice, when targeting at least $\lambda=128$ bits of security. -->

## Future development

The current prototype implementation can be extended in several ways.
TODOs are left in-line, however, the broad strokes include:

- [ ] Change iterative FFT to recursive as the recursive FFT is more efficient.
- [ ] Merge the codes of [src/mal_gf64_trace_bench.c](src/mal_gf64_trace_bench.c) and [src/mal_gf128_trace_bench.c](src/mal_gf128_trace_bench.c) as the code structures are very similar.

## ⚠️ Important Warning

<b>This implementation is intended for _research purposes only_. The code has NOT been reviewed by security experts.
As such, no portion of the code should be used in any real-world or production setting!</b>
