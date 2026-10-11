Measured on 2026-10-11 against the unchanged neighboring `multicritical`
checkout at commit `0f439443d1fa34cbd02e6964b18b19ae7bf869c8`.
The native conversion is faster on all three supplied SCC samples, for both
path and logpath. All six canonical outputs match byte for byte with timers
both enabled and disabled. Every measured repetition also matches the same
structural fingerprint, generator count and nonzero count.
The assertion-enabled native checks also pass for the built-in fixtures and all
three samples, including gradedness, chain equations and SCC roundtrips.

The project's standard mode is **path**. Use the path rows below as the baseline
for performance decisions.
The initial measurements precede the implemented copy removals; the follow-up
at the end records their effect on path mode.

Conversion-only medians, with timers disabled in both implementations
(15 measured conversions per case):

| Sample | Variant | Original (ms) | Native (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| `compressed_deg_rips_100_points` | path | 323.51 | 244.22 | 1.32x |
| `compressed_deg_rips_100_points` | logpath | 900.18 | 763.13 | 1.18x |
| `full_deg_rips_20_points` | path | 303.43 | 221.53 | 1.37x |
| `full_deg_rips_20_points` | logpath | 662.48 | 552.11 | 1.20x |
| `wheel` | path | 253.49 | 210.93 | 1.20x |
| `wheel` | logpath | 64.19 | 36.62 | 1.75x |

Parsing followed immediately by the first conversion, with timers disabled
(median of five separate process runs per case):

| Sample | Variant | Original (ms) | Native (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| `compressed_deg_rips_100_points` | path | 344.14 | 278.87 | 1.23x |
| `compressed_deg_rips_100_points` | logpath | 920.55 | 802.05 | 1.15x |
| `full_deg_rips_20_points` | path | 326.77 | 261.16 | 1.25x |
| `full_deg_rips_20_points` | logpath | 683.95 | 592.92 | 1.15x |
| `wheel` | path | 283.29 | 243.74 | 1.16x |
| `wheel` | logpath | 76.63 | 55.94 | 1.37x |

Native parsing is slower on these files, but the combined load-and-convert
operation remains faster in every tested case. This second measurement includes
canonicalizing upset degrees, which the native reader performs before conversion.

Both binaries use Apple clang 17.0.0, targeting x86_64-apple-darwin24.6.0,
with `-std=c++17 -O3 -DNDEBUG -DGRLINA_ENABLE_CHECKS=0`. No OpenMP is enabled.
These are wall-clock measurements on the same local machine, with no concurrent
benchmark runs. Each case has five batches; each batch starts a fresh process,
parses once, performs one warmup conversion, then measures three conversions.
Execution order alternates between original and native. Original parser cursors
are reset outside the timed region; the native const input is reused directly.

The conversion timer covers the `free_resolution` call, including its internal
preparation, sorting, output construction and temporary cleanup. Parsing,
result fingerprinting, serialization and destruction of the returned result
are outside that timer. The combined measurement starts just before parsing
and ends when the first result returns; opening the input file is excluded.
All inputs have already been read for correctness comparisons, so filesystem
caches are warm. These timings measure the refactored API as used: native
matrix indices are 32-bit `int`, while original PHAT indices are 64-bit.
They include the degree, input and output storage changes together.

The original header has two unguarded timer objects in its logpath loop and
will not compile normally with timers disabled. The benchmark supplies no-op
objects for those calls in that configuration; the original headers are
unchanged. The native header already guards those calls. A separate comparison
uses real Boost.Timer instrumentation in both implementations:

| Sample | Variant | Original (ms) | Native (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| `compressed_deg_rips_100_points` | path | 334.02 | 249.55 | 1.34x |
| `compressed_deg_rips_100_points` | logpath | 1158.82 | 1011.70 | 1.15x |
| `full_deg_rips_20_points` | path | 306.77 | 221.38 | 1.39x |
| `full_deg_rips_20_points` | logpath | 949.62 | 834.85 | 1.14x |
| `wheel` | path | 258.03 | 218.64 | 1.18x |
| `wheel` | logpath | 65.13 | 38.03 | 1.71x |

With instrumentation enabled, preparing the internal input takes roughly
2.5–5.2 ms natively versus 14.9–26.2 ms originally. Constructing the output
matrices also takes less time in all six cases. These stage measurements
support the interpretation that removing parser adapters and constructing
native output matrices directly accounts for much of the improvement.

The resulting complex sizes are identical between implementations:

| Sample | Variant | Generators across all chain groups | Boundary nonzeros |
| --- | --- | ---: | ---: |
| `compressed_deg_rips_100_points` | path | 85,170 | 371,062 |
| `compressed_deg_rips_100_points` | logpath | 158,178 | 902,871 |
| `full_deg_rips_20_points` | path | 71,891 | 328,413 |
| `full_deg_rips_20_points` | logpath | 116,879 | 641,668 |
| `wheel` | path | 15,001 | 6,272,500 |
| `wheel` | logpath | 19,991 | 47,475 |

The wheel illustrates why variant choice matters independently of the port:
path produces 6,272,500 boundary entries, while logpath produces 47,475.

[Raw measurements](multicritical-benchmark-2026-10-11.csv) contain 360 timed
conversions. Each batch's parsing and combined measurement is repeated on its
three conversion rows; it should be counted once when aggregating those fields.
The initial instrumented pass did not record combined time (blank CSV field).
Stage times are zero when instrumentation is disabled because they are not
collected in that configuration.

The reusable [benchmark driver](../tests/multicritical_benchmark.cpp) accepts
`input.scc path|logpath repetitions [--dump]`. The dump option writes the
canonical native SCC output for exact comparisons, without reporting timings.
The original result adapter is used only for that dump, outside timing.

To rebuild from the Persistence-Algebra directory with the local Boost install:

```sh
boost_prefix=/Users/Jendrysiak/anaconda3
mkdir -p /tmp/grlina-multicritical-benchmark
c++ -std=c++17 -O3 -DNDEBUG -DGRLINA_ENABLE_CHECKS=0 -DMULTI_CRITICAL_TIMERS=0 \
  -Iinclude -I"$boost_prefix/include" tests/multicritical_benchmark.cpp \
  -L"$boost_prefix/lib" -Wl,-rpath,"$boost_prefix/lib" -lboost_timer -lboost_chrono \
  -o /tmp/grlina-multicritical-benchmark/native-no-timers
c++ -std=c++17 -O3 -DNDEBUG -DGRLINA_ENABLE_CHECKS=0 -DMULTI_CRITICAL_TIMERS=0 \
  -DGRLINA_MULTICRITICAL_ORIGINAL -Iinclude -I../multicritical/include \
  -I../multicritical/mpp_utils_mod/include -I../multicritical/phat_mod/include \
  -I../multicritical/scc_mod/include -I"$boost_prefix/include" \
  tests/multicritical_benchmark.cpp -L"$boost_prefix/lib" \
  -Wl,-rpath,"$boost_prefix/lib" -lboost_timer -lboost_chrono \
  -o /tmp/grlina-multicritical-benchmark/original-no-timers
/tmp/grlina-multicritical-benchmark/original-no-timers ../multicritical/sample_files/wheel.scc path 15
/tmp/grlina-multicritical-benchmark/native-no-timers ../multicritical/sample_files/wheel.scc path 15
```

Use `-DMULTI_CRITICAL_TIMERS=1` in both builds for the instrumented comparison.
For the batch protocol used above, run each executable with three repetitions,
five times per sample and variant, alternating which executable runs first.

Native conversion header SHA-256 at measurement time:
`a354e101e6c5c626ffe1d2707adfcff861c4a298eb3608ca069854887860267b`.


The copy removals were implemented on 2026-10-11. This follow-up compares the
native implementation immediately before and after that change, using **path**
in both. Both builds use the same 32-bit native matrix indices, Apple clang 17,
`-std=c++17 -O3 -DNDEBUG -DGRLINA_ENABLE_CHECKS=0`, and disabled timers.
Each sample has three separate process batches, each with one warmup and three
measured conversions (nine measurements per version). Execution order alternates
between versions; parsing and result checks are outside the timed call.

| Sample | Before copy removals (ms) | After copy removals (ms) | Time saved |
| --- | ---: | ---: | ---: |
| `compressed_deg_rips_100_points` | 321.15 | 276.41 | 13.9% |
| `full_deg_rips_20_points` | 302.93 | 239.28 | 21.0% |
| `wheel` | 271.79 | 251.32 | 7.5% |

The conversion reads input column views, constructs internal vectors from known
ranges, appends relation paths directly into the destination, moves completed
map images into storage, and reserves known vector sizes. Path is still the
default. Canonical SCC output matches byte for byte before and after on all
three samples. Assertion-enabled checks pass for both algorithm variants,
including chain equations, SCC roundtrips, unchanged input, CSC column views,
64-bit indices, F2 coefficients and index overflow.

[Raw copy-removal measurements](multicritical-copy-removal-2026-10-11.csv)
contain all 54 measured conversions. Wall times vary between benchmark sessions;
compare the paired before/after measurements within this follow-up.

Native conversion header SHA-256 after copy removals:
`0e8e8d435efa13f508072712547de7cb31b4abb8c8a38c82a0d30f528f9a3ce5`.
