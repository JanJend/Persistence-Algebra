# Free resolutions of multicritical chain complexes

This header library computes free resolutions of multicritical bifiltered chain
complexes over F2. It takes a chain complex of upset-graded matrices and produces
a free chain complex graded over R2 with the same persistent homology. Each input
generator may have several incomparable birth degrees; each output generator has
a single birth degree. Both the path and log-path algorithms are available, with
path as the default.

## Authorship and acknowledgments

**Michael Kerber (TU Graz) wrote the original multicritical implementation.**
We thank him very warmly for writing and sharing this code. I (Jan Jendrysiak)
am also deeply grateful to him for teaching me; my gratitude goes well beyond
this implementation.

We also warmly thank **Ulrich Bauer, Tamal K. Dey, Florian Russold, and Matthias
Söls**, who developed these algorithms together with Michael Kerber. Their paper,
[Fast Free Resolutions of Bifiltered Chain Complexes](https://doi.org/10.4230/LIPIcs.SoCG.2026.10),
describes the algorithms and their analysis. An
[arXiv version](https://arxiv.org/abs/2512.08652) is also available. Please cite
the paper when using these algorithms in research.

The original [multi-critical repository](https://bitbucket.org/mkerber/multi_critical/src/main/)
is hosted on Bitbucket. This integration is part of
[Persistence-Algebra on GitHub](https://github.com/JanJend/Persistence-Algebra).

## Usage

`free_resolution.hpp` accepts a const
`ChainComplex<UpsetGradedSparseMatrix<double, Index, Storage>>` and returns a
`ChainComplex<CoordGradedSparseMatrix<double, 2, Index>>`.
The input remains unchanged. The coordinate scalar is `double`, as in the
original algorithm; the caller's index type is retained.

```cpp
#include <multicritical/free_resolution.hpp>

using UpsetMatrix = graded_linalg::UpsetGradedSparseMatrix<double, int>;
using UpsetComplex = graded_linalg::ChainComplex<UpsetMatrix>;

const auto input = UpsetComplex::from_file("input.scc");
auto output = multi_critical::free_resolution(input); // path by default
output.to_file("free.scc");
```

Parsing and writing use the existing `ChainComplex` SCC functions. The conversion
reads C0 from the first differential's row degrees, and each higher group from
its differential's columns and upset degrees. The input must be a graded chain
complex with canonical upset degrees. This is already the degree representation
produced by the upset SCC reader.

The computation uses generator copies, relations, syzygies and homotopy maps.
Internal degrees use `CoordinateDegree<double, 2>` and
`r2upset<double>`, with comparisons and joins supplied by `Degree_traits`.
The resulting free matrices are constructed directly with `col_degrees`,
`row_degrees` and sparse columns. Stable degree sorting shares the same basis
order between adjacent matrices. Both input and output use grlina's `d1`, `d2`,
... ordering; C0 is stored in d1's rows, without a separate `C0 -> 0` matrix.
Empty higher groups produced by the original algorithm are retained.

The consolidated headers `mpp_utils.hpp`, `phat.hpp` and `scc.hpp` are reference
copies with their original namespaces and implementations. The free-resolution
header uses grlina's parser and matrix types directly. `Map`/`Dim_Map` and the
copy/relation/syzygy metadata serve as internal algorithm bookkeeping.

Input boundaries are read through column views and copied once into the internal
index type. Relation paths append directly to their destination, completed map
images are moved into storage, and known vector sizes are reserved before filling.

## Provenance and licensing

The original code was copied at commit
`0f439443d1fa34cbd02e6964b18b19ae7bf869c8` and adapted to grlina's degree,
matrix and chain-complex types. The original copyright notices and source-boundary
comments are retained; the license texts are in [COPYING](COPYING) and
[COPYING.LESSER](COPYING.LESSER).

## Checks and performance

Timing is optional: define `MULTI_CRITICAL_TIMERS=1` and link Boost.Timer to use
the copied timer functions. Header definitions are inline for use from multiple
translation units.

Build and run the native checks through the existing CMake project:

```sh
cmake --build build --target multicritical_copy_test
ctest --test-dir build -R multicritical_copy --output-on-failure
```

The check validates gradedness, chain equations, compatible bases, both algorithm
variants, native SCC roundtrips, 64-bit indices, F2 coefficients and index-range
overflow. If the original sample files are present, CTest also exercises all
three. Define `GRLINA_MULTICRITICAL_ORIGINAL` when compiling the same check with
the original project's include directories and Boost.Timer to compare against
its implementation. That comparison executable translates the original result
into grlina matrices only inside the test; the library conversion is fully native.
Pass `--dump` followed by sample paths to either executable to compare their
native SCC output byte-for-byte (compile the original comparison with `NDEBUG`
to suppress its diagnostic printouts).

The [performance comparison](../../docs/multicritical-benchmark-2026-10-11.md)
benchmarks both variants on the original project's three samples, with identical
compiler settings and checks outside the timed conversion. It includes raw
measurements and build commands for the reusable benchmark driver.
