# Persistence-Algebra

Current version: **2.0.0**

The compatibility-preserving persistence-module API is documented in
[`docs/module-framework.md`](docs/module-framework.md). Existing presentation
matrix APIs remain available; new clients should start with
`#include <grlina/graded_linalg.hpp>` and `R2Module<index>`.

Coordinate and grid matrices are available through `<grlina/matrices/coord.hpp>` and
`<grlina/matrices/grid.hpp>`. Their types live in `graded_linalg` and
take the coordinate scalar, dimension, and index type as template arguments:
`CoordGradedSparseMatrix<double, 3, int>`. Both support graded kernels and
integration with `Module` and `ChainComplex`. The [header guide](#library-structure)
below lists the implementation headers by responsibility.

`GridGradedSparseMatrix` is graded over integer degrees `D`. Its `grid_` maps
these degrees into real space; `GridD` is the type of their images. `Grid` is
an array of coordinate vectors, one monotone map per axis. Methods
such as `dim_at(degree)` and `append_column(column, degree)` take coordinate
values. Use `dim_at_grid_degree(indices)` or
`append_column_at_grid_degree(column, indices)` for stored indices. This rule
also applies when `Scalar` and `Index` are the same integer type.

All graded matrices expose `degree_type`, `row_degree(i)`, and `col_degree(j)`
for actual degrees. Ordinary matrices return their stored degree; grid matrices
decode the stored indices through their grid. `base_degree<Matrix>` is the type
stored in `row_degrees` and `col_degrees`; `Matrix::degree_type` is the actual
degree type. These types coincide for ordinary matrices.

### Optional compact column storage

`<grlina/storage/compact.hpp>` provides `CompactStorage<Index>` for mutable
sparse matrices. A 32-bit column occupies 16 bytes and stores up to three entries
inline; longer columns own an array. Signed 64-bit indices are also supported.
The entry count of a single column is limited to `2^31 - 1`.

```cpp
using Matrix = graded_linalg::R2Matrix<int, graded_linalg::CompactStorage<int>>;
using Grid = graded_linalg::GridGradedSparseMatrix<double, 2, int,
                                                 graded_linalg::CompactStorage<int>>;
```

The matrix operations, kernel and solver use the selected backend without
changes to calling algorithms. `column()` borrows entries; `get_col()` returns
an owning vector. `move_col_from()` transfers compact column buffers directly.
Compact XOR runs on the stored columns, including temporary bitset reduction.
Conversions to vector and CSC storage use `assign_data()`.

Configure with `-DGRLINA_USE_COMPACT_COLUMNS=ON` to select compact storage for
the `R2Matrix<>` and `R3Matrix<>` aliases throughout a linked application.
The CMake target propagates this choice to all translation units. The default
remains vector storage; the other matrix templates keep their existing defaults.

### Optional CSC matrix storage

Include `<grlina/matrices/sparse.hpp>` for `CSCMatrix`, the sparse matrix alias
with contiguous compressed sparse columns:

```cpp
#include <grlina/matrices/sparse.hpp>
#include <grlina/graded_linalg.hpp>

using namespace graded_linalg;
using PackedMatrix = R2GradedSparseMatrix<int, CSCStorage<int>>;
using PackedModule = R2Module<int, CSCStorage<int>>;

PackedMatrix presentation("presentation.scc");
const PackedModule module(presentation);
auto copy = presentation;
copy.shift({1, 2}); // Changes degrees; the entry buffers remain shared.
copy.col_op(0, 1);  // Supported, but warns about CSC mutation cost.
```

`SparseMatrix<int>`, `R2GradedSparseMatrix<int>`, and `R2Module<int>` keep their
vector-of-vectors storage. Graded matrices always inherit from
`SparseMatrix<index, Storage>`; the added template argument selects only storage
at compile time. `const` controls mutability and does not choose the
representation. The same option is available through the generic
`GradedSparseMatrix` base. Shared
algorithms dispatch to storage primitives, without virtual calls or a runtime
variant. CSC storage, column views and diagnostics live in
`grlina/storage/csc.hpp`; `grlina/matrices/sparse.hpp` defines the `CSCMatrix` alias.
Other matrix types use the generic storage traits.

Matrix templates use their template parameter names directly. Generic code can
obtain types from the matrix interface with `matrix_index_t<Matrix>`,
`matrix_storage_t<Matrix>`, `matrix_column_t<Matrix>`, `base_degree<Matrix>`, and
`typename Matrix::degree_type`.

CSC uses a contiguous row-index array and a `std::size_t` column-offset array.
Matrix copies share these two buffers; a write detaches them first. Degree arrays,
row caches, and algorithm workspaces retain their ordinary copy semantics, so
copying an entire graded matrix or module is not necessarily constant-time.
Changing a column's length can move the remaining entries and update all later
offsets. Same-length edits avoid those shifts, but still detach shared buffers.
Bulk transforms rebuild the packed buffers once. No column operation implicitly
converts the entire matrix to vector storage.

Both warning channels are enabled by default:

- **Compile time:** expensive CSC primitives carry a C++17 `[[deprecated]]`
  diagnostic labelled “CSC performance warning”. The operation remains supported;
  the attribute communicates cost, not planned removal. Higher-level algorithms
  that instantiate these primitives also warn, even if a runtime branch does not
  eventually execute them. Read-only module construction can also warn because
  its virtual methods instantiate mutation paths. Compilers treating deprecation
  warnings as errors will reject these instantiations until the warning is
  acknowledged.
- **Runtime:** the first expensive edit or detach of nonempty shared buffers emits
  one message to `std::cerr`, across CSC index types and translation units.

Define `GRLINA_CSC_COMPILE_WARNINGS=0` and/or
`GRLINA_CSC_RUNTIME_WARNINGS=0` before including matrix headers to acknowledge the
cost and disable either channel independently. Keep these definitions consistent
across translation units. Plain `CSCMatrix` construction, reads, and copies do not
warn. Graded/module loading can produce the conservative compile diagnostics
described above; disable compile warnings and retain runtime warnings if this is
too noisy for a read-only client.

`matrix.column(i)` and `matrix.data[i]` provide borrowed, read-only columns.
Mutation may invalidate these views. `matrix.get_col(i)` returns an independent
vector; use `set_col`, `append_entry`, or `col_op` to edit the matrix. Existing
vector-backed matrices still expose their mutable `data` as before.

SCC readers append directly into the selected storage. To adopt buffers received
from another program, move them into `CSCStorage` (offsets start at zero and end at
the entry count), then assign the storage. Raw offset validity is checked only
when `GRLINA_ENABLE_CHECKS` is enabled:

```cpp
std::vector<std::size_t> offsets{0, 2, 2, 3};
std::vector<int> entries{0, 2, 1};
CSCMatrix<int> matrix(3, 3); // columns, rows
matrix.assign_data(CSCStorage<int>(std::move(offsets), std::move(entries)));
```

As with the existing low-level `data` API, storage primitives do not update matrix
dimensions, grades, or algorithm caches. Maintain these when assembling a matrix;
after appending columns, `compute_num_cols()` updates the column count.

A C++17 header-only library for computational multiparameter persistent homology. It provides flexible, efficient data structures and algorithms for working with **presentations of persistence modules** — graded linear maps between free modules over arbitrary posets — with a focus on the two-parameter case over ℝ².

This library forms the algebraic backbone of the author's PhD research and is used directly by three companion projects:

- **[AIDA](https://github.com/JanJend/AIDA)** — Approximation and Interval Decomposition Algorithm
- **[Skyscraper-Invariant](https://github.com/JanJend/Skyscraper-Invariant)** — computation of the skyscraper invariant for 2-parameter persistence modules
- **[Stable-Decomposition](https://github.com/JanJend/Stable-Decomposition)** — stable decomposition of multiparameter persistence modules

---

## Table of Contents

- [Background](#background)
- [Citing This Work](#citing-this-work)
- [Related Projects](#related-projects)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation and Build](#installation-and-build)
- [File Formats](#file-formats)
- [Library Structure](#library-structure)
- [Command-Line Tools](#command-line-tools)
- [Usage Examples](#usage-examples)
- [License](#license)

---

## Background

A **persistence module** over a poset *P* is a functor from *P* to vector spaces over a field (here 𝔽₂ for now). We store them via **presentations**: graded linear maps

```
d : F₁ → F₀
```

between free graded modules, whose cokernel is the persistence module of interest. This is because in practice these matrices are much smaller than storing all of the necessary structure maps. This library provides the data structures and algorithms needed to construct, manipulate, minimise, and analyse such presentations, as well as to compute derived algebraic objects such as kernels, resolutions, and homomorphism spaces.

---

## Citing This Work

If you use this library in your research, please cite it via its DOI. The full citation metadata is in [`CITATION.cff`](CITATION.cff).

```bibtex
@software{Jendrysiak_Persistence_Algebra,
  author       = {Jendrysiak, Jan},
  title        = {Persistence-Algebra},
  version      = {2.0.0},
  year         = {2026},
  license      = {LGPL-3.0-or-later},
  doi          = {10.4230/artifacts.23283},
  orcid        = {https://orcid.org/0000-0002-3761-3463}
}
```

If you use this library through or alongside **AIDA**, please also cite the accompanying paper:

```bibtex
@InProceedings{djk25,
  author    = {Dey, Tamal K. and Jendrysiak, Jan and Kerber, Michael},
  title     = {{Decomposing Multiparameter Persistence Modules}},
  booktitle = {41st International Symposium on Computational Geometry (SoCG 2025)},
  pages     = {41:1--41:19},
  series    = {Leibniz International Proceedings in Informatics (LIPIcs)},
  ISBN      = {978-3-95977-370-6},
  ISSN      = {1868-8969},
  year      = {2025},
  volume    = {332},
  editor    = {Aichholzer, Oswin and Wang, Haitao},
  publisher = {Schloss Dagstuhl -- Leibniz-Zentrum f{\"u}r Informatik},
  address   = {Dagstuhl, Germany},
  doi       = {10.4230/LIPIcs.SoCG.2025.41}
}
```

A preprint is also available on arXiv: [arXiv:2504.08119](https://arxiv.org/abs/2504.08119).

The kernel computation for ℝ²-graded matrices is adapted from the **MPfree** algorithm by Michael Kerber and Alexander Rolle; please cite the relevant MPfree paper if that component is central to your use.

@inProceedings{mpfree,
author = {Michael Kerber and Alexander Rolle},
title = {Fast Minimal Presentations of Bi-graded Persistence Modules},
booktitle = {Algorithm Engineering and Experiments (ALENEX)},
year = {2021},
doi = {10.1137/1.9781611976472.16},
}

---

## Related Projects

This library is used as a dependency in the following projects, all part of the author's PhD thesis at TU Graz:

- **[AIDA](https://github.com/JanJend/AIDA)** — Computing indecomposable decompositions of multiparameter persistence modules; published at [SoCG 2025](https://doi.org/10.4230/LIPIcs.SoCG.2025.41) (Dey, Jendrysiak, Kerber)
- **[Skyscraper-Invariant](https://github.com/JanJend/Skyscraper-Invariant)** — efficient computation of the skyscraper invariant (Fersztand, Jendrysiak)
- **[Stable-Decomposition](https://github.com/JanJend/Stable-Decomposition)** — algorithm to compute the Pruning (Bjerkevik '24) to stabilise decomposition of multiparameter modules
(Bjerkevik, Jendrysiak, Lenzen)
---

## Features

- **Flexible matrix types** over 𝔽₂, with columns stored as sorted index vectors, sets, or dense bitsets
- **Graded sparse matrices** parametrised by an arbitrary degree type `D`, with existing specialisations for ℝ¹, ℝ², and ℝ³
- **Column reduction algorithms**: standard, triangular, graded, with and without tracking of performed operations
- **Kernel computation** for ℝ²-graded matrices, adapted from the MPfree algorithm (Kerber, Rolle - Implementation by Michael Kerber, TU Graz)
- **Isomorphism Testing** for persistence modules
- **Minimisation** of graded presentations
- **Free resolutions** up to second syzygies
- **Hom-space computation** between graded modules
- **Snap-to-grid** operations for discretising presentations
- **Quiver representation** extraction from presentations of the corresponding representations of quivers
- Reading and writing of `.scc` and `.firep` file formats (compatible with RIVET, mpfree, and related tools)
- OpenMP parallelisation for matrix transformations
- Comprehensive template design: column type, index type, and degree type are all template parameters

---

## Prerequisites

- A C++17-compatible compiler (GCC ≥ 9, Clang ≥ 9, MSVC ≥ 19.14)
- [CMake](https://cmake.org/) ≥ 3.15
- [Boost](https://www.boost.org/) ≥ 1.70 (components: `timer`, `system`; `dynamic_bitset` is header-only)

Hilbert PNG export (`invariants/hilbert_plot.hpp`) uses the bundled
`grlina/third_party/stb_image_write.h` (v1.16), installed with the library.
Its original license notices are retained in the header. No separate image
library or companion project is required.

On Ubuntu/Debian:

```bash
sudo apt install cmake libboost-all-dev
```

On macOS with Homebrew:

```bash
brew install cmake boost
```

---

## Installation and Build

Clone the repository and build with CMake:

```bash
git clone https://github.com/JanJend/Persistence-Algebra.git
cd Persistence-Algebra
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

For a debug build (produces executables with a `_debug` suffix):

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

To install the command-line tools system-wide:

```bash
sudo make install
```

### Isomorphism tests and runtime statistics

From the repository root, use a Release build for timings:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --target isomorphism_test isomorphism_benchmark -j 2
ctest --test-dir build -R '^isomorphism_' --output-on-failure
python3 tests/isomorphism_statistics.py --executable build/isomorphism_benchmark \
    --output build/isomorphism-statistics
```

The regression tests compare both Hom implementations against an independent
exhaustive oracle on tiny modules, cover all 67 spaces of 2×2 matrices over F₂,
and include non-isomorphic modules with equal Betti degrees and Hilbert functions.
Two real presentations are also tested after thousands of admissible row/column
additions and basis permutations; reversing the additions checks the construction.

The statistics script uses sparse random presentations of doubling size and
exact generator-degree multiplicities `k=1,2,4,8`, with three seeds per case.
It writes `report.md`, raw and aggregated CSVs, log-log regressions with bootstrap
intervals, and machine/command metadata. Options include `--sizes`,
`--multiplicities`, `--repeats`, `--seed`, `--timeout`, `--file-timeout`, and
`--files` (paths to two-parameter SCC presentations). Python needs only its
standard library. Each case runs in a separate process with a deadline;
timeouts remain unknown decisions and are excluded from fits.

The current exact algorithm enumerates a local matrix space for repeated degrees
of multiplicity at least three. This can take exponential time in `k²`:
multiplicity-eight positive cases and large real inputs can time out. The
benchmark includes all-singular block spaces to expose this limit. Empirical
size regressions for small `k` are not general complexity bounds.

### Graded linear system scaling benchmark

The benchmark compares the queued graded solver with the previous per-column
solver and computing the full kernel of `(A | B)` followed by solution extraction.
It checks the reference coefficients, grading, and A X = B outside the timed
region. The functional tests also compare the methods on random systems and
check that differences between nonunique solutions lie in the graded kernel.

```bash
cmake --build build --target graded_linear_system_benchmark -j 2
ctest --test-dir build -R '^graded_linear_system_benchmark$' --output-on-failure
./build/graded_linear_system_benchmark 256 3 all > build/graded-linear-system-scaling.csv
./build/graded_linear_system_benchmark 1024 1 mixed 256 > build/graded-linear-system-large.csv
./build/graded_linear_system_benchmark 1024 6 uniform 0 > build/graded-linear-system-balanced.csv
```

Arguments are maximum size, repetitions per size, and case (`uniform`, `mixed`,
`late_basis`, `nonunique`, or `all`). Sizes double from 32; increase 256 to 512 or beyond for
larger runs. The optional fourth argument caps the old solver's size so the
new solver can be tested further; skipped old timings are `nan`.
With the column baseline skipped, an even repetition count balances the order
of the two remaining methods. CSV output
reports median milliseconds for all three methods, speedups relative to the
direct solver, growth when size doubles, observed exponents, and times normalized
by n⁴ and n³. `kernel_ms` includes matrix construction and solution extraction;
`kernel_only_ms` times the kernel call alone. Exact coefficient comparisons and
`A X = B` checks are reported separately (`column_exact_match=-1` means skipped).
Quartic growth predicts a factor of 16 per doubling; cubic growth predicts a factor of 8.
`uniform` uses one RHS grade; `mixed` uses n incomparable grades. `late_basis`
also puts late duplicate A columns before the admissible basis. In `nonunique`,
both duplicate blocks are admissible: the column solver chooses late coordinates,
while the other two choose early coordinates. These are different valid solutions.
The direct solver's two-parameter cases all share
the kernel algorithm's queued traversal and column reduction, with no per-degree
fallback. The solver maintains an A helper and a slave initialized by `(I | 0)`;
B columns never become reducers. Each zero RHS's slave column gives its solution
in the original A basis. Reduction stops after the last requested RHS, without
constructing any kernel generators. A `graded_kernel()` method is not required.
For three or more parameters it reuses the existing kernel's two-parameter slice
restrictions at the frozen prefixes requested by B. Timing estimates do not
establish worst-case bounds in general dimensions.

### Chain-complex minimization benchmark against multi-chunk

The optional benchmark uses the multi-chunk checkout at `../multi-chunk`.
Set `GRLINA_MULTI_CHUNK_ROOT` to use another location. It compares
`ChainComplex::minimize(false)` on sorted input, the default `minimize()` call
including its sorting step, and multi-chunk's compression routine. Both use
one thread and omit map tracking. Timings are median milliseconds after a
warm-up, with execution order rotated. Input copies, initial sorting, format
conversion, and output checks are outside the timers; work performed inside
each minimizer's public call is included.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGRLINA_BUILD_TESTS=ON
cmake --build build --target chain_complex_minimization_benchmark -j 2
ctest --test-dir build -R '^chain_complex_minimization_benchmark' --output-on-failure
./build/chain_complex_minimization_benchmark 4000 5 all > build/chain-minimization.csv
./build/chain_complex_minimization_benchmark file ../multi-chunk/sample_files/hand.scc 3 > build/chain-minimization-hand.csv
```

Synthetic arguments are maximum cancellable pairs, repetitions, and family
(`independent`, `path`, or `all`). `independent` has isolated local pairs;
`path` has long chains of substitutions during compression. Both have a
nonzero neighboring differential and a known reduced complex, checked exactly.
CSV includes input/output generator and entry counts, all three timings,
and `multi_over_grlina` (greater than one means grlina was faster).

File mode accepts ordinary two-parameter SCC files with one degree per
generator and grades for every chain group. Multicritical SCC files require
conversion to a free complex before this comparison. The file is read once,
with sparse boundary entries sorted and combined over F2 outside the timers;
the adapter supplies multi-chunk's terminal zero map even when the file lacks
a trailing zero rank. Every output is checked for grading, chain equations,
and absence of equal-degree pairs, and the two methods must agree on generator
counts per degree and chain group. General files may yield different bases;
`exact_matrix_match` reports matrix equality separately. These checks do not
construct an isomorphism between differing outputs.

### Using the library as a header-only dependency

Since the core functionality lives entirely in the `include/grlina/` headers, you can integrate this into your own CMake project by adding the include path:

```cmake
target_include_directories(your_target PRIVATE path/to/Persistence-Algebra/include)
find_package(Boost REQUIRED COMPONENTS timer system)
target_link_libraries(your_target Boost::timer Boost::system)
```

---

## File Formats

The library reads and writes two standard formats used in the multiparameter persistence community.

### `.scc` (Sparse Column-Compressed)

```
scc2020
2
<num_cols> <num_rows> 0
<x1> <y1> ; <entry1> <entry2> ...
...
<x1> <y1> ;
...
```

The first block (num_cols lines) gives the relations with their degrees and non-zero row indices. The second block (num_rows lines) gives the generator degrees with no entries.

### `.firep` (Free-to-Free Presentation)

Similar structure but with a different header and an additional line describing the ambient chain complex level.

Both formats are compatible with [RIVET](https://github.com/rivetTDA/rivet), [mpfree](https://bitbucket.org/mkerber/mpfree), and [multipers](https://github.com/DavidLapous/multipers).

---

## Library Structure

Matrix classes, conversions and the shared matrix/presentation algorithms live
in `matrices/`. Storage internals and representation-specific arithmetic live in
`storage/`. Module classes and constructions live in `module/`; landscape and
Hilbert plotting live in `invariants/`. Degree definitions and poset helpers live
in `posets/`. Graded kernels, solvers, isomorphism tests and the `hom_`
homomorphism headers remain directly under `grlina/`. `graded_linalg.hpp` is the
aggregate include for the library.

```text
include/grlina/
├── matrices/
│   ├── base.hpp              # MatrixUtil interface and basic coefficient operations
│   ├── sparse.hpp            # SparseMatrix and CSCMatrix alias
│   ├── dense.hpp             # DenseMatrix
│   ├── graded.hpp            # Generic GradedSparseMatrix CRTP base
│   ├── upset.hpp             # UpsetGradedSparseMatrix
│   ├── r2.hpp                # R2GradedSparseMatrix
│   ├── coord.hpp             # Coordinate-graded matrices
│   ├── grid.hpp              # Grid-graded matrices and degree/grid adapters
│   ├── conversions.hpp       # Dense/sparse matrix conversions
│   ├── matrix_algorithms.hpp # Free matrix-family algorithms and column-space comparison
│   └── io.hpp                # SCC adapters and matrix stream helpers
├── storage/
│   ├── columns.hpp           # Column representations and Column_traits arithmetic
│   ├── traits.hpp            # MatrixStorageTraits and vector-backed primitives
│   ├── bitset_algebra.hpp    # Bitset helpers
│   ├── compact.hpp           # Inline columns and compact storage traits
│   └── csc.hpp               # CSCStorage, borrowed columns, traits and diagnostics
├── graded_kernel.hpp         # Graded kernels in two or more parameters
├── graded_kernel_scheduler.hpp # Grade scheduling for graded kernel reductions
├── graded_linear_system.hpp     # Degree-admissible linear systems
├── subspaces.hpp             # Subspace enumeration, decomposition data and serialization
├── posets/
│   ├── traits.hpp            # Degree_traits capabilities and linear-order contract
│   ├── coordinate.hpp        # CoordinateDegree, product orders and R2 arithmetic
│   ├── upset.hpp             # Scalar-templated 2D upsets, reverse inclusion
│   └── poset_operations.hpp  # Degree Hasse diagrams and CSV export
├── hom_algorithms.hpp        # Full and optimized Hom-space algorithms
├── hom_core_class.hpp        # Homomorphism class and resolution lifts
├── hom_interface.hpp         # Module Hom bases and canonical shift interfaces
├── module/
│   ├── module_base.hpp       # Module, resolutions, Hilbert queries and homology
│   ├── submodule.hpp         # Submodule, parent/generator maps and matrix adapters
│   ├── constructions.hpp     # Sums, quotients, pullbacks and pushouts with maps
│   └── epsilon.hpp           # Epsilon images and kernels
├── isomorphism_test.hpp       # Exact module isomorphism decisions
├── quiver_representations.hpp # Quiver types, I/O and presentation conversion
├── invariants/
│   ├── hilbert_plot.hpp      # Hilbert function PNG visualization
│   └── landscape.hpp         # Rank landscapes, PNG bytes and R2 CSV/SVG export
├── proj_chain_complex.hpp    # ChainComplex, SCC I/O and ChainBasisMaps
├── graded_linalg.hpp         # Aggregate library API
├── general.hpp               # Timing, files, containers, sorting and formatting
├── graphs.hpp                # Graph construction, traversal and condensation
└── general_checks.hpp        # Diagnostic-check build policy
```

Class and function names are unchanged. Matrix headers define their member
operations directly:

```cpp
#include <grlina/matrices/sparse.hpp>

graded_linalg::SparseMatrix<int> matrix(2, 2, {{0}, {1}});
auto kernel = matrix.get_kernel();
```

For graded kernels and methods that use them (including full presentation
minimization and submodules), also include `graded_kernel.hpp`. Matrix-family
algorithms and column-space comparison live in `matrices/matrix_algorithms.hpp`;
induced quiver representations require `quiver_representations.hpp`.
`graded_linalg.hpp` exports these algorithms and the module API. For a focused
module include, use
`module/module_base.hpp`, `module/submodule.hpp`, or the relevant construction
header. Hilbert PNG export requires `invariants/hilbert_plot.hpp` separately.
`shifted_identity` lives beside `GradedSparseMatrix` in `matrices/graded.hpp`.
The dense/sparse conversion functions live in `matrices/conversions.hpp`.
The old matrix and storage header paths have been removed.

`ChainBasisMaps` and its identity helper live with `ChainComplex` in
`proj_chain_complex.hpp`. Euler-characteristic Hilbert helpers live with
`Module` in `module/module_base.hpp`. Hom-space algorithms share `hom_algorithms.hpp`.
Submodule containment is `S.contains(T)` in `module/submodule.hpp`, with a check
that both submodules have the same parent object. Equality checks containment
in both directions; `present_same_submodule` adapts this to presentation and
generator matrices. The graded linear solver lives in `graded_linear_system.hpp`.
The former module umbrella and presentation-adapter headers have been removed.

### Key types at a glance

| Type | Description |
|---|---|
| `SparseMatrix<index>` | Ungraded 𝔽₂ matrix, columns as sorted `vec<index>` |
| `GradedSparseMatrix<D, index, DERIVED>` | Matrix graded by degree type `D`; CRTP base for concrete degree specialisations |
| `R2GradedSparseMatrix<index>` | ℝ²-graded matrix; primary working type for 2-parameter persistence |
| `CoordGradedSparseMatrix<Scalar, Dimension, Index>` | Coordinate-graded matrix with a compile-time dimension |
| `GridGradedSparseMatrix<Scalar, Dimension, Index>` | Grid-graded matrix with physical coordinate grids |
| `ChainComplex<Matrix>` | A chain complex of graded matrices |
| `Module<Matrix>` | A persistence module owning its resolutions and Hilbert queries |
| `Degree_traits<D>` | Policy struct defining the partial order, lexicographic sort, join/meet, I/O for degree type `D` |
| `Column_traits<COLUMN, index>` | Policy struct for column arithmetic (add, pivot, scalar product, etc.) |

---

## Command-Line Tools

After building, the following executables are available in `build/`:

| Executable | Description |
|---|---|
| `minimize` | Compute a minimal presentation from an input `.scc`/`.firep` file |
| `resolution` | Compute a free resolution (up to second syzygies) |
| `hom` | Compute the homomorphism space between two presentations |
| `shift_endo` | Compute the shift-endomorphism of a presentation |
| `snap_grid` | Snap all degrees to a coarser grid |
| `submodule_at` | Extract the submodule generated at a given degree |
| `pres_to_quiver` | Convert a presentation to a quiver representation |
| `analyse_ind` | Analyse indecomposable summands |
| `thickness` | Compute the thickness invariant |
| `size` | Print the size (number of generators and relations) of a presentation |
| `cut_module` | Cut a module by removing degrees above a threshold |

Each tool reads from a file path passed as the first argument and writes to stdout or an output file. Run any tool with no arguments for usage information.

---

## Usage Examples

### Computing a persistence landscape

```cpp
#define STB_IMAGE_WRITE_IMPLEMENTATION // in one .cpp, before any grlina include
#include <grlina/invariants/landscape.hpp>
using namespace graded_linalg;

Module<R2Matrix<>> module("presentation_or_resolution.scc");
Landscape landscape(module); // 1% resolution; degree bounds enlarged by 20%
auto png = landscape.image(1, 800, 600); // layer 1 as PNG bytes; no file created
auto inferno = landscape.image(1, 800, 600, ColourMap::inferno); // optional Matplotlib palette
auto shared_scale = landscape.image(1, 800, 600, ColourMap::blue, 0.5); // linear colours on [0,0.5]
// save_png(png, "landscape.png"); // explicitly save when wanted
landscape.save("landscape", 3); // barcode CSV, grid CSV, and SVG with three layers
// landscape.save("landscape_inferno", 3, ColourMap::inferno); // same palette in SVG

Landscape cropped(module, {0, 0, 1, 1}, 0.01); // xmin,ymin,xmax,ymax; relative resolution
auto heights = cropped.at({0.5, 0.5}, 3);
```

The optional final `image` argument fixes the maximum colour value; its default
of zero selects each image's sampled maximum. `sample(layer, width, height)`
returns the same numerical grid used for rendering. To obtain the PNG and its
sampled maximum together, use `image_with_data(layer, width, height, colormap,
colour_maximum, include_values)`: its result has `png`, `maximum`, and optional
`values` (`values[x][y]`). `maximum` is always the sampled maximum, even when the
colour scale is fixed separately.

For a shared scale, compute each grid once and render it after choosing the maximum:

```cpp
auto grid = landscape.sample_grid(1, 500, 500); // grid.values, grid.maximum
auto png = grid.image(ColourMap::blue, shared_maximum); // no algebra recomputation
```

The `landscape_image` CLI offers `--json` for `{maximum, png, bounds, layer}`
(PNG encoded as base64), `--include-values` to also return the array, and
`--values-only` to return the grid without rendering. Feed that grid JSON to
`landscape_image --render-values --scale MAX` on stdin to render at a shared
scale without reading a module or recomputing a barcode. Add `--json` to receive
the PNG and maximum together. Raw PNG output and `--maximum-only` remain available.

`Landscape` owns a copy of the presentation and cached barcodes on parallel
diagonal slices. Each query evaluates its own line's barcode, reusing a cached
line when available. There is no interpolation between slices; `error_bound()`
is zero, apart from ordinary floating-point rounding. Grid points on the same
line share one barcode. With an N-by-N grid and direction along the box diagonal,
only 2N-1 distinct lines are needed. Vineyard updates are enabled by default:
one initial reduction stores `R = D V`, then adjacent generator/relation order
swaps update it between slices. Coefficients retain their original IDs to avoid
rewriting matrix entries on swaps. Each sampling call owns its update state.
Pass `false` as the constructor's final argument, or `--no-vineyard` to the CLI,
to use independent reductions as a reference. Other dimensions or directions can require
more lines, up to one per pixel. Resolution sets the initial cache spacing
relative to the longest box side in direction units and the export grid spacing;
it does not limit the accuracy of point queries or PNG samples. The default box
uses `bounding_box()` with 10% padding on each side; these are
Betti-degree bounds for a minimal presentation. Constant axes use a unit span.

`save` writes the barcodes of the export grid's actual lines to
`landscape.barcodes.csv`, with bounds/direction/resolution metadata,
`sampling=grid`, and `slice,offset,birth,death` rows. Offsets need not be uniform.
Empty slices have blank endpoints; infinite deaths are retained. It also writes
the sampled `landscape.csv` (`x,y,layer,value`) and labelled `landscape.svg` views.
Only rendering/exporting materializes a grid. Bounds never clip bars or the module itself.

`image(layer = 1, width = 800, height = 600)` returns PNG bytes for one
1-based layer, evaluated directly at each pixel. Cached lines are reused;
additional grid lines are processed in order with vineyard updates. Width and
height count heatmap pixels; labels add 265 horizontal and 185 vertical pixels.
Heights use a linear scale: white at zero, then pale blue through muted blue to
black, or the optional inferno palette. Layers above those present in
the barcodes give zero images. The bytes can be passed to Jupyter's
`IPython.display.Image(data=png_bytes)`, just like Hilbert images. Define
`STB_IMAGE_WRITE_IMPLEMENTATION` before any grlina include in one translation
unit that supplies the PNG encoder; numerical queries and CSV/SVG exports do
not require it.

This follows [Vipond's diagonal-slice construction, Proposition 41](https://jmlr.org/papers/volume21/19-054/19-054.pdf).
The standalone `landscape_at` remains available for exact individual queries
in any positive coordinate dimension. A positive direction argument selects
coordinate weights; the default is `(1,...,1)`.

### Reading and minimising a presentation

```cpp
#include <grlina/graded_linalg.hpp>
using namespace graded_linalg;

int main() {
    // Read a presentation from an scc file
    R2GradedSparseMatrix<int> M("path/to/presentation.scc");

    std::cout << "Input: " << M.get_num_cols() << " relations, "
              << M.get_num_rows() << " generators\n";

    // Sort columns and rows lexicographically (required before minimisation)
    M.sort_columns_lexicographically();
    M.sort_rows_lexicographically();

    // Minimise in place
    M.minimize();

    std::cout << "Minimal: " << M.get_num_cols() << " relations, "
              << M.get_num_rows() << " generators\n";

    // Write the result
    std::ofstream out("minimal.scc");
    M.to_stream_r2(out);
}
```

### Computing the kernel (graded syzygy module)

```cpp
#include <grlina/graded_linalg.hpp>
using namespace graded_linalg;

int main() {
    R2GradedSparseMatrix<int> M("presentation.scc");
    M.sort_columns_lexicographically();
    M.sort_rows_lexicographically();

    // Kernel gives a presentation of the first syzygy module
    R2GradedSparseMatrix<int> K = M.graded_kernel();

    K.sort_columns_lexicographically();
    K.minimize();

    std::ofstream out("syzygies.scc");
    K.to_stream_r2(out);
}
```

### Building a free resolution

```cpp
#include <grlina/graded_linalg.hpp>
using namespace graded_linalg;

int main() {
    R2GradedSparseMatrix<int> d1("presentation.scc");
    d1.sort_columns_lexicographically();
    d1.sort_rows_lexicographically();

    Module<R2GradedSparseMatrix<int>> module(d1);
    module.compute_projective_resolution();

    std::ofstream out("resolution.scc");
    module.to_stream(out);

    r2degree alpha = {1.5, 2.0};
    std::cout << "dim at (1.5, 2.0) = " << module.dimension_at(alpha) << "\n";
}
```

### Snapping to a coarser grid

```cpp
#include <grlina/matrices/r2.hpp>
#include <grlina/matrices/matrix_algorithms.hpp>
using namespace graded_linalg;

int main() {
    R2GradedSparseMatrix<int> M("presentation.scc");

    // Snap all degrees to a uniform 10x10 grid within the bounding box
    M.snap_to_equidistant_grid(10);

    std::ofstream out("snapped.scc");
    M.to_stream_r2(out);
}
```

### Extending the library to a new degree type

Specialise `Degree_traits<D>` only for the capabilities your degree type has.
The checks in `posets/traits.hpp` distinguish the comparison interface from sorting,
lattice operations, and group operations:

| Check | Required `Degree_traits<D>` interface |
| --- | --- |
| `is_poset_degree_v<D>` | `equals`, `smaller`, `greater`, `smaller_equal`, `greater_equal`, each returning `bool` |
| `is_degree_v<D>` | Poset comparisons plus `lex_order(a, b)` and `lex_lambda()` for a compatible linear extension |
| `is_lattice_degree_v<D>` | Poset comparisons plus `join(a, b)` and `meet(a, b)`, each returning `D` |
| `is_ordered_abelian_group_degree_v<D>` | Poset comparisons plus `zero() -> D`, `add(amount, value) -> void`, and `subtract(amount, value) -> void` |
| `is_lattice_ordered_abelian_group_degree_v<D>` | Both the lattice and ordered abelian group interfaces |

`is_degree_v` is the minimum for generic graded matrices. Hasse-diagram code
accepts `is_poset_degree_v` because it receives an already sorted list.
The algebraic checks do not require sorting, serialization, printing, or
coordinates. Individual operations can be checked using `has_degree_join_v`,
`has_degree_meet_v`, `has_degree_zero_v`, `has_degree_add_v`, and
`has_degree_subtract_v`. For example, shifting requires subtraction; it does not
require joins or meets.
Full relation minimization and free resolutions additionally need a
poset-specific `graded_kernel` implementation on the matrix type; comparisons
alone support local pair cancellation and degreewise dimension queries.

The stronger mathematical structure is a **lattice-ordered abelian group**
(abelian **ℓ-group**). Every pair has a least upper bound (join) and a greatest
lower bound (meet), and translation preserves the partial order:
`a <= b` implies `a + c <= b + c`. A total order is not required;
coordinatewise order on ℤⁿ is an example. These C++ checks inspect signatures,
not algebraic laws. Implementations must satisfy the partial-order, lattice,
and abelian-group laws as appropriate. `add(amount, value)` changes `value` to
`value + amount`; subtraction is analogous. The group identity is supplied
explicitly by `zero()`, rather than assumed to be a default-constructed degree.

This is a complete minimum specialization without arithmetic or lattice helpers:

```cpp
#include <grlina/posets/traits.hpp>
using namespace graded_linalg;

struct MyDegree {
    int x, y, z;
    bool operator==(const MyDegree& b) const {
        return x == b.x && y == b.y && z == b.z;
    }
};

template<>
struct Degree_traits<MyDegree> {
    static bool smaller_equal(const MyDegree& a, const MyDegree& b) {
        return a.x <= b.x && a.y <= b.y && a.z <= b.z;
    }
    static bool equals(const MyDegree& a, const MyDegree& b) {
        return a == b;
    }
    static bool smaller(const MyDegree& a, const MyDegree& b) {
        return smaller_equal(a, b) && !equals(a, b);
    }
    static bool greater(const MyDegree& a, const MyDegree& b) { return smaller(b, a); }
    static bool greater_equal(const MyDegree& a, const MyDegree& b) { return smaller_equal(b, a); }
    static bool lex_order(const MyDegree& a, const MyDegree& b) {
        return a.x < b.x || (a.x == b.x &&
            (a.y < b.y || (a.y == b.y && a.z < b.z)));
    }
    static auto lex_lambda() {
        return [](const MyDegree& a, const MyDegree& b) { return lex_order(a, b); };
    }
};

static_assert(is_degree_v<MyDegree>);
static_assert(!is_lattice_ordered_abelian_group_degree_v<MyDegree>);
// Usable with GradedSparseMatrix<MyDegree, int, MyDerivedMatrix>.
```

Optional `has_degree_io_v<D>` checks `poset_id`, `write_degree(out, degree)`,
and `from_stream(in)`. `has_degree_position_v<D>` checks
`position(degree) -> std::vector<double>`, and `has_degree_print_v<D>` checks
`print_degree(degree) -> void`. Serialization and visualization require these
only when used. Built-in coordinate traits supply every capability, including
`zero()`.

With C++20, the corresponding concepts are `PosetDegree`, `Degree`,
`LatticeDegree`, `OrderedAbelianGroupDegree`, and
`LatticeOrderedAbelianGroupDegree`.

`posets/upset.hpp` defines `r2upset<Scalar> = vec<CoordinateDegree<Scalar, 2>>`,
using the corresponding coordinate degree traits. Generators must be sorted
lexicographically by x, then y, with all redundant generators removed;
`Degree_traits<r2upset<Scalar>>::remove_redundant` removes duplicate and dominated
generators in place. Comparisons use reverse
inclusion: `greater_equal(a, b)` means every generator of `a` lies above a
generator of `b`. Consequently, `meet` computes union and `join` computes
intersection. Both operations and comparisons take linear time in the input
lengths. The empty vector is the greatest element. Lexicographic and
colexicographic sorting put longer common prefixes first to preserve this order.
SCC I/O follows multicritical's extended format: the parameter count is `2`, and
each degree contains x/y pairs up to the mandatory semicolon, without a generator
count. Parsing sorts the pairs and removes redundant generators.
These degrees have no addition, subtraction, group identity, or fixed-dimensional
geometric position.

`UpsetGradedSparseMatrix<Scalar, Index>` in `matrices/upset.hpp` uses these upsets as
row and column degrees. It inherits the generic graded matrix operations and
accepts stream or file-path constructors. `to_stream` and `to_file` write extended
SCC data with enough precision to preserve the scalar values:

```cpp
#include <grlina/matrices/upset.hpp>
using namespace graded_linalg;
using Matrix = UpsetGradedSparseMatrix<double, int>;

Matrix matrix("presentation.scc");
matrix.to_file("copy.scc");
auto complex = ChainComplex<Matrix>::from_file("multicritical_chain.scc");
complex.to_file("chain_copy.scc");
```

Full chain complexes use the existing `ChainComplex` interface, which stores
differentials from low to high homological degree. Readers accept comments and
blank lines; optional integer boundary coefficients are interpreted over F2.
Degrees must be provided for every nonzero chain group, including the final
group: SCC permits omitting that group's degrees, but then the file does not
specify the graded matrix completely. No upset-specific graded kernel is supplied.

---

## License

This library is released under the **GNU Lesser General Public License v3.0** (LGPL-3.0). See [`LICENSE`](LICENSE) for the full text.

Copyright © 2025 Jan Jendrysiak, TU Graz.

You are free to use, modify, and distribute this library, including in proprietary software, provided that any modifications to the library itself are shared under the same licence.
