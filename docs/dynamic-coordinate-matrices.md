# NEW: runtime coordinate matrices

`<grlina/dynamic_coordinate_matrix.hpp>` introduces
`DynamicCoordinateGradedSparseMatrix<Scalar, Index, Storage>` alongside the
existing fixed-dimension matrix classes. The number of grading parameters is
an instance property, so matrices with two, three, or five parameters have the
same C++ type. The sparse entries remain binary, as in `SparseMatrix`;
`Scalar` is the type of the degree coordinates.

```cpp
#include <grlina/dynamic_coordinate_matrix.hpp>
#include <grlina/module.hpp>

using namespace graded_linalg;
using Matrix = DynamicCoordinateGradedSparseMatrix<double, int>;
using Degree = Matrix::degree_type;

// Constructors take (columns, rows, parameter_count), in that order.
Matrix presentation(2, 1, 3, {{0}, {0}},
                    {{1, 0, 0}, {0, 1, 0}}, {{0, 0, 0}});
Matrix empty_five_parameter_map(0, 0, 5);
presentation.sort_compatibly();
Module<Matrix> module(presentation);
auto dimension = module.dimension_at(Degree{0, 0, 0}); // 1
```

The new matrix derives from the existing `SparseMatrix<Index, Storage>`,
independently of the `GradedSparseMatrix` base. Its graded operations work with
flat degree tables. Existing fixed-dimension classes and their algorithms
remain available.

The additions are marked `NEW` in their header comments:

- `dynamic_coordinate_degree.hpp`: owning degrees, read-only views, flat tables.
- `graded_matrix_algorithms.hpp`: graded algorithms using accessors supplied by a matrix.
- `graded_matrix_io.hpp`: SCC metadata and parsing with runtime dimension context.
- `dynamic_coordinate_matrix.hpp`: the public runtime coordinate matrix.

Each degree table owns one contiguous coordinate vector. For example, the
two column degrees above occupy `{1, 0, 0, 0, 1, 0}`. `row_degree(i)` and
`col_degree(i)` return read-only views into these buffers. Save an independent
degree with `to_degree()` before operations that replace, reorder, or resize
the table:

```cpp
Degree saved = presentation.col_degree(0).to_degree();
presentation.set_col_degree(0, Degree{2, 0, 0});
// saved still contains {1, 0, 0}.
```

Use `set_row_degree` and `set_col_degree` to copy coordinates into the matrix.
The table-level `row_degrees.set(i, degree)` and `col_degrees.set(i, degree)`
also exist. After direct table edits, call `invalidate_compatible_sorting()`;
after direct sparse-storage edits, call `invalidate_cached_rows()`. Use
`validate()` to check the resulting matrix.
`row_degrees.coordinates()` and `col_degrees.coordinates()` expose const
references to the flat buffers. The runtime dimension is fixed for an
instance; degrees, queries, and connected matrices must agree with it.
Zero parameters are supported: each degree is an empty tuple, while the
tables still retain their logical row and column counts.

The default sparse storage is `vec<vec<Index>>`. To select CSC storage, include
its header before the dynamic matrix header. `Index` must be a signed integral
type, because the existing sparse algorithms use the `-1` sentinel:

```cpp
#include <grlina/csc_matrix.hpp>
#include <grlina/dynamic_coordinate_matrix.hpp>
using CSC = graded_linalg::DynamicCoordinateGradedSparseMatrix<
    double, int, graded_linalg::CSCStorage<int>>;
```

Sorting, degree queries, row and column deletion, appending, restrictions,
transpose, multiplication, shifts, and local pair cancellation are supported.
As in the legacy API, a transpose swaps the degree lists; those swapped degrees
can violate the original product order, in which case `validate()` rejects it.
`shift(degree)` subtracts the supplied coordinates, following the existing
matrix convention. `Module<Matrix>` and `ChainComplex<Matrix>` support the new
type, including SCC I/O. The SCC header records the runtime dimension even
for a map with no rows or columns. Integer coordinate types use the existing
`Z` suffix in the poset identifier.

`semi_minimize()` cancels equal-degree generator/relation pairs.
`ChainComplex<Matrix>::minimize()` transports those cancellations to adjacent
differentials. `graded_kernel()`, full presentation minimization, and automatic
projective resolutions are supported for any runtime dimension. Direct-coordinate
kernels use an exact ordinal grid embedding, preserving the scalar coordinates
without rounding or narrowing. The source matrix is preserved.

## NEW: runtime grid matrices

`<grlina/dynamic_grid_matrix.hpp>` adds
`DynamicGridGradedSparseMatrix<Scalar, Index, Storage>`. It uses the same
runtime coordinate storage and algorithms, with `FlatDegreeTable<Index>` for
both degree tables. `Scalar` belongs to the grid and to real queries:

```cpp
#include <grlina/dynamic_grid_matrix.hpp>
using GridMatrix = graded_linalg::DynamicGridGradedSparseMatrix<
    double, int, graded_linalg::CSCStorage<int>>;
GridMatrix matrix(2, 1, 3, {{0}, {0}},
                  {{0.5, 4.25, 1}, {2.75, 1.5, 1}}, {{-1, 0, 0}});
auto stored = matrix.col_degree(0).to_degree(); // integer grid indices
// grid(axis)[stored[axis]] is the actual coordinate.
auto real = matrix.real_col_degree(0); // {0.5, 4.25, 1}
auto dimension = matrix.dim_at(GridMatrix::real_degree_type{1, 2, 1});
```

`grids` holds one strictly increasing scalar vector per parameter.
`set_real_degrees(columns, rows)` constructs grids and index degrees together.
`grid_degree(real_degree)` requires exact grid points; `query_degree(real_degree)`
finds the preceding point on each axis and uses `-1` below the first point.
Typed `degree_type` arguments query integer grid coordinates, while
`real_degree_type` arguments query geometry when these types differ.
`Module<GridMatrix>::degree_type` always denotes real geometry, including when
`Scalar` and `Index` are the same type.

`empty_like`, restrictions, products, submodules, kernels, and projective
resolutions preserve the embedding, including unused grid points. Products and
appends align independent grids by real coordinates. Real shifts translate the
grid; they never subtract offsets from the stored indices. SCC files contain
real degrees, so loading reconstructs the grid from the serialized coordinates.
Unused grid points have no SCC representation.

The grid type supports full minimization and automatic projective resolutions.
Two parameters use the existing `Grid_scheduler` kernel algorithm directly on
the flat tables. Higher dimensions freeze the first `d-2` coordinates at
combinations of distinct column-coordinate thresholds, then run that same
two-parameter algorithm on the remaining coordinates. Slice generators are
lifted to the original domain and full degrees; an eligible linear-span check
removes redundant generators. Zero and one parameter use the two-parameter
routine with zero padding.

The returned matrix gives homogeneous kernel generators. In dimensions three
and above the kernel need not be free, so its generators can have relations;
repeated `graded_kernel()` calls compute those next differentials. Following
the legacy R2 API, the two-parameter vector-storage routine uses its source as
reduction scratch space and may leave entries inconsistent with the original
degrees. Call it on a copy when the source must be reused. Other dimensions and
CSC inputs preserve their source.

Further additions are marked `NEW` in their headers:

- `dynamic_coordinate_matrix_base.hpp`: shared coordinate storage and result factories.
- `dynamic_grid_matrix.hpp`: the public grid matrix and geometry operations.
- `dynamic_grid_kernel.hpp`: grid-aware kernels.
- `sliced_coordinate_kernel.hpp`: arbitrary-dimensional slice construction and redundancy filtering.
- `dynamic_coordinate_kernel.hpp`: direct-coordinate kernels through an exact grid embedding.
- `matrix_geometry.hpp`: real degrees at module and homomorphism boundaries.
- `runtime_hom_spaces.hpp`: shared legacy/runtime hom-space implementations.
- `runtime_matrix_io.hpp`: runtime SCC presentation and sum loading.

AIDA's default matrix now uses the runtime grid type with vector-of-columns
storage. AIDA, Skyscraper, and Stable accept CSC inputs by copying them into
editable vector-of-columns storage at the algorithm boundary. Their iterations
and repeated kernels run on that editable copy; CSC results are restored in
one batch where the API returns the input storage type. Skyscraper explicitly
requires two parameters for its geometric invariant. Stable pruning remains
generic over `Module<Matrix>`.

AIDA also accepts other scalar and index types with `std::vector<Matrix>` as
its summand output. Its adapter uses exact grid ordinals and restores the
original scalar grid, so it preserves `long double` coordinates without
narrowing them to `double`. Ranks and grid-axis sizes must fit AIDA's existing
`int` core.

Both runtime matrix classes provide `editable_copy()` and
`to_storage<TargetStorage>()`. They preserve the parameter count, degree tables,
sorting metadata, and (for the grid class) every grid point:

```cpp
auto editable = matrix.editable_copy(); // vec<vec<Index>> entries
auto packed = editable.to_storage<graded_linalg::CSCStorage<int>>();
```

For `d >= 3`, the slice count is the product of the threshold counts on the
first `d-2` axes, at most `n^(d-2)` for `n` columns. With 20 thresholds on each axis in
five dimensions, that means `20^3 = 8,000` two-parameter kernel calls. This
baseline reuses the proven two-axis routine; high-dimensional workloads with
many thresholds can still be expensive. It uses exact column thresholds, so
unused grid points do not add slices and real coordinates are never sampled.

For example, equal columns born at `(1,0,0)` and `(0,1,0)` give a kernel
generator at `(1,1,0)`, their coordinatewise maximum. The independent test in
`general_coordinate_kernel_test.cpp` verifies fibre rank/nullity and spanning
in dimensions zero through four, including successive 3D and 4D kernels.

## Packed exchange and CMake integration (NEW)

Include `grlina/packed_matrix.hpp` for
`from_packed_csc<Matrix>(columns, rows, parameters, offsets, entries,
column_coordinates, row_coordinates, grids)` and `to_packed_csc(matrix)`.
Construction retains the supplied bases. The returned packed matrix owns its
CSC buffers (`data.offsets()`, `data.entries()`) and contiguous degree buffers
(`col_degrees.coordinates()`, `row_degrees.coordinates()`), flattened degree
by degree. Grid matrices store index coordinates and their complete scalar
`grids`; direct matrices store scalar coordinates and take no grids. Offsets
use `size_t`, entries use the matrix's signed `Index`, and coefficients remain
over F2. Empty matrices retain the explicit parameter count. Buffer references
remain valid only while their owner lives and the corresponding storage is unchanged.

For grid matrices, `graded_kernel()` on a mutable object retains its destructive
behavior. The const overload copies and invokes that same algorithm. Direct
coordinate kernels already use a const ordinal-grid adapter. No alternate
kernel algorithm is introduced by these overloads.

With `add_subdirectory(Persistence-Algebra)`, link `grlina::grlina`. Only the
header-only library target is enabled by default when embedded. Opt into
`GRLINA_BUILD_PROGRAMS` and `GRLINA_BUILD_TESTS` if needed. C++17 is a target
requirement; a parent project's newer standard and compiler flags are retained.
Installed consumers can use `find_package(grlina CONFIG REQUIRED)` and the same
target. The package declares Boost.Timer and, when enabled, OpenMP dependencies.
