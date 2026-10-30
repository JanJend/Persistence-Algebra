# Persistence-module framework

This document describes the compatibility-preserving module layer introduced
on the `module-framework` branch. Presentation matrices remain public and all
pre-existing matrix algorithms remain available. New code should own modules
and cross the matrix boundary only when calling a legacy algorithm.

## Public types

All types live in `graded_linalg`.

- `ChainComplex<Matrix>` stores `d1, d2, ...` in that order and trusts supplied
  matrices and adjacent bases. `is_chain_complex()` explicitly computes
  `d_i d_{i+1}` and tests that each product is zero.
- `Module<Matrix>` (`PersistenceModule` is a compatibility alias) owns optional projective
  and injective chain complexes. `R2Module<index>` is the standard alias.
  For other parameter counts and coordinate domains, use
  `Module<CoordGradedSparseMatrix<Scalar, Index>>` or
  `Module<GridGradedSparseMatrix<Scalar, Index>>` with the corresponding runtime
  matrix header.
- `Submodule<Matrix>` publicly inherits `Module<Matrix>` and stores its embedding
  as a `Homomorphism<Matrix>` named `generator_map_`. `generator_map()` exposes
  the inclusion and `parent()` returns its target. The generator lift's row
  degrees must exactly equal the parent's presentation row degrees.
- `Homomorphism<Matrix>` (`ModuleMorphism` and `ModuleFunction` are compatibility aliases) owns pointers to its
  domain and target. `resolution_maps()` returns the stored components of a
  possibly partial chain map between their projective resolutions: entry 0 maps
  generators, entry 1 maps relations, and later entries map higher syzygies.
  This accessor computes nothing; `lift_to_resolution()` extends the stored map.
- `module_hom_space_basis` and `module_endomorphism_basis` adapt the established
  Hom algorithms and return typed module homomorphisms.

The implementation is split across `proj_chain_complex.hpp`, `module/module_base.hpp`,
`module/submodule.hpp`, `hom_core_class.hpp`, and `hom_interface.hpp`.
Including `grlina/graded_linalg.hpp` loads the whole public layer.
See [the correction/review guide](module-framework-review.md) for the latest
algorithm contracts, categorical operations and a suggested walkthrough.

## Conventions and invariants

Every matrix type accepted by `ChainComplex`, `PersistenceModule`, `Submodule`,
or `ModuleMorphism` is checked at compile time to inherit
`GradedSparseMatrix<D, index, Matrix>` via CRTP. The inherited constructions
therefore return the concrete `Matrix` type and dispatch genuinely
poset-specific steps through it (for example, `graded_kernel()`).

`ChainComplex<Matrix>::differential(1)` is the presentation
`d1 : F1 -> F0`. Consequently, its column degrees are relations and its row
degrees are module generators. Higher differentials follow the same convention.

A module created from a presentation has a one-differential projective
resolution. A module created from a chain complex trusts its structure and the
caller's exactness. The stored chains `proj_res_` and
`inj_res_`, and the completeness flag `res_complete_`, are public.
Direct edits through `presentation()` leave flag and higher-map maintenance to
the caller. Generator/relation edits and support bounding discard stale models.
Grid snapping and cutting instead apply to every stored differential.
`minimize_presentation()` also discards higher projective maps; standard `minimize()` instead minimizes the stored
resolution when it has multiple maps. `compute_projective_resolution()` completes
the stored sequence by repeated kernels until the terminal map is injective,
for matrix types whose `graded_kernel()` returns that same matrix type. Existing
bases are preserved. A uniform `shift()` is
applied to every stored projective and injective differential and therefore
preserves the resolutions.

`GradedSparseMatrix` retains the boolean `compatibly_sorted` flag.
`sort_compatibly()` sorts both degree lists by the linear extension supplied by
`Degree_traits` and sets the flag. No comparator or sortedness validator runs.
Explicit-degree constructors and unsorted readers leave the flag false.
Partial sorts conservatively leave it false unless the untouched axis has at
most one degree. Appends and arbitrary permutations invalidate it; copies,
transposes, deletions and uniform coordinate shifts preserve known ordering.
Callers who directly edit public degrees must maintain the flag themselves.
Set `compatibly_sorted = true` to record a caller guarantee without scanning.
Algorithms requiring sorted input check only the flag in diagnostic builds.
Module `minimize()` sorts by default; `minimize(false)` uses the supplied order.

`general_checks.hpp` selects diagnostic checks automatically: enabled without optimization,
disabled when GCC/Clang define `__OPTIMIZE__` (including `-Og`) or when `NDEBUG`
is defined. `-DGRLINA_ENABLE_CHECKS=1` or `=0` overrides this. Keep the setting
consistent across translation units. On compilers without `__OPTIMIZE__`, use
`NDEBUG` or the explicit setting. File-token parsing and I/O errors remain active
in every mode. Matrix, chain, Hom, submodule and grid validation methods have
been removed. See [the validation audit](validation-audit.md).

Minimization first cancels equal-degree generator/relation pairs, clearing the
entire pivot row by column operations before deletion. It then uses the concrete
graded kernel to remove redundant relations. Ordinary graded column reduction
alone is insufficient at incomparable grades. `minimize_variant()` performs that
cheap reduction as a preliminary optimization, then the standard algorithm.
`semi_minimize()` performs only local pair cancellations.

`Submodule::reduce_generators_lazy()` is a kernel-free preprocessing operation:
look up parent relations with the same pivot as a generator, add one only if
its degree is <= the generator degree, and continue while the pivot decreases.
Zero generators are removed together at the end. It changes representatives
modulo parent relations but neither the submodule nor its ambient row basis.
It does not reduce the parent relations or combine submodule generators with
one another, and it need not detect all redundant generators. No degree sorting
is required because every column addition explicitly checks admissibility.

`Submodule::minimize_generators()` runs this cheap pass by default before the
exact syzygy computation; `minimize_generators(false)` disables it. During exact
elimination, a pivot generator row is cleared in all other syzygies and its pivot
syzygy column is cleared. The now-unused row remains in place, so indices stay
fixed and no syzygy row/column compaction is needed. Redundant generator columns
are removed in one final batch. If preprocessing leaves no generators, the kernel
computation is skipped entirely.

`ChainComplex::minimize()` only cancels contractible equal-degree pairs, transporting
basis changes into both adjacent differentials. It preserves every homology module,
does not assume exactness and needs no graded kernel. `Module::minimize_resolution()`
applies that operation directly to a complete resolution. For a truncation, it
first appends the kernel of the last differential, minimizes the extended complex,
and drops the added differential. This exposes redundant terminal generators as
contractible pairs while preserving the represented module and exactness below
truncation. Extending a truncation requires a graded-kernel implementation.

## Hilbert functions and notebook images

The public operations are members of `Module`; their definitions and numerical
helpers live in `grlina/invariants/hilbert_function.hpp`. The module header also
includes these definitions, so existing numerical callers need no extra include.

```cpp
using namespace graded_linalg;
int dimension = module.dimension_at({0.5, 1.0});
auto dimensions = module.dimension_at(vec<r2degree>{{0,0}, {1,1}});
auto function = module.hilbert_function();
auto sampled = module.hilbert_function({0.0, 0.5, 2.0}, {0.0, 1.0});
int thickness = function.maximum;
```

`HilbertFunction<Scalar, Index>` owns `x_grid`, `y_grid`, `values` and `maximum`.
`values[x][y]` is the integer dimension at `(x_grid[x], y_grid[y])`; coordinates
retain the matrix's actual scalar type, including integer and float coordinates.
With no explicit axes, the function uses all presentation-induced coordinates.
For a grid matrix it uses the full stored coordinate grid, including unused
coordinates. Its maximum is the module's thickness. An explicitly sampled table's
maximum is only the maximum among its samples.

Axes must be finite and sorted; repeated coordinates are allowed. Empty axes are allowed for numerical
queries. Explicit grid queries compute the induced table once and resample by the
greatest coordinate less than or equal to each query, independently on each axis.
The dimension is zero when either query coordinate precedes its first induced
coordinate. `function.on_grid(xs, ys)` reuses an existing table without performing
algebra; when that table contains only coarse samples, omitted changes cannot be
recovered from it. Point-list queries preserve order and duplicates and return
only the integer dimensions. Use an explicit `vec<degree_type>` for an empty or
single-element batch to distinguish it from a single degree.

Numerical queries never compute or modify a projective resolution. If
`res_complete_` is true, the dimension is the alternating sum of free generators
born at or below the degree. On a two-parameter grid, signed birth events and 2D
prefix sums take O(B(log X + log Y) + XY), for B free summands and X, Y axis
lengths. Complete two-parameter point-list queries use an x-sweep and Fenwick
prefix counts. Otherwise dimensions come from local presentations; generic
point/list queries also work for other posets and numbers of parameters.

Completeness is distinct from storing several differentials. Use
`ResolutionCompleteness::complete` when supplying a known complete exact
resolution. `compute_projective_resolution()` establishes completeness explicitly;
queries never infer it from a terminal zero group. SCC has no completeness
marker. Generator/relation edits and support bounding invalidate completeness;
sorting, shifts and full-resolution minimization preserve it. Direct matrix edits
require updating the flag yourself.

For PNG output, define `STB_IMAGE_WRITE_IMPLEMENTATION` **before any grlina
include** in one translation unit. Numerical queries do not require an encoder
implementation. Images return bytes and create no file:

```cpp
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <grlina/invariants/hilbert_function.hpp>

// Width and height count heatmap pixels; labels add 265 x 185 pixels.
auto png = module.hilbert_function_image(800, 600); // 5% padding by default
auto cropped = module.hilbert_function_image({0.0, 2.0}, {0.0, 1.0}, 800, 600);
graded_linalg::save_png(png, "hilbert.png"); // only this explicitly creates a file

// Reuse numerical work for further images or resolutions.
auto function = module.hilbert_function();
auto another_png = graded_linalg::hilbert_function_image(function, 1000, 800);
```

Automatic viewing bounds are the induced/stored grid's bounds, padded on both
sides. A constant axis uses a total reference span of 2, giving a margin of 0.1
with default padding. Explicit axes restrict the picture to their endpoints,
without padding. Images always sample the full induced table at equidistant
pixels, preserving changes omitted by the supplied plotting axes. A singleton
axis displays its one coordinate. Free and zero modules are supported. Width
and height must each be at least two. The standalone renderer uses its numerical
table's bounds without padding; optional shared positive minimum/maximum
arguments provide comparable logarithmic colour scales.

The `hilbert_image` executable connects this API to a notebook without Python
bindings or temporary PNG files. It explicitly completes the resolution for
fast Euler evaluation. Build it with programs enabled:

```sh
cmake -S Persistence-Algebra -B build -DGRLINA_BUILD_PROGRAMS=ON
cmake --build build --target hilbert_image
```

```python
import subprocess
from IPython.display import Image, display

png = subprocess.run(
    ["./build/hilbert_image", "input.scc", "800", "600"],
    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
).stdout
display(Image(data=png))

# Only when a separate image file is wanted:
# from pathlib import Path
# Path("hilbert.png").write_bytes(png)
```

Debug builds name the executable `hilbert_image_debug`. Diagnostics use standard
error; standard output contains only the PNG bytes.

Append `--bounds xmin ymin xmax ymax` to select a plotting window, and
`--scale min max` to share a positive logarithmic colour range between images.
Both options can be combined with the requested heatmap width and height.

Landscapes offer the same in-memory PNG output, with a 1-based layer selector:

```cpp
// Define STB_IMAGE_WRITE_IMPLEMENTATION before any grlina include in one .cpp.
#include <grlina/invariants/landscape.hpp>
graded_linalg::Landscape landscape(module);
auto png = landscape.image(1, 800, 600); // layer, heatmap width, heatmap height
auto inferno = landscape.image(1, 800, 600, graded_linalg::ColourMap::inferno);
// graded_linalg::save_png(png, "landscape.png"); // only when explicitly wanted
```

This evaluates each output pixel on its own line, sharing a barcode between
points on the same line and reusing cached lines. Each sequence of additional
lines uses one initial reduction followed by vineyard updates; there is no interpolation between slices. Colours are
linear in height; the visible range is the landscape's existing bounds. The returned bytes
work with the same Python `Image(data=png_bytes)` call. The default palette matches
the Hilbert plots: white at zero, then pale blue through muted blue to black.
`ColourMap::inferno` uses Matplotlib's 256 colours, from near-black through red
and yellow to pale yellow. `landscape.save(prefix, layers, ColourMap::inferno)`
selects the same palette for SVG; CSV values do not depend on the palette.

Build `landscape_image` alongside `hilbert_image` to use this from Python:

```python
png = subprocess.run(
    ["./build/landscape_image", "input.scc", "500", "500", "--layer", "1"],
    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
).stdout
display(Image(data=png))
```

Optional `--bounds xmin ymin xmax ymax`, `--direction vx vy`, and
`--resolution relative_spacing` select the landscape's window, positive direction,
and relative cache spacing (default 0.01). Heatmap dimensions set the grid of
direct evaluations separately. The executable constructs the landscape, calls `image`,
and returns PNG bytes on standard output without creating files.
Append `--colormap inferno` for inferno, or `--colormap blue` for the default.
Use `--json` to receive the sampled `maximum` alongside a base64 `png`, and
`--include-values` to also include `values[x][y]`. For a shared scale, first call
`--values-only` once per module. After finding the common maximum, feed each
returned JSON grid to `landscape_image --render-values --scale MAX` on stdin.
That path only renders the retained values; it never recomputes persistence.
Every image then uses the same linear value-to-colour map on `[0, MAX]`, including
images whose landscape is zero. In C++, pass MAX as the final argument to
`landscape.image(layer, width, height, colormap, MAX)`; zero retains automatic scaling.
`landscape.image_with_data(...)` returns PNG bytes and the sampled maximum together,
with an optional final `include_values` boolean. Alternatively retain
`landscape.sample_grid(layer, width, height)` and call its `image(colormap, MAX)`.
Vineyard updates are enabled by default; `--no-vineyard` uses independent reductions.

## SCC I/O

The second SCC line is written from `Degree_traits<D>::poset_id`. Current IDs
are:

| Degree/matrix | Poset ID |
| --- | --- |
| `r2degree`, `R2GradedSparseMatrix` | `2` |
| `CoordinateDegree<double, N>` / runtime matrices with real scalar coordinates | `N` |
| `CoordinateDegree<Integer, N>` / runtime matrices with integer scalar coordinates | `NZ` |

Custom degree traits may use any stable string. The chain dimensions on line
three determine the number of differentials. A one-matrix presentation keeps
the historical trailing zero (`relations generators 0`). The poset ID is strict:
there is no chain-length heuristic. `0 0 0` means a zero module represented by
one 0-by-0 presentation, not an absent presentation.

`ChainComplex::to_stream`, `from_stream`, `to_file`, and `from_file` are the
canonical generic SCC operations. Older matrix SCC readers also check the ID.

`homology_module(complex, k)` computes a presentation of
`ker(d_k) / im(d_(k+1))`: incoming boundary columns are lifted degree by degree
to coordinates in the concrete matrix type's graded-kernel basis. This is the
path used by `mpfree_clone`.

## Typical use

```cpp
#include <grlina/invariants/hilbert_function.hpp>

using Matrix = graded_linalg::R2GradedSparseMatrix<int>;
using Module = graded_linalg::R2Module<int>;

Module module("input.scc");
module.minimize();
int dimension = module.dimension_at({0.5, 1.0});
auto hilbert = module.hilbert_function();
module.compute_projective_resolution();
module.to_file("resolution.scc");
```

Submodules and maps use shared module ownership so their ambient objects cannot
silently disappear:

```cpp
auto source = std::make_shared<Module>("source.scc");
auto target = std::make_shared<Module>("target.scc");
Matrix lift = /* generators(source) -> generators(target) */;

graded_linalg::Homomorphism<Matrix> f(source, target, std::move(lift));
auto image = f.image();
auto kernel = f.kernel();
Module image_as_module = image.presented_module();
```

`module->whole_submodule()` and `module->zero_submodule()` return the canonical
submodules with that exact module as their shared parent. Both methods are const,
require a stored presentation, and require the module to be owned by a
`std::shared_ptr`; calling them on a stack-allocated module throws
`std::bad_weak_ptr`. Include `grlina/module/submodule.hpp` or `grlina/graded_linalg.hpp` to use
them. The returned submodules keep their parent alive.

No separate module object is needed for a submodule's own presentation/resolution:

```cpp
auto image = f.image();
image.compute_presentation();         // stores its own d1 in the Module base
image.compute_projective_resolution(); // completes/stores its own resolution
auto dimension = image.dimension_at({0.5, 1.0});
image.to_file("image-resolution.scc");
```

`Submodule::compute_projective_resolution()` also obtains a missing presentation
automatically, including through a `Module<Matrix>&`. `compute_presentation()`
belongs only to `Submodule`; resolution computation dispatches virtually.
`Module` has a virtual destructor and defaulted copy/move operations. Existing
binaries using these header types should be rebuilt.

By default, `Submodule::compute_presentation(false)` keeps its generator basis:
the stored presentation's rows correspond exactly to `generator_map().generator_lift()`'s columns.
`compute_presentation(true)` removes redundant inclusion generators and stores the
presentation in that same basis. The domain of `generator_map()` is this object's
`Module` base, whether or not a presentation exists. `domain()` never computes it.
After computation, `number_of_embedding_generators()` and `num_of_gens()`
agree. Normal minimization through a `Module&` dispatches to the submodule's
presentation/resolution minimizer, preserving the inclusion basis.

Generator reductions invalidate the submodule's projective storage, preserving its
injective resolution; they do not modify the parent. Explicit `compute_presentation`
computes the kernel of the combined generators and parent relations, then stores
the projected relations in the base. There is no second presentation cache.
`presented_module()` explicitly computes and returns a standalone copy; its const
overload performs this work on a temporary submodule.

As with parents of existing homomorphisms, do not use arbitrary inherited module
edits to change the represented module while retaining a fixed embedding. In
particular, shifting/editing the stored module alone does not shift/edit its parent
or defining generator matrix. This update supports computing and processing the
same module's representations, not automatic transport of embeddings under such edits.

## Migrated clients

Stable-Decomposition's reusable helpers are exposed in
`grlina/module/submodule.hpp`, `grlina/hom_interface.hpp`, and
`grlina/matrices/base.hpp`, all included by
`grlina/graded_linalg.hpp`. In particular:

- `S.contains(T)`, `S.is_contained_in(T)`, and `S.equals(T)` test exact submodule
  membership modulo their common parent's relations, without requiring a kernel.
  `S.contains(T)` means that S contains T, including equality. It first checks
  parent identity in every build, then tests whether `[parent relations, S] X = T`
  has a graded solution. Equality calls containment in both directions, also
  through the `present_same_submodule` matrix adapter.
- `Homomorphism<Matrix>::canonical_shift(M, amount)` builds the structure map
  with identity lifts on all stored projective groups. `f.shifted(amount)`
  translates both endpoint modules and every stored lift.
- `f.image(S)` computes the image of a submodule of its domain.
- Matrix-level adapters cover zero/whole/sum/reduction, canonical shift lifts,
  and equality in a presented parent. They preserve
  ambient row coordinates; containment needs no sorting.
- `homomorphism_lift_basis` and `End_2d_0` intentionally
  operate on spaces of generator lifts. Use `module_hom_space_basis` when maps
  differing by target relations should be identified. `reduce_matrix_family_modulo`
  is coefficient-vector linear algebra, not a categorical Hom quotient.

These additions are tested with handmade examples in
`tests/presentation_operations_test.cpp`. The generic timing utility now lives
in `grlina/general.hpp`, which also retains its historical global name.

- Persistence-Algebra command-line targets now construct modules for file I/O,
  minimization, resolutions, Hilbert/thickness calculations, Hom calculations,
  grid operations, and submodule presentations.
- AIDA's functor has a module overload. Stream and Multipers entry points own
  modules while the matrix overload and decomposition core remain compatible.
- Stable-Decomposition exposes module and owned-submodule pruning overloads;
  its executable uses them. Its relation-aware submodule reduction is also
  available as `Submodule::minimize_generators()`.
- Skyscraper-Invariant accepts module containers through a presentation adapter;
  its input and standalone executable paths now create modules.

## Deliberate current limits

- The new runtime direct-coordinate and grid matrices support graded kernels
  and projective resolutions in arbitrary dimensions. Three parameters use
  muphasa's signature-based Groebner reduction; dimensions above three use
  slices of the existing two-parameter routine. Higher-dimensional kernels need
  not be free; repeated kernels compute their relations. See
  [runtime coordinate matrices](dynamic-coordinate-matrices.md) for storage
  choices and slice costs. The legacy fixed-dimension R³, Z², Z³, R⁴, and Z⁴
  classes still provide only operations that do not require syzygies.
- Injective resolutions can be stored, read, written, sorted, and replaced, but
  no injective-resolution algorithm existed to wrap.
- Exactness of supplied resolutions is trusted. Homomorphism constructors trust
  endpoints, lift dimensions, degrees, gradedness and equations without scanning
  or refreshing sorting flags. `check_lifts()` remains an explicit equation
  query and is never called automatically. `lift_to_relations` can also check
  a supplied generator lift.
- Hom adapters provide generator lifts; `lift_to_resolution()` can extend them
  through the common available projective resolution using graded linear systems.

## Verification

`tests/modules_test.cpp` exercises CRTP enforcement, sorting flags and
caller-managed invalidation, correct unit cancellation and redundant-relation
deletion, strict SCC round trips, real fixture loading, chain
equations, module Hilbert functions, resolution invalidation/recomputation,
submodule reduction/presentation/quotients/intersections, morphism
image/kernel/Hom adapters, homology presentations, R³ colex sorting, and
R⁴/Z⁴ I/O.
It is registered as `module_framework_test` with CTest. The established dense,
sparse, graded-matrix, and Hom tests are also registered, so the new layer and
the compatibility API run together. `cli_programs_test` runs all 15 installed
Persistence-Algebra executables on small hand-computed fixtures and compares
their SCC/quiver output or exact mathematical invariants. The fixture with a
three-parameter header but two-coordinate degrees is now explicitly rejected;
its round-trip test corrects only an in-memory copy. The additional
`module_operations_test` covers categorical maps, exact syzygy minimization,
resolution cancellation, sorting and explicit homomorphism-equation queries.

## Presentation operations on modules

The following adapters reuse the graded-matrix algorithms. Generator/relation edits
and support bounding discard higher projective maps and the injective model. Presentation
minimization retains the injective model because the module is unchanged.
Queries leave the module unchanged.

| Matrix operation | Module API |
|---|---|
| `cull_columns(count, false)` | `quotient_by_tail_generators(count)` — retain the first `count` generators, kill the rest |
| `map_at_degree_pair` | `local_presentation_at(degree)` — local matrix and original row indices |
| `print_graded` | `print_presentation()` |
| `delete_columns` | `remove_relations(indices)` |
| `delete_rows` | `quotient_by_generators(indices)` |
| `print_degrees` | `print_degrees()` |
| `semi_minimize` | `semi_minimize_presentation()` |
| `is_minimal` | `is_presentation_minimal()` |
| `append_column` | `add_relation(coefficients, degree)` |
| `submodule_generated_at` | `submodule_generated_at(shared_parent, degree)` — free function returning an embedded `Submodule` |
| `induced_quiver_rep` | `to_quiver(vertices = {}, edges = {})` |
| `bounding_box` | `bounding_box()` |
| `get_equidistant_grid` | `get_equidistant_grid(n)` |
| `snap_to_grid` | `snap_to_grid(x_grid, y_grid)` or `snap_to_grid(axes)` for coordinate/grid matrices |
| `snap_to_equidistant_grid` | `snap_to_equidistant_grid(n, end_inclusive = false)` |
| `bound_support` | `bound_support(bound)` |
| `cut_above` | `cut_above(x_cutoff, y_cutoff)` or `cut_above(cutoffs)` for coordinate/grid matrices |

`bound_support(bound)` modifies only the presentation, discards higher projective
maps, clears the injective resolution, and resets `res_complete_`. Its new boundary
relations require recomputing the resolution with `compute_projective_resolution()`.

Grid snapping and cutting operate independently on every
projective and injective differential, retaining both vectors and `res_complete_`.
These matrix methods can change bases and dimensions independently; equidistant
snapping and cutting also use each matrix's own bounding box. Consequently,
retaining the maps and flag does not establish that the transformed sequence is
a resolution. The caller must maintain compatible adjacent bases and exactness.

The local matrix presents the fibre; the fibre itself is its cokernel.
Generator and relation indices refer to the current presentation. Removing
relations can enlarge the module; deleting generators takes a quotient.
The generated-at free function accepts `shared_ptr<const Module<Matrix>>`,
so the returned submodule retains the exact parent and its generator coordinates.

Quiver conversion defaults to the unique presentation degrees and their Hasse
edges. Explicit vertices with omitted edges must be lexicographically sorted;
explicit adjacency lists must follow the degree order. Supply one empty
adjacency list per vertex to request no edges.

Bounds and sampling grids currently use the R2 matrix implementation. They
describe presentation degrees, not the potentially unbounded module support.
Empty presentations have no bounds (an exception is thrown) and yield an empty
sampling grid; `n = 0` also yields an empty grid, and negative `n` is rejected.
Minimality uses the graded-kernel minimizer; partial minimization only cancels
local pairs and therefore also works without a graded kernel.

### Typed additional shifted lifts

Include `grlina/hom_interface.hpp` and call `End_2d_0(module, amount)` with a
`std::shared_ptr<Module<Matrix>>` or `std::shared_ptr<const Module<Matrix>>`.
It returns `std::vector<Homomorphism<Matrix>>`, retaining the original domain
and sharing one presentation-only target `M(amount)` across the results.
The computation preserves the matrix overload’s quotient of lift spaces; it
does not identify lifts modulo target relations. Result matrices are moved
into homomorphisms, and both Hom computations reuse one source row cache.
Both the matrix and module overloads are defined side by side in
`hom_interface.hpp`. Their algorithms are unchanged.

`hom_interface.hpp` provides `module_hom_space_basis` and
`module_endomorphism_basis`. Include `hom_interface.hpp` directly; the former
`homomorphism.hpp` and `module_homomorphisms.hpp` forwarding headers were removed.


### Identity lifts and composition

`Homomorphism::id_matrix()` reports a known identity coefficient matrix on
module generators; the endpoints and their degrees can still differ. Ordinary
constructors always leave it `false`; there is no public flag override.
`identity`, `canonical_shift`, the whole-submodule embedding and
`quotient_projection` (used by `as_quotient`) set it by construction.
No coefficient scan checks or discovers identity.
Shifting preserves the flag; addition conservatively clears it.

`f.compose(g)` means **apply f, then g**, so its matrix is normally `g * f`.
The former `then` spelling is retained as a compatibility wrapper. Composition
and `f.image(I, false)` copy coefficients and adjust degrees whenever an identity
lift makes multiplication unnecessary. Copies still cost time proportional to
the matrix storage; the shortcut avoids the general multiplication. Higher lifts
are tracked separately: extending a quotient map to relations does not mark
those newly computed lifts as identities.

### Quotients and parent minimization

For submodules `I` and `K` with the same parent `X`:

```cpp
auto image_in_quotient = I.submodule_quotient(K);              // no minimization
auto lazy = I.submodule_quotient(K, true);                    // no graded kernel
auto full = I.submodule_quotient(K, false, true);              // includes graded kernel
```

The result is a submodule of a newly constructed `X/K`, representing
`I/(I intersect K)` (in particular `I/K` if `K` is contained in `I`). There is
no containment check or presentation computation. By default, the parent
presentation is formed by appending K's generators to X's relations, and I's
generator matrix is copied directly: this is the identity projection shortcut
without constructing a projection object. Neither input nor its parent changes.
The flags are `(lazy_minimize = false, minimize = false)`; full minimization
wins if both are true.

`S.lazy_minimize_parent()` changes S in place: it builds a new parent, applies
one shared ambient row permutation, cancels equal-degree generator/relation
pairs while substituting into S's generator columns, and performs graded column
reduction with deletion on the remaining relations. It does not use a graded
kernel. `S.minimize_parent()` additionally removes redundant relations via the
graded kernel. Both preserve the defining generator column degrees and order,
including zero or redundant columns; neither computes S's own presentation.
Both clear S's cached projective representation, preserve its injective resolution,
and leave other objects sharing the old parent unchanged. Failed minimization leaves S unchanged.

`presentation().remove_redundant_relations()` performs the graded-kernel
relation-removal step while preserving ambient generator coordinates and their
order. Callers must discard higher projective maps and reset `res_complete_` if
they edit a stored resolution this way. `minimize_presentation()` uses this matrix
operation after local cancellation. To obtain the final module in pruning:

```cpp
auto quotient_submodule = I.submodule_quotient(K, true);
auto result = quotient_submodule.presented_module(true);
result.shift({-epsilon, -epsilon});
```


### Submodule generator maps

```cpp
Submodule<Mat> I(parent, generators);
const auto& inclusion = I.generator_map();
assert(inclusion.domain().get() == static_cast<const Module<Mat>*>(&I));
assert(!inclusion.domain()->has_presentation());
I.compute_presentation();                       // explicit kernel computation
assert(inclusion.domain()->has_presentation());
```

The homomorphism's domain is the submodule's own Module base, not a separately
materialized module. Accessing either endpoint or its lift never computes a
presentation. `presentation()` requires a supplied/computed presentation.
`check_lifts()` needs presentations; callers must provide them before querying
its equations.

The member inclusion holds a non-owning alias to its containing object, avoiding
an ownership cycle. Copy/move construction and assignment of a Submodule rebind
that alias to the destination object's base. Copies of the inclusion borrow the
original submodule and must not outlive it, or be used after its generator basis
changes. They no longer preserve snapshots across submodule destruction or mutation.
Ordinary homomorphisms still retain the shared endpoint owners supplied by callers.
Lift matrices are stored directly in a vector with ordinary value semantics.

Use `I.inclusion()` when exporting an inclusion that must outlive `I`. It
materializes a presentation without changing the inclusion's generator basis.
For a shared, presented submodule it retains that same endpoint; for a stack
object it owns a copy. The internal `generator_map()` remains a borrowed accessor.

Explicit minimization updates the inclusion generators and presentation together.
Sorting a Submodule directly, or invoking default sorting through a Module reference,
also permutes its inclusion columns. Arbitrary inherited edits that change the
represented module or its basis require corresponding embedding updates; use a
standalone Module copy for edits without a parent interpretation. Custom-comparator
sorting through a statically typed Module reference remains a base-class operation.

`as_subobject(I)` explicitly copies I into an owning Submodule, computes its
presentation, and returns that same object with an owning inclusion. This keeps
categorical results such as `kernel(f)` valid after local temporaries disappear;
it does not add a hidden domain object to I.

`hom_core_class.hpp` defines the map class before Submodule is complete;
`module/submodule.hpp` then defines Submodule with its member map. Include the latter when
using both types. The empty `hom_operations.hpp` forwarding header was removed.


Identity image/composition shortcuts retain the copied sorting metadata: their
factory-created identity lifts only preserve or uniformly translate degrees.
Trusted submodule construction does not refresh sorting metadata. Parent-row
permutations conservatively leave embedding sorting unknown unless established
by an explicit sorting operation; no automatic scan attempts to recertify it.


Presentation minimization, local cancellation and removal of redundant relations
preserve the module, so they retain its independently stored injective resolution.
Submodule parent minimization copies that resolution to the new parent as well.
Recomputing a submodule's presentation preserves its own injective model.
Arbitrary generator/relation edits and support bounding still invalidate it; a
quotient does not inherit the original module's injective resolution. Grid snapping
and cutting transform each stored injective map as described above.

### Optional cycle representatives and direct coordinate updates

`homology_module(C, q, minimize = true, keep_cycles = false)` returns a `Module`.
When requested, its `std::optional<Matrix> cycles` has one column per presentation
generator, expressed in the original `C_q` coordinates. Its image lies in the
cycle group; it is a lift from the free generators, not a map from homology to
cycles. An engaged optional with zero columns represents requested zero homology.
The construction includes relations among kernel generators, also for nonfree
kernels. Degree zero is supported, including an empty complex.

Sorting and minimization update stored cycles directly. Module methods accept
an optional incoming matrix (`elements`) and a list of outgoing matrices
(`images`). Incoming matrices have one row per module generator; outgoing
matrices have one column per generator. The existing matrix algorithms apply
coordinate changes to these matrices during reduction. No inverse matrix or
final matrix multiplication is needed. For example:

```cpp
M.minimize(true, &incoming, {&outgoing});
M.sort_compatibly(&incoming, {&outgoing});
```

Initialize `outgoing` to the identity on the old generator degrees to obtain
the new-to-old generator map; initialize `incoming` to identity for the
old-to-new map. Use distinct matrices with matching generator bases. These
arguments are optional and ignored by ordinary calls; no tracking matrices are
allocated unless the caller supplies them. Cycle storage is always updated
when present, including when other outgoing maps are supplied.

Generator deletion, support bounding, cutting and grid snapping carry the same
column changes into cycles. Snapping updates the cycle source degrees while
retaining the original ambient chain coordinates. Shifting shifts both degree
lists of the cycle matrix, interpreting its target as the shifted chain group.
Relation-only edits preserve the representatives, but arbitrary quotients or
relation edits no longer claim to compute homology of the original complex.
Direct mutation through `presentation()` or `proj_res_` bypasses module-level
cycle maintenance; pass accompanying matrices to the lower-level operations.

Submodules do not store separate cycle matrices. Their representatives are
`parent.cycles * inclusion.generator_lift()`. Parent minimization transports
both the parent cycles and the inclusion. New categorical constructions do not
automatically acquire cycle storage.

The removed runtime storage-conversion API is no longer available. Current grid
matrices use vector columns; R2 matrices can use `CSCStorage<Index>`.
Stable-Decomposition converts R2 CSC modules to editable columns at its pruning
boundary and packs the results back, preserving stored resolutions and completeness.
