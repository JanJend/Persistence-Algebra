# Persistence-module framework

This document describes the compatibility-preserving module layer introduced
on the `module-framework` branch. Presentation matrices remain public and all
pre-existing matrix algorithms remain available. New code should own modules
and cross the matrix boundary only when calling a legacy algorithm.

## Public types

All types live in `graded_linalg`.

- `ChainComplex<Matrix>` stores `d1, d2, ...` in that order. It validates matrix
  dimensions, degrees, gradedness, adjacent chain groups, and can check
  `d_i d_{i+1} = 0` with `is_chain_complex()`.
- `Module<Matrix>` (`PersistenceModule` is a compatibility alias) owns optional projective
  and injective chain complexes. `R2Module<index>` is the standard alias.
  The aggregate header also provides `R3Module`, `Z2Module`, `Z3Module`,
  `R4Module`, and `Z4Module` aliases.
- `Submodule<Matrix>` publicly inherits `Module<Matrix>` and owns a non-null `shared_ptr` to its parent module and a
  generator-coordinate matrix. Its row degrees must exactly equal the row
  degrees of the parent's presentation.
- `Homomorphism<Matrix>` (`ModuleMorphism` and `ModuleFunction` are compatibility aliases) owns pointers to its
  domain and target and stores the lifts to their projective resolutions.
- `module_hom_space_basis` and `module_endomorphism_basis` adapt the established
  Hom algorithms and return typed module homomorphisms.

The implementation is split across `chain_complex.hpp`, `module.hpp`,
`submodule.hpp`, `homomorphism.hpp`, and `module_homomorphisms.hpp`.
Including `grlina/modules.hpp` loads the whole public layer.
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
resolution. A module created from a chain complex trusts the caller's exactness,
as requested, but still validates structural compatibility. Calling
`mutable_presentation()` discards higher projective differentials first, because
an arbitrary edit would make them stale. `minimize_presentation()` also discards
higher projective maps; standard `minimize()` instead minimizes the stored
resolution when it has multiple maps. `compute_projective_resolution()` completes
the stored sequence by repeated kernels until the terminal map is injective,
for matrix types whose `graded_kernel()` returns that same matrix type. Existing
bases are preserved. A uniform `shift()` is
applied to every stored projective and injective differential and therefore
preserves the resolutions.

`GradedSparseMatrix` has `compatibly_sorted`, false when no ordering has been
certified. Calling `sort_compatibly()` sorts rows and columns with the mandatory
linear extension from `Degree_traits`; the comparator overload supports any
other compatible linear extension. The matrix retains the certifying
comparator, so sorted-input algorithms re-check the actual degree vectors and
detect a stale flag caused by legacy direct writes. Explicit-degree
constructors and SCC readers check their input; chain-complex readers accept a
`sort_if_needed` argument. Appending, arbitrary permutation, and degree edits
invalidate the certificate. `minimize`, `minimize_variant`, `semi_minimize`,
and graded column reduction throw `std::invalid_argument` when their required
certificate is absent or stale. A module's `minimize()` sorts by default;
`minimize(false)` selects strict rejection instead.

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
first invokes that operation, then minimizes the terminal generating set via its
kernel. That extra step is valid for a truncated resolution of a module, but can
change its terminal homology, so it is not part of chain-complex minimization.

## Euler-characteristic Hilbert queries

`dimension_at` uses the alternating sum of free chain-group generators born at
or below the queried degree when `has_complete_projective_resolution()` is true.
Otherwise a point query retains the local-presentation method and does not
compute a resolution just for that point.

Completeness is distinct from merely storing several differentials. Use
`ResolutionCompleteness::complete` when supplying a known full projective
resolution, or `set_projective_resolution_completeness` after reading one.
The guarantee is trusted, like exactness. Unmarked sequences are treated as
truncated unless their terminal free group is explicitly zero. SCC itself has
no completeness marker. Computed resolutions are marked complete; presentation
edits/replacement invalidate the guarantee. Sorting, shifts and full-resolution
minimization preserve it.

R2 `hilbert_function_on_induced_grid()` and `hilbert_function_on_grid(xs, ys)`
automatically complete a missing/truncated resolution when a graded kernel is
available, then use a signed birth histogram and two-dimensional prefix sums.
The axes of an explicit grid must be sorted and unique. For B total free
summands and an X-by-Y grid this takes O(B(log X + log Y) + XY), excluding the
one-time resolution computation. Arbitrary R2 location lists use an incremental
x-sweep and y-prefix counts when the resolution is complete; output order and
repeated locations are preserved. Generic point queries use the poset comparison.

Mutable grid queries retain the computed resolution for reuse. Const grid queries
compute on a private copy, so querying a shared const module does not invalidate
references to its bases. Without a graded kernel, incomplete modules fall back to
local presentation evaluations. A supplied complete resolution still enables Euler
queries even when that matrix type cannot compute its own kernel. Cartesian-grid
helpers remain R2-specific; generic point/list queries work with other degree traits.

## SCC I/O

The second SCC line is written from `Degree_traits<D>::poset_id`. Current IDs
are:

| Degree/matrix | Poset ID |
| --- | --- |
| `r2degree`, `R2GradedSparseMatrix` | `2` |
| `triple`, `R3GradedSparseMatrix` | `3` |
| `z2degree`, `Z2GradedSparseMatrix` | `2Z` |
| `z3degree`, `Z3GradedSparseMatrix` | `3Z` |
| `r4degree`, `R4GradedSparseMatrix` | `4` |
| `z4degree`, `Z4GradedSparseMatrix` | `4Z` |

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
using Matrix = graded_linalg::R2GradedSparseMatrix<int>;
using Module = graded_linalg::R2Module<int>;

Module module("input.scc");
module.minimize();
int dimension = module.dimension_at({0.5, 1.0});
auto hilbert = module.hilbert_function_on_induced_grid();
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

No separate module object is needed for a submodule's own presentation/resolution:

```cpp
auto image = f.image();
image.compute_presentation();         // stores its own d1 in the Module base
image.compute_projective_resolution(); // completes/stores its own resolution
auto dimension = image.dimension_at({0.5, 1.0});
image.to_file("image-resolution.scc");
```

`compute_projective_resolution()` also obtains a missing presentation automatically,
including through a `Module<Matrix>&`. `Module::compute_presentation` is a virtual
construction hook; `Module` has a virtual destructor and defaulted copy/move
operations. Existing binaries using these header types should be rebuilt.

By default, `Submodule::compute_presentation(false)` keeps its generator basis:
the stored presentation's rows correspond exactly to `generators()`' columns.
`compute_presentation(true)` minimizes the stored module, without replacing the
defining family in parent coordinates. Consequently that family is not necessarily
the generator lift from a subsequently minimized/sorted presentation. Use
`number_of_embedding_generators()` for its size; after a presentation is stored,
`number_of_generators()` reports that presentation's size, consistently with Module.
The categorical adapter reconstructs the defining basis before building its inclusion.

Generator reductions invalidate the submodule's own projective/injective storage;
they do not invalidate or modify the parent's storage. A later computation rebuilds
the submodule presentation. Explicit `compute_presentation` recomputes from the
defining family and replaces older stored resolutions. The compatibility method
`presented_module()` now stores the result on a mutable submodule and still returns
a standalone copy; its const overload computes on a temporary without modifying it.

As with parents of existing homomorphisms, do not use arbitrary inherited module
edits to change the represented module while retaining a fixed embedding. In
particular, shifting/editing the stored module alone does not shift/edit its parent
or defining generator matrix. This update supports computing and processing the
same module's representations, not automatic transport of embeddings under such edits.

## Migrated clients

Stable-Decomposition's reusable helpers are exposed in
`grlina/presentation_operations.hpp` (included by `modules.hpp`) and
`grlina/matrix_family.hpp`. In particular:

- `S.contains(T)`, `S.is_contained_in(T)`, and `S.equals(T)` test exact submodule
  membership modulo their common parent's relations, without requiring a kernel.
- `Homomorphism<Matrix>::canonical_shift(M, amount)` builds the structure map
  with identity lifts on all stored projective groups. `f.shifted(amount)`
  translates both endpoint modules and every stored lift.
- `f.image(S)` computes the image of a submodule of its domain.
- Matrix-level adapters cover zero/whole/sum/reduction, canonical shift lifts,
  free-target image inclusion and equality in a presented parent. They preserve
  ambient row coordinates and validate grading; containment needs no sorting.
- `homomorphism_lift_basis` and `shifted_endomorphism_lift_complement` intentionally
  operate on spaces of generator lifts. Use `module_hom_space_basis` when maps
  differing by target relations should be identified. `reduce_matrix_family_modulo`
  is coefficient-vector linear algebra, not a categorical Hom quotient.

These additions are tested with handmade examples in
`tests/presentation_operations_test.cpp`. The generic timing utility now lives
in `grlina/progress.hpp`; `general.hpp` retains its historical global name.

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

- The verified graded-kernel/projective-resolution computation remains R²-only.
  R³, Z², Z³, R⁴, and Z⁴ provide degree arithmetic, product order, lex/colex sort,
  graded matrix operations that do not require syzygies, and SCC I/O. No
  unverified higher-dimensional kernel algorithm was added.
- Injective resolutions can be stored, read, written, sorted, and replaced, but
  no injective-resolution algorithm existed to wrap.
- Exactness of supplied resolutions and the homomorphism equations for manual
  lifts are trusted. Structural dimensions, degrees and gradedness are checked.
  Optional `lift_to_relations` and `Homomorphism::check_lifts` validate equations.
- Hom adapters provide generator lifts; `lift_to_resolution()` can extend them
  through the common available projective resolution using graded linear systems.

## Verification

`tests/modules_test.cpp` exercises CRTP enforcement, sorting certificates and
stale-flag detection, correct unit cancellation and redundant-relation
deletion, strict SCC round trips, real fixture loading, chain
validation, module Hilbert functions, resolution invalidation/recomputation,
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
resolution cancellation, sorting and optional homomorphism validation.
