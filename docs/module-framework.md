# Persistence-module framework

This document describes the compatibility-preserving module layer introduced
on the `module-framework` branch. Presentation matrices remain public and all
pre-existing matrix algorithms remain available. New code should own modules
and cross the matrix boundary only when calling a legacy algorithm.

## Public types

All types live in `graded_linalg`.

- `ChainComplex<Matrix>` stores `d1, d2, ...` in that order. Structural checks
  run automatically in diagnostic builds and explicitly via `validate_structure()`
  in every build. `is_chain_complex()` checks `d_i d_{i+1} = 0`.
- `Module<Matrix>` (`PersistenceModule` is a compatibility alias) owns optional projective
  and injective chain complexes. `R2Module<index>` is the standard alias.
  The aggregate header also provides `R3Module`, `Z2Module`, `Z3Module`,
  `R4Module`, and `Z4Module` aliases.
- `Submodule<Matrix>` publicly inherits `Module<Matrix>` and stores its embedding
  as a `Homomorphism<Matrix>` named `generator_map_`. `generator_map()` exposes
  the inclusion and `parent()` returns its target. The generator lift's row
  degrees must exactly equal the parent's presentation row degrees.
- `Homomorphism<Matrix>` (`ModuleMorphism` and `ModuleFunction` are compatibility aliases) owns pointers to its
  domain and target and stores the lifts to their projective resolutions.
- `module_hom_space_basis` and `module_endomorphism_basis` adapt the established
  Hom algorithms and return typed module homomorphisms.

The implementation is split across `chain_complex.hpp`, `module.hpp`,
`submodule.hpp`, `homomorphism_core.hpp`, and `hom_interface.hpp`.
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
with structural compatibility checked automatically only in diagnostic builds. Calling
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
comparator. Diagnostic builds re-check actual degree vectors and detect stale
flags caused by legacy direct writes. Optimized builds read cached flags in
constant time: callers that directly edit public degree vectors must call
`invalidate_compatible_sorting()` (or explicitly refresh/sort). Uniform shifts
assume a translation-invariant degree order, as supplied by the coordinate
lex/colex orders. Explicit-degree constructors and SCC readers establish initial
sorting metadata. Appending and arbitrary permutation invalidate it.
Sorted-input algorithms diagnose missing/stale certificates in diagnostic builds;
in optimized builds, sorted input is a caller precondition. A module's
`minimize()` still sorts by default; `minimize(false)` trusts the supplied order
in optimized builds.

`checks.hpp` selects diagnostic checks automatically: enabled without optimization,
disabled when GCC/Clang define `__OPTIMIZE__` (including `-Og`) or when `NDEBUG`
is defined. `-DGRLINA_ENABLE_CHECKS=1` or `=0` overrides this. Keep the setting
consistent across translation units. On compilers without `__OPTIMIZE__`, use
`NDEBUG` or the explicit setting. Explicit validators and file-input checks
remain active in every mode. See [the validation audit](validation-audit.md)
for the decisions by subsystem and the retained sorting scans.

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

`module->whole_submodule()` and `module->zero_submodule()` return the canonical
submodules with that exact module as their shared parent. Both methods are const,
require a stored presentation, and require the module to be owned by a
`std::shared_ptr`; calling them on a stack-allocated module throws
`std::bad_weak_ptr`. Include `grlina/submodule.hpp` or `grlina/modules.hpp` to use
them. The returned submodules keep their parent alive.

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
the stored presentation's rows correspond exactly to `generator_map().generator_lift()`'s columns.
`compute_presentation(true)` removes redundant inclusion generators and stores the
presentation in that same basis. The domain of `generator_map()` is this object's
`Module` base, whether or not a presentation exists. `domain()` never computes it.
After computation, `number_of_embedding_generators()` and `number_of_generators()`
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
`grlina/presentation_operations.hpp` (included by `modules.hpp`) and
`grlina/hom_interface.hpp` and `grlina/matrix_family.hpp`. In particular:

- `S.contains(T)`, `S.is_contained_in(T)`, and `S.equals(T)` test exact submodule
  membership modulo their common parent's relations, without requiring a kernel.
- `Homomorphism<Matrix>::canonical_shift(M, amount)` builds the structure map
  with identity lifts on all stored projective groups. `f.shifted(amount)`
  translates both endpoint modules and every stored lift.
- `f.image(S)` computes the image of a submodule of its domain.
- Matrix-level adapters cover zero/whole/sum/reduction, canonical shift lifts,
  free-target image inclusion and equality in a presented parent. They preserve
  ambient row coordinates; containment needs no sorting.
- `homomorphism_lift_basis` and `End_2d_0` intentionally
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
- Exactness of supplied resolutions is trusted. Homomorphism constructors trust
  endpoints, lift dimensions, degrees, gradedness and equations without scanning
  or refreshing sorting flags. For untrusted input, explicitly call
  `Homomorphism::validate()` to check structure, then `check_lifts()` to check
  equations. `lift_to_relations` can also check a supplied generator lift.
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

## Presentation operations on modules

The following adapters reuse the graded-matrix algorithms. Edits that change the
module discard higher projective maps and the injective model. Presentation
minimization retains the injective model because the module is unchanged.
Queries leave the module unchanged.

| Matrix operation | Module API |
|---|---|
| `cull_columns(count, false)` | `quotient_by_tail_generators(count)` — retain the first `count` generators, kill the rest |
| `map_at_degree_pair` | `local_presentation_at(degree)` — local matrix and original row indices |
| `map_at_degree` | `relations_at(degree, relation_indices)` — available relations, with all original rows |
| `print_graded` | `print_presentation()` |
| `delete_columns` | `remove_relations(indices)` |
| `delete_rows` | `quotient_by_generators(indices)` |
| `print_degrees` | `print_degrees()` |
| `semi_minimize` | `semi_minimize_presentation()` |
| `is_minimal` | `is_presentation_minimal()` |
| `append_column` | `add_relation(coefficients, degree)` |
| `submodule_generated_at` | `submodule_generated_at(shared_parent, degree)` — free function returning an embedded `Submodule` |
| `induced_quiver_rep` | `to_quiver(vertices = {}, edges = {})` |
| `bounding_box` | `presentation_degree_bounds()` |
| `get_equidistant_grid` | `equidistant_presentation_grid(n)` |

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
`module_endomorphism_basis`. The former header names `homomorphism.hpp` and
`module_homomorphisms.hpp` remain as compatibility includes.


### Identity lifts and composition

`Homomorphism::id_matrix()` reports a known identity coefficient matrix on
module generators; the endpoints and their degrees can still differ. Ordinary
constructors always leave it `false`; there is no public flag override.
`identity`, `canonical_shift`, the whole-submodule embedding and
`quotient_projection` (used by `as_quotient`) set it by construction.
No coefficient scan checks or discovers identity, including in `validate()`.
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

`Module::remove_extra_rels()` performs only the graded-kernel relation-removal
step. It preserves ambient generator coordinates, including their original
order, so existing submodule generator matrices remain valid. Higher projective
maps are discarded; the independent injective resolution is preserved. `minimize_presentation()` reuses this
step after local cancellation. To obtain the final module in pruning:

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
presentation. `presentation()` throws if none was supplied/computed. Homomorphism
`validate()` checks available endpoint bases without constructing missing ones;
`check_lifts()` needs presentations and throws if they are missing.

The member inclusion holds a non-owning alias to its containing object, avoiding
an ownership cycle. Copy/move construction and assignment of a Submodule rebind
that alias to the destination object's base. Copies of the inclusion borrow the
original submodule and must not outlive it, or be used after its generator basis
changes. They no longer preserve snapshots across submodule destruction or mutation.
Ordinary homomorphisms still retain the shared endpoint owners supplied by callers.
Lift matrices are stored directly in a vector with ordinary value semantics.

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

`homomorphism_core.hpp` defines the map class before Submodule is complete;
`submodule.hpp` then defines Submodule with its member map. Include the latter when
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
Arbitrary presentation edits and operations that change the module still invalidate
it; a quotient does not inherit the original module's injective resolution.
