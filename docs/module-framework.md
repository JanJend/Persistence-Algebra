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
- `PersistenceModule<Matrix>` (also `Module<Matrix>`) owns optional projective
  and injective chain complexes. `R2Module<index>` is the standard alias.
  The aggregate header also provides `R3Module`, `Z2Module`, `Z3Module`,
  `R4Module`, and `Z4Module` aliases.
- `Submodule<Matrix>` owns a non-null `shared_ptr` to its parent module and a
  generator-coordinate matrix. Its row degrees must exactly equal the row
  degrees of the parent's presentation.
- `ModuleMorphism<Matrix>` (also `ModuleFunction<Matrix>`) owns pointers to its
  domain and target and stores the lifts to their projective resolutions.
- `module_hom_space_basis` and `module_endomorphism_basis` adapt the established
  Hom algorithms and return typed module morphisms.

The implementation is split across `chain_complex.hpp`, `module.hpp`,
`submodule.hpp`, `module_morphism.hpp`, and `module_homomorphisms.hpp`.
Including `grlina/modules.hpp` loads the whole public layer.

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
an arbitrary edit would make them stale. `minimize()` has the same invalidation
rule. `compute_projective_resolution()` currently computes `d2` for matrix types
whose `graded_kernel()` returns that same matrix type. A uniform `shift()` is
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

Minimization now uses graded basis operations. It admissibly column-reduces and
removes zero relation columns, then finds unit entries where a relation and a
generator have equal degree. Before deleting such a row/column pair it adds the
pivot relation to every other relation containing that generator, making the
full pivot row zero outside the pivot column. The three historical minimizer
names share this correctness-first implementation.

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
the historical trailing zero (`relations generators 0`). The reader also
accepts legacy SCC files where line two was the number of nonzero chain groups;
new output always uses the poset ID.

`ChainComplex::to_stream`, `from_stream`, `to_file`, and `from_file` are the
canonical generic SCC operations. The older matrix and `R2Resolution` readers
and writers are unchanged.

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

graded_linalg::ModuleMorphism<Matrix> f(source, target, std::move(lift));
auto image = f.image();
auto kernel = f.kernel();
Module image_as_module = image.presented_module();
```

## Migrated clients

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
- Exactness of supplied resolutions and the chain-map equations for supplied
  higher morphism lifts are trusted. Structural dimensions, degrees, and
  gradedness are checked.
- A morphism created by the Hom adapter currently contains the generator lift;
  users may supply higher lifts explicitly when available.

## Verification

`tests/modules_test.cpp` exercises CRTP enforcement, sorting certificates and
stale-flag detection, correct unit cancellation and redundant-relation
deletion, legacy and canonical SCC round trips, real fixture loading, chain
validation, module Hilbert functions, resolution invalidation/recomputation,
submodule reduction/presentation/quotients/intersections, morphism
image/kernel/Hom adapters, homology presentations, R³ colex sorting, and
R⁴/Z⁴ I/O.
It is registered as `module_framework_test` with CTest. The established dense,
sparse, graded-matrix, and Hom tests are also registered, so the new layer and
the compatibility API run together. `cli_programs_test` runs all 15 installed
Persistence-Algebra executables on small hand-computed fixtures and compares
their SCC/quiver output or exact mathematical invariants. A fixture sweep successfully loads all 95
valid `scc2020` files in `test_presentations` (excluding the intentionally
ungraded examples and the SCC-sum container), and the module/Hom tests pass
with AddressSanitizer and UndefinedBehaviorSanitizer enabled.
