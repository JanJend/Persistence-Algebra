# Follow-up corrections and code-review map

Read this alongside [the API overview](module-framework.md). This guide records
the corrections requested during review, not a claim of exhaustive verification
of every older library routine. Arithmetic and categorical formulas are over F2.

## 1. Types, ownership and homomorphisms

- Actual classes: `Module` in `module.hpp`, `Homomorphism` in `homomorphism.hpp`.
  Earlier names remain compatibility aliases; `module_morphism.hpp` forwards to
  the new header. `module_homomorphisms.hpp` returns the canonical type.
- CRTP enforcement requires `GradedSparseMatrix<D, index, Matrix>`. Generic
  constructions call the concrete `graded_kernel`, not a poset-blind substitute.
  The index type must be signed integral, with -1 available as a sentinel.
- Shared const module pointers keep parents/domain/target alive. Null pointers
  fail before dereference, including both Hom-space adapters. Do not mutate a
  parent through another alias once submodule or homomorphism coordinates refer
  to its basis; these objects are not automatically transported by minimization.
  Arbitrary mutable presentation access also invalidates the stored injective
  model, since an edit can change the represented module.
- Homomorphism lifts are a vector of f_i, not a chain complex in themselves:
  their equations involve the source and target resolutions.
- Manual construction validates sparse storage, dimensions and graded degrees,
  but trusts the caller's homomorphism guarantee. Library constructions guarantee
  the equations algebraically; regression tests independently check their maps.

`graded_linear_system.hpp` adds `solve_graded_linear_system(A, B)` for A X = B.
Each right-hand column uses only A columns born at or below its degree.
`homomorphisms.hpp` adds `lift_to_relations(sourceP, targetP, f0)`, returning
an optional f1 with `targetP * f1 = f0 * sourceP`; no graded solution gives
`nullopt`. `is_homomorphism` is its boolean form. These are optional checks for
untrusted input, not extra work imposed on every trusted constructor.

`Homomorphism::check_lifts()` checks existence of f1 and all stored equations.
`lift_to_resolution()` extends through the common stored part of both projective
resolutions using graded linear systems. Identity, addition and composition
preserve every common available lift level.

## 2. Sorting and basis coordinates

`graded_matrix.hpp` keeps public degree vectors for compatibility. Consequently,
`compatibly_sorted` is a boolean-compatible checked proxy rather than a literal
bool: reads recheck the stored comparator and invalidate a stale certificate.
Ordinary boolean uses/assignments work; binding a bool reference does not.
A flag read costs linear time. This avoids silently trusting direct legacy
vector edits without replacing the public containers.

Custom comparators are checked for equality consistency, totality, transitivity
and extension of the finite partial order before sorting. The finite check is
quadratic; trusted built-in trait orders bypass it. Exact floating-point equality
is unchanged. Sorted-input reducers/minimizers reject absent or stale certificates.

`chain_complex.hpp` sorts each chain group once and transports its permutation to
both adjacent maps, with stable treatment of repeated grades. Workspace callers
of old `sort_*_with_output` functions were inspected before changes. Their old
directions remain intact: lexicographic column output is old-to-new, while R2
colexicographic column output is new-to-old. New `sort_*_with_permutation` APIs
provide both named directions.

The R2 kernel previously restored the original input column basis, then sorted
its output rows again, losing those coordinates. It now sorts only output
columns after restoration. Fixed ambient rows are also preserved in submodule
reduction, inverse images and intersections. Kernel work clears stale reduction
state before running.

## 3. Correct minimization and the counterexample

Let a=e0+e2 have degree (0,1), b=e1+e2 degree (1,0), and c=e0+e1 degree (1,1).
The first two columns have incomparable grades and the same pivot; c has another
pivot. Ordinary graded reduction can miss c=a+b. The new test explicitly proves
the cheap reducer retains three columns and the kernel-based minimizer retains
two, while the hand-computed Hilbert values agree.

Presentation `minimize()` first cancels equal-degree generator/relation pairs,
clearing the whole pivot row with admissible column operations before deletion.
It then computes the concrete graded kernel and uses equal-degree syzygy units
to remove redundant relations. `minimize_variant()` first tries the cheap graded
reduction, then runs this standard algorithm. `semi_minimize()` only cancels
local pairs. `Submodule::minimize_generators()` computes syzygies of the combined
parent relations and supplied generators, deleting only redundant submodule
generators and preserving all ambient row coordinates.
`Submodule::is_zero()` checks vanishing modulo the parent relations with the
graded solver, so it is correct even before generator minimization.

Submodule reduction now has `reduce_generators_lazy()`, using only the parent's
existing relation columns at matching pivots and checking degree admissibility
at every addition. It never rebases ambient rows or modifies the parent. Multiple
relations can share a pivot; an inadmissible first candidate does not prevent
trying another. This works without a graded kernel and does not require sorted
degrees, since admissibility is checked directly. It intentionally misses
dependencies that require first forming combinations of parent relations.
`minimize_generators(bool lazy_preprocessing = true)` uses it by default to reduce
the kernel input; passing false selects the direct exact method.

The repeated syzygy row/column deletions have been replaced with deferred
elimination. For A=[P S], a syzygy q with an equal-degree unit in generator row r
expresses that generator using the others modulo P. Adding q to each other
syzygy containing r clears that coordinate. We can then clear q's column and
leave row r as an unused zero row: subsequent additions cannot reintroduce it.
Original indices stay fixed throughout. Only the selected columns of S are
physically removed, in a single final batch; the temporary syzygy matrix is
discarded after the loop. This is dependency bookkeeping, not an assertion that
arbitrary row/column deletions preserve a chain complex's homology.

`tests/submodule_reduction_test.cpp` checks pivot cascades, inadmissible and
incomparable relation degrees, several relations sharing a pivot, fixed unsorted
ambient coordinates, no-kernel R4 preprocessing, a dependency deliberately missed
by lazy reduction, and preprocessing on/off. Instrumented matrices verify smaller
kernel inputs, skipping the kernel for an empty result, and one final compaction
for 127 exact eliminations with no syzygy row deletions.

`ChainComplex::minimize()` isolates equal-degree units using graded row/column
operations, transports inverse basis changes into both adjacent maps, and removes
only contractible summands. It preserves the chain-homotopy type and all homology,
without assuming exactness or requiring a graded kernel. Empty complexes are a
no-op. A complete projective resolution is a special case of this operation.

Standard `Module::minimize()` dispatches on the stored projective map count:
one map means presentation minimization; multiple maps mean
`Module::minimize_resolution()`. This module method first calls the chain-complex
minimizer, then removes terminal redundant generators using their graded kernel.
The additional step preserves the resolved module and exactness below truncation,
but can change the terminal homology of a truncated resolution. Exactness is the
caller's guarantee. Both operations work on copies and check d*d=0, committing
only on success. Explicit `minimize_presentation()` ignores/discards higher maps.

For example, `[x x] : S(-1,0)^2 -> S` is already a minimal chain complex with
nonzero H1, whereas its presentation of S/(x) has a redundant relation. Chain
minimization retains both columns; module minimization can delete one. The
previous `ChainComplex::minimize_resolution` method mixed these contracts and
has been replaced by this separation, not retained as a misleading alias.
With both projective and injective representations present, standard minimization
acts on the projective one.

### Euler Hilbert-query follow-up

`module.hpp` now tracks completeness separately from the presence of resolution
maps. Supplied resolutions default to truncated/unknown unless explicitly marked
`ResolutionCompleteness::complete` (or terminated by a zero free group). Exactness
and an explicit completeness assertion remain the caller's guarantees. This is
essential: duplicate terminal syzygies can make the Euler sum of a truncation
incorrect even though the represented module is unchanged.

`compute_projective_resolution()` extends the stored maps without rebasing them
and computes kernels until the terminal map is injective. It then records
completeness. `dimension_at()` uses signed chain-group counts for complete
resolutions and local presentation evaluation otherwise. `hilbert_euler.hpp`
contains the degree-only routines: generic point evaluation, incremental R2
query sweeps, and R2 Cartesian-grid prefix sums. Full-grid queries compute a
resolution by default when the concrete matrix offers a graded kernel. Mutable
queries retain it; const queries use a private copy. Without a kernel they retain
the local-evaluation fallback. No new SCC metadata or injective convention is assumed.

`tests/module_hilbert_test.cpp` checks square-interval dimensions on induced,
non-induced and 100-by-100 grids, duplicates and unordered point lists, a
three-map completion with a nonzero F3 contribution, empty/free modules,
completeness invalidation, const queries, explicit R4 completeness and the
no-kernel fallback. Instrumented CRTP matrices count calls and prohibit local
evaluation on Euler paths, proving that the optimization is actually selected.

### Submodules as modules

`Submodule<Matrix>` now publicly inherits `Module<Matrix>`. The base stores the
submodule's own projective/injective representations, separately from its parent
pointer and defining parent-coordinate generator matrix. `compute_presentation()`
stores d1 directly; inherited `compute_projective_resolution()` invokes the virtual
presentation-construction hook if needed and retains all resulting maps in that
same object. This works through a Module reference/pointer. A virtual destructor
supports polymorphic ownership; explicit defaulted copy/move operations preserve
value semantics. Rebuild clients after this header-level layout change.

Default presentation construction keeps the defining generator basis. Requested
module minimization can choose a different basis without changing the defining
family. `number_of_embedding_generators()` always counts that family, whereas
`number_of_generators()` counts stored presentation generators once available.
`generator_map()` is now the typed inclusion from a stable source in the defining
basis. Its `generator_lift()` need not match a separately minimized presentation
cached in the Module base; the categorical adapter reuses the map's own source. Arbitrary inherited edits that change the module itself (such
as shifting only its presentation) do not update the embedding or parent and
must not be used as parent-aware submodule edits.

Lazy/exact generator reductions clear the submodule's stored projective representation,
preventing stale bases; the parent's representations remain unchanged. Mutable
`presented_module()` remains a copy-returning adapter but also populates the base
storage. Const calls preserve their old non-mutating behavior. Zero submodules
can store their 0-by-0 presentation/resolution even without a graded kernel.

`tests/submodule_module_test.cpp` covers in-place computation, polymorphic dispatch
and destruction, conversion to homomorphism domain/target pointers, nested
submodules, inherited Hilbert/SCC operations, copy/move, the two generator bases,
cache invalidation and the legacy copy-returning API.

## 4. Categorical operations

The Stable-Decomposition extraction adds exact parent-aware Submodule
containment/equality, canonical and translated Homomorphism shifts retaining
all stored lifts, and images of submodules. New presentation adapters are in
`presentation_operations.hpp`; matrix-family quotient reduction is in
`matrix_family.hpp`. The latter uses coefficient spaces, not Hom classes modulo
relations. Tests cover this distinction, incomparable degrees, rectangular
families and preservation of ambient row bases. The old kernel-degree-multiset
containment heuristic is not carried over: membership is solved directly by
degree-admissible linear systems. See `module-framework.md` for the public APIs.

`module_operations.hpp` returns objects together with their canonical maps:

| Operation | Construction / returned maps |
| --- | --- |
| Binary direct sum, product, coproduct | Block presentation, both inclusions and projections |
| Kernel, image | Presented submodule and inclusion |
| Cokernel, coimage | Quotient and projection |
| Equalizer, coequalizer | Kernel/cokernel of f+g |
| Pullback | Kernel of [f,g]: A+B -> C, with both projections |
| Pushout | Cokernel of (f,g): C -> A+B, with both structure maps |

These presentations are not minimized, keeping the explicit bases used by the
canonical maps. Minimize a copy when only the object is needed. `Submodule`
also supplies intersection/quotient; `Homomorphism` supplies image/preimage/kernel
as parent-owned submodules and object-only cokernel/coimage methods.
Universal factorization methods and the coimage-to-image isomorphism are not
part of this revision.

## 5. SCC and client changes

SCC line two is strictly the trait's poset ID: 2/3/4 for the continuous coordinate
posets and 2Z/3Z/4Z for the discrete ones. Other traits supply their own explicit
string. The chain-length heuristic is removed. Wrong IDs are rejected even for
zero matrices, as are malformed coordinates/separators/entries. `0 0 0` reads
as a zero module with a 0-by-0 presentation, not an absent presentation. The
existing full_rips fixture with ID 3 but two-coordinate degrees is rejected;
the round-trip test corrects only an in-memory header, leaving the fixture intact.

- AIDA retains matrix and module overloads. `making_examples.cpp` now uses
  argv[1] or a filename relative to the current directory for output.
  Remaining executable defaults referencing developers' home directories were
  removed across AIDA, Stable and Skyscraper: input programs require a supplied
  path, and decomposition helper data uses `listsof_decompositions` relative to
  the working directory.
- Stable-Decomposition's `reduce_submodule` delegates to the exact module-layer
  implementation. Handcrafted tests cover the incomparable-grade syzygy and
  zero-scale pruning of a square interval.
- Skyscraper adapts a working copy of the presentation instead of discarding
  the stored resolution. `Uni_B1` reuses stored d2 and computes it only if absent.
  The duplicated one-generator resolution shortcut was consolidated, fixing
  its indexing and free-module case. Repeated grid-adapted copies are a possible
  performance improvement for later, not a correctness requirement.
- Nested Persistence-Algebra submodules are not removed or advanced by this
  revision. Builds use the sibling library; unrelated workspace edits are preserved.

Multipers does default to Persistence-Algebra for its algebra operations, but
not exclusively: optional Muphasa support supplies kernel and image. See
[`_algebra_op` in Multipers ops.py](https://github.com/DavidLapous/multipers/blob/main/multipers/ops.py).
Its default wrappers are therefore not an independent correctness oracle for PA.

## 6. Explicit boundaries and tests

R3, Z2, Z3, R4 and Z4 have arithmetic, ordering, SCC and generic operations not
requiring syzygies. Their concrete graded kernels are not implemented here.
Kernel-dependent methods contain throwing implementation frames/comments for
Jan, rather than an incorrect ordinary-reduction substitute. Injective
minimization likewise has an explicit `minimize_injective_resolution` frame
pending agreement on injective/cochain conventions.

Start the test review with `tests/module_operations_test.cpp`: it contains the
dependency example above, cancellation through three adjacent maps, repeated
grade permutations, cyclic comparator rejection, stale flags, graded versus
ungraded-only f1 solvability, null pointers, lift extension/composition and SCC
errors. Its categorical examples use free F born at (0,0) and its quotient M
with one relation at (1,1). For F -> M, kernel is free at (1,1), image/coimage
are M and cokernel is zero. The self-pullback has dimensions 1 before the
relation and 2 after; the self-pushout is M. Tests verify these values and the
canonical-map equations, not merely agreement between two wrappers.

`modules_test.cpp` covers the wider module API and fixtures; old dense/sparse/
graded/Hom tests remain enabled. `cli_programs_test` exercises all 15 installed
PA executables on handcrafted fixtures. Stable and Skyscraper add downstream
regressions. Consult the final handoff for the builds and sanitizer runs actually
completed on this revision; coverage is not exhaustive for all legacy routines.
