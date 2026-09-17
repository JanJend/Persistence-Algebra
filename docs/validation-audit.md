# Validation and sorting audit

Scope: every header in `include/grlina` and the CMake test configuration,
with CLI build and regression verification. The decisions below separate explicit queries, algorithmic work,
and defensive checks of caller-supplied invariants.

## Build policy

`checks.hpp` controls `GRLINA_DEBUG_CHECK(...)` and `GRLINA_ASSERT(...)`.
Without an override, checks are disabled when `__OPTIMIZE__` or `NDEBUG` is
present, and enabled otherwise. Thus GCC/Clang `-O1`, `-O2`, `-O3`, `-Os`, and
`-Og` disable them even without `-DNDEBUG`. `-O0` enables them unless `NDEBUG`
is set. On compilers without `__OPTIMIZE__`, use the conventional `NDEBUG` or
an explicit `-DGRLINA_ENABLE_CHECKS=0`/`=1` setting. Set it consistently in all
translation units using these headers.

Explicit `validate()`, `validate_structure()`, `squares_to_zero()`,
`is_chain_complex()`, degree-order queries, `refresh_compatible_sorted()`,
`require_linear_extension()`, and `require_compatibly_sorted()` retain their
checks in optimized builds. `ChainComplex(matrices, true)` also explicitly
requests validation. Its default follows the build policy; `false` skips it.
Homomorphism constructors remain trusted in both modes, as previously requested.

## Decisions by call site / operation

| Header / operation | Decision and reason |
| --- | --- |
| `graded_matrix.hpp`: `SortedFlag::operator bool`, `compatible_sorting_is_verified` | Optimized reads trust the cached flag and comparator, without walking degrees. Diagnostic reads detect stale public edits. Removed the second scan in `compatible_sorting_is_verified`. |
| `graded_matrix.hpp`: sorted-input preconditions in batches, statistics, graph construction, cancellation, minimization, and graded reduction | Diagnostic-only. Correct ordering is a caller precondition; the actual reductions and degree comparisons that determine admissibility remain active. |
| `graded_matrix.hpp`: custom-comparator validation in sorting and permutation helpers | Diagnostic-only. The quadratic proof that a comparator is a compatible total order is not part of sorting. Explicit `require_linear_extension` still checks it. |
| `graded_matrix.hpp`: full `sort_compatibly` | Performs the permutations and sparse-entry sorting, then records the established certificate. No optimized rescan of the result. |
| `graded_matrix.hpp`: partial lex sorts, permutation-returning row sort | Check only the untouched degree axis to establish the combined certificate. Sorting one axis does not prove the other is sorted. |
| `r2graded_matrix.hpp`, `r3graded_matrix.hpp`, `coordinate_degree.hpp`: partial colex sorts | Same rule: check only the untouched axis. |
| `graded_matrix.hpp`: `shift`, `shift_generators`, `cull_columns`, `transposed_copy` | Preserve metadata; defensive shift/truncation scans are diagnostic-only. Transposition copies the certificate because exchanging the two degree lists preserves their ordering. Coordinate lex/colex shifts preserve order. |
| `graded_matrix.hpp`: explicit-degree constructors, parsing, restricted-domain copies, matrix products and direct sums | Retain initial certificate discovery when ordering is not established by the operation. Restricted-domain indices may reorder columns; products combine degree lists from distinct matrices. |
| `graded_matrix.hpp`: `basisLifting` | The unused subset-consistency computation is diagnostic-only. The computation that produces the result remains active. |
| `chain_complex.hpp`: in-memory constructors | Default structural validation and certificate repair follow the build policy. Explicit `true` still checks. |
| `chain_complex.hpp`: `validate_structure` | Always available explicitly. Removed duplicate dimension and grading scans already done by `matrix.validate()`. Adjacent basis compatibility still checked. |
| `chain_complex.hpp`: `push_differential` | Diagnostic builds validate only the appended matrix and adjacent degrees, rather than rescanning the whole chain. Cheap adjacent dimension check retained. |
| `chain_complex.hpp`: sorting and minimization | Structural pre/postchecks and `d*d=0` verification are diagnostic-only. Sorting establishes certificates directly. Chain cancellation still updates both neighbors. |
| `chain_complex.hpp`: SCC read/write | Retain boundary validation, including malformed sparse indices, dimensions, grading, and adjacency. File parsing explicitly validates regardless of build mode. |
| `module.hpp`: resolution constructors/setters, `edit_presentation`, `shift` | Repeated structural validation is diagnostic-only. Invalidation of obsolete resolutions remains unconditional because it changes object state correctly. |
| `module.hpp`: resolution minimization/completion and `homology_module` | Expensive automatic `d*d=0` checks are diagnostic-only. Kernel calculations and boundary-coordinate solving remain active: these compute the answer. |
| `module.hpp`: `add_relation` | Diagnostic-only temporary matrix construction and validation. Optimized builds append directly, without allocating a matrix just to check it. |
| `module.hpp`: Hilbert grid axis validation | Diagnostic-only. Sorted, unique, non-NaN axes are the API precondition. Grid sorting, resolution selection, and dimension computations remain active. |
| `submodule.hpp`: constructors and operations | Trust supplied structure. Removed automatic calls to submodule/parent validation and the minimizer's extra sorted-input check. `validate()` remains opt-in. Parent-identity checks and degree admissibility used by the algorithms remain. |
| `submodule.hpp`: zero/whole factories | Inherit sorting metadata from the parent presentation instead of rediscovering it. Generator deletion preserves existing certificates. |
| `homomorphism_core.hpp`: constructors and `validate` | Constructors remain trusted and scan-free. Explicit `validate` checks structure without repairing sorting flags. `check_lifts` computes the equations when requested. |
| `homomorphism_core.hpp`: identity, zero, shifts | Identity inherits the relevant differential's sorting certificate. Zero maps between independent endpoints retain initial sorting discovery. General shifts no longer repeat grading checks; canonical shifts retain a single nonnegative-amount check. |
| `hom_interface.hpp`: both `End_2d_0` overloads | Coefficient-family reduction does not need inherited endomorphism degree relabeling or sorting refreshes, so those were removed. The typed overload checks the sign once, instead of every generator/relation degree. |
| `homomorphisms.hpp`: `lift_to_relations`, `is_homomorphism` | Retain validation: these are explicit checks of manually supplied maps. Hom-space kernel solving, homotopy reduction, and admissibility tests compute the result and stay active. |
| `graded_linear_system.hpp` | Input scans and target-degree equality are diagnostic-only. Degree selection, solving, no-solution detection, sorting solution indices, and initial output certificate discovery remain active. |
| `matrix_family.hpp` | Per-matrix full validation is diagnostic-only. Cheap shape/alias checks remain. Pivot searches, coefficient reduction, and survivor selection remain active. |
| `presentation_operations.hpp`: `image_contained_in_image` | Removed duplicate wrapper validation; the solver performs diagnostic input validation. |
| `matrix_base.hpp`, `sparse_matrix.hpp`, `dense_matrix.hpp`, `column_types.hpp`, `bitset_algebra.hpp`, `grid_scheduler.hpp`, `to_quiver.hpp`, matrix/Hom kernels | Existing runtime assertions now follow the same optimization-aware policy, including sortedness assertions and expensive `is_invertible()` assertions. Compile-time `static_assert`s remain unchanged. |
| `to_quiver.hpp`, `graded_matrix.hpp`: user-supplied quiver vertices/edges and path dimensions | Retain public boundary checks. Actual graph construction and path-action checks must not be confused with matrix sorting certificates. |
| `module_operations.hpp`, `epsilon_functors.hpp`, remaining constructors/accessors | Retain cheap null/parent-identity, parameter-range, dimension, unavailable-presentation, and unsupported-operation checks. These avoid misuse without matrix-wide scans. |
| `hilbert_euler.hpp`, `orders_and_graphs.hpp`, `r2graded_matrix.hpp`: mathematical branches | Keep ordering-dependent kernel selection, zero/nonzero tests, degree eligibility, graph comparisons, and rank/dimension logic: results depend on them. |
| `r4graded_matrix.hpp`, `z2graded_matrix.hpp`, `z3graded_matrix.hpp`, `z4graded_matrix.hpp` | Use the shared coordinate/matrix implementation; no additional validation loops to change. |
| `graded_linalg.hpp`, `general.hpp`, `draw_hf.hpp`, `progress.hpp`, `modules.hpp` | I/O/format errors and compile-time constraints retained; no additional repeated structural-validation loops. |

## Preconditions in optimized builds

Direct edits to public `row_degrees`/`col_degrees` must invalidate or explicitly
refresh sorting metadata. Existing mutation methods that change order keep
invalidating it. Custom comparators must actually be compatible total orders;
when retaining certificates across shifts they must also be translation-invariant.
A missing certificate means “not certified”, not necessarily “unsorted”.

Removing defensive checks does not remove the need for valid sparse storage,
compatible bases, graded matrices, and sorted input where an algorithm requires
it. Explicit validators remain useful at application input boundaries.

## Verification

Existing algebra and invalid-input regression suites compile with
`GRLINA_ENABLE_CHECKS=1`, independently of optimization. Four policy-test targets
exercise default `-O0`, default `-O2`, forced checks on, and forced checks off.
They check that disabled assertions do not evaluate their expressions, optimized
sorting-flag reads make no comparator calls, explicit validators/file readers
still reject malformed input, and minimization, resolutions, submodules, Hom
lifts, linear solving, and serialization give the same expected results.
