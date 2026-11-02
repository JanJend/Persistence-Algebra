# Presentation minimisation comparison

Measured 2026-11-02 on this machine; 21 repetitions per method and input.

Both public methods cancel local generator–relation pairs first. They are adjacent in
`include/grlina/matrices/graded.hpp`: `minimize()` and `minimize_by_image_generators()`.
This compares presentation minimisers directly, rather than the complete chromatic pipeline.

| Parameters | Family | Input rows / columns | After cancellation | Normal (ms) | Image generators (ms) | Image / normal |
|---:|---|---:|---:|---:|---:|---:|
| 2 | minimal | 128 / 128 | 128 / 128 | 0.023875 | 0.010500 | 0.44× |
| 3 | minimal | 64 / 64 | 64 / 64 | 0.014459 | 0.305417 | 21.12× |
| 2 | duplicates | 128 / 512 | 128 / 512 | 1.177041 | 0.071291 | 0.06× |
| 3 | duplicates | 64 / 256 | 64 / 256 | 1.250250 | 1.085042 | 0.87× |
| 2 | incomparable | 128 / 512 | 128 / 512 | 0.738125 | 0.058000 | 0.08× |
| 3 | incomparable | 64 / 256 | 64 / 256 | 1.439417 | 1.242500 | 0.86× |
| 2 | local_pairs | 256 / 640 | 128 / 512 | 0.890042 | 0.101667 | 0.11× |
| 3 | local_pairs | 128 / 320 | 64 / 256 | 1.178917 | 1.047458 | 0.89× |

Ratios below one favour image generation. Times are medians in milliseconds;
sub-millisecond measurements and their ratios are sensitive to machine noise.

## What the methods compute

Both first cancel all equal-degree generator–relation pairs, transporting supplied
cycle representatives and generator maps. This leaves the same generator coordinates
in both paths. Both then remove redundant relation generators without changing the cokernel.

- **Normal:** compute the graded kernel of the remaining presentation. A kernel
  generator with a coefficient at the same degree as a relation identifies a
  redundant relation. Clear that coefficient from the other syzygies, then delete
  the redundant relation and its matching syzygy. This constructs and updates
  coefficient vectors for dependencies even though a presentation does not retain them.
- **Image generators, two coordinates:** run the existing scheduled bigraded image
  reducer with an empty auxiliary matrix. Retain generators of the image without
  building syzygy coefficient vectors. The reducer shares work across grade events.
- **Image generators, other degrees:** process columns in a linear extension.
  At degree d, reduce a candidate against the span of previously retained columns
  born at degrees <= d. Discard it exactly when it is already in that span.
  The existing ungraded solver is called without an operation matrix. Dependencies
  are still detected, but their coefficient vectors are not retained.

The general fallback repeatedly constructs and reduces admissible spans. This is
especially wasteful on an already-minimal presentation: there is no redundancy
to remove, while the normal kernel computation can quickly return an empty kernel.
Here it is about 21 times slower on the three-parameter minimal case. In the
redundant three-parameter cases it saves only about 11–14%. The scheduled
two-parameter path is about 2.3–16.5 times faster on these controlled inputs.
These examples do not establish a universal speedup, and the general default
has not been switched to the fallback.

After local cancellation every remaining nonzero presentation entry has strictly
smaller row degree than its column degree. Degree-admissible combinations cannot
create a new equal-degree entry, so removing redundant columns does not require
a second local-cancellation pass.

## Inputs and checks

- `minimal`: independent triangular columns, all born at the same positive degree.
- `duplicates`: four comparable births of each triangular column.
- `incomparable`: the early copies have incomparable births and must be retained;
  later copies are redundant. A separate fixed correctness example requires
  combining different columns at their common upper birth to detect redundancy.
- `local_pairs`: one cancellable pair per surviving row, with later relations
  touching cancelled rows before compression, followed by the duplicate family.

The test compares exact graded image containment in both directions after the
common cancellation, equal remaining relation counts, and absence of local pairs.
It covers fixed and seeded random inputs in one through four parameters, legacy R2,
ordinary coordinate, CSC, and grid-backed matrices. A counted matrix confirms
that only the normal method calls `graded_kernel()`. Additional checks exercise
the alternative on matrix types without that method and on a non-coordinate
fork-shaped degree poset.

The timing harness warms up each method, alternates execution order, and keeps
input copies, validation, and output outside each timed public call. Every timed
result must match that method's verified warm-up result exactly. Library diagnostics
are disabled for timing; test assertions remain enabled. No competing builds were
running during these measurements.

The module wrapper is now `minimize_presentation_by_image_generators()`, supporting
all parameter counts. It rejects stored higher projective maps unconditionally;
use `minimize_resolution()` when higher maps need transport. The standard module
minimiser also performs cancellation before its cheap graded-column preprocessing.

## Reproduction

```sh
cmake --build build-native-optimization --target minimization_comparison_test -j2
ctest --test-dir build-native-optimization -R '^minimization_comparison_test$' --output-on-failure
./build-native-optimization/minimization_comparison_test --benchmark 21
```

The comparison test and 25 other relevant Persistence-Algebra suites passed.
All eight selected chromatic integration suites passed.
Raw per-run samples are in `results.csv`; compiler flags, source hashes, binary
hash, and execution command are in `environment.json`.
