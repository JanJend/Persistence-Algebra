#!/usr/bin/env python3
"""Reproducible isolated cases, CSV statistics and log-log runtime regression.

Example (Release executable):
  python3 tests/isomorphism_statistics.py --executable build/isomorphism_benchmark \
      --output build/isomorphism-statistics
No third-party Python packages are needed. Timeouts/errors are recorded, never
converted to successful decisions or substituted into the runtime regression.
"""
import argparse
import csv
import json
import hashlib
import os
import math
import platform
import random
import statistics as st
import subprocess
import time
from pathlib import Path


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def regression(points):
    xs, ys = zip(*[(math.log(n), math.log(t)) for n, t in points])
    mx, my = st.mean(xs), st.mean(ys)
    xx = sum((x-mx)**2 for x in xs)
    slope = sum((x-mx)*(y-my) for x, y in zip(xs, ys))/xx
    intercept = my-slope*mx
    residual = sum((y-intercept-slope*x)**2 for x, y in zip(xs, ys))
    total = sum((y-my)**2 for y in ys)
    return slope, intercept, 1-residual/total if total else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument('--multiplicities', nargs='+', type=int, default=[1, 2, 4, 8])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=15)
    parser.add_argument('--file-timeout', type=float, default=120)
    parser.add_argument('--files', nargs='*', type=Path)
    parser.add_argument('--seed', type=int, default=1001)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0 or args.file_timeout <= 0:
        parser.error('repeats and timeouts must be positive')
    if any(n <= 0 for n in args.sizes) or any(k <= 0 for k in args.multiplicities):
        parser.error('sizes and multiplicities must be positive')
    exe = args.executable.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    files = args.files
    if files is None:
        fixtures = Path(__file__).resolve().parent.parent/'test_presentations'
        files = [fixtures/name for name in [
            'points_wo_density_20_dim2_k_fold_10_min_pres.scc',
            'presentation/noisy_annulus_socg_bifilt.scc',
            'minpres_dim_0_torus_1000_3_0.10.scc']]
    cache = exe.parent/'CMakeCache.txt'
    build_settings = [line for line in cache.read_text().splitlines()
                      if line.startswith(('CMAKE_BUILD_TYPE:', 'CMAKE_CXX_COMPILER:', 'CMAKE_CXX_FLAGS'))] if cache.exists() else []
    (args.output/'environment.json').write_text(json.dumps({
        'platform': platform.platform(), 'processor': platform.processor(),
        'machine': platform.machine(), 'logical_cpus': os.cpu_count(), 'cmake': build_settings,
        'executable_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
        'python': platform.python_version(), 'command': vars(args)}, default=str, indent=2)+'\n')
    rows = []

    def run(command, timeout, **metadata):
        start = time.monotonic()
        row = dict(metadata)
        try:
            proc = subprocess.run([str(exe), *map(str, command)], capture_output=True, text=True, timeout=timeout)
            for line in proc.stdout.splitlines():
                if line.startswith('{'):
                    row.update(json.loads(line))
            row['status'] = 'pass' if proc.returncode == 0 and 'iso_seconds' in row else 'error'
            row['error'] = proc.stderr[-1500:] if proc.returncode else ''
            row['returncode'] = proc.returncode
        except subprocess.TimeoutExpired as error:
            for line in (error.stdout or b'').decode().splitlines():
                if line.startswith('{'):
                    row.update(json.loads(line))
            row.update(status='timeout', error='Process deadline; decision unknown')
        row.update(wall_seconds=time.monotonic()-start, deadline_seconds=timeout)
        rows.append(row)
        write_csv(args.output/'raw.csv', rows)
        print(metadata, row['status'], f"{row['wall_seconds']:.3g}s", flush=True)

    for n in sorted(set(args.sizes)):
        for k in sorted(set(args.multiplicities)):
            if n%k or n < 2*k:
                continue
            for kind in ['positive', 'negative']:
                for trial in range(args.repeats):
                    seed = args.seed+trial
                    run(['random', n, k, seed, kind], args.timeout,
                        family='random', kind=kind, n=n, k=k, seed=seed)
    for path in files:
        for trial in range(args.repeats):
            seed = args.seed+trial
            run(['file', path.resolve(), seed], args.file_timeout,
                family='file', kind='positive', file=str(path), seed=seed)
    # A direct worst-case local-space probe: dim=k(k-1), every matrix singular.
    for k in [2, 3, 4, 5, 6, 8]:
        run(['block', k], min(args.timeout, 5), family='block', kind='negative', k=k)

    return summarize(rows, args)


def summarize(rows, args):
    groups = {}
    for row in rows:
        key = tuple(row.get(k, '') for k in ['family', 'kind', 'n', 'k', 'file'])
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, trials in groups.items():
        group = dict(zip(['family', 'kind', 'n', 'k', 'file'], key))
        good = [r for r in trials if r['status'] == 'pass']
        group.update(passed=len(good), timed_out=sum(r['status'] == 'timeout' for r in trials),
                     errors=sum(r['status'] == 'error' for r in trials), trials=len(trials))
        for metric in ['iso_seconds', 'default_seconds', 'setup_seconds', 'nnz_a', 'nnz_b',
                       'relations', 'row_ops', 'column_ops']:
            values = [r[metric] for r in good if metric in r]
            if values:
                group.update({metric+'_median': st.median(values), metric+'_min': min(values),
                              metric+'_max': max(values), metric+'_mean': st.mean(values),
                              metric+'_sd': st.stdev(values) if len(values)>1 else 0})
        group['identical_shortcuts'] = sum(r.get('identical', 0) for r in good)
        summaries.append(group)
    write_csv(args.output/'summary.csv', summaries)

    fits = []
    rng = random.Random(args.seed)
    for kind in ['positive', 'negative']:
        for k in sorted(set(args.multiplicities)):
            for metric in ['iso_seconds', 'default_seconds']:
                samples = {}
                for row in rows:
                    if (row['family'] == 'random' and row['kind'] == kind and row['k'] == k
                            and row['status'] == 'pass' and not row['identical']):
                        samples.setdefault(row['n'], []).append(row[metric])
                if len(samples) < 3:
                    continue  # No credible size fit from only one or two sizes.
                points = [(n, st.median(v)) for n, v in sorted(samples.items())]
                slope, intercept, r2 = regression(points)
                boot = sorted(regression([(n, st.median(rng.choices(v, k=len(v))))
                                          for n, v in samples.items()])[0] for _ in range(1000))
                fits.append(dict(kind=kind, k=k, metric=metric, exponent=slope,
                                 prefactor=math.exp(intercept), r_squared=r2,
                                 bootstrap_low=boot[24], bootstrap_high=boot[974],
                                 min_n=min(samples), max_n=max(samples), sizes=len(samples),
                                 completed_samples=sum(map(len, samples.values()))))
    write_csv(args.output/'regression.csv', fits)
    passed = sum(r['status'] == 'pass' for r in rows)
    timeout = sum(r['status'] == 'timeout' for r in rows)
    errors = len(rows)-passed-timeout
    lines = ['# Isomorphism runtime study', '',
             f'{passed} completed correctly; {timeout} timed out; {errors} errors.', '',
             'Sparse random presentations start with up to 3 nonzeros per relation and 2n relations. '
             'The recorded relation count is after full minimization. Generator multiplicity is exactly k. '
             'Copies undergo up to 20(n+m) admissible additions plus independent basis permutations; '
             'fill-in is capped at 12 entries per column (or the original maximum for files). '
             'Every sequence is verified by applying its inverse.', '',
             'Positive pairs are isomorphic by construction. Negative pairs have equal Betti degrees '
             'and equal Hilbert functions; a reserved summand on 2k generators gives different structure-map ranks. '
             'The block probes separately expose the exponential singular-space search.', '',
             'Timing uses steady_clock inside the executable. iso_seconds includes copies, validation, '
             'sorting, Hom and block tests with assume_minimal=true; default_seconds additionally '
             'includes the algorithm\'s minimization, which can also change the basis and subsequent Hom cost. '
             'Generation, transformations and inverse checks '
             'are outside both timers. Minimal timing runs first, so cache/order effects can affect '
             'the second timing. Each repetition uses a different reproducible seed.', '',
             'Timeouts cover the entire process (including setup and both calls), are unknown decisions, '
             'and are excluded rather than replaced by deadline values. Fits use log(median time) '
             '= log(c) + p log(n), separately per k and case, excluding identical-matrix shortcuts. '
             'Bootstrap intervals resample seeds within sizes; they do not cover model misspecification '
             'or missing timed-out cases. These are empirical scaling fits, not complexity bounds.', '',
             '| Case | k | Timing | n range | p | Bootstrap 95% interval | R² |',
             '|---|---:|---|---|---:|---|---:|']
    for f in fits:
        lines.append(f"| {f['kind']} | {f['k']} | {f['metric']} | {f['min_n']}–{f['max_n']} | "
                     f"{f['exponent']:.2f} | {f['bootstrap_low']:.2f}–{f['bootstrap_high']:.2f} | {f['r_squared']:.3f} |")
    for kind in ['positive', 'negative']:
        lines += ['', f'## Sparse random {kind} cases', '',
                  'Each entry is median minimal-input seconds [completed/total]. Timeouts are unknown.', '',
                  '| n | ' + ' | '.join(f'k={k}' for k in sorted(set(args.multiplicities))) + ' |',
                  '|---:|' + '---:|'*len(set(args.multiplicities))]
        for n in sorted(set(args.sizes)):
            cells = []
            for k in sorted(set(args.multiplicities)):
                group = next((g for g in summaries if g['family']=='random' and g['kind']==kind
                              and g['n']==n and g['k']==k), None)
                if not group:
                    cells.append('—')
                else:
                    t = group.get('iso_seconds_median')
                    cells.append((f'{t:.5g}' if t is not None else 'timeout') +
                                 f" [{group['passed']}/{group['trials']}]")
            lines.append('| ' + str(n) + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Real presentations', '',
              '| File | Generators | k | Completed | Timed out | Median iso s | Median row / column additions |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for g in summaries:
        if g['family']=='file':
            timing = f"{g['iso_seconds_median']:.5g}" if 'iso_seconds_median' in g else '—'
            ops = f"{g['row_ops_median']:g} / {g['column_ops_median']:g}" if 'row_ops_median' in g else '—'
            lines.append(f"| {Path(g['file']).name} | {g['n']} | {g['k']} | {g['passed']} | "
                         f"{g['timed_out']} | {timing} | {ops} |")
    lines += ['', '## Local all-singular block probe', '',
              '| k | Span dimension | Status | Seconds (completed only) |', '|---:|---:|---|---:|']
    for r in rows:
        if r['family']=='block':
            timing = f"{r['iso_seconds']:.5g}" if r['status']=='pass' else '—'
            lines.append(f"| {r['k']} | {r['k']*(r['k']-1)} | {r['status']} | {timing} |")
    lines += ['', 'Raw trials: [raw.csv](raw.csv). Aggregated timings, sparsity and operation counts: '
              '[summary.csv](summary.csv). Regression data: [regression.csv](regression.csv). '
              'Machine and command: [environment.json](environment.json).']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
