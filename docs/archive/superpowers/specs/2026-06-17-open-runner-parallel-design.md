# Open Runner Parallel Execution Design

## Goal

Reduce wall-clock time for full open CPMP benchmark runs without changing the solver behavior for any individual case. The target entry point is `scripts/open_cpmp_benchmarks.py`; the default serial behavior must remain unchanged.

## Decision

Use case-level process parallelism, exposed as `--jobs N`, rather than Python thread-level parallelism inside one solver run.

The solver is CPU-bound Python code with heavy successor generation, ranking, dictionaries, and sorting. A thread pool would mostly fight the GIL and would add ordering risk without much speedup. Separate worker processes avoid the GIL, keep each case isolated, and fit the current runner structure because benchmark cases are independent.

## Scope

In scope:

- Add `--jobs` to `scripts/open_cpmp_benchmarks.py`, defaulting to `1`.
- Keep the existing serial code path when `--jobs <= 1`.
- For `--jobs > 1`, dispatch independent case runs through `concurrent.futures.ProcessPoolExecutor`.
- Preserve deterministic output row order by sorting worker results by the original case index and variant run index before writing CSV and summary files.
- Preserve existing portfolio handling by treating each top-level `(case, variant)` run as one worker task; portfolio child variants stay inside that worker so no-regret selection remains local and unchanged.
- Keep `CASE_START` / `CASE_DONE` progress output useful, but accept that worker completion order differs from serial order.
- Make worker payloads simple and picklable: paths, scalar args, selected variant, case metadata, and case index. Workers should load their own instance data inside the process.
- Add tests that verify `--jobs` parsing, serial fallback, row ordering, and consistency of parallel results on a tiny DLTS fixture.

Out of scope:

- No threading inside `run_compact_beam_instance`.
- No parallel expansion inside a single beam layer.
- No shared cross-process successor or Wang feature cache.
- No default change to v5 quality profiles.

## Architecture

The current runner does:

1. discover cases;
2. load each instance;
3. loop through variants;
4. run `_run_open_variant_row(...)`;
5. append rows;
6. write CSV and summary.

The parallel path should preserve the same logical units:

1. discover and filter cases in the parent process;
2. expand top-level tasks as `(case_index, case, variant)`;
3. submit those tasks to a process pool;
4. each worker loads the instance, calls the existing `_run_open_variant_row(...)` or portfolio-local logic, and returns one or more row records plus progress metadata;
5. parent prints completion progress and merges rows in stable task order;
6. existing `_write_open_csv(...)` and `_write_summary(...)` remain unchanged.

The safest helper split is:

- `_run_open_task(task_payload) -> OpenTaskResult`
- `_run_open_variant_rows_for_case(case, args, variant) -> list[dict[str, object]]`
- `_serial_open_rows(...)` keeps the current readable loop.
- `_parallel_open_rows(...)` owns `ProcessPoolExecutor`.

`argparse.Namespace` can be passed if it remains picklable, but a plain dict payload is preferred for worker stability on Windows spawn mode.

## Error Handling

Worker exceptions should propagate to the parent and fail the run. This is preferable to silently producing partial benchmark artifacts.

`--skip-invalid` behavior must match serial mode. If a case is skipped in a worker, return a skipped result with no rows and a printable reason.

The parent should not write CSV/summary until all tasks finish successfully.

## Testing

Add focused unit tests in `tests/test_qubo.py`:

- `build_parser().parse_args(["--jobs", "2"])` sets `jobs=2`, default is `1`.
- A tiny fixture with two cases and one variant returns the same sorted rows for `jobs=1` and `jobs=2`.
- A multi-variant or portfolio-like path preserves row order by case order and variant order.
- Serial fallback remains the default.

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py -k "open_runner or parallel" --basetemp .tmp_pytest_open_parallel -p no:cacheprovider -q
```

Then:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py --basetemp .tmp_pytest_full -p no:cacheprovider -q
```

## Benchmark Verification

First verify a small slice:

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test --limit-per-family 4 --variant qubo_scheduler_state_v5_wang_rescue_polish --time-limit 20 --alpha 10 --jobs 1 --output docs\experiments\scratch_open_parallel_serial_5x5_cv1_4_t20_20260617.csv
```

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test --limit-per-family 4 --variant qubo_scheduler_state_v5_wang_rescue_polish --time-limit 20 --alpha 10 --jobs 4 --output docs\experiments\scratch_open_parallel_jobs4_5x5_cv1_4_t20_20260617.csv
```

The rows should match on case, variant, final validity, moves, full quality penalty, and solver/profile fields. Runtime fields can differ.

Then verify the full six-family test120 line:

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_wang_rescue_polish --time-limit 20 --alpha 10 --jobs 4 --output docs\experiments\scratch_open_parallel_jobs4_test120_t20_20260617.csv
```

Success bar:

- same validity count as the serial v5 run under the same code and time limit;
- no row-order instability in CSV;
- clear wall-clock reduction versus serial execution;
- no change to default `--jobs 1` behavior.

## Risks

The main risk is memory pressure. Each worker has its own process, caches, and solver state. Start with `--jobs 4`; `--jobs 8` should be treated as an experiment, not a default.

Another risk is time-boundary drift: parallel workers can compete for CPU, which may change cases near the 20-second cutoff. For quality claims, compare both coverage and penalty, not just wall-clock time.

## Final Recommendation

Implement `--jobs` as case-level process parallelism in the open benchmark runner. Do not attempt solver-internal threading until the runner-level parallel path is measured and committed.
