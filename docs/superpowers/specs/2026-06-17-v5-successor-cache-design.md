# V5 Successor Cache Design

## Goal

Speed up the v5 Wang-rescue-polish path without changing the default best-quality profile. The final landing is an opt-in variant, `qubo_scheduler_state_v5_wang_rescue_successor_cache_polish`, mapped to `scheduler_state_v5_wang_rescue_successor_cache_polish`.

## Evidence

A `cProfile` run on `7x5_cv1_test_12` with `time-limit=20` showed runtime dominated by classical successor work, not QUBO sampling:

- total runtime: about 20.0 seconds
- QUBO time: about 0.13 seconds
- `_successors_with_n_cached(...)`: about 8.57 cumulative seconds
- post-polish window/suffix search: about 7.53 seconds with no improvement on that case

The first safe speed target is therefore repeated successor generation. The main beam remains streaming; post-polish successor caching is gated behind an explicit profile flag because it can shift runtime and quality through time-boundary effects.

## Design

Keep `_successors_with_n_cached(code, current_n, stack_count, max_height)` as a local helper for repeated successor consumers, but do not force the main beam expansion through the full-state cache. The main beam expansion should keep a streaming uncached helper so one-off frontier states do not pay tuple storage and LRU bookkeeping cost.

Implementation details:

- Add `_successors_with_n_uncached(...)` with the original streaming generator behavior.
- Add a private `_successor_tuple_cached(...)` helper decorated with bounded `lru_cache`.
- Implement `_successor_tuple_cached(...)` as `tuple(_successors_with_n_uncached(...))`.
- Return a tuple, preserving the current nested source/destination iteration order and deduplication behavior.
- Use `_successors_with_n_uncached(...)` in the main beam expansion and non-polish one-step lookahead paths.
- Keep `_successors_with_n_cached(...)` for post-polish suffix/bridge repair paths and explicit diagnostics/tests.
- Gate post-polish cache use with `post_polish_successor_cache`, defaulting to `False` in `speed_v2` and in the default v5 Wang-rescue-polish profile.
- Add `scheduler_state_v5_wang_rescue_successor_cache_polish` as an opt-in profile that sets `post_polish_successor_cache=True`.
- Add `qubo_scheduler_state_v5_wang_rescue_successor_cache_polish` to the open benchmark runner choices and Wang research variant sets.
- Serialize `post_polish_successor_cache` in both regular compact rows and priority-group open rows so repeated-priority DLTS cases report the cache gate correctly.
- Keep chunk-level caches unchanged.
- Expose `cache_clear` and `cache_info` on `_successors_with_n_cached` by aliasing the private helper methods so tests and future diagnostics can inspect the full-state cache.

This is intended to be behavior-preserving:

- The same input state yields exactly the same successor sequence.
- Existing callers can still iterate over the result.
- Since the cached result is immutable, callers cannot corrupt shared cache state.

## Tests

Add tests in `tests/test_qubo.py`:

- verify the first call misses and the second identical call hits the full-state cache;
- verify the uncached streaming helper matches the reference and does not fill the full-state cache;
- verify the cached tuple remains exactly equal to the reference `_successors_with_n(...)` output;
- verify post-polish successor-cache use is controlled by the explicit gate;
- verify the open runner accepts the opt-in variant and priority-group rows serialize the cache flag;
- keep existing deduplication and chunk-cache tests passing.

## Verification

Run focused tests first:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py -k "successors or chunk_pop_push" --basetemp .tmp_pytest_successors -p no:cacheprovider -q
```

Then run full QUBO tests:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py --basetemp .tmp_pytest_full -p no:cacheprovider -q
```

Benchmark progression:

1. single slow case `7x5_cv1_test_12`, `time-limit=20`;
2. 20-case 7x5 slice if the slow case preserves validity and moves;
3. full open120 `time-limit=20` only if the earlier checks do not show a quality regression.

## Final Results

Unit tests:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py --basetemp .tmp_pytest_full -p no:cacheprovider -q
```

Result: `207 passed in 0.50s`.

Single slow case:

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 7x5_cv1_test --case-ids 7x5_cv1_test_12 --variant qubo_scheduler_state_v5_wang_rescue_successor_cache_polish --time-limit 20 --alpha 10 --output docs\experiments\scratch_v5_successor_cache_profile_7x5cv1_12_t20_20260617.csv
```

Result: `goal_ok=True`, `business_ok=True`, `moves=42`.

Six-family test120:

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_wang_rescue_successor_cache_polish --time-limit 20 --alpha 10 --output docs\experiments\scratch_v5_successor_cache_profile_test120_t20_20260617_v2.csv
```

Compared with `docs/experiments/scratch_wang_baseline_clean_open120_t20_20260616.csv`:

| profile | valid | moves | full quality penalty | runtime sec |
|---|---:|---:|---:|---:|
| v5 Wang rescue polish baseline | 118/120 | 3687 | 3868 | 578.981 |
| successor-cache polish opt-in | 118/120 | 3690 | 3871 | 524.447 |

The opt-in profile preserves the same two residual failures (`5x5_cv1_test_4`, `5x5_cv2_test_10`) and improves runtime by `54.534s`, with a `+3` move/full-penalty tradeoff. Keep the original v5 Wang-rescue-polish route as the best-quality baseline; use the successor-cache profile when runtime is the priority.
