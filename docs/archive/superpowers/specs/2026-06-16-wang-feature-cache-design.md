# Wang Feature Cache Design

## Goal

Reduce per-case runtime for `qubo_scheduler_state_v5_wang_rescue_polish` without changing beam widths, QUBO parameters, rescue gates, polish behavior, or candidate ordering semantics.

## Context

The current best single-route profile remains `qubo_scheduler_state_v5_wang_rescue_polish`. A full `time-limit=20` open120 rerun recovered `118/120`; the two remaining failures are `5x5_cv1_test_4` and `5x5_cv2_test_10`. The 7x5 drift cases are time-boundary sensitive and benefit from more runtime, so speed work should focus on preserving the same search decisions while reducing repeated feature cost.

Profiling a slow 7x5 case showed `_compact_wang_fix_pair_score(...)` as a major hotspot. The full-code Wang score functions already have `lru_cache`, so a second whole-code cache would add little. The better first cut is a smaller reusable stack-level feature cache because neighboring compact states often share most stack chunks.

## Design

Add a private immutable stack feature record in `scripts/benchmark_compact_beam_qubo_n0.py`:

- `fixed_prefix`: the Wang fixed-prefix length for one compact stack chunk.
- `acceptor_top`: the top box for a clean stack with capacity, or `-1` if the stack cannot accept.
- `dirty_boxes`: dirty boxes in the same order used by the current fix-pair score.

Add `_compact_wang_stack_features(chunk: bytes, max_height: int)` with `lru_cache`. Rewrite `_compact_wang_fixed_prefix_score(...)` and `_compact_wang_fix_pair_score(...)` to reuse stack features after `_split(...)`.

This preserves all public behavior:

- The same compact code, stack count, and max height produce the same scores.
- Candidate sorting still uses the same score tuple.
- QUBO model construction and seeds stay unchanged.
- Cache lifetime is process-local and pure-function only; no cross-instance mutable search state is introduced.

## Tests

Add focused tests in `tests/test_qubo.py` near existing Wang tests:

- Stack feature cache preserves known Wang fixed-prefix behavior.
- Two different compact states that share stack chunks produce stack-feature cache hits.
- Wang pair-QUBO supplement still selects the same high fix-pair candidate.

## Verification

Run focused pytest first:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py -k "wang_feature or wang_pair_qubo_supplement or fixed_prefix" --basetemp .tmp_pytest -q
```

Then benchmark:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 7x5_cv1_test --case-ids 7x5_cv1_test_12 --variant qubo_scheduler_state_v5_wang_rescue_polish --time-limit 20 --alpha 10 --output docs\experiments\scratch_wang_feature_cache_7x5cv1_12_t20_20260616.csv
```

Finally compare a full open120 `time-limit=20` run against the current clean baseline. The required safety bar is no validity regression from `118/120`.
