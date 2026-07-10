# V5 Successor Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in successor-cache speed profile for the v5 Wang-rescue-polish search path while keeping the best default v5 profile unchanged.

**Architecture:** Split successor generation into an uncached streaming helper for main beam expansion and a private bounded `lru_cache` tuple helper for repeated post-polish/local repair consumers. The cache is exposed through `scheduler_state_v5_wang_rescue_successor_cache_polish`, not enabled on the default `scheduler_state_v5_wang_rescue_polish` baseline.

**Tech Stack:** Python, `functools.lru_cache`, pytest, existing compact CPMP benchmark scripts.

---

### Task 1: Full-State Successor Cache

**Files:**
- Modify: `tests/test_qubo.py`
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`

- [x] **Step 1: Write the failing cache-hit test**

Add this test near `test_compact_cached_successors_match_reference_successors`:

```python
def test_compact_full_state_successor_cache_hits_on_repeated_state():
    from benchmark_compact_beam_qubo_n0 import _successors_with_n_cached
    from run_business_exact_target_search_sqlite import _encode, _successors_with_n

    _successors_with_n_cached.cache_clear()
    code, n_value = _encode([[5, 2], [4, 3], [1], []], 4, 3)

    first = tuple(_successors_with_n_cached(code, n_value, 4, 3))
    after_first = _successors_with_n_cached.cache_info()
    second = tuple(_successors_with_n_cached(code, n_value, 4, 3))
    after_second = _successors_with_n_cached.cache_info()

    assert first == tuple(_successors_with_n(code, n_value, 4, 3))
    assert second == first
    assert after_first.misses == 1
    assert after_second.hits == 1
```

- [x] **Step 2: Verify the test fails**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py::test_compact_full_state_successor_cache_hits_on_repeated_state --basetemp .tmp_pytest_successors -p no:cacheprovider -q
```

Expected: failure because `_successors_with_n_cached` does not expose `cache_clear` or `cache_info`.

- [x] **Step 3: Implement the tuple-backed cache**

Change `scripts/benchmark_compact_beam_qubo_n0.py` by adding:

```python
@lru_cache(maxsize=65536)
def _successor_tuple_cached(
    code: bytes,
    current_n: int,
    stack_count: int,
    max_height: int,
) -> tuple[tuple[bytes, int], ...]:
    ...
```

Move the current `_successors_with_n_cached(...)` body into this helper, append each `(nxt, nxt_n)` pair to a list, and return `tuple(out)`.

Then redefine:

```python
def _successors_with_n_cached(code: bytes, current_n: int, stack_count: int, max_height: int):
    return _successor_tuple_cached(code, current_n, stack_count, max_height)


_successors_with_n_cached.cache_clear = _successor_tuple_cached.cache_clear
_successors_with_n_cached.cache_info = _successor_tuple_cached.cache_info
```

- [x] **Step 4: Verify focused tests pass**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py -k "successors or chunk_pop_push" --basetemp .tmp_pytest_successors -p no:cacheprovider -q
```

Expected: all selected tests pass.

- [x] **Step 5: Add the streaming helper guard test**

Add `test_compact_uncached_successor_stream_matches_reference_without_filling_full_cache` to prove the main-search helper can match reference successors without filling the full-state cache.

- [x] **Step 6: Route main expansion through the streaming helper**

Use `_successors_with_n_uncached(...)` for main beam expansion and non-polish one-step lookahead paths. Keep `_successors_with_n_cached(...)` for post-polish suffix/bridge repair and explicit diagnostics.

### Task 2: Regression and Benchmark Verification

**Files:**
- Create: `docs/experiments/scratch_v5_successor_cache_profile_test120_t20_20260617_v2.csv`
- Create: `docs/experiments/scratch_v5_successor_cache_profile_test120_t20_20260617_v2_summary.md`

- [x] **Step 1: Run the full unit regression**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py --basetemp .tmp_pytest_full -p no:cacheprovider -q
```

Expected: all tests pass.

- [x] **Step 2: Run the profiled slow case**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 7x5_cv1_test --case-ids 7x5_cv1_test_12 --variant qubo_scheduler_state_v5_wang_rescue_polish --time-limit 20 --alpha 10 --output docs\experiments\scratch_v5_successor_cache_7x5cv1_12_t20_20260617.csv
```

Observed with the new opt-in variant:

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 7x5_cv1_test --case-ids 7x5_cv1_test_12 --variant qubo_scheduler_state_v5_wang_rescue_successor_cache_polish --time-limit 20 --alpha 10 --output docs\experiments\scratch_v5_successor_cache_profile_7x5cv1_12_t20_20260617.csv
```

Result: `goal_ok=True`, `business_ok=True`, `moves=42`, `post_polish_successor_cache=True`.

- [x] **Step 3: Run test120/open120-equivalent verification**

Run the six-family test120 command against the opt-in profile:

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_wang_rescue_successor_cache_polish --time-limit 20 --alpha 10 --output docs\experiments\scratch_v5_successor_cache_profile_test120_t20_20260617_v2.csv
```

Result versus `docs/experiments/scratch_wang_baseline_clean_open120_t20_20260616.csv`:

- baseline v5: `118/120`, moves `3687`, penalty `3868`, runtime `578.981s`
- successor-cache profile: `118/120`, moves `3690`, penalty `3871`, runtime `524.447s`
- delta: same coverage, `+3` moves/penalty, `-54.534s` runtime
- residual failures unchanged: `5x5_cv1_test_4`, `5x5_cv2_test_10`
- telemetry: `post_polish_successor_cache=True` on all 120 rows

Conclusion: keep default v5 as the best quality baseline; keep successor-cache polish as an opt-in speed profile.

- [ ] **Step 4: Commit the fixed point**

Stage only the successor-cache code, tests, docs, and selected verification artifacts. Do not stage `AGENTS.md`.
