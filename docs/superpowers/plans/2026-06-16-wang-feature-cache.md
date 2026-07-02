# Wang Feature Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stack-level Wang feature caching so `qubo_scheduler_state_v5_wang_rescue_polish` spends less time recomputing fixed-prefix and fix-pair scores.

**Architecture:** Keep the existing solver profile, QUBO model, candidate ordering, and whole-code `lru_cache` functions intact. Add a smaller cached helper for per-stack compact chunks, then route the existing Wang score functions through that helper.

**Tech Stack:** Python 3.13, `functools.lru_cache`, existing compact-byte CPMP benchmark code, pytest.

---

## File Structure

- Modify `scripts/benchmark_compact_beam_qubo_n0.py`: add `_WangStackFeatures`, `_compact_wang_stack_features(...)`, and rewrite the two existing Wang score functions to call it.
- Modify `tests/test_qubo.py`: add cache behavior coverage near the existing Wang probe tests.
- Create benchmark CSVs under `docs/experiments/` only during verification.

### Task 1: Add Failing Cache Behavior Test

**Files:**
- Modify: `tests/test_qubo.py`

- [ ] **Step 1: Add a test that expects stack-level cache hits**

Add this test after `test_wang_probe_fixed_prefix_allows_repeated_priority_groups`:

```python
def test_wang_stack_feature_cache_reuses_shared_chunks():
    from benchmark_compact_beam_qubo_n0 import (
        _compact_wang_fix_pair_score,
        _compact_wang_fixed_prefix_score,
        _compact_wang_stack_features,
    )
    from run_business_exact_target_search_sqlite import _encode

    _compact_wang_fix_pair_score.cache_clear()
    _compact_wang_fixed_prefix_score.cache_clear()
    _compact_wang_stack_features.cache_clear()

    first, _first_n = _encode([[5, 4], [3, 1, 2], []], 3, 3)
    second, _second_n = _encode([[5, 4], [3, 2, 1], []], 3, 3)

    assert _compact_wang_fixed_prefix_score(first, 3, 3) >= 0
    assert _compact_wang_fix_pair_score(first, 3, 3) >= 0
    before = _compact_wang_stack_features.cache_info()

    assert _compact_wang_fixed_prefix_score(second, 3, 3) >= 0
    assert _compact_wang_fix_pair_score(second, 3, 3) >= 0
    after = _compact_wang_stack_features.cache_info()

    assert after.hits > before.hits
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py::test_wang_stack_feature_cache_reuses_shared_chunks --basetemp .tmp_pytest -q
```

Expected: fail with `ImportError` for `_compact_wang_stack_features`.

### Task 2: Implement Stack-Level Wang Feature Cache

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`

- [ ] **Step 1: Add the import and feature record**

Change the imports near the top:

```python
from typing import NamedTuple
```

Add this record before `_wang_fixed_prefix_len(...)`:

```python
class _WangStackFeatures(NamedTuple):
    fixed_prefix: int
    acceptor_top: int
    dirty_boxes: tuple[int, ...]
```

- [ ] **Step 2: Add the cached stack helper**

Add this helper after `_wang_fixed_prefix_len(...)`:

```python
@lru_cache(maxsize=1048576)
def _compact_wang_stack_features(chunk: bytes, max_height: int) -> _WangStackFeatures:
    height = _chunk_height_raw(chunk)
    dirty_count = _stack_n(chunk, max_height)
    fixed_prefix = _wang_fixed_prefix_len(chunk)
    acceptor_top = -1
    if dirty_count == 0 and height < max_height:
        acceptor_top = int(chunk[height - 1]) if height > 0 else 255
    dirty_boxes: tuple[int, ...] = ()
    if dirty_count > 0:
        start = height - dirty_count
        dirty_boxes = tuple(int(box) for box in chunk[start:height])
    return _WangStackFeatures(
        fixed_prefix=fixed_prefix,
        acceptor_top=acceptor_top,
        dirty_boxes=dirty_boxes,
    )
```

- [ ] **Step 3: Rewrite fixed-prefix score**

Replace `_compact_wang_fixed_prefix_score(...)` body with:

```python
def _compact_wang_fixed_prefix_score(code: bytes, stack_count: int, max_height: int) -> int:
    if stack_count <= 0 or max_height <= 0:
        return 0
    return sum(
        _compact_wang_stack_features(chunk, max_height).fixed_prefix
        for chunk in _split(code, stack_count, max_height)
    )
```

Keep the existing `@lru_cache(maxsize=262144)` decorator.

- [ ] **Step 4: Rewrite fix-pair score**

Replace `_compact_wang_fix_pair_score(...)` body with:

```python
def _compact_wang_fix_pair_score(code: bytes, stack_count: int, max_height: int) -> int:
    if stack_count <= 0 or max_height <= 0:
        return 0
    features = [
        _compact_wang_stack_features(chunk, max_height)
        for chunk in _split(code, stack_count, max_height)
    ]
    acceptors = [
        (stack_idx, feature.acceptor_top)
        for stack_idx, feature in enumerate(features)
        if feature.acceptor_top >= 0
    ]
    score = 0
    for origin, feature in enumerate(features):
        for box in feature.dirty_boxes:
            score += sum(1 for stack_idx, top in acceptors if stack_idx != origin and box <= top)
    return score
```

Keep the existing `@lru_cache(maxsize=262144)` decorator.

- [ ] **Step 5: Run the new test**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py::test_wang_stack_feature_cache_reuses_shared_chunks --basetemp .tmp_pytest -q
```

Expected: pass.

### Task 3: Regression Tests

**Files:**
- Test: `tests/test_qubo.py`
- Test: `scripts/benchmark_compact_beam_qubo_n0.py`

- [ ] **Step 1: Run Wang-focused tests**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py -k "wang_probe or wang_pair_qubo_supplement or unified_wang_qubo_features or wang_feature" --basetemp .tmp_pytest -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run the changed-file broader test slice**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py --basetemp .tmp_pytest -q
```

Expected: all tests in `tests/test_qubo.py` pass.

### Task 4: Benchmark Verification

**Files:**
- Create: `docs/experiments/scratch_wang_feature_cache_7x5cv1_12_t20_20260616.csv`
- Create: `docs/experiments/scratch_wang_feature_cache_open120_t20_20260616.csv`

- [ ] **Step 1: Run one slow case**

Run:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 7x5_cv1_test --case-ids 7x5_cv1_test_12 --variant qubo_scheduler_state_v5_wang_rescue_polish --time-limit 20 --alpha 10 --output docs\experiments\scratch_wang_feature_cache_7x5cv1_12_t20_20260616.csv
```

Expected: the command writes the CSV and summary. Record runtime and validity; do not require identical move count because the existing profile has small runtime-sensitive drift.

- [ ] **Step 2: Run full open120 at 20s**

Run:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_wang_rescue_polish --time-limit 20 --alpha 10 --output docs\experiments\scratch_wang_feature_cache_open120_t20_20260616.csv
```

Expected safety bar: valid count is at least `118/120`, and failures do not expand beyond the two known 5x5 cases without explicit investigation.

- [ ] **Step 3: Compare against clean baseline**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'; @'
import csv
from pathlib import Path

def read(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def ok(row):
    return row.get('final_n') == '0' and row.get('goal_validation_ok') == 'True' and row.get('business_ok') == 'True'

for label, path in [
    ('baseline', 'docs/experiments/scratch_wang_baseline_clean_open120_t20_20260616.csv'),
    ('cache', 'docs/experiments/scratch_wang_feature_cache_open120_t20_20260616.csv'),
]:
    rows = read(path)
    valid = sum(1 for row in rows if ok(row))
    invalid = [row['case'] for row in rows if not ok(row)]
    runtime = sum(float(row.get('runtime_sec') or 0.0) for row in rows)
    print(label, valid, invalid, round(runtime, 3))
'@ | python -
```

Expected: the cache run keeps validity at or above the baseline safety bar and reports lower or comparable total runtime.
