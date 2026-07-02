# Strict-Cap Dual Scorecard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the paper `alpha=10`, `t=20` scorecard count and select only cap-valid, in-time `N=0` paths while retaining feasibility-only paths as explicit diagnostics.

**Architecture:** Keep solver behavior unchanged. Add paper-runner scorecard fields immediately after existing replay validation, and make the shared open-row selector honor `primary_ok` only when that field is present. The paper runner then uses those fields for CSVs, portfolio selection, penalties, and dual summaries; ordinary open benchmark rows omit the field and retain their existing behavior.

**Tech Stack:** Python 3.10+, `argparse`, `csv`, `pytest`.

---

## File Structure

- Modify: `scripts/open_cpmp_benchmarks.py`
  - Add paper scorecard columns to `OPEN_EXTRA_FIELDS` so the paper CSV serializes them.
  - Make `_open_row_valid()` use `primary_ok` only for rows that explicitly provide it; preserve the legacy open-row predicate otherwise.
- Modify: `scripts/benchmark_compact_paper_set.py`
  - Derive `feasibility_ok`, `time_limit_ok`, and `primary_ok` at the runner boundary.
  - Reject nonpositive paper time budgets, use strict rows for portfolio control flow, and emit primary plus feasibility summaries.
- Modify: `tests/test_bs_frg_paper_benchmarks.py`
  - Add scorecard, exception, portfolio, CSV, summary, and parser regression coverage.
- Modify: `tests/test_qubo.py`
  - Add one compatibility test proving rows without `primary_ok` retain the existing open-runner validity behavior.

### Task 1: Pin the strict row contract with failing paper-runner tests

**Files:**
- Modify: `tests/test_bs_frg_paper_benchmarks.py`
- Modify: `tests/test_qubo.py:8302-8333`

- [ ] **Step 1: Write the failing scorecard tests**

Add these tests near the existing exception and lazy-portfolio tests. Import the new helper name from `benchmark_compact_paper_set` exactly as shown.

```python
import pytest


@pytest.mark.parametrize(
    ("row", "time_limit", "expected"),
    [
        (
            {"moves": 17, "final_n": 0, "goal_validation_ok": True, "business_ok": False, "runtime_sec": 1.0},
            20.0,
            {"feasibility_ok": True, "time_limit_ok": True, "primary_ok": False},
        ),
        (
            {"moves": 17, "final_n": 0, "goal_validation_ok": True, "business_ok": True, "runtime_sec": 20.01},
            20.0,
            {"feasibility_ok": True, "time_limit_ok": False, "primary_ok": False},
        ),
        (
            {"moves": 17, "final_n": 0, "goal_validation_ok": True, "business_ok": True, "runtime_sec": 20.0},
            20.0,
            {"feasibility_ok": True, "time_limit_ok": True, "primary_ok": True},
        ),
        (
            {"moves": 17, "final_n": 0, "goal_validation_ok": True, "business_ok": True, "runtime_sec": "bad"},
            20.0,
            {"feasibility_ok": True, "time_limit_ok": False, "primary_ok": False},
        ),
    ],
)
def test_paper_scorecard_distinguishes_feasibility_cap_and_time(row, time_limit, expected):
    from benchmark_compact_paper_set import _apply_paper_scorecard

    scored = _apply_paper_scorecard(dict(row), time_limit=time_limit)

    assert {key: scored[key] for key in expected} == expected
```

Add a strict-budget test:

```python
def test_paper_runner_rejects_nonpositive_time_limit():
    from benchmark_compact_paper_set import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--time-limit", "0"])
```

Extend `test_paper_exception_row_preserves_failed_case_metadata()` with:

```python
assert row["feasibility_ok"] is False
assert row["time_limit_ok"] is False
assert row["primary_ok"] is False
```

Add this compatibility test adjacent to `test_open_no_regret_row_keeps_valid_baseline_and_selects_shorter_valid_guarded()` in `tests/test_qubo.py`:

```python
def test_open_row_valid_uses_primary_ok_only_when_the_field_is_present():
    from open_cpmp_benchmarks import _open_row_valid

    legacy_row = {
        "moves": 10,
        "final_n": 0,
        "business_ok": True,
        "goal_validation_ok": True,
    }
    strict_row = {**legacy_row, "primary_ok": False}

    assert _open_row_valid(legacy_row) is True
    assert _open_row_valid(strict_row) is False
```

Add this lazy-portfolio regression after the existing lazy tests. Its first child is cap-valid but over time, so it proves the stop condition reads `primary_ok` rather than the legacy open validity predicate:

```python
def test_paper_lazy_portfolio_skips_over_time_feasibility_child(monkeypatch, tmp_path: Path):
    from argparse import Namespace

    from benchmark_bs_frg_paper import PaperCase
    from benchmark_compact_paper_set import PaperRunTask, _run_task

    bridge_variant = "qubo_scheduler_state_v5_qubo_diversity_union_terminal_bridge_polish"
    wide_variant = "qubo_scheduler_state_v5_qubo_diversity_union_tail_lookahead_wide_polish"
    lazy_variant = "qubo_scheduler_state_v5_qubo_diversity_union_terminal_escape_lazy_portfolio"
    case = PaperCase("BF", "BFx", "case_1", tmp_path / "case_1.bay")
    calls = []

    def fake_load(_case_arg):
        return object()

    def fake_run(case_arg, _args_arg, variant, instance=None):
        del instance
        calls.append(variant)
        primary = variant == wide_variant
        return {
            "dataset": case_arg.dataset,
            "family": case_arg.family,
            "case": case_arg.case_id,
            "variant": variant,
            "solver_profile": variant,
            "moves": 9 if primary else 7,
            "final_n": 0,
            "runtime_sec": 1.0 if primary else 21.0,
            "business_ok": True,
            "goal_validation_ok": True,
            "feasibility_ok": True,
            "time_limit_ok": primary,
            "primary_ok": primary,
            "full_quality_penalty": 9 if primary else 99,
        }

    monkeypatch.setattr("benchmark_compact_paper_set._load_paper_case", fake_load)
    monkeypatch.setattr("benchmark_compact_paper_set._run_paper_variant_row", fake_run)

    result = _run_task(PaperRunTask(0, case, lazy_variant), vars(Namespace(alpha=10.0)))

    assert calls == [bridge_variant, wide_variant]
    assert result.row["portfolio_selected_profile"] == wide_variant
    assert result.row["primary_ok"] is True
    assert result.row["portfolio_runtime_sec"] == "22.000000"
```

- [ ] **Step 2: Run the focused tests and confirm the expected red failure**

Run:

```powershell
python -m pytest tests/test_bs_frg_paper_benchmarks.py -q --basetemp .tmp_pytest_strict_scorecard_red
```

Expected: FAIL during collection or assertion because `_apply_paper_scorecard` and the strict parser validation do not yet exist; existing tests may still pass.

- [ ] **Step 3: Commit the failing tests**

```powershell
git add tests/test_bs_frg_paper_benchmarks.py
git commit -m "test: define strict paper scorecard contract"
```

### Task 2: Add derived scorecard fields and strict time-budget parsing

**Files:**
- Modify: `scripts/open_cpmp_benchmarks.py:44-62`
- Modify: `scripts/open_cpmp_benchmarks.py:847-852`
- Modify: `scripts/benchmark_compact_paper_set.py:70-147`
- Modify: `scripts/benchmark_compact_paper_set.py:319-322`

- [ ] **Step 1: Add the scorecard columns and explicit-primary selector rule**

In `OPEN_EXTRA_FIELDS`, insert these fields immediately after `goal_validation_ok`:

```python
    "feasibility_ok",
    "time_limit_ok",
    "primary_ok",
```

Replace `_open_row_valid()` with a backward-compatible explicit-field check:

```python
def _open_row_valid(row: dict[str, object]) -> bool:
    if "primary_ok" in row:
        return _is_true(row.get("primary_ok"))
    return (
        _is_true(row.get("business_ok"))
        and _is_true(row.get("goal_validation_ok"))
        and _optional_int(row.get("final_n")) == 0
        and _optional_int(row.get("moves")) is not None
    )
```

This keeps Open120 semantics unchanged because its rows do not contain `primary_ok`; it makes existing shared penalty and portfolio helpers strict for paper rows.

- [ ] **Step 2: Add the paper scorecard helpers**

In `scripts/benchmark_compact_paper_set.py`, add a local move-presence helper and the scorecard transformer after `_is_true()`:

```python
def _paper_has_move_path(row: dict[str, object]) -> bool:
    return row.get("moves") not in (None, "")


def _paper_time_limit_ok(row: dict[str, object], *, time_limit: float) -> bool:
    try:
        runtime = float(row.get("runtime_sec"))
    except (TypeError, ValueError):
        return False
    return runtime <= float(time_limit)


def _apply_paper_scorecard(row: dict[str, object], *, time_limit: float) -> dict[str, object]:
    feasibility_ok = (
        _is_true(row.get("goal_validation_ok"))
        and row.get("final_n") == 0
        and _paper_has_move_path(row)
    )
    time_limit_ok = feasibility_ok and _paper_time_limit_ok(row, time_limit=time_limit)
    row.update(
        {
            "feasibility_ok": feasibility_ok,
            "time_limit_ok": time_limit_ok,
            "primary_ok": feasibility_ok and _is_true(row.get("business_ok")) and time_limit_ok,
        }
    )
    return row
```

Call it after the validator fields are added in `_run_paper_variant_row()` and before `_open_full_quality_penalty()` is calculated:

```python
row = _apply_paper_scorecard(row, time_limit=args.time_limit)
row["full_quality_penalty"] = _open_full_quality_penalty(row)
```

Give `row_from_paper_exception()` all three scorecard fields set to `False`.

Add a parser converter near `build_parser()`:

```python
def _strict_positive_time_limit(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("strict paper scorecard requires --time-limit > 0")
    return parsed
```

Then use `type=_strict_positive_time_limit` for `--time-limit`.

- [ ] **Step 3: Run the focused tests and confirm green**

Run:

```powershell
python -m pytest tests/test_bs_frg_paper_benchmarks.py -q --basetemp .tmp_pytest_strict_scorecard_green
```

Expected: PASS. The new scorecard tests show over-cap and over-time rows as feasibility-only, and the exception row exposes all three false fields.

- [ ] **Step 4: Commit the minimal scorecard implementation**

```powershell
git add scripts/open_cpmp_benchmarks.py scripts/benchmark_compact_paper_set.py tests/test_bs_frg_paper_benchmarks.py
git commit -m "bench: add strict paper scorecard fields"
```

### Task 3: Make paper summaries report both scorecards

**Files:**
- Modify: `tests/test_bs_frg_paper_benchmarks.py`
- Modify: `scripts/benchmark_compact_paper_set.py:241-309`

- [ ] **Step 1: Add failing summary and CSV tests**

Add a test that writes three synthetic rows through `_write_csv()` and `_write_summary()`:

```python
def test_paper_summary_reports_primary_and_feasibility_counts(tmp_path: Path):
    from argparse import Namespace

    from benchmark_compact_paper_set import _write_csv, _write_summary

    rows = [
        {"dataset": "CV", "family": "10-6", "case": "strict", "moves": 9, "final_n": 0,
         "runtime_sec": 1.0, "business_ok": True, "goal_validation_ok": True,
         "feasibility_ok": True, "time_limit_ok": True, "primary_ok": True, "full_quality_penalty": 9},
        {"dataset": "CV", "family": "10-6", "case": "over_cap", "moves": 20, "final_n": 0,
         "runtime_sec": 1.0, "business_ok": False, "goal_validation_ok": True,
         "feasibility_ok": True, "time_limit_ok": True, "primary_ok": False, "full_quality_penalty": 99},
        {"dataset": "CV", "family": "10-6", "case": "over_time", "moves": 21, "final_n": 0,
         "runtime_sec": 20.1, "business_ok": True, "goal_validation_ok": True,
         "feasibility_ok": True, "time_limit_ok": False, "primary_ok": False, "full_quality_penalty": 99},
    ]
    output = tmp_path / "paper.csv"
    args = Namespace(variant="union", sample_per_family=10, time_limit=20.0)

    _write_csv(rows, output)
    summary = _write_summary(rows, output, tmp_path, args)

    assert "| ALL | 3 | 1 | 9 |" in summary.read_text(encoding="utf-8")
    assert "| feasibility-only over cap | 1 |" in summary.read_text(encoding="utf-8")
    assert "| feasibility-only over time | 1 |" in summary.read_text(encoding="utf-8")
    assert "primary_ok" in output.read_text(encoding="utf-8").splitlines()[0]
```

- [ ] **Step 2: Run the new summary test and confirm red**

Run:

```powershell
python -m pytest tests/test_bs_frg_paper_benchmarks.py::test_paper_summary_reports_primary_and_feasibility_counts -q --basetemp .tmp_pytest_strict_summary_red
```

Expected: FAIL because the current summary has a single goal-only `solved` column and no feasibility-only breakdown.

- [ ] **Step 3: Implement dual aggregation and summary rendering**

Replace `_is_solved_row()` with a `primary_ok` predicate:

```python
def _is_solved_row(row: dict[str, object]) -> bool:
    return _is_true(row.get("primary_ok"))
```

Extend each `_summary_groups()` item with `feasibility`, `over_cap`, and `over_time` counters. For every row, increment:

```python
feasibility = _is_true(row.get("feasibility_ok"))
primary = _is_solved_row(row)
over_cap = feasibility and not _is_true(row.get("business_ok"))
over_time = feasibility and _is_true(row.get("business_ok")) and not _is_true(row.get("time_limit_ok"))
```

Use `primary` for solved moves and penalties. Preserve all-row average runtime. Change the ALL, dataset, and family headers to `primary solved` and add `feasibility`, `over cap`, and `over time` columns. Add a `## Feasibility-Only Breakdown` table with the two exclusive categories so the test's exact rows appear.

In the failure list, label a feasibility-only row as `feasibility_only` instead of presenting it as an ordinary unsolved case.

- [ ] **Step 4: Run the paper-runner test file and confirm green**

Run:

```powershell
python -m pytest tests/test_bs_frg_paper_benchmarks.py -q --basetemp .tmp_pytest_strict_summary_green
```

Expected: PASS, including the CSV header and primary-versus-feasibility summary assertions.

- [ ] **Step 5: Commit the dual summary work**

```powershell
git add scripts/benchmark_compact_paper_set.py tests/test_bs_frg_paper_benchmarks.py
git commit -m "bench: report strict paper scorecards"
```

### Task 4: Protect open-runner compatibility and run regression tests

**Files:**
- Test: `tests/test_qubo.py:8302-8333`

- [ ] **Step 1: Run the prewritten compatibility test and confirm green**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_open_row_valid_uses_primary_ok_only_when_the_field_is_present -q --basetemp .tmp_pytest_open_primary_compat
```

Expected: PASS because Task 2 added the explicit-field fallback.

- [ ] **Step 2: Run the affected regression suite**

Run:

```powershell
python -m pytest tests/test_bs_frg_paper_benchmarks.py tests/test_current_mainline_union.py tests/test_qubo.py -q --basetemp .tmp_pytest_strict_scorecard_regression
python scripts/benchmark_compact_paper_set.py --help
git diff --check
```

Expected: all tests pass, help displays `--time-limit`, and `git diff --check` has no output.

- [ ] **Step 3: Commit the compatibility coverage**

```powershell
git add tests/test_qubo.py
git commit -m "test: preserve open scorecard compatibility"
```

### Task 5: Establish the strict paired paper baseline

**Files:**
- Create: ignored `docs/experiments/strict_union_sample10_t20_jobs1_YYYYMMDD.csv`
- Create: ignored `docs/experiments/strict_union_sample10_t20_jobs8_YYYYMMDD.csv`
- Create: ignored `docs/experiments/strict_lazy_sample10_t20_jobs1_YYYYMMDD.csv`
- Create: ignored `docs/experiments/strict_lazy_sample10_t20_jobs8_YYYYMMDD.csv`

- [ ] **Step 1: Run the strict union baseline serially**

```powershell
python scripts/benchmark_compact_paper_set.py --datasets BF CV --sample-per-family 10 --variant qubo_scheduler_state_v5_qubo_diversity_union_polish --alpha 10 --time-limit 20 --jobs 1 --output docs/experiments/strict_union_sample10_t20_jobs1_YYYYMMDD.csv
```

Expected: CSV and markdown summary contain primary, feasibility, over-cap, and over-time totals.

- [ ] **Step 2: Run the strict union baseline at eight-way concurrency**

```powershell
python scripts/benchmark_compact_paper_set.py --datasets BF CV --sample-per-family 10 --variant qubo_scheduler_state_v5_qubo_diversity_union_polish --alpha 10 --time-limit 20 --jobs 8 --output docs/experiments/strict_union_sample10_t20_jobs8_YYYYMMDD.csv
```

Expected: the same dual scorecard format; BF differences from the serial run remain visible as evidence rather than being averaged away.

- [ ] **Step 3: Run the lazy portfolio only as a strict experiment**

```powershell
python scripts/benchmark_compact_paper_set.py --datasets BF CV --sample-per-family 10 --variant qubo_scheduler_state_v5_qubo_diversity_union_terminal_escape_lazy_portfolio --alpha 10 --time-limit 20 --jobs 1 --output docs/experiments/strict_lazy_sample10_t20_jobs1_YYYYMMDD.csv
python scripts/benchmark_compact_paper_set.py --datasets BF CV --sample-per-family 10 --variant qubo_scheduler_state_v5_qubo_diversity_union_terminal_escape_lazy_portfolio --alpha 10 --time-limit 20 --jobs 8 --output docs/experiments/strict_lazy_sample10_t20_jobs8_YYYYMMDD.csv
```

Expected: lazy remains an experiment; compare only `primary_ok`, feasibility-only reasons, full penalty, and portfolio runtime against the paired union baselines.

- [ ] **Step 4: Record the decision without promoting an algorithm**

Write a short ignored experiment note beside the CSVs that states the strict primary totals separately from feasibility totals and names any concurrency-sensitive BF cases. Do not alter `docs/guides/current-mainline-algorithm.md` unless a later algorithm change passes the strict promotion rule.
