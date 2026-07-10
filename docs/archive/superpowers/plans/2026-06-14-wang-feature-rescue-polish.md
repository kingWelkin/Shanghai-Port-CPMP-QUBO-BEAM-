# Wang Feature Rescue Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Wang fixed-prefix/fix-pair QUBO scoring, rescue-then-polish profiles, and focused diagnostics for the remaining open QUBO-family failures.

**Architecture:** Keep compact byte-state parent chains as the source of truth. Add Wang feature scoring as an optional linear-score modifier, expose it through explicit profiles and open-runner aliases, then add diagnostics as a reproducible experiment utility rather than production solver behavior.

**Tech Stack:** Python, argparse, pytest, compact byte-state helpers in `scripts/benchmark_compact_beam_qubo_n0.py`, open benchmark orchestration in `scripts/open_cpmp_benchmarks.py`, Wang baseline helpers in `container_reshuffle/solvers/classical_cpmp.py`.

---

## File Structure

- Modify `scripts/benchmark_compact_beam_qubo_n0.py`
  - Add Wang feature fields to result CSV metadata.
  - Add `_wang_qubo_feature_values(...)` and `_wang_feature_adjusted_group_score(...)`.
  - Wire feature-adjusted scoring into `_group_qubo_linear_score(...)`.
  - Add CLI flags and solver profiles:
    - `scheduler_state_v5_wang_features`
    - `scheduler_state_v5_wang_rescue_polish`
- Modify `scripts/open_cpmp_benchmarks.py`
  - Add variant aliases:
    - `qubo_scheduler_state_v5_wang_features`
    - `qubo_scheduler_state_v5_wang_rescue_polish`
  - Expand `wang_all` to include the new variants.
- Create `scripts/diagnose_wang_qubo_failures.py`
  - Emit structured CSV rows for exact open cases.
  - Compare compact QUBO signal fields and Wang BS-B early moves where available.
- Modify `tests/test_qubo.py`
  - Add TDD tests for Wang feature scoring, profile wiring, and open variant wiring.
- Create `tests/test_wang_failure_diagnostics.py`
  - Add TDD tests for diagnostic row creation and exact case selection.
- Modify `docs/experiments/wang_qubo_full_quality_trial_summary.md`
  - Add a short "Next probes" section after benchmarks finish.

---

### Task 1: Add Wang Feature Scoring Tests

**Files:**
- Modify: `tests/test_qubo.py`
- Later modify: `scripts/benchmark_compact_beam_qubo_n0.py`

- [ ] **Step 1: Write failing tests for score preservation and fixed-prefix preference**

Add these tests near the existing Wang/QUBO tests in `tests/test_qubo.py`:

```python
def test_wang_feature_adjustment_preserves_group_score_with_zero_weights():
    from benchmark_compact_beam_qubo_n0 import _group_qubo_linear_score
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    code, n_value = _encode([[4], [3, 2], [1]], stack_count, max_height)
    rows = [(code, (n_value, 7, b"parent"))]
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        qubo_group_lookahead_weight=0.0,
        qubo_group_branch_weight=0.0,
        wang_feature_fixed_prefix_weight=0.0,
        wang_feature_fix_pair_weight=0.0,
        wang_feature_risk_weight=0.0,
        wang_feature_same_parent_weight=0.0,
    )

    assert _group_qubo_linear_score(rows, args) == float(n_value * 1000 + 7)


def test_wang_feature_fixed_prefix_weight_lowers_better_prefix_score():
    from benchmark_compact_beam_qubo_n0 import _group_qubo_linear_score
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    better_prefix, n_value = _encode([[5, 4], [3], [2, 1]], stack_count, max_height)
    worse_prefix, same_n = _encode([[4, 5], [3], [2, 1]], stack_count, max_height)
    assert same_n == n_value
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        qubo_group_lookahead_weight=0.0,
        qubo_group_branch_weight=0.0,
        wang_feature_fixed_prefix_weight=25.0,
        wang_feature_fix_pair_weight=0.0,
        wang_feature_risk_weight=0.0,
        wang_feature_same_parent_weight=0.0,
    )

    better_score = _group_qubo_linear_score([(better_prefix, (n_value, 4, b"p1"))], args)
    worse_score = _group_qubo_linear_score([(worse_prefix, (n_value, 4, b"p2"))], args)

    assert better_score < worse_score
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_wang_feature_adjustment_preserves_group_score_with_zero_weights tests/test_qubo.py::test_wang_feature_fixed_prefix_weight_lowers_better_prefix_score -q --basetemp .pytest_tmp_wang_features_red
```

Expected result:

```text
FAILED ... unexpected keyword argument 'wang_feature_fixed_prefix_weight'
```

or:

```text
FAILED ... AssertionError
```

The failure must be due to missing Wang feature support, not an import typo.

- [ ] **Step 3: Add result metadata fields and profile defaults**

In `scripts/benchmark_compact_beam_qubo_n0.py`, add these names to `COMPACT_EXTRA_FIELDS` after `wang_pair_qubo_time_sec`:

```python
    "wang_feature_fixed_prefix_weight",
    "wang_feature_fix_pair_weight",
    "wang_feature_risk_weight",
    "wang_feature_same_parent_weight",
```

In `_SOLVER_PROFILES["speed_v2"]`, add defaults after `wang_pair_qubo_same_parent_penalty`:

```python
        "wang_feature_fixed_prefix_weight": 0.0,
        "wang_feature_fix_pair_weight": 0.0,
        "wang_feature_risk_weight": 0.0,
        "wang_feature_same_parent_weight": 0.0,
```

In `_row_dict_from_instance_result(...)`, add:

```python
        "wang_feature_fixed_prefix_weight": getattr(args, "wang_feature_fixed_prefix_weight", 0.0),
        "wang_feature_fix_pair_weight": getattr(args, "wang_feature_fix_pair_weight", 0.0),
        "wang_feature_risk_weight": getattr(args, "wang_feature_risk_weight", 0.0),
        "wang_feature_same_parent_weight": getattr(args, "wang_feature_same_parent_weight", 0.0),
```

In `main()`, add parser flags after the `--wang-pair-qubo-same-parent-penalty` argument:

```python
    parser.add_argument("--wang-feature-fixed-prefix-weight", type=float, default=0.0)
    parser.add_argument("--wang-feature-fix-pair-weight", type=float, default=0.0)
    parser.add_argument("--wang-feature-risk-weight", type=float, default=0.0)
    parser.add_argument("--wang-feature-same-parent-weight", type=float, default=0.0)
```

- [ ] **Step 4: Add feature helper and score adjustment**

Add this dataclass near other small scoring structures:

```python
@dataclass(frozen=True)
class WangQuboFeatures:
    fixed_prefix: int
    fix_pair: int
    risk: float
    same_parent: float
```

Add these helpers near `_candidate_wang_probe_score(...)`:

```python
def _wang_qubo_feature_values(
    rows: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
) -> WangQuboFeatures:
    if not rows:
        return WangQuboFeatures(0, 0, 0.0, 0.0)
    best_code, best_data = rows[0]
    _score, fix_pairs, fixed_prefix, target_gap = _candidate_wang_probe_score(best_code, args)
    parent_counts: dict[bytes, int] = {}
    for _code, data in rows:
        parent_counts[data[2]] = parent_counts.get(data[2], 0) + 1
    same_parent = float(max(parent_counts.values(), default=1) - 1)
    risk = float(target_gap)
    return WangQuboFeatures(
        fixed_prefix=int(fixed_prefix),
        fix_pair=int(fix_pairs),
        risk=risk,
        same_parent=same_parent,
    )


def _wang_feature_adjusted_group_score(
    base_score: float,
    rows: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
) -> float:
    fixed_weight = float(getattr(args, "wang_feature_fixed_prefix_weight", 0.0) or 0.0)
    pair_weight = float(getattr(args, "wang_feature_fix_pair_weight", 0.0) or 0.0)
    risk_weight = float(getattr(args, "wang_feature_risk_weight", 0.0) or 0.0)
    parent_weight = float(getattr(args, "wang_feature_same_parent_weight", 0.0) or 0.0)
    if fixed_weight == 0.0 and pair_weight == 0.0 and risk_weight == 0.0 and parent_weight == 0.0:
        return base_score
    features = _wang_qubo_feature_values(rows, args)
    return (
        base_score
        - fixed_weight * float(features.fixed_prefix)
        - pair_weight * float(features.fix_pair)
        + risk_weight * float(features.risk)
        + parent_weight * float(features.same_parent)
    )
```

Change the end of `_group_qubo_linear_score(...)` from:

```python
    return score
```

to:

```python
    return _wang_feature_adjusted_group_score(score, rows, args)
```

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_wang_feature_adjustment_preserves_group_score_with_zero_weights tests/test_qubo.py::test_wang_feature_fixed_prefix_weight_lowers_better_prefix_score -q --basetemp .pytest_tmp_wang_features_green
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit Task 1**

Run:

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py tests\test_qubo.py
git commit -m "Add Wang feature scoring for group QUBO"
```

---

### Task 2: Add Wang Feature and Rescue-Polish Profiles

**Files:**
- Modify: `tests/test_qubo.py`
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `scripts/open_cpmp_benchmarks.py`

- [ ] **Step 1: Write failing profile tests**

Add these tests near existing solver-profile tests in `tests/test_qubo.py`:

```python
def test_solver_profile_scheduler_state_v5_wang_features_enables_generic_features(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_wang_features")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v5_wang_features"
    assert args.scheduler_state_controller == "state_v2"
    assert args.group_qubo_supplement is True
    assert args.wang_feature_fixed_prefix_weight > 0.0
    assert args.wang_feature_fix_pair_weight > 0.0
    assert args.wang_pair_qubo_enabled is False


def test_solver_profile_scheduler_state_v5_wang_rescue_polish_combines_rescue_and_window_polish(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_wang_rescue_polish")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.rescue_enabled is True
    assert args.wang_pair_qubo_enabled is True
    assert args.post_polish_qubo is True
    assert args.post_polish_mode == "qubo_windows_v2"
    assert args.post_polish_accept == "better"
    assert args.post_polish_window_select >= 1
```

- [ ] **Step 2: Run profile tests to verify RED**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_solver_profile_scheduler_state_v5_wang_features_enables_generic_features tests/test_qubo.py::test_solver_profile_scheduler_state_v5_wang_rescue_polish_combines_rescue_and_window_polish -q --basetemp .pytest_tmp_wang_profiles_red
```

Expected:

```text
FAILED ... invalid choice: 'scheduler_state_v5_wang_features'
```

or a `KeyError`/assertion showing the profiles do not exist yet.

- [ ] **Step 3: Add compact solver profiles**

In `_SOLVER_PROFILES`, add:

```python
    "scheduler_state_v5_wang_features": {
        "extends": "scheduler_state_v5",
        "group_qubo_supplement": True,
        "max_group_qubo_calls": 64,
        "qubo_group_max_vars": 96,
        "qubo_group_select": 18,
        "qubo_group_take": 3,
        "qubo_group_lookahead_weight": 0.25,
        "qubo_group_branch_weight": 0.05,
        "group_qubo_risk_gate": True,
        "group_qubo_risk_min_candidates": 12000,
        "group_qubo_risk_parent_share": 0.18,
        "group_qubo_risk_plateau_window": 3,
        "group_qubo_risk_plateau_delta": 0,
        "wang_feature_fixed_prefix_weight": 18.0,
        "wang_feature_fix_pair_weight": 8.0,
        "wang_feature_risk_weight": 1.0,
        "wang_feature_same_parent_weight": 0.35,
        "wang_pair_qubo_enabled": False,
    },
    "scheduler_state_v5_wang_rescue_polish": {
        "extends": "scheduler_state_v5_wang_rescue",
        "post_polish_qubo": True,
        "post_polish_mode": "qubo_windows_v2",
        "post_polish_accept": "better",
        "post_polish_passes": 3,
        "post_polish_suffix_max_candidates": 72,
        "post_polish_suffix_select": 7,
        "post_polish_suffix_max_qubo_vars": 64,
        "post_polish_window_select": 2,
        "post_polish_window_max_size": 8,
        "post_polish_window_budget": 96,
        "post_polish_window_qubo_vars": 72,
        "post_polish_window_repair_expanded": 6000,
        "post_polish_window_repair_depth_extra": 0,
    },
```

- [ ] **Step 4: Run profile tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_solver_profile_scheduler_state_v5_wang_features_enables_generic_features tests/test_qubo.py::test_solver_profile_scheduler_state_v5_wang_rescue_polish_combines_rescue_and_window_polish -q --basetemp .pytest_tmp_wang_profiles_green
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Write failing open-runner variant tests**

Add to `tests/test_qubo.py` near `test_open_runner_accepts_experimental_solver_profiles`:

```python
def test_open_runner_accepts_wang_feature_and_rescue_polish_variants():
    from open_cpmp_benchmarks import _args_for_variant, _variant_names

    outer = argparse.Namespace(
        adaptive_candidate_pool=0,
        adaptive_high_frontier=800,
        adaptive_high_n=16,
        adaptive_low_frontier=1200,
        adaptive_low_n=8,
        adaptive_mid_frontier=1000,
        alpha=10.0,
        max_moves=None,
        solver_profile="speed_v2",
        time_limit=10.0,
        no_qubo_beam_width=1200,
    )

    feature_args = _args_for_variant(outer, "qubo_scheduler_state_v5_wang_features")
    polish_args = _args_for_variant(outer, "qubo_scheduler_state_v5_wang_rescue_polish")

    assert feature_args.solver_profile == "scheduler_state_v5_wang_features"
    assert polish_args.solver_profile == "scheduler_state_v5_wang_rescue_polish"
    assert "qubo_scheduler_state_v5_wang_features" in _variant_names("wang_all")
    assert "qubo_scheduler_state_v5_wang_rescue_polish" in _variant_names("wang_all")
```

- [ ] **Step 6: Run open-runner test to verify RED**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_open_runner_accepts_wang_feature_and_rescue_polish_variants -q --basetemp .pytest_tmp_wang_open_variants_red
```

Expected:

```text
FAILED ... AssertionError
```

or a parser/variant failure showing the aliases are not wired yet.

- [ ] **Step 7: Add open runner aliases**

In `scripts/open_cpmp_benchmarks.py`, update `_variant_names("wang_all")` to include:

```python
            "qubo_scheduler_state_v5_wang_features",
            "qubo_scheduler_state_v5_wang_rescue_polish",
```

In `_args_for_variant(...)`, add:

```python
    if variant == "qubo_scheduler_state_v5_wang_features":
        overrides.update({"solver_profile": "scheduler_state_v5_wang_features"})
    if variant == "qubo_scheduler_state_v5_wang_rescue_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_wang_rescue_polish"})
```

In the `--variant` choices list, add:

```python
            "qubo_scheduler_state_v5_wang_features",
            "qubo_scheduler_state_v5_wang_rescue_polish",
```

- [ ] **Step 8: Run open-runner test to verify GREEN**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_open_runner_accepts_wang_feature_and_rescue_polish_variants -q --basetemp .pytest_tmp_wang_open_variants_green
```

Expected:

```text
1 passed
```

- [ ] **Step 9: Commit Task 2**

Run:

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py tests\test_qubo.py
git commit -m "Add Wang feature and rescue polish profiles"
```

---

### Task 3: Add Focused Failure Autopsy Utility

**Files:**
- Create: `scripts/diagnose_wang_qubo_failures.py`
- Create: `tests/test_wang_failure_diagnostics.py`

- [ ] **Step 1: Write failing tests for diagnostic row creation**

Create `tests/test_wang_failure_diagnostics.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_case_selector_accepts_exact_open_case_ids():
    from diagnose_wang_qubo_failures import _case_selected

    assert _case_selected("5x5_cv1_test_4", {"5x5_cv1_test_4"})
    assert not _case_selected("5x5_cv1_test_5", {"5x5_cv1_test_4"})
    assert _case_selected("5x5_cv1_test_6", set())


def test_diagnostic_row_degrades_when_wang_data_is_missing():
    from diagnose_wang_qubo_failures import _diagnostic_row

    row = _diagnostic_row(
        case_id="5x5_cv1_test_4",
        family="5x5_cv1_test",
        compact_status="NO_FEASIBLE_FOUND",
        compact_final_n=7,
        compact_moves=None,
        compact_best_n_history=[12, 10, 10, 10, 9],
        fixed_prefix=4,
        fix_pair=2,
        parent_top_share=0.42,
        wang_moves=None,
        wang_first_moves=[],
    )

    assert row["case_id"] == "5x5_cv1_test_4"
    assert row["compact_plateau_tail"] == "10,10,9"
    assert row["wang_moves"] == ""
    assert row["wang_first_moves"] == ""
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests/test_wang_failure_diagnostics.py -q --basetemp .pytest_tmp_wang_diag_red
```

Expected:

```text
FAILED ... ModuleNotFoundError: No module named 'diagnose_wang_qubo_failures'
```

- [ ] **Step 3: Create the diagnostic script**

Create `scripts/diagnose_wang_qubo_failures.py`:

```python
"""Focused diagnostics for Wang-vs-QUBO open CPMP failure cases."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

DEFAULT_CASES = ("5x5_cv1_test_4", "5x5_cv1_test_6")
FIELDS = [
    "case_id",
    "family",
    "compact_status",
    "compact_final_n",
    "compact_moves",
    "compact_plateau_tail",
    "fixed_prefix",
    "fix_pair",
    "parent_top_share",
    "wang_moves",
    "wang_first_moves",
]


def _case_selected(case_id: str, exact_cases: set[str]) -> bool:
    return not exact_cases or case_id in exact_cases


def _plateau_tail(history: Iterable[int], width: int = 3) -> str:
    values = list(history)[-max(1, width):]
    return ",".join(str(value) for value in values)


def _diagnostic_row(
    *,
    case_id: str,
    family: str,
    compact_status: str,
    compact_final_n: int | None,
    compact_moves: int | None,
    compact_best_n_history: list[int],
    fixed_prefix: int,
    fix_pair: int,
    parent_top_share: float,
    wang_moves: int | None,
    wang_first_moves: list[str],
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "family": family,
        "compact_status": compact_status,
        "compact_final_n": "" if compact_final_n is None else compact_final_n,
        "compact_moves": "" if compact_moves is None else compact_moves,
        "compact_plateau_tail": _plateau_tail(compact_best_n_history),
        "fixed_prefix": fixed_prefix,
        "fix_pair": fix_pair,
        "parent_top_share": f"{parent_top_share:.6f}",
        "wang_moves": "" if wang_moves is None else wang_moves,
        "wang_first_moves": ";".join(wang_first_moves),
    }


def _write_rows(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--output", type=Path, default=Path("docs/experiments/wang_qubo_failure_autopsy.csv"))
    args = parser.parse_args()
    exact = set(args.cases)
    rows = [
        _diagnostic_row(
            case_id=case_id,
            family=case_id.rsplit("_", 1)[0],
            compact_status="diagnostic_not_run",
            compact_final_n=None,
            compact_moves=None,
            compact_best_n_history=[],
            fixed_prefix=0,
            fix_pair=0,
            parent_top_share=0.0,
            wang_moves=None,
            wang_first_moves=[],
        )
        for case_id in DEFAULT_CASES
        if _case_selected(case_id, exact)
    ]
    _write_rows(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_wang_failure_diagnostics.py -q --basetemp .pytest_tmp_wang_diag_green
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add scripts\diagnose_wang_qubo_failures.py tests\test_wang_failure_diagnostics.py
git commit -m "Add Wang QUBO failure diagnostics scaffold"
```

---

### Task 4: Connect Diagnostics to Real Open Runs

**Files:**
- Modify: `scripts/diagnose_wang_qubo_failures.py`
- Modify: `tests/test_wang_failure_diagnostics.py`

- [ ] **Step 1: Write failing tests for first-move formatting and CSV write**

Append to `tests/test_wang_failure_diagnostics.py`:

```python
def test_format_first_moves_keeps_stack_and_box_fields():
    from container_reshuffle.core import Move
    from diagnose_wang_qubo_failures import _format_first_moves

    moves = [
        Move(step=1, box=7, from_stack=0, to_stack=2, from_level=3, to_level=1),
        Move(step=2, box=5, from_stack=1, to_stack=0, from_level=2, to_level=4),
    ]

    assert _format_first_moves(moves, limit=2) == ["1:7:0->2", "2:5:1->0"]
```

- [ ] **Step 2: Run the new test to verify RED**

Run:

```powershell
python -m pytest tests/test_wang_failure_diagnostics.py::test_format_first_moves_keeps_stack_and_box_fields -q --basetemp .pytest_tmp_wang_diag_real_red
```

Expected:

```text
FAILED ... ImportError: cannot import name '_format_first_moves'
```

- [ ] **Step 3: Add first-move formatter and real-run hooks**

In `scripts/diagnose_wang_qubo_failures.py`, add imports:

```python
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_compact_beam_qubo_n0 import make_compact_args, run_compact_beam_instance
from container_reshuffle.core import Move
from container_reshuffle.features import future_disorder_count_stacks
from container_reshuffle.solvers.base import SolverConfig
from container_reshuffle.solvers.classical_cpmp import WangBsBabyMovesSolver
from open_cpmp_benchmarks import discover_dlts_cases, ensure_dlts_source, load_dlts_dat
```

Add:

```python
def _format_first_moves(moves: list[Move], limit: int = 8) -> list[str]:
    return [
        f"{move.step}:{move.box}:{move.from_stack}->{move.to_stack}"
        for move in moves[: max(0, limit)]
    ]
```

Replace `main()` row construction with real case discovery:

```python
    parser.add_argument("--source", type=Path, default=ROOT / "external" / "DLTS")
    parser.add_argument("--families", nargs="+", default=["5x5_cv1_test"])
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--wang-move-limit", type=int, default=8)
```

Inside `main()` after parsing:

```python
    source = ensure_dlts_source(args.source, auto_clone=False, refresh=False)
    cases = discover_dlts_cases(source, families=args.families, limit_per_family=None, max_total=None)
    exact = set(args.cases)
    rows: list[dict[str, object]] = []
    for case in cases:
        if not _case_selected(case.case_id, exact):
            continue
        instance = load_dlts_dat(case.instance_path, max_height=case.reference.max_height)
        compact_args = make_compact_args(
            solver_profile="scheduler_state_v5_wang_rescue",
            alpha=10.0,
            time_limit=args.time_limit,
            max_moves=None,
        )
        compact_result = run_compact_beam_instance(case.case_id, instance, compact_args)
        wang_solution = WangBsBabyMovesSolver().solve(
            instance,
            SolverConfig(
                max_depth=200,
                beam_width=100,
                max_candidates_per_layer=64,
                time_limit_sec=args.time_limit,
            ),
        )
        rows.append(
            _diagnostic_row(
                case_id=case.case_id,
                family=case.family,
                compact_status=compact_result.status,
                compact_final_n=(0 if compact_result.moves else future_disorder_count_stacks(instance.stacks)),
                compact_moves=(len(compact_result.moves) if compact_result.moves else None),
                compact_best_n_history=list(getattr(compact_args, "_best_n_history", [])),
                fixed_prefix=0,
                fix_pair=0,
                parent_top_share=float(getattr(compact_args, "_layer_parent_top_share", 0.0)),
                wang_moves=len(wang_solution.moves) if wang_solution.moves else None,
                wang_first_moves=_format_first_moves(wang_solution.moves, limit=args.wang_move_limit),
            )
        )
```

Keep the scaffold behavior by allowing `--source` to point at an existing DLTS source; if the source is absent, the script should fail clearly because this diagnostic is not allowed to silently network-clone in tests.

- [ ] **Step 4: Run diagnostics tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_wang_failure_diagnostics.py -q --basetemp .pytest_tmp_wang_diag_real_green
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Smoke the diagnostic CLI without network**

Run:

```powershell
python scripts\diagnose_wang_qubo_failures.py --source external\DLTS --families 5x5_cv1_test --cases 5x5_cv1_test_4 5x5_cv1_test_6 --time-limit 1 --output docs\experiments\scratch_wang_qubo_failure_autopsy.csv
```

Expected:

```text
exit code 0
```

If `external\DLTS` is absent, stop and report that the diagnostic CLI requires the existing local DLTS source. Do not enable auto-clone in this task.

- [ ] **Step 6: Commit Task 4**

Run:

```powershell
git add scripts\diagnose_wang_qubo_failures.py tests\test_wang_failure_diagnostics.py
git commit -m "Connect Wang failure diagnostics to open cases"
```

---

### Task 5: Benchmark and Documentation

**Files:**
- Modify: `docs/experiments/wang_qubo_full_quality_trial_summary.md`
- Optionally create ignored local CSVs under `docs/experiments/`

- [ ] **Step 1: Run focused test suite**

Run:

```powershell
python -m pytest tests/test_qubo.py tests/test_wang_failure_diagnostics.py tests/test_classical_cpmp_baselines.py tests/test_classical_cpmp_runner.py -q --basetemp .pytest_tmp_wang_feature_full
```

Expected:

```text
passed
```

Record the exact pass count in the final summary.

- [ ] **Step 2: Run syntax check**

Run:

```powershell
python -m py_compile scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py scripts\diagnose_wang_qubo_failures.py scripts\benchmark_classical_cpmp_baselines.py container_reshuffle\solvers\classical_cpmp.py
```

Expected: exit code `0`.

- [ ] **Step 3: Run small open smoke for new variants**

Run:

```powershell
python scripts\open_cpmp_benchmarks.py --families 5x5_cv1_test --limit-per-family 2 --variant qubo_scheduler_state_v5_wang_features --time-limit 3 --output docs\experiments\scratch_wang_features_smoke.csv
python scripts\open_cpmp_benchmarks.py --families 5x5_cv1_test --limit-per-family 2 --variant qubo_scheduler_state_v5_wang_rescue_polish --time-limit 3 --output docs\experiments\scratch_wang_rescue_polish_smoke.csv
```

Expected: each command exits `0` and writes its ignored CSV plus summary markdown.

- [ ] **Step 4: Update experiment summary**

Append this section to `docs/experiments/wang_qubo_full_quality_trial_summary.md`:

```markdown
## Follow-Up Probes

The next Wang/QUBO probes are intentionally separated:

- `scheduler_state_v5_wang_features` moves fixed-prefix and fix-pair signals into the generic group-QUBO linear score.
- `scheduler_state_v5_wang_rescue_polish` keeps Wang rescue as a feasibility mechanism, then uses suffix/window polish to reduce long rescue paths.
- `scripts/diagnose_wang_qubo_failures.py` focuses on `5x5_cv1_test_4` and `5x5_cv1_test_6`, comparing compact plateau signals with Wang BS-B early moves.

These probes should be judged by full-case quality first: valid count, invalid count, full-quality penalty, solved moves, and runtime.
```

- [ ] **Step 5: Run whitespace check**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors. CRLF warnings and user-level git ignore permission warnings may appear and should be reported as warnings, not failures.

- [ ] **Step 6: Commit Task 5**

Run:

```powershell
git add docs\experiments\wang_qubo_full_quality_trial_summary.md
git commit -m "Document Wang feature follow-up probes"
```

---

## Final Verification

- [ ] **Step 1: Run final tests**

Run:

```powershell
python -m pytest tests/test_qubo.py tests/test_wang_failure_diagnostics.py tests/test_classical_cpmp_baselines.py tests/test_classical_cpmp_runner.py tests/test_dlts_original_runner.py tests/test_paper_bbs_probe.py -q --basetemp .pytest_tmp_wang_feature_final
```

Expected: all selected tests pass.

- [ ] **Step 2: Run final syntax check**

Run:

```powershell
python -m py_compile scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py scripts\diagnose_wang_qubo_failures.py scripts\benchmark_classical_cpmp_baselines.py scripts\benchmark_qubo_no_regret_portfolio_n0.py container_reshuffle\solvers\classical_cpmp.py
```

Expected: exit code `0`.

- [ ] **Step 3: Check repository status**

Run:

```powershell
git status --short --untracked-files=all
```

Expected: no uncommitted implementation files. The local `AGENTS.md` may remain untracked if it is still intentionally kept out of repository commits.
