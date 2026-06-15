# Feasibility DLTS Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an experimental feasibility-escape profile and DLTS-ready feature export path on top of the current compact QUBO CPMP solver.

**Architecture:** Keep `scripts/benchmark_compact_beam_qubo_n0.py` as the compact search owner, but move reusable feasibility and DLTS signal math into `container_reshuffle/solvers/feasibility_signals.py`. The new profile extends `scheduler_state_v5_wang_rescue_polish`; it adds only legal compact successor states and never splices Wang or DLTS move paths.

**Tech Stack:** Python dataclasses, pytest, existing compact byte-state helpers in `scripts/benchmark_compact_beam_qubo_n0.py`, open benchmark orchestration in `scripts/open_cpmp_benchmarks.py`, CSV experiment outputs under `docs/experiments/`.

---

## Scope Check

This plan implements one coherent vertical slice:

- feasibility signal computation,
- explicit profile and runner surface,
- bounded compact escape selection,
- DLTS-ready feature export,
- focused verification and experiment documentation.

It does not train a neural model, add ML dependencies, or make the new profile a default.

## File Structure

- Create `container_reshuffle/solvers/feasibility_signals.py`
  - Owns layout-only signal computation, escape scoring, and deterministic DLTS policy/value adapter.
- Create `tests/test_feasibility_signals.py`
  - Tests the signal module without importing benchmark runners.
- Modify `scripts/benchmark_compact_beam_qubo_n0.py`
  - Adds profile defaults, CLI flags, result fields, escape selector, and DLTS feature export.
- Modify `scripts/open_cpmp_benchmarks.py`
  - Adds the open runner alias `qubo_scheduler_state_v5_feasibility_escape_polish`.
- Modify `tests/test_qubo.py`
  - Adds profile, runner alias, selector wiring, and CSV serialization tests.
- Modify `docs/experiments/wang_qubo_full_quality_trial_summary.md`
  - Adds a short curated note after verification.

---

### Task 1: Add Feasibility Signal Module

**Files:**
- Create: `container_reshuffle/solvers/feasibility_signals.py`
- Create: `tests/test_feasibility_signals.py`

- [ ] **Step 1: Write failing tests for layout signals and deterministic adapter**

Create `tests/test_feasibility_signals.py`:

```python
from __future__ import annotations

from container_reshuffle.solvers.feasibility_signals import (
    DltsSignalAdapter,
    DltsStateFeatures,
    compute_feasibility_signals,
    feasibility_escape_score,
)


def test_feasibility_signals_capture_stability_slack_and_dead_end_risk():
    signals = compute_feasibility_signals(
        stacks=[[5, 4, 1], [3, 6], [2], []],
        max_height=3,
    )

    assert signals.stable_prefix_total == 5
    assert signals.empty_slots == 5
    assert signals.blocked_fixed_candidates == 1
    assert signals.temporary_slack_moves > 0
    assert signals.low_tier_risk > 0
    assert 0.0 <= signals.dead_end_risk <= 1.0


def test_feasibility_escape_score_prefers_slack_without_low_tier_damage():
    risky = compute_feasibility_signals(
        stacks=[[5, 4, 3], [2, 6], []],
        max_height=3,
    )
    flexible = compute_feasibility_signals(
        stacks=[[5, 4], [3, 1], [2]],
        max_height=3,
    )

    assert feasibility_escape_score(flexible, n_value=2, depth=10) < feasibility_escape_score(
        risky,
        n_value=2,
        depth=10,
    )


def test_dlts_signal_adapter_policy_and_value_are_deterministic():
    signals = compute_feasibility_signals(
        stacks=[[5, 4], [3, 1], [2]],
        max_height=3,
    )
    features = DltsStateFeatures(
        n_value=2,
        depth=7,
        parent_rank=3,
        selected=False,
        signals=signals,
    )
    adapter = DltsSignalAdapter()

    assert adapter.policy_score(features) == adapter.policy_score(features)
    assert adapter.value_estimate(features) >= 0.0
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests/test_feasibility_signals.py -q --basetemp .pytest_tmp_feas_signals_red -p no:cacheprovider
```

Expected:

```text
FAILED ... ModuleNotFoundError: No module named 'container_reshuffle.solvers.feasibility_signals'
```

- [ ] **Step 3: Create the signal module**

Create `container_reshuffle/solvers/feasibility_signals.py`:

```python
"""Feasibility and DLTS-ready signals for CPMP layouts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeasibilitySignals:
    fixed_prefix_total: int
    stable_prefix_total: int
    target_gap: int
    empty_slots: int
    blocked_fixed_candidates: int
    temporary_slack_moves: int
    low_tier_risk: int
    dead_end_risk: float


@dataclass(frozen=True)
class DltsStateFeatures:
    n_value: int
    depth: int
    parent_rank: int
    selected: bool
    signals: FeasibilitySignals


def _stable_prefix_len(stack: list[int]) -> int:
    if not stack:
        return 0
    prefix = 1
    for idx in range(1, len(stack)):
        if stack[idx - 1] > stack[idx]:
            prefix += 1
        else:
            break
    return prefix


def _future_disorder_count(stacks: list[list[int]]) -> int:
    total = 0
    for stack in stacks:
        for idx in range(1, len(stack)):
            if stack[idx] > stack[idx - 1]:
                total += len(stack) - idx
                break
    return total


def _move_top(stacks: list[list[int]], source: int, target: int) -> list[list[int]]:
    copied = [list(stack) for stack in stacks]
    copied[target].append(copied[source].pop())
    return copied


def compute_feasibility_signals(
    stacks: list[list[int]],
    max_height: int,
    target_gap: int = 0,
) -> FeasibilitySignals:
    stable_lengths = [_stable_prefix_len(stack) for stack in stacks]
    stable_prefix_total = sum(stable_lengths)
    empty_slots = sum(max(0, max_height - len(stack)) for stack in stacks)
    current_n = _future_disorder_count(stacks)
    blocked_fixed_candidates = 0
    temporary_slack_moves = 0
    low_tier_risk = 0

    for source, stack in enumerate(stacks):
        if not stack:
            continue
        if stable_lengths[source] < len(stack):
            blocked_fixed_candidates += len(stack) - stable_lengths[source]
        source_top_is_stable = stable_lengths[source] == len(stack) and len(stack) > 1
        for target, dst in enumerate(stacks):
            if source == target or len(dst) >= max_height:
                continue
            moved = _move_top(stacks, source, target)
            if _future_disorder_count(moved) <= current_n:
                temporary_slack_moves += 1
            if source_top_is_stable:
                low_tier_risk += 1

    capacity = max(1, len(stacks) * max_height)
    n_pressure = min(1.0, float(current_n) / float(capacity))
    slack_pressure = 1.0 - min(1.0, float(empty_slots) / float(max(1, len(stacks))))
    escape_pressure = 1.0 if temporary_slack_moves == 0 and current_n > 0 else 0.0
    dead_end_risk = max(0.0, min(1.0, 0.45 * n_pressure + 0.35 * slack_pressure + 0.20 * escape_pressure))

    return FeasibilitySignals(
        fixed_prefix_total=stable_prefix_total,
        stable_prefix_total=stable_prefix_total,
        target_gap=max(0, int(target_gap)),
        empty_slots=empty_slots,
        blocked_fixed_candidates=blocked_fixed_candidates,
        temporary_slack_moves=temporary_slack_moves,
        low_tier_risk=low_tier_risk,
        dead_end_risk=dead_end_risk,
    )


def feasibility_escape_score(signals: FeasibilitySignals, n_value: int, depth: int) -> float:
    return (
        float(n_value) * 1000.0
        + float(depth)
        + signals.dead_end_risk * 120.0
        + float(signals.low_tier_risk) * 25.0
        + float(signals.target_gap) * 2.0
        - float(signals.temporary_slack_moves) * 30.0
        - float(signals.stable_prefix_total) * 3.0
        - float(signals.blocked_fixed_candidates) * 2.0
    )


class DltsSignalAdapter:
    def policy_score(self, features: DltsStateFeatures) -> float:
        return feasibility_escape_score(features.signals, features.n_value, features.depth) + float(features.parent_rank)

    def value_estimate(self, features: DltsStateFeatures) -> float:
        signals = features.signals
        value = (
            float(features.n_value)
            + signals.dead_end_risk * 3.0
            + float(signals.target_gap) * 0.10
            - float(signals.temporary_slack_moves) * 0.25
            - float(signals.stable_prefix_total) * 0.05
        )
        return max(0.0, value)
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_feasibility_signals.py -q --basetemp .pytest_tmp_feas_signals_green -p no:cacheprovider
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add container_reshuffle\solvers\feasibility_signals.py tests\test_feasibility_signals.py
git commit -m "Add CPMP feasibility signal helpers"
```

---

### Task 2: Add Profile And Open Runner Surface

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `scripts/open_cpmp_benchmarks.py`
- Modify: `tests/test_qubo.py`

- [ ] **Step 1: Write failing profile and open-runner tests**

Add to `tests/test_qubo.py` near existing Wang profile tests:

```python
def test_solver_profile_scheduler_state_v5_feasibility_escape_polish_extends_rescue_polish(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_feasibility_escape_polish")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v5_feasibility_escape_polish"
    assert args.rescue_enabled is True
    assert args.wang_pair_qubo_enabled is True
    assert args.post_polish_qubo is True
    assert args.post_polish_mode == "qubo_windows_v2"
    assert args.feasibility_escape_enabled is True
    assert args.feasibility_escape_supplement_width > 0
    assert args.feasibility_escape_dead_end_threshold > 0.0


def test_open_runner_accepts_feasibility_escape_variant():
    from open_cpmp_benchmarks import _args_for_variant, _variant_names, build_parser

    outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v5_feasibility_escape_polish"])
    args = _args_for_variant(outer, "qubo_scheduler_state_v5_feasibility_escape_polish")

    assert args.solver_profile == "scheduler_state_v5_feasibility_escape_polish"
    assert "qubo_scheduler_state_v5_feasibility_escape_polish" in _variant_names("wang_research_all")


def test_open_runner_case_ids_filter_accepts_full_or_numeric_ids():
    from open_cpmp_benchmarks import _case_id_selected, build_parser

    args = build_parser().parse_args(["--case-ids", "4,5x5_cv2_test_10"])

    assert _case_id_selected("5x5_cv1_test_4", args.case_ids)
    assert _case_id_selected("5x5_cv2_test_10", args.case_ids)
    assert not _case_id_selected("5x5_cv1_test_5", args.case_ids)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_solver_profile_scheduler_state_v5_feasibility_escape_polish_extends_rescue_polish tests/test_qubo.py::test_open_runner_accepts_feasibility_escape_variant tests/test_qubo.py::test_open_runner_case_ids_filter_accepts_full_or_numeric_ids -q --basetemp .pytest_tmp_feas_profile_red -p no:cacheprovider
```

Expected:

```text
FAILED ... invalid choice: 'scheduler_state_v5_feasibility_escape_polish'
```

or a missing variant/helper assertion.

- [ ] **Step 3: Add compact defaults, profile fields, and parser flags**

In `scripts/benchmark_compact_beam_qubo_n0.py`, add these names to `COMPACT_EXTRA_FIELDS` after `unified_wang_fix_pair_selected`:

```python
    "feasibility_escape_enabled",
    "feasibility_escape_supplement_width",
    "feasibility_escape_focus_n",
    "feasibility_escape_n_window",
    "feasibility_escape_dead_end_threshold",
    "feasibility_escape_min_temporary_slack",
    "feasibility_escape_calls",
    "feasibility_escape_added_total",
```

In `_SOLVER_PROFILES["speed_v2"]`, add after unified Wang defaults:

```python
        "feasibility_escape_enabled": False,
        "feasibility_escape_supplement_width": 0,
        "feasibility_escape_focus_n": 8,
        "feasibility_escape_n_window": 1,
        "feasibility_escape_dead_end_threshold": 0.45,
        "feasibility_escape_min_temporary_slack": 1,
```

After `_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"]`, add:

```python
_SOLVER_PROFILES["scheduler_state_v5_feasibility_escape_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "feasibility_escape_enabled": True,
    "feasibility_escape_supplement_width": 4,
    "feasibility_escape_focus_n": -1,
    "feasibility_escape_n_window": 2,
    "feasibility_escape_dead_end_threshold": 0.35,
    "feasibility_escape_min_temporary_slack": 1,
}
```

Add parser flags near the Wang/unified parser flags:

```python
    parser.add_argument("--feasibility-escape-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--feasibility-escape-supplement-width", type=int)
    parser.add_argument("--feasibility-escape-focus-n", type=int)
    parser.add_argument("--feasibility-escape-n-window", type=int)
    parser.add_argument("--feasibility-escape-dead-end-threshold", type=float)
    parser.add_argument("--feasibility-escape-min-temporary-slack", type=int)
```

- [ ] **Step 4: Add result fields and row serialization**

In `CompactBeamResult`, add after `unified_wang_fix_pair_selected`:

```python
    feasibility_escape_calls: int = 0
    feasibility_escape_added_total: int = 0
```

In `_row_dict_from_instance_result(...)`, add after unified Wang result fields:

```python
            "feasibility_escape_enabled": getattr(args, "feasibility_escape_enabled", False),
            "feasibility_escape_supplement_width": getattr(args, "feasibility_escape_supplement_width", 0),
            "feasibility_escape_focus_n": getattr(args, "feasibility_escape_focus_n", 8),
            "feasibility_escape_n_window": getattr(args, "feasibility_escape_n_window", 1),
            "feasibility_escape_dead_end_threshold": getattr(args, "feasibility_escape_dead_end_threshold", 0.45),
            "feasibility_escape_min_temporary_slack": getattr(args, "feasibility_escape_min_temporary_slack", 1),
            "feasibility_escape_calls": result.feasibility_escape_calls,
            "feasibility_escape_added_total": result.feasibility_escape_added_total,
```

In `tests/test_qubo.py`, update `_compact_args(...)` defaults with the same six `feasibility_escape_*` settings used in `speed_v2`.

- [ ] **Step 5: Add open-runner alias**

In `scripts/open_cpmp_benchmarks.py`, add `"qubo_scheduler_state_v5_feasibility_escape_polish"` to:

- the `build_parser()` `--variant` choices,
- `_variant_names("wang_research_all")`.

In `_args_for_variant(...)`, add:

```python
    if variant == "qubo_scheduler_state_v5_feasibility_escape_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_feasibility_escape_polish"})
```

Add these fields to the existing pass-through loop near the Wang/unified fields:

```python
        "feasibility_escape_enabled",
        "feasibility_escape_supplement_width",
        "feasibility_escape_focus_n",
        "feasibility_escape_n_window",
        "feasibility_escape_dead_end_threshold",
        "feasibility_escape_min_temporary_slack",
```

Add this helper near `_variant_names(...)`:

```python
def _case_id_selected(case_id: str, case_ids_csv: str) -> bool:
    requested = {part.strip() for part in str(case_ids_csv or "").split(",") if part.strip()}
    if not requested:
        return True
    numeric = case_id.rsplit("_", 1)[-1]
    return case_id in requested or numeric in requested
```

In `build_parser()`, add after `--limit-per-family`:

```python
    parser.add_argument("--case-ids", default="")
```

In `run_open_benchmark(...)`, after `discover_dlts_cases(...)`, add:

```python
    cases = [case for case in cases if _case_id_selected(case.case_id, args.case_ids)]
```

- [ ] **Step 6: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_solver_profile_scheduler_state_v5_feasibility_escape_polish_extends_rescue_polish tests/test_qubo.py::test_open_runner_accepts_feasibility_escape_variant tests/test_qubo.py::test_open_runner_case_ids_filter_accepts_full_or_numeric_ids -q --basetemp .pytest_tmp_feas_profile_green -p no:cacheprovider
```

Expected:

```text
3 passed
```

- [ ] **Step 7: Commit Task 2**

Run:

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py tests\test_qubo.py
git commit -m "Add feasibility escape profile surface"
```

---

### Task 3: Wire Bounded Feasibility Escape Into Compact Frontier

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `tests/test_qubo.py`

- [ ] **Step 1: Write failing selector unit test**

Add to `tests/test_qubo.py` near `_select_enhanced_qubo_frontier` tests:

```python
def test_feasibility_escape_supplement_prefers_lower_escape_score(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as bench
    from container_reshuffle.solvers.feasibility_signals import FeasibilitySignals

    low = b"low"
    high = b"high"
    args = _compact_args(
        feasibility_escape_min_temporary_slack=1,
    )

    def fake_signals(code, selector_args):
        if code == low:
            return FeasibilitySignals(3, 3, 0, 2, 1, 3, 0, 0.20)
        return FeasibilitySignals(1, 1, 4, 0, 0, 0, 3, 0.80)

    monkeypatch.setattr(bench, "_feasibility_signals_for_code", fake_signals)

    selected = bench._select_feasibility_escape_supplement(
        [(high, (2, 7, b"p0")), (low, (2, 7, b"p1"))],
        args,
        target=1,
    )

    assert selected == [low]
```

- [ ] **Step 2: Write failing frontier wiring test**

Add to `tests/test_qubo.py`:

```python
def test_feasibility_escape_selector_is_used_inside_enhanced_frontier(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as bench
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    chosen_code, chosen_n = _encode([[5, 4], [3, 1], [2]], stack_count, max_height)
    other_code, other_n = _encode([[4, 5], [3, 1], [2]], stack_count, max_height)
    ranked = [
        (other_code, (other_n, 8, b"p0")),
        (chosen_code, (chosen_n, 8, b"p1")),
    ]
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        use_qubo=True,
        qubo_min_candidates=99,
        feasibility_escape_enabled=True,
        feasibility_escape_supplement_width=1,
        feasibility_escape_focus_n=-1,
        feasibility_escape_n_window=2,
        _best_n_history=[chosen_n, chosen_n, chosen_n],
        _layer_depth_slack=1,
        _layer_parent_top_share=0.50,
    )

    def fake_layer_active(selector_args, best_n):
        return True

    def fake_selector(rows, selector_args, target):
        assert rows
        assert target == 1
        return [chosen_code]

    monkeypatch.setattr(bench, "_feasibility_escape_layer_active", fake_layer_active)
    monkeypatch.setattr(bench, "_select_feasibility_escape_supplement", fake_selector)

    selected, selected_by_qubo, qcalls, qvars, qtime = bench._select_enhanced_qubo_frontier(
        ranked,
        candidate_count=len(ranked),
        args=args,
        beam_width=2,
        core_width=1,
        qubo_supplement_width=0,
    )

    assert chosen_code in selected
    assert chosen_code not in selected_by_qubo
    assert qcalls == 0
    assert qvars == 0
    assert qtime == 0.0
    assert args._feasibility_escape_calls_done == 1
    assert args._feasibility_escape_added_total == 1
```

- [ ] **Step 3: Run tests to verify RED**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_feasibility_escape_supplement_prefers_lower_escape_score tests/test_qubo.py::test_feasibility_escape_selector_is_used_inside_enhanced_frontier -q --basetemp .pytest_tmp_feas_selector_red -p no:cacheprovider
```

Expected:

```text
FAILED ... AttributeError: module 'benchmark_compact_beam_qubo_n0' has no attribute '_select_feasibility_escape_supplement'
```

- [ ] **Step 4: Add imports and selector helpers**

In `scripts/benchmark_compact_beam_qubo_n0.py`, add near other imports:

```python
from container_reshuffle.solvers.feasibility_signals import (  # noqa: E402
    FeasibilitySignals,
    compute_feasibility_signals,
    feasibility_escape_score,
)
```

Add near `_select_unified_wang_qubo_supplement(...)`:

```python
def _feasibility_signals_for_code(code: bytes, args: argparse.Namespace) -> FeasibilitySignals:
    stacks = _split(code, int(args._stack_count), int(args._max_height))
    target_gap = _candidate_target_gap(code, args) if getattr(args, "_layout_target_groups", ()) else 0
    return compute_feasibility_signals(stacks, int(args._max_height), target_gap=target_gap)


def _feasibility_escape_layer_active(args: argparse.Namespace, best_n: int) -> bool:
    if not getattr(args, "feasibility_escape_enabled", False):
        return False
    focus_n = int(getattr(args, "feasibility_escape_focus_n", 8))
    if focus_n >= 0 and int(best_n) > focus_n:
        return False
    history = list(getattr(args, "_best_n_history", []) or [])
    if len(history) < 3 or min(history[-3:]) != history[-3]:
        return False
    if int(getattr(args, "_layer_depth_slack", 10**9)) > int(getattr(args, "rescue_cap_slack", 26)):
        return False
    if float(getattr(args, "_layer_parent_top_share", 0.0)) < 0.18:
        return False
    return True


def _select_feasibility_escape_supplement(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    target: int,
) -> list[bytes]:
    target = max(0, min(int(target), len(ranked)))
    if target <= 0:
        return []
    threshold = float(getattr(args, "feasibility_escape_dead_end_threshold", 0.45) or 0.45)
    min_slack = int(getattr(args, "feasibility_escape_min_temporary_slack", 1) or 1)
    scored: list[tuple[float, int, int, bytes]] = []
    for code, data in ranked:
        signals = _feasibility_signals_for_code(code, args)
        if signals.dead_end_risk < threshold and signals.temporary_slack_moves < min_slack:
            continue
        score = feasibility_escape_score(signals, int(data[0]), int(data[1]))
        scored.append((score, int(data[0]), int(data[1]), code))
    return [item[3] for item in sorted(scored)[:target]]
```

- [ ] **Step 5: Wire selector into `_select_enhanced_qubo_frontier(...)`**

Insert this block after the unified Wang block and before the generic QUBO block:

```python
    feasibility_width = max(0, int(getattr(args, "feasibility_escape_supplement_width", 0) or 0))
    if (
        _feasibility_escape_layer_active(args, int(best_n))
        and feasibility_width > 0
        and len(selected) < effective_beam
    ):
        n_window = max(0, int(getattr(args, "feasibility_escape_n_window", 1) or 0))
        remaining = [
            (code, data)
            for code, data in ranked
            if code not in selected and data[0] <= int(best_n) + n_window
        ]
        target = min(feasibility_width, effective_beam - len(selected))
        boosted = _select_feasibility_escape_supplement(remaining, args, target)
        if boosted:
            if not hasattr(args, "_feasibility_escape_calls_done"):
                args._feasibility_escape_calls_done = 0
            if not hasattr(args, "_feasibility_escape_added_total"):
                args._feasibility_escape_added_total = 0
            args._feasibility_escape_calls_done += 1
            for code in boosted:
                before = len(selected)
                selected.setdefault(code, None)
                if len(selected) > before:
                    args._feasibility_escape_added_total += 1
                if len(selected) >= effective_beam:
                    break
```

In `run_compact_beam_instance(...)`, initialize counters near other `_..._calls_done` counters:

```python
    args._feasibility_escape_calls_done = 0
    args._feasibility_escape_added_total = 0
```

When constructing `CompactBeamResult`, pass:

```python
        feasibility_escape_calls=args._feasibility_escape_calls_done,
        feasibility_escape_added_total=args._feasibility_escape_added_total,
```

- [ ] **Step 6: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_feasibility_escape_supplement_prefers_lower_escape_score tests/test_qubo.py::test_feasibility_escape_selector_is_used_inside_enhanced_frontier -q --basetemp .pytest_tmp_feas_selector_green -p no:cacheprovider
```

Expected:

```text
2 passed
```

- [ ] **Step 7: Commit Task 3**

Run:

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py tests\test_qubo.py
git commit -m "Wire feasibility escape into compact frontier"
```

---

### Task 4: Add DLTS Feature Export

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `tests/test_qubo.py`

- [ ] **Step 1: Write failing export helper test**

Add to `tests/test_qubo.py`:

```python
def test_dlts_feature_row_contains_feasibility_and_selection_fields(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as bench
    from container_reshuffle.solvers.feasibility_signals import FeasibilitySignals

    code = b"candidate"
    args = _compact_args(_case_id="case_a")

    monkeypatch.setattr(
        bench,
        "_feasibility_signals_for_code",
        lambda row_code, row_args: FeasibilitySignals(4, 4, 2, 3, 1, 5, 0, 0.25),
    )

    row = bench._dlts_feature_row(
        case="case_a",
        layer=2,
        code=code,
        data=(3, 9, b"parent"),
        parent_rank=7,
        selected=True,
        args=args,
    )

    assert row["case"] == "case_a"
    assert row["layer"] == 2
    assert row["n_value"] == 3
    assert row["depth"] == 9
    assert row["parent_rank"] == 7
    assert row["selected"] == 1
    assert row["stable_prefix_total"] == 4
    assert row["temporary_slack_moves"] == 5
    assert row["on_final_path"] == 0
```

- [ ] **Step 2: Run export helper test to verify RED**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_dlts_feature_row_contains_feasibility_and_selection_fields -q --basetemp .pytest_tmp_dlts_export_red -p no:cacheprovider
```

Expected:

```text
FAILED ... AttributeError: module 'benchmark_compact_beam_qubo_n0' has no attribute '_dlts_feature_row'
```

- [ ] **Step 3: Add feature row helpers**

In `scripts/benchmark_compact_beam_qubo_n0.py`, add near other field constants:

```python
DLTS_FEATURE_FIELDS = [
    "case",
    "solver_profile",
    "layer",
    "candidate_code",
    "n_value",
    "depth",
    "parent_rank",
    "selected",
    "on_final_path",
    "final_status",
    "fixed_prefix_total",
    "stable_prefix_total",
    "target_gap",
    "empty_slots",
    "blocked_fixed_candidates",
    "temporary_slack_moves",
    "low_tier_risk",
    "dead_end_risk",
]
```

Add helpers near `_write_beam_survivor_rows(...)`:

```python
def _dlts_feature_row(
    *,
    case: str,
    layer: int,
    code: bytes,
    data: tuple[int, int, bytes],
    parent_rank: int,
    selected: bool,
    args: argparse.Namespace,
) -> dict[str, object]:
    signals = _feasibility_signals_for_code(code, args)
    return {
        "case": case,
        "solver_profile": getattr(args, "solver_profile", ""),
        "layer": layer,
        "candidate_code": code.hex(),
        "n_value": int(data[0]),
        "depth": int(data[1]),
        "parent_rank": int(parent_rank),
        "selected": 1 if selected else 0,
        "on_final_path": 0,
        "final_status": "",
        "fixed_prefix_total": signals.fixed_prefix_total,
        "stable_prefix_total": signals.stable_prefix_total,
        "target_gap": signals.target_gap,
        "empty_slots": signals.empty_slots,
        "blocked_fixed_candidates": signals.blocked_fixed_candidates,
        "temporary_slack_moves": signals.temporary_slack_moves,
        "low_tier_risk": signals.low_tier_risk,
        "dead_end_risk": f"{signals.dead_end_risk:.6f}",
    }


def _finalize_dlts_feature_rows(rows: list[dict[str, object]], search_path_codes: list[bytes], status: str) -> None:
    path_hex = {code.hex() for code in search_path_codes}
    for row in rows:
        row["on_final_path"] = 1 if row["candidate_code"] in path_hex else 0
        row["final_status"] = status


def _write_dlts_feature_rows(rows: list[dict[str, object]], path: Path) -> None:
    output = path if path.is_absolute() else ROOT / path
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DLTS_FEATURE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
```

- [ ] **Step 4: Run export helper test to verify GREEN**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_dlts_feature_row_contains_feasibility_and_selection_fields -q --basetemp .pytest_tmp_dlts_export_green -p no:cacheprovider
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Add CLI flag and collect rows during search**

Add parser flag near output flags:

```python
    parser.add_argument("--dlts-feature-output", type=Path)
```

In `run_compact_beam_instance(...)`, add near `beam_survivor_records`:

```python
    dlts_feature_rows: list[dict[str, object]] = []
    dlts_feature_output = getattr(args, "dlts_feature_output", None)
```

After `selected_codes, selected_by_qubo, add_qcalls, add_qvars, add_qtime = _select_next_frontier(candidates, args)`, add:

```python
        if dlts_feature_output is not None:
            parent_rank_by_code = {code: idx + 1 for idx, (code, _n_value, _depth) in enumerate(frontier)}
            selected_set = set(selected_codes)
            for rank, (code, data) in enumerate(_rank_bucket_topk_candidates(candidates, args)[: min(256, len(candidates))], start=1):
                dlts_feature_rows.append(
                    _dlts_feature_row(
                        case=case,
                        layer=layer_index,
                        code=code,
                        data=data,
                        parent_rank=parent_rank_by_code.get(data[2], rank),
                        selected=code in selected_set,
                        args=args,
                    )
                )
```

Before returning `CompactBeamResult`, after reconstructing `search_path_codes`, add:

```python
    if dlts_feature_output is not None:
        _finalize_dlts_feature_rows(dlts_feature_rows, search_path_codes, status)
        _write_dlts_feature_rows(dlts_feature_rows, dlts_feature_output)
```

If `search_path_codes` is currently local only inside a branch, define it once as an empty list before reconstruction and assign it where path reconstruction succeeds.

- [ ] **Step 6: Add a CLI smoke test with monkeypatched writer**

Add to `tests/test_qubo.py`:

```python
def test_make_compact_args_accepts_dlts_feature_output(tmp_path):
    from benchmark_compact_beam_qubo_n0 import make_compact_args

    output = tmp_path / "features.csv"
    args = make_compact_args(dlts_feature_output=output)

    assert args.dlts_feature_output == output
    assert "dlts_feature_output" in args._profile_explicit_keys
```

- [ ] **Step 7: Run export tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_dlts_feature_row_contains_feasibility_and_selection_fields tests/test_qubo.py::test_make_compact_args_accepts_dlts_feature_output -q --basetemp .pytest_tmp_dlts_export_all_green -p no:cacheprovider
```

Expected:

```text
2 passed
```

- [ ] **Step 8: Commit Task 4**

Run:

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py tests\test_qubo.py
git commit -m "Add DLTS-ready feature export"
```

---

### Task 5: Verification, Smoke, And Experiment Note

**Files:**
- Modify: `docs/experiments/wang_qubo_full_quality_trial_summary.md`

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m pytest tests/test_feasibility_signals.py tests/test_qubo.py::test_solver_profile_scheduler_state_v5_feasibility_escape_polish_extends_rescue_polish tests/test_qubo.py::test_open_runner_accepts_feasibility_escape_variant tests/test_qubo.py::test_feasibility_escape_supplement_prefers_lower_escape_score tests/test_qubo.py::test_feasibility_escape_selector_is_used_inside_enhanced_frontier tests/test_qubo.py::test_dlts_feature_row_contains_feasibility_and_selection_fields tests/test_qubo.py::test_make_compact_args_accepts_dlts_feature_output -q --basetemp .pytest_tmp_feas_final_focused -p no:cacheprovider
```

Expected:

```text
passed
```

Record the exact pass count in the final report.

- [ ] **Step 2: Run syntax check**

Run:

```powershell
python -m py_compile container_reshuffle\solvers\feasibility_signals.py scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py scripts\diagnose_wang_qubo_failures.py
```

Expected: exit code `0`.

- [ ] **Step 3: Run narrow residual smoke if local DLTS exists**

First run:

```powershell
Test-Path external\DLTS
```

If it prints `True`, run:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test --case-ids 4,10 --variant qubo_scheduler_state_v5_feasibility_escape_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_feasibility_escape_residual.csv
```

Expected: exit code `0`, with `docs\experiments\scratch_feasibility_escape_residual.csv` and summary markdown written.

If `external\DLTS` is absent, do not auto-clone in this task. Report that residual smoke was skipped because the local DLTS source is absent.

- [ ] **Step 4: Run DLTS feature export smoke on a tiny internal case**

Run:

```powershell
python scripts\benchmark_compact_beam_qubo_n0.py --cases A1_01 --solver-profile scheduler_state_v5_feasibility_escape_polish --time-limit 1 --dlts-feature-output docs\experiments\scratch_dlts_features_A1_01.csv --output docs\experiments\scratch_feasibility_escape_A1_01.csv
```

Expected: exit code `0`, and `docs\experiments\scratch_dlts_features_A1_01.csv` contains the header fields from `DLTS_FEATURE_FIELDS`.

- [ ] **Step 5: Append curated experiment note**

Append to `docs/experiments/wang_qubo_full_quality_trial_summary.md`:

```markdown
## Feasibility Escape And DLTS Feature Export Probe

`scheduler_state_v5_feasibility_escape_polish` extends the current `scheduler_state_v5_wang_rescue_polish` line with a bounded feasibility escape selector. It uses deterministic state-feasibility, stability, temporary-slack, and dead-end-risk signals. It does not call original DLTS, train a neural model, or splice external move paths into compact parent chains.

The first focused commands are:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test --case-ids 4,10 --variant qubo_scheduler_state_v5_feasibility_escape_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_feasibility_escape_residual.csv
python scripts\benchmark_compact_beam_qubo_n0.py --cases A1_01 --solver-profile scheduler_state_v5_feasibility_escape_polish --time-limit 1 --dlts-feature-output docs\experiments\scratch_dlts_features_A1_01.csv --output docs\experiments\scratch_feasibility_escape_A1_01.csv
```

Promotion requires a full-case quality improvement against `qubo_scheduler_state_v5_wang_rescue_polish`, or solving one residual hard case without adding new failures in a clean paired run.
```

If the residual smoke ran, add one sentence after this block with the exact solved count, full penalty, and remaining failures from the generated summary.

- [ ] **Step 6: Run whitespace check**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 7: Commit Task 5**

Run:

```powershell
git add docs\experiments\wang_qubo_full_quality_trial_summary.md
git commit -m "Document feasibility escape probe"
```

---

## Final Verification

- [ ] **Step 1: Run final focused suite**

Run:

```powershell
python -m pytest tests/test_feasibility_signals.py tests/test_qubo.py tests/test_wang_failure_diagnostics.py tests/test_dlts_original_runner.py -q --basetemp .pytest_tmp_feas_dlt_final -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 2: Run final syntax check**

Run:

```powershell
python -m py_compile container_reshuffle\solvers\feasibility_signals.py scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py scripts\benchmark_dlts_original_cpmp.py scripts\diagnose_wang_qubo_failures.py container_reshuffle\solvers\classical_cpmp.py
```

Expected: exit code `0`.

- [ ] **Step 3: Check repository status**

Run:

```powershell
git status --short --untracked-files=all
```

Expected: no uncommitted implementation files. The local untracked `AGENTS.md` may remain untracked if it is still intentionally kept out of commits.
