# FRG DLTS Paper Ideas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the latest fill-and-reduce greedy paper ideas plus DLTS-style policy/value guidance to the current best CPMP solver.

**Architecture:** Keep `qubo_scheduler_state_v5_wang_rescue_polish` as the baseline and keep the compact byte-state parent chain as the only search contract. The latest FRG paper contributes fill/reduce local structure: unblocked bad containers, ordered-stack filling, reduce-stack choice, and bounded compound continuation. DLTS contributes deterministic policy/value ranking and telemetry only; no neural model is trained or called.

**Tech Stack:** Python, pytest, compact beam runner `scripts/benchmark_compact_beam_qubo_n0.py`, open benchmark runner `scripts/open_cpmp_benchmarks.py`, signal helpers under `container_reshuffle/solvers/`, experiment notes under `docs/experiments/`.

---

## Scope Check

This plan replaces the mistakenly drafted Wang2017 plan. It targets the latest FRG/fill-and-reduce greedy paper and the DLTS/deep-learning search-guidance paper.

It does not:

- reimplement a standalone FRG solver,
- train a neural model,
- call original DLTS inside compact search,
- splice arbitrary multi-step paths without parent-chain support,
- promote any new profile unless it beats the current best single-route baseline.

Promotion baseline:

- `qubo_scheduler_state_v5_wang_rescue_polish`
- solved `118/120`
- full-quality penalty `3873`
- solved moves `3692`
- solved gap sum `350`
- average runtime `3.901s`

Promotion gate:

- New single route must keep at least `118/120` and reduce full-quality penalty below `3873`.
- A residual-specialist route may be kept as an ablation only if it solves `5x5_cv1_test_4` or `5x5_cv2_test_10` without new failures.
- A trigger/portfolio route must approach the `3851` quality-portfolio penalty while cutting actual dual-run cost.

---

## File Structure

- Create `container_reshuffle/solvers/frg_signals.py`
  - Computes FRG-style layout and transition signals from plain stack layouts.
  - Owns scoring for fill moves, reduce moves, unblocked bad containers, and slack-preserving stack choice.

- Create `tests/test_frg_signals.py`
  - Unit tests for FRG signal math independent from benchmark runners.

- Modify `scripts/benchmark_compact_beam_qubo_n0.py`
  - Adds FRG profile parameters.
  - Adds compact helpers to convert code to stack layouts.
  - Adds FRG candidate supplement and bounded compound continuation.
  - Adds deterministic DLTS policy/value scoring as a ranking supplement.
  - Adds result telemetry fields.

- Modify `scripts/open_cpmp_benchmarks.py`
  - Adds public variant alias `qubo_scheduler_state_v5_frg_dlts_polish`.

- Modify `tests/test_qubo.py`
  - Adds profile, selector, alias, and serialization tests.

- Create `docs/experiments/frg_dlts_paper_ideas_results.md`
  - Records residual, open-120, and promotion decisions.

---

### Task 1: Add FRG Signal Module

**Files:**
- Create: `container_reshuffle/solvers/frg_signals.py`
- Create: `tests/test_frg_signals.py`

- [ ] **Step 1: Write failing FRG signal tests**

Create `tests/test_frg_signals.py`:

```python
from __future__ import annotations

from container_reshuffle.solvers.frg_signals import (
    FrgTransitionSignals,
    compute_frg_layout_signals,
    compute_frg_transition_signals,
    frg_transition_score,
)


def test_frg_layout_signals_detect_unblocked_bad_and_fillable_ordered_stack():
    signals = compute_frg_layout_signals(
        stacks=[[6, 5], [4, 7, 2], [3], []],
        max_height=3,
    )

    assert signals.stable_prefix_total == 5
    assert signals.unblocked_bad_containers >= 1
    assert signals.fillable_ordered_stacks >= 2
    assert signals.empty_slots == 3


def test_frg_transition_prefers_fill_over_destabilizing_reduce():
    fill = compute_frg_transition_signals(
        before=[[6, 5], [4, 7, 2], [3], []],
        after=[[6, 5, 2], [4, 7], [3], []],
        max_height=3,
    )
    reduce = compute_frg_transition_signals(
        before=[[6, 5], [4, 7, 2], [3], []],
        after=[[6], [4, 7, 2], [3, 5], []],
        max_height=3,
    )

    assert fill.fill_gain > reduce.fill_gain
    assert frg_transition_score(fill, n_value=3, depth=10) < frg_transition_score(reduce, n_value=3, depth=10)


def test_frg_transition_signal_defaults_are_value_objects():
    signals = FrgTransitionSignals(
        fill_gain=1,
        reduce_gain=0,
        unblocked_bad_delta=1,
        slack_delta=0,
        stable_prefix_delta=1,
        max_height_ordered_penalty=0,
        low_tier_damage=0,
    )

    assert frg_transition_score(signals, n_value=2, depth=5) < 2100.0
```

- [ ] **Step 2: Run RED test**

Run:

```powershell
python -m pytest tests/test_frg_signals.py -q --basetemp .pytest_tmp_frg_signals_red -p no:cacheprovider
```

Expected:

```text
FAILED ... ModuleNotFoundError: No module named 'container_reshuffle.solvers.frg_signals'
```

- [ ] **Step 3: Implement `frg_signals.py`**

Create `container_reshuffle/solvers/frg_signals.py`:

```python
"""FRG-style fill-and-reduce signals for CPMP layouts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrgLayoutSignals:
    stable_prefix_total: int
    unblocked_bad_containers: int
    fillable_ordered_stacks: int
    empty_slots: int
    max_height_ordered_stacks: int
    low_tier_risk: int


@dataclass(frozen=True)
class FrgTransitionSignals:
    fill_gain: int
    reduce_gain: int
    unblocked_bad_delta: int
    slack_delta: int
    stable_prefix_delta: int
    max_height_ordered_penalty: int
    low_tier_damage: int


def _stable_prefix_len(stack: list[int]) -> int:
    if not stack:
        return 0
    prefix = 1
    for idx in range(1, len(stack)):
        if stack[idx] <= stack[idx - 1]:
            prefix += 1
        else:
            break
    return prefix


def _is_ordered(stack: list[int]) -> bool:
    return _stable_prefix_len(stack) == len(stack)


def _unblocked_bad_count(stack: list[int]) -> int:
    stable = _stable_prefix_len(stack)
    if stable >= len(stack):
        return 0
    return len(stack) - stable


def _low_tier_risk(stacks: list[list[int]]) -> int:
    risk = 0
    for stack in stacks:
        stable = _stable_prefix_len(stack)
        if stable == len(stack) and len(stack) > 1:
            risk += 1
    return risk


def compute_frg_layout_signals(stacks: list[list[int]], max_height: int) -> FrgLayoutSignals:
    stable_prefix_total = sum(_stable_prefix_len(stack) for stack in stacks)
    unblocked_bad_containers = sum(_unblocked_bad_count(stack) for stack in stacks)
    fillable_ordered_stacks = sum(1 for stack in stacks if len(stack) < max_height and _is_ordered(stack))
    empty_slots = sum(max(0, max_height - len(stack)) for stack in stacks)
    max_height_ordered_stacks = sum(1 for stack in stacks if len(stack) >= max_height and _is_ordered(stack))
    return FrgLayoutSignals(
        stable_prefix_total=stable_prefix_total,
        unblocked_bad_containers=unblocked_bad_containers,
        fillable_ordered_stacks=fillable_ordered_stacks,
        empty_slots=empty_slots,
        max_height_ordered_stacks=max_height_ordered_stacks,
        low_tier_risk=_low_tier_risk(stacks),
    )


def compute_frg_transition_signals(
    before: list[list[int]],
    after: list[list[int]],
    max_height: int,
) -> FrgTransitionSignals:
    b = compute_frg_layout_signals(before, max_height)
    a = compute_frg_layout_signals(after, max_height)
    return FrgTransitionSignals(
        fill_gain=max(0, a.stable_prefix_total - b.stable_prefix_total),
        reduce_gain=max(0, b.unblocked_bad_containers - a.unblocked_bad_containers),
        unblocked_bad_delta=b.unblocked_bad_containers - a.unblocked_bad_containers,
        slack_delta=a.empty_slots - b.empty_slots,
        stable_prefix_delta=a.stable_prefix_total - b.stable_prefix_total,
        max_height_ordered_penalty=max(0, a.max_height_ordered_stacks - b.max_height_ordered_stacks),
        low_tier_damage=max(0, a.low_tier_risk - b.low_tier_risk),
    )


def frg_transition_score(signals: FrgTransitionSignals, n_value: int, depth: int) -> float:
    return (
        float(n_value) * 1000.0
        + float(depth)
        - 26.0 * float(signals.fill_gain)
        - 18.0 * float(signals.reduce_gain)
        - 10.0 * float(signals.unblocked_bad_delta)
        - 6.0 * float(signals.stable_prefix_delta)
        - 3.0 * float(signals.slack_delta)
        + 18.0 * float(signals.max_height_ordered_penalty)
        + 20.0 * float(signals.low_tier_damage)
    )
```

- [ ] **Step 4: Run GREEN test**

Run:

```powershell
python -m pytest tests/test_frg_signals.py -q --basetemp .pytest_tmp_frg_signals_green -p no:cacheprovider
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit**

```powershell
git add container_reshuffle\solvers\frg_signals.py tests\test_frg_signals.py
git commit -m "feat: add frg layout signals"
```

---

### Task 2: Add FRG Candidate Supplement To Compact Beam

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `tests/test_qubo.py`

- [ ] **Step 1: Add failing profile test**

Append to `tests/test_qubo.py`:

```python
def test_solver_profile_scheduler_state_v5_frg_dlts_polish_extends_current_best(monkeypatch):
    args = _compact_args(solver_profile="scheduler_state_v5_frg_dlts_polish")

    assert args.solver_profile == "scheduler_state_v5_frg_dlts_polish"
    assert args.post_polish_qubo is True
    assert args.post_polish_mode == "qubo_windows_v2"
    assert args.wang_pair_qubo_enabled is True
    assert args.frg_candidate_supplement_enabled is True
    assert args.frg_candidate_supplement_width == 8
    assert args.frg_compound_escape_enabled is False
```

- [ ] **Step 2: Run RED profile test**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_solver_profile_scheduler_state_v5_frg_dlts_polish_extends_current_best -q --basetemp .pytest_tmp_frg_profile_red -p no:cacheprovider
```

Expected:

```text
FAILED ... unknown solver_profile
```

- [ ] **Step 3: Add defaults and profile**

In `scripts/benchmark_compact_beam_qubo_n0.py`, add default args near other supplement flags:

```python
"frg_candidate_supplement_enabled": False,
"frg_candidate_supplement_width": 0,
"frg_candidate_focus_n": -1,
"frg_candidate_n_window": 2,
"frg_candidate_max_ranked": 96,
"frg_compound_escape_enabled": False,
"frg_compound_escape_width": 0,
"frg_compound_escape_focus_n": -1,
"frg_compound_escape_plateau_window": 5,
"frg_compound_escape_plateau_delta": 0,
"frg_compound_escape_cap_slack": 24,
"frg_compound_escape_min_candidates": 7000,
"dlts_value_supplement_enabled": False,
"dlts_value_supplement_width": 0,
```

Add profile after `scheduler_state_v5_wang_rescue_polish`:

```python
_SOLVER_PROFILES["scheduler_state_v5_frg_dlts_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "frg_candidate_supplement_enabled": True,
    "frg_candidate_supplement_width": 8,
    "frg_candidate_focus_n": -1,
    "frg_candidate_n_window": 2,
    "frg_candidate_max_ranked": 96,
    "frg_compound_escape_enabled": False,
    "frg_compound_escape_width": 0,
    "dlts_value_supplement_enabled": True,
    "dlts_value_supplement_width": 4,
}
```

- [ ] **Step 4: Add compact-to-stacks helper and selector**

In `scripts/benchmark_compact_beam_qubo_n0.py`, import:

```python
from container_reshuffle.solvers.frg_signals import (
    compute_frg_transition_signals,
    frg_transition_score,
)
```

Add helpers:

```python
def _code_to_plain_stacks(code: bytes, stack_count: int, max_height: int) -> list[list[int]]:
    return [list(chunk[: _chunk_height_raw(chunk)]) for chunk in _split(code, stack_count, max_height)]


def _select_frg_candidate_supplement(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    target: int,
) -> list[bytes]:
    if not getattr(args, "frg_candidate_supplement_enabled", False):
        return []
    if target <= 0 or not ranked:
        return []
    best_n = int(ranked[0][1][0])
    focus_n = int(getattr(args, "frg_candidate_focus_n", -1))
    if focus_n >= 0 and best_n > focus_n:
        return []
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return []
    n_window = max(0, int(getattr(args, "frg_candidate_n_window", 2)))
    max_ranked = max(1, int(getattr(args, "frg_candidate_max_ranked", 96)))
    scored: list[tuple[float, int, int, bytes]] = []
    for code, data in ranked[:max_ranked]:
        if data[0] > best_n + n_window:
            continue
        parent_code = data[2]
        before = _code_to_plain_stacks(parent_code, stack_count, max_height)
        after = _code_to_plain_stacks(code, stack_count, max_height)
        signals = compute_frg_transition_signals(before, after, max_height)
        score = frg_transition_score(signals, n_value=data[0], depth=data[1])
        scored.append((score, data[0], data[1], code))
    scored.sort()
    return [code for _score, _n, _depth, code in scored[:target]]
```

- [ ] **Step 5: Wire selector into enhanced frontier**

In `_select_enhanced_qubo_frontier`, after Wang pair-QUBO supplement and before generic QUBO, add:

```python
frg_width = max(0, int(getattr(args, "frg_candidate_supplement_width", 0) or 0))
if frg_width > 0 and len(selected) < effective_beam:
    target = min(frg_width, effective_beam - len(selected))
    boosted = _select_frg_candidate_supplement(
        [(code, data) for code, data in ranked if code not in selected],
        args,
        target,
    )
    for code in boosted:
        selected.setdefault(code, None)
        if len(selected) >= effective_beam:
            break
```

- [ ] **Step 6: Run profile test**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_solver_profile_scheduler_state_v5_frg_dlts_polish_extends_current_best -q --basetemp .pytest_tmp_frg_profile_green -p no:cacheprovider
```

Expected:

```text
1 passed
```

- [ ] **Step 7: Commit**

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py tests\test_qubo.py
git commit -m "feat: add frg candidate supplement"
```

---

### Task 3: Add Bounded FRG Compound Escape As An Ablation Profile

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `tests/test_qubo.py`

- [ ] **Step 1: Add failing compound profile test**

Append to `tests/test_qubo.py`:

```python
def test_solver_profile_scheduler_state_v5_frg_compound_polish_is_gated(monkeypatch):
    args = _compact_args(solver_profile="scheduler_state_v5_frg_compound_polish")

    assert args.solver_profile == "scheduler_state_v5_frg_compound_polish"
    assert args.frg_candidate_supplement_enabled is True
    assert args.frg_compound_escape_enabled is True
    assert args.frg_compound_escape_width == 2
    assert args.frg_compound_escape_plateau_window == 5
```

- [ ] **Step 2: Run RED test**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_solver_profile_scheduler_state_v5_frg_compound_polish_is_gated -q --basetemp .pytest_tmp_frg_compound_red -p no:cacheprovider
```

Expected:

```text
FAILED ... unknown solver_profile
```

- [ ] **Step 3: Add compound profile**

Add:

```python
_SOLVER_PROFILES["scheduler_state_v5_frg_compound_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_frg_dlts_polish"],
    "frg_compound_escape_enabled": True,
    "frg_compound_escape_width": 2,
    "frg_compound_escape_focus_n": -1,
    "frg_compound_escape_plateau_window": 5,
    "frg_compound_escape_plateau_delta": 0,
    "frg_compound_escape_cap_slack": 24,
    "frg_compound_escape_min_candidates": 7000,
}
```

- [ ] **Step 4: Implement compound gate**

Add:

```python
def _frg_compound_escape_active(args: argparse.Namespace, best_n: int) -> bool:
    if not getattr(args, "frg_compound_escape_enabled", False):
        return False
    focus_n = int(getattr(args, "frg_compound_escape_focus_n", -1))
    if focus_n >= 0 and best_n > focus_n:
        return False
    if int(getattr(args, "_layer_candidate_count", 0)) < int(getattr(args, "frg_compound_escape_min_candidates", 7000)):
        return False
    if int(getattr(args, "_layer_depth_slack", 10**9)) > int(getattr(args, "frg_compound_escape_cap_slack", 24)):
        return False
    history = list(getattr(args, "_best_n_history", []) or [])
    window = max(2, int(getattr(args, "frg_compound_escape_plateau_window", 5)))
    if len(history) < window:
        return False
    recent = history[-window:]
    return max(recent) - min(recent) <= int(getattr(args, "frg_compound_escape_plateau_delta", 0))
```

- [ ] **Step 5: Implement one-step continuation append using FRG score**

Use the existing `wang_rollout` shape: append only legal one-step compact children after the selected frontier. Add:

```python
def _frg_best_successor(
    code: bytes,
    n_value: int,
    depth: int,
    args: argparse.Namespace,
) -> tuple[bytes, tuple[int, int, bytes]] | None:
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return None
    before = _code_to_plain_stacks(code, stack_count, max_height)
    best: tuple[float, bytes, int] | None = None
    for child_code, child_n in _successors_with_n_cached(code, n_value, stack_count, max_height):
        after = _code_to_plain_stacks(child_code, stack_count, max_height)
        signals = compute_frg_transition_signals(before, after, max_height)
        score = frg_transition_score(signals, n_value=child_n, depth=depth + 1)
        item = (score, child_code, child_n)
        if best is None or item < best:
            best = item
    if best is None:
        return None
    return best[1], (best[2], depth + 1, code)
```

Add:

```python
def _append_frg_compound_frontier(
    selected_codes: list[bytes],
    candidates: dict[bytes, tuple[int, int, bytes]],
    args: argparse.Namespace,
    depth_cap: int,
    best_depth: dict[bytes, int] | None = None,
) -> set[bytes]:
    if not selected_codes:
        return set()
    best_n = min(candidates[code][0] for code in selected_codes if code in candidates)
    if not _frg_compound_escape_active(args, best_n):
        return set()
    width = max(0, int(getattr(args, "frg_compound_escape_width", 0) or 0))
    if width <= 0:
        return set()
    added: set[bytes] = set()
    selected_set = set(selected_codes)
    origins = sorted(
        [code for code in selected_codes if code in candidates],
        key=lambda code: _candidate_sort_key((code, candidates[code]), args),
    )
    for code in origins:
        if len(added) >= width:
            break
        n_value, depth, _parent = candidates[code]
        child = _frg_best_successor(code, n_value, depth, args)
        if child is None:
            continue
        child_code, child_data = child
        child_n, child_depth, _child_parent = child_data
        if child_code in selected_set or child_code in added:
            continue
        if child_depth + child_n > depth_cap:
            continue
        old_best_depth = best_depth.get(child_code) if best_depth is not None else None
        if old_best_depth is not None and child_depth >= old_best_depth:
            continue
        old = candidates.get(child_code)
        if old is not None and old[1] <= child_depth:
            continue
        candidates[child_code] = child_data
        selected_codes.append(child_code)
        added.add(child_code)
    return added
```

- [ ] **Step 6: Call compound append in the main loop**

In `run_compact_beam_instance`, after `_append_wang_rollout_frontier(...)`, add:

```python
frg_appended_codes = _append_frg_compound_frontier(selected_codes, candidates, args, depth_cap, best_depth)
```

Record `frg_appended_codes` count into telemetry in Task 5.

- [ ] **Step 7: Run compound profile test**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_solver_profile_scheduler_state_v5_frg_compound_polish_is_gated -q --basetemp .pytest_tmp_frg_compound_green -p no:cacheprovider
```

Expected:

```text
1 passed
```

- [ ] **Step 8: Commit**

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py tests\test_qubo.py
git commit -m "feat: add bounded frg compound escape"
```

---

### Task 4: Add DLTS Policy/Value Ranking As A Small Supplement

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `tests/test_qubo.py`

- [ ] **Step 1: Add failing DLTS supplement test**

Append to `tests/test_qubo.py`:

```python
def test_frg_dlts_profile_enables_small_dlts_value_supplement(monkeypatch):
    args = _compact_args(solver_profile="scheduler_state_v5_frg_dlts_polish")

    assert args.dlts_value_supplement_enabled is True
    assert args.dlts_value_supplement_width == 4
```

- [ ] **Step 2: Run test**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_frg_dlts_profile_enables_small_dlts_value_supplement -q --basetemp .pytest_tmp_frg_dlts_value -p no:cacheprovider
```

Expected:

```text
1 passed
```

- [ ] **Step 3: Implement DLTS value supplement**

Add:

```python
def _select_dlts_value_supplement(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    target: int,
) -> list[bytes]:
    if not getattr(args, "dlts_value_supplement_enabled", False):
        return []
    if target <= 0 or not ranked:
        return []
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return []
    scored: list[tuple[float, int, int, bytes]] = []
    for code, data in ranked[:64]:
        parent_code = data[2]
        before = _code_to_plain_stacks(parent_code, stack_count, max_height)
        after = _code_to_plain_stacks(code, stack_count, max_height)
        signals = compute_frg_transition_signals(before, after, max_height)
        value_score = (
            1000.0 * float(data[0])
            + float(data[1])
            - 12.0 * float(signals.fill_gain)
            - 8.0 * float(signals.reduce_gain)
            + 16.0 * float(signals.low_tier_damage)
        )
        scored.append((value_score, data[0], data[1], code))
    scored.sort()
    return [code for _score, _n, _depth, code in scored[:target]]
```

Wire it after FRG candidate supplement and before generic QUBO:

```python
dlts_width = max(0, int(getattr(args, "dlts_value_supplement_width", 0) or 0))
if dlts_width > 0 and len(selected) < effective_beam:
    target = min(dlts_width, effective_beam - len(selected))
    boosted = _select_dlts_value_supplement(
        [(code, data) for code, data in ranked if code not in selected],
        args,
        target,
    )
    for code in boosted:
        selected.setdefault(code, None)
        if len(selected) >= effective_beam:
            break
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests/test_frg_signals.py tests/test_qubo.py::test_solver_profile_scheduler_state_v5_frg_dlts_polish_extends_current_best tests/test_qubo.py::test_frg_dlts_profile_enables_small_dlts_value_supplement -q --basetemp .pytest_tmp_frg_dlts_focused -p no:cacheprovider
```

Expected:

```text
5 passed
```

- [ ] **Step 5: Commit**

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py tests\test_qubo.py
git commit -m "feat: add dlts value supplement for frg profile"
```

---

### Task 5: Add Open Runner Alias And Telemetry

**Files:**
- Modify: `scripts/open_cpmp_benchmarks.py`
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `tests/test_qubo.py`

- [ ] **Step 1: Add failing open runner alias test**

Append to `tests/test_qubo.py`:

```python
def test_open_runner_accepts_frg_dlts_variant():
    from scripts.open_cpmp_benchmarks import _args_for_variant

    outer = SimpleNamespace(
        time_limit=10.0,
        alpha=10,
        solver_profile=None,
    )
    args = _args_for_variant(outer, "qubo_scheduler_state_v5_frg_dlts_polish")

    assert args.solver_profile == "scheduler_state_v5_frg_dlts_polish"
```

- [ ] **Step 2: Run RED alias test**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_open_runner_accepts_frg_dlts_variant -q --basetemp .pytest_tmp_frg_alias_red -p no:cacheprovider
```

Expected:

```text
FAILED
```

- [ ] **Step 3: Add variant alias**

In `scripts/open_cpmp_benchmarks.py`, add `qubo_scheduler_state_v5_frg_dlts_polish` to explicit variant lists.

In `_args_for_variant`, add:

```python
if variant == "qubo_scheduler_state_v5_frg_dlts_polish":
    overrides.update({"solver_profile": "scheduler_state_v5_frg_dlts_polish"})
```

- [ ] **Step 4: Add telemetry fields**

In `CompactBeamResult`, add:

```python
frg_candidate_selected_total: int = 0
frg_compound_appended_total: int = 0
dlts_value_selected_total: int = 0
```

Initialize counters in `run_compact_beam_instance`:

```python
args._frg_candidate_selected_total = 0
args._frg_compound_appended_total = 0
args._dlts_value_selected_total = 0
```

Increment counters where FRG/DLTS supplements add codes. Return them through `CompactBeamResult` and serialize them in CSV row dictionaries.

- [ ] **Step 5: Run alias test**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_open_runner_accepts_frg_dlts_variant -q --basetemp .pytest_tmp_frg_alias_green -p no:cacheprovider
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Commit**

```powershell
git add scripts\open_cpmp_benchmarks.py scripts\benchmark_compact_beam_qubo_n0.py tests\test_qubo.py
git commit -m "feat: expose frg dlts benchmark variant"
```

---

### Task 6: Run Residual And Open-120 Benchmarks

**Files:**
- Create: `docs/experiments/frg_dlts_paper_ideas_results.md`
- Create: `docs/experiments/scratch_frg_dlts_residual.csv`
- Create: `docs/experiments/scratch_frg_dlts_open120.csv`
- Create: `docs/experiments/scratch_frg_compound_residual.csv`

- [ ] **Step 1: Create result note**

Create `docs/experiments/frg_dlts_paper_ideas_results.md`:

```markdown
# FRG DLTS Paper Ideas Results

## Baseline

| variant | solved | full-quality penalty | solved moves | solved gap sum | avg runtime sec |
|---|---:|---:|---:|---:|---:|
| `qubo_scheduler_state_v5_wang_rescue_polish` | 118/120 | 3873 | 3692 | 350 | 3.901 |

## Promotion Rule

Promote only if the new single route keeps at least `118/120` and reduces full-quality penalty below `3873`, or if a residual ablation solves a current residual case without adding open-120 failures.

## Results

No benchmark has been run for this plan yet.
```

- [ ] **Step 2: Run residual smoke for FRG/DLTS candidate profile**

Run:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test --case-ids 4,10 --variant qubo_scheduler_state_v5_frg_dlts_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_frg_dlts_residual.csv
```

Expected:

```text
summary table printed
```

- [ ] **Step 3: Run residual smoke for compound ablation**

Run:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test --case-ids 4,10 --variant qubo_scheduler_state_v5_frg_compound_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_frg_compound_residual.csv
```

Expected:

```text
summary table printed
```

- [ ] **Step 4: Run clean open-120 for the safer candidate profile**

Run:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_frg_dlts_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_frg_dlts_open120.csv
```

Expected:

```text
summary table printed
```

- [ ] **Step 5: Append measured results**

Append a measured table to `docs/experiments/frg_dlts_paper_ideas_results.md` using the printed summaries and CSV totals:

```markdown
## Measured Results

| variant | solved | full-quality penalty | solved moves | solved gap sum | avg runtime sec | reading |
|---|---:|---:|---:|---:|---:|---|
```

Add one row per run with actual values from the generated summaries.

- [ ] **Step 6: Commit**

```powershell
git add docs\experiments\frg_dlts_paper_ideas_results.md docs\experiments\scratch_frg_dlts_residual.csv docs\experiments\scratch_frg_compound_residual.csv docs\experiments\scratch_frg_dlts_open120.csv
git commit -m "exp: benchmark frg dlts paper ideas"
```

---

### Task 7: Decide Promotion Or Cleanup

**Files:**
- Modify: `docs/experiments/frg_dlts_paper_ideas_results.md`
- Modify: `docs/experiments/wang_qubo_direction_summary.md`

- [ ] **Step 1: Add decision section**

Append to `docs/experiments/frg_dlts_paper_ideas_results.md`:

```markdown
## Decision

Use these rules:

- Promote `qubo_scheduler_state_v5_frg_dlts_polish` only if it keeps `>=118/120` and full-quality penalty is below `3873`.
- Keep `scheduler_state_v5_frg_compound_polish` as an ablation if it solves either residual case but fails open-120 promotion.
- Close the FRG compound route if it adds no residual solve or introduces new open-120 failures.
- Keep DLTS value scoring only if it reduces moves or improves trigger selectivity without reducing coverage.
```

- [ ] **Step 2: Update direction summary**

Append to `docs/experiments/wang_qubo_direction_summary.md`:

```markdown
## FRG + DLTS Latest-Paper Probe

This probe applies the latest fill-and-reduce greedy idea as compact FRG candidate scoring and a bounded compound escape ablation. It also applies DLTS-style policy/value guidance as a deterministic supplement, not a learned model.

Current baseline remains `qubo_scheduler_state_v5_wang_rescue_polish` unless the FRG/DLTS route beats `118/120`, full-quality penalty `3873`, solved moves `3692`, and gap sum `350` in a clean open-120 run.
```

- [ ] **Step 3: Commit**

```powershell
git add docs\experiments\frg_dlts_paper_ideas_results.md docs\experiments\wang_qubo_direction_summary.md
git commit -m "docs: classify frg dlts latest-paper probe"
```

---

## Self-Review

Spec coverage:

- Latest FRG paper idea is covered by fill/reduce signals, unblocked bad containers, ordered-stack filling, reduce-style transition scoring, and a gated compound ablation.
- DLTS paper idea is covered by deterministic policy/value ranking, not learning.
- Current best algorithm remains the mandatory baseline.
- Failed-route cleanup is included through explicit promotion and closure rules.

Placeholder scan:

- The plan contains no implementation placeholders. Experiment rows are added after measurements, not prefilled with placeholder values.

Type consistency:

- `FrgLayoutSignals`, `FrgTransitionSignals`, `compute_frg_transition_signals`, and `frg_transition_score` are defined before use.
- Internal profile names use `scheduler_state_v5_frg_dlts_polish` and `scheduler_state_v5_frg_compound_polish`.
- Public open-runner variant uses `qubo_scheduler_state_v5_frg_dlts_polish`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-16-frg-dlts-paper-ideas.md`.

Two execution options:

1. Subagent-Driven (recommended) - Dispatch a fresh subagent per task, review between tasks, fast iteration.

2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints.
