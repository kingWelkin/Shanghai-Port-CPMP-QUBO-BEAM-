# Wang2017 DLTS V2 Paper Ideas Implementation Plan

> Superseded on 2026-06-16: the requested "latest paper" plan is `docs/superpowers/plans/2026-06-16-frg-dlts-paper-ideas.md`. Do not execute this Wang2017 plan unless explicitly requested.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the useful ideas from Wang2017 and DLTS-style learning guidance to the current best CPMP solver without replacing `qubo_scheduler_state_v5_wang_rescue_polish`.

**Architecture:** Keep the compact byte-state beam and parent-chain contract as the only search backbone. Wang2017 contributes a residual-prefix escape teacher and compact legal continuation signals; DLTS contributes deterministic policy/value scoring and feature export, not neural training. New profiles are experimental and must beat the current best single-route baseline before promotion.

**Tech Stack:** Python, pytest, existing compact beam runner in `scripts/benchmark_compact_beam_qubo_n0.py`, feasibility signal helpers in `container_reshuffle/solvers/feasibility_signals.py`, open benchmark runner in `scripts/open_cpmp_benchmarks.py`, experiment notes under `docs/experiments/`.

---

## Scope Check

This plan targets one vertical slice: a conservative Wang2017 + DLTS-guided residual escape profile on top of the current best v5 line.

It explicitly does not:

- train a neural network,
- call original DLTS inside search,
- splice Wang BS-B paths into compact parent chains,
- make Wang logic unconditional,
- promote anything unless it beats `qubo_scheduler_state_v5_wang_rescue_polish`.

Current promotion baseline:

- Single-route best: `qubo_scheduler_state_v5_wang_rescue_polish`, `118/120`, full-quality penalty `3873`, solved moves `3692`, avg runtime `3.901s`.
- Quality portfolio upper bound: `qubo_scheduler_state_v5_wang2017_task_guarded_portfolio`, `118/120`, penalty `3851`, but actual dual-run avg `8.866s`.

Promotion gate for this plan:

- Single-route promotion requires `>=118/120` and penalty `<3873`, or solving `5x5_cv1_test_4` or `5x5_cv2_test_10` with no new failures in a clean open-120 run.
- Trigger/portfolio promotion requires matching or beating penalty `3851` while materially reducing dual-run cost.

---

## File Structure

- Modify `scripts/diagnose_wang_qubo_failures.py`
  - Add residual prefix extraction for the two current unsolved cases.
  - Export compact-equivalent Wang prefix deltas, temporary slack changes, stable prefix changes, and first-move patterns.

- Modify `container_reshuffle/solvers/feasibility_signals.py`
  - Add V2 signals: `slack_created`, `stable_prefix_delta`, `fixed_pair_delta`, `stack_fillability`, `dead_end_relief`, and DLTS policy/value V2 scoring.

- Modify `tests/test_feasibility_signals.py`
  - Pin deterministic V2 policy/value ordering.

- Modify `scripts/benchmark_compact_beam_qubo_n0.py`
  - Add a bounded `wang_prefix_escape` supplement inside the existing compact frontier selector.
  - Add a `dlts_policy_value` supplement that scores only legal compact candidates.
  - Add profile `scheduler_state_v5_wang_dlts_escape_polish`.
  - Add telemetry fields for escape calls, candidates added, active layers, and policy/value selected counts.

- Modify `scripts/open_cpmp_benchmarks.py`
  - Add public variant alias `qubo_scheduler_state_v5_wang_dlts_escape_polish`.

- Modify `tests/test_qubo.py`
  - Add profile wiring tests, selector gate tests, and serialization tests.

- Create `docs/experiments/wang2017_dlts_v2_plan_results.md`
  - Record residual run, open-120 run, paired baseline, and promotion decision.

---

### Task 1: Baseline Guard And Residual Teacher Snapshot

**Files:**
- Modify: `scripts/diagnose_wang_qubo_failures.py`
- Create: `docs/experiments/wang2017_dlts_v2_plan_results.md`

- [ ] **Step 1: Capture current baseline commands in the results note**

Create `docs/experiments/wang2017_dlts_v2_plan_results.md`:

```markdown
# Wang2017 DLTS V2 Results

## Fixed Baseline

Current best single route:

| variant | solved | full-quality penalty | solved moves | solved gap sum | avg runtime sec |
|---|---:|---:|---:|---:|---:|
| `qubo_scheduler_state_v5_wang_rescue_polish` | 118/120 | 3873 | 3692 | 350 | 3.901 |

Promotion must compare against this row, not plain `scheduler_state_v5`.

## Residual Cases

- `5x5_cv1_test_4`
- `5x5_cv2_test_10`

## Commands

```powershell
python scripts\diagnose_wang_qubo_failures.py --source external\DLTS --families 5x5_cv1_test 5x5_cv2_test --case-ids 4,10 --output docs\experiments\scratch_wang2017_dlts_v2_residual_teacher.csv
```

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test --case-ids 4,10 --variant qubo_scheduler_state_v5_wang_dlts_escape_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_wang2017_dlts_v2_residual.csv
```

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_wang_dlts_escape_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_wang2017_dlts_v2_open120.csv
```

## Decision

Status: pending.
```

- [ ] **Step 2: Extend residual diagnostic columns**

In `scripts/diagnose_wang_qubo_failures.py`, add columns to each residual diagnostic row:

```python
WANG_DTLS_V2_FIELDS = [
    "wang_prefix_moves_json",
    "wang_prefix_n_trace",
    "wang_prefix_slack_trace",
    "wang_prefix_stable_prefix_trace",
    "wang_first_move_source_height",
    "wang_first_move_target_height",
    "wang_first_move_creates_empty_stack",
]
```

Add these fields to the CSV writer field list beside the existing compact/Wang diagnostics.

- [ ] **Step 3: Run residual teacher export**

Run:

```powershell
python scripts\diagnose_wang_qubo_failures.py --source external\DLTS --families 5x5_cv1_test 5x5_cv2_test --case-ids 4,10 --output docs\experiments\scratch_wang2017_dlts_v2_residual_teacher.csv
```

Expected:

```text
diagnostic CSV written
```

and the CSV has non-empty Wang prefix fields for both residual cases.

- [ ] **Step 4: Commit**

```powershell
git add scripts\diagnose_wang_qubo_failures.py docs\experiments\wang2017_dlts_v2_plan_results.md docs\experiments\scratch_wang2017_dlts_v2_residual_teacher.csv
git commit -m "exp: capture wang dlts residual teacher signals"
```

---

### Task 2: Add V2 Feasibility And DLTS Policy Signals

**Files:**
- Modify: `container_reshuffle/solvers/feasibility_signals.py`
- Modify: `tests/test_feasibility_signals.py`

- [ ] **Step 1: Write failing tests for V2 signals**

Append to `tests/test_feasibility_signals.py`:

```python
from container_reshuffle.solvers.feasibility_signals import (
    DltsPolicyValueWeights,
    compute_transition_feasibility_signals,
    dlts_policy_value_score,
)


def test_transition_signals_reward_slack_creation_and_stability_gain():
    before = [[5, 4, 1], [3, 6], [2], []]
    after = [[5, 4], [3, 6], [2], [1]]

    signals = compute_transition_feasibility_signals(before, after, max_height=3)

    assert signals.slack_created >= 0
    assert signals.stable_prefix_delta >= 0
    assert signals.dead_end_relief >= 0.0


def test_dlts_policy_value_score_prefers_lower_risk_candidate():
    weights = DltsPolicyValueWeights()
    risky = compute_transition_feasibility_signals(
        [[5, 4], [3, 1], [2]],
        [[5], [3, 1, 4], [2]],
        max_height=3,
    )
    safer = compute_transition_feasibility_signals(
        [[5, 4], [3, 1], [2]],
        [[5], [3, 1], [2, 4]],
        max_height=3,
    )

    assert dlts_policy_value_score(safer, n_value=2, depth=8, weights=weights) < dlts_policy_value_score(
        risky,
        n_value=2,
        depth=8,
        weights=weights,
    )
```

- [ ] **Step 2: Run RED test**

Run:

```powershell
python -m pytest tests/test_feasibility_signals.py -q --basetemp .pytest_tmp_wang_dlts_v2_red -p no:cacheprovider
```

Expected:

```text
FAILED ... cannot import name 'DltsPolicyValueWeights'
```

- [ ] **Step 3: Add V2 dataclasses and scoring**

In `container_reshuffle/solvers/feasibility_signals.py`, add:

```python
@dataclass(frozen=True)
class TransitionFeasibilitySignals:
    stable_prefix_delta: int
    fixed_pair_delta: int
    slack_created: int
    stack_fillability: int
    dead_end_relief: float
    low_tier_damage: int


@dataclass(frozen=True)
class DltsPolicyValueWeights:
    n_weight: float = 1000.0
    depth_weight: float = 1.0
    stable_prefix_weight: float = -14.0
    fixed_pair_weight: float = -10.0
    slack_weight: float = -18.0
    fillability_weight: float = -4.0
    dead_end_relief_weight: float = -35.0
    low_tier_damage_weight: float = 20.0
```

Add deterministic helpers:

```python
def _fixed_pair_count(stacks: list[list[int]]) -> int:
    count = 0
    for stack in stacks:
        for idx in range(1, len(stack)):
            if stack[idx - 1] > stack[idx]:
                count += 1
    return count


def _fillability(stacks: list[list[int]], max_height: int) -> int:
    score = 0
    for stack in stacks:
        if len(stack) >= max_height:
            continue
        if not stack:
            score += 2
        elif _stable_prefix_len(stack) == len(stack):
            score += 1
    return score


def compute_transition_feasibility_signals(
    before: list[list[int]],
    after: list[list[int]],
    max_height: int,
) -> TransitionFeasibilitySignals:
    before_signals = compute_feasibility_signals(before, max_height)
    after_signals = compute_feasibility_signals(after, max_height)
    slack_created = after_signals.empty_slots - before_signals.empty_slots
    stable_prefix_delta = after_signals.stable_prefix_total - before_signals.stable_prefix_total
    fixed_pair_delta = _fixed_pair_count(after) - _fixed_pair_count(before)
    fillability_delta = _fillability(after, max_height) - _fillability(before, max_height)
    dead_end_relief = max(0.0, before_signals.dead_end_risk - after_signals.dead_end_risk)
    low_tier_damage = max(0, after_signals.low_tier_risk - before_signals.low_tier_risk)
    return TransitionFeasibilitySignals(
        stable_prefix_delta=stable_prefix_delta,
        fixed_pair_delta=fixed_pair_delta,
        slack_created=slack_created,
        stack_fillability=fillability_delta,
        dead_end_relief=dead_end_relief,
        low_tier_damage=low_tier_damage,
    )


def dlts_policy_value_score(
    signals: TransitionFeasibilitySignals,
    n_value: int,
    depth: int,
    weights: DltsPolicyValueWeights | None = None,
) -> float:
    w = weights or DltsPolicyValueWeights()
    return (
        w.n_weight * float(n_value)
        + w.depth_weight * float(depth)
        + w.stable_prefix_weight * float(signals.stable_prefix_delta)
        + w.fixed_pair_weight * float(signals.fixed_pair_delta)
        + w.slack_weight * float(signals.slack_created)
        + w.fillability_weight * float(signals.stack_fillability)
        + w.dead_end_relief_weight * float(signals.dead_end_relief)
        + w.low_tier_damage_weight * float(signals.low_tier_damage)
    )
```

- [ ] **Step 4: Run GREEN test**

Run:

```powershell
python -m pytest tests/test_feasibility_signals.py -q --basetemp .pytest_tmp_wang_dlts_v2_green -p no:cacheprovider
```

Expected:

```text
5 passed
```

- [ ] **Step 5: Commit**

```powershell
git add container_reshuffle\solvers\feasibility_signals.py tests\test_feasibility_signals.py
git commit -m "feat: add dlts policy value feasibility signals"
```

---

### Task 3: Add Bounded Wang Prefix Escape Selector

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `tests/test_qubo.py`

- [ ] **Step 1: Add failing profile test**

Append to `tests/test_qubo.py`:

```python
def test_solver_profile_scheduler_state_v5_wang_dlts_escape_polish_extends_current_best(monkeypatch):
    args = _compact_args(solver_profile="scheduler_state_v5_wang_dlts_escape_polish")

    assert args.solver_profile == "scheduler_state_v5_wang_dlts_escape_polish"
    assert args.post_polish_qubo is True
    assert args.post_polish_mode == "qubo_windows_v2"
    assert args.wang_pair_qubo_enabled is True
    assert args.wang_prefix_escape_enabled is True
    assert args.dlts_policy_value_enabled is True
    assert args.wang_prefix_escape_supplement_width <= 8
```

- [ ] **Step 2: Run RED profile test**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_solver_profile_scheduler_state_v5_wang_dlts_escape_polish_extends_current_best -q --basetemp .pytest_tmp_wang_dlts_profile_red -p no:cacheprovider
```

Expected:

```text
FAILED ... unknown solver_profile
```

- [ ] **Step 3: Add profile defaults**

In `scripts/benchmark_compact_beam_qubo_n0.py`, add to default args:

```python
"wang_prefix_escape_enabled": False,
"wang_prefix_escape_supplement_width": 0,
"wang_prefix_escape_focus_n": -1,
"wang_prefix_escape_n_window": 2,
"wang_prefix_escape_plateau_window": 5,
"wang_prefix_escape_plateau_delta": 0,
"wang_prefix_escape_cap_slack": 24,
"wang_prefix_escape_min_candidates": 7000,
"dlts_policy_value_enabled": False,
"dlts_policy_value_supplement_width": 0,
"dlts_policy_value_max_vars": 48,
```

Add profile:

```python
_SOLVER_PROFILES["scheduler_state_v5_wang_dlts_escape_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "wang_prefix_escape_enabled": True,
    "wang_prefix_escape_supplement_width": 6,
    "wang_prefix_escape_focus_n": -1,
    "wang_prefix_escape_n_window": 2,
    "wang_prefix_escape_plateau_window": 5,
    "wang_prefix_escape_plateau_delta": 0,
    "wang_prefix_escape_cap_slack": 24,
    "wang_prefix_escape_min_candidates": 7000,
    "dlts_policy_value_enabled": True,
    "dlts_policy_value_supplement_width": 8,
    "dlts_policy_value_max_vars": 48,
}
```

- [ ] **Step 4: Implement conservative gate**

Add helper:

```python
def _wang_prefix_escape_layer_active(args: argparse.Namespace, best_n: int) -> bool:
    if not getattr(args, "wang_prefix_escape_enabled", False):
        return False
    focus_n = int(getattr(args, "wang_prefix_escape_focus_n", -1))
    if focus_n >= 0 and best_n > focus_n:
        return False
    if int(getattr(args, "_layer_candidate_count", 0)) < int(getattr(args, "wang_prefix_escape_min_candidates", 7000)):
        return False
    if int(getattr(args, "_layer_depth_slack", 10**9)) > int(getattr(args, "wang_prefix_escape_cap_slack", 24)):
        return False
    history = list(getattr(args, "_best_n_history", []) or [])
    window = max(2, int(getattr(args, "wang_prefix_escape_plateau_window", 5)))
    if len(history) < window:
        return False
    recent = history[-window:]
    return max(recent) - min(recent) <= int(getattr(args, "wang_prefix_escape_plateau_delta", 0))
```

- [ ] **Step 5: Implement selector without path splicing**

Add helper:

```python
def _select_wang_prefix_escape_supplement(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    target: int,
) -> list[bytes]:
    if target <= 0 or not ranked:
        return []
    best_n = int(ranked[0][1][0])
    if not _wang_prefix_escape_layer_active(args, best_n):
        return []
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return []
    n_window = max(0, int(getattr(args, "wang_prefix_escape_n_window", 2)))
    pool = [(code, data) for code, data in ranked if data[0] <= best_n + n_window]
    scored = []
    for code, data in pool:
        pair_score = _compact_wang_fix_pair_score(code, stack_count, max_height)
        prefix_score = _compact_wang_fixed_prefix_score(code, stack_count, max_height)
        if pair_score <= 0 and prefix_score <= 0:
            continue
        score = (-pair_score, -prefix_score, data[0], data[1], data[2], code)
        scored.append((score, code))
    scored.sort(key=lambda item: item[0])
    return [code for _score, code in scored[:target]]
```

This selector chooses only candidates already generated by the compact legal successor loop.

- [ ] **Step 6: Wire selector into enhanced frontier**

In `_select_enhanced_qubo_frontier`, after Wang pair-QUBO and before generic QUBO, add:

```python
prefix_width = max(0, int(getattr(args, "wang_prefix_escape_supplement_width", 0) or 0))
if prefix_width > 0 and len(selected) < effective_beam:
    target = min(prefix_width, effective_beam - len(selected))
    boosted = _select_wang_prefix_escape_supplement(
        [(code, data) for code, data in ranked if code not in selected],
        args,
        target,
    )
    for code in boosted:
        selected.setdefault(code, None)
        if len(selected) >= effective_beam:
            break
```

- [ ] **Step 7: Run profile test**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_solver_profile_scheduler_state_v5_wang_dlts_escape_polish_extends_current_best -q --basetemp .pytest_tmp_wang_dlts_profile_green -p no:cacheprovider
```

Expected:

```text
1 passed
```

- [ ] **Step 8: Commit**

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py tests\test_qubo.py
git commit -m "feat: add bounded wang prefix escape profile"
```

---

### Task 4: Add DLTS Policy/Value Candidate Supplement

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `tests/test_qubo.py`

- [ ] **Step 1: Add failing selector test**

Append to `tests/test_qubo.py`:

```python
def test_dlts_policy_value_supplement_is_disabled_by_default():
    args = _compact_args()

    assert getattr(args, "dlts_policy_value_enabled", False) is False
    assert getattr(args, "dlts_policy_value_supplement_width", 0) == 0
```

- [ ] **Step 2: Run test**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_dlts_policy_value_supplement_is_disabled_by_default -q --basetemp .pytest_tmp_dlts_default -p no:cacheprovider
```

Expected:

```text
1 passed
```

- [ ] **Step 3: Implement policy/value supplement**

In `scripts/benchmark_compact_beam_qubo_n0.py`, import:

```python
from container_reshuffle.solvers.feasibility_signals import (
    DltsPolicyValueWeights,
    compute_transition_feasibility_signals,
    dlts_policy_value_score,
)
```

Add helper:

```python
def _select_dlts_policy_value_supplement(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    target: int,
) -> list[bytes]:
    if not getattr(args, "dlts_policy_value_enabled", False):
        return []
    if target <= 0 or not ranked:
        return []
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return []
    weights = DltsPolicyValueWeights()
    scored = []
    for code, data in ranked[: max(1, int(getattr(args, "dlts_policy_value_max_vars", 48)))]:
        parent_code = data[2]
        before = _decode(parent_code, stack_count, max_height)
        after = _decode(code, stack_count, max_height)
        signals = compute_transition_feasibility_signals(before, after, max_height)
        score = dlts_policy_value_score(signals, n_value=data[0], depth=data[1], weights=weights)
        scored.append((score, data[0], data[1], code))
    scored.sort()
    return [code for _score, _n, _depth, code in scored[:target]]
```

If `_decode` has a different local signature, adapt this helper to the existing compact-code decode utility and keep the test focused on behavior, not helper internals.

- [ ] **Step 4: Wire policy/value supplement**

In `_select_enhanced_qubo_frontier`, after Wang prefix escape and before generic QUBO, add:

```python
dlts_width = max(0, int(getattr(args, "dlts_policy_value_supplement_width", 0) or 0))
if dlts_width > 0 and len(selected) < effective_beam:
    target = min(dlts_width, effective_beam - len(selected))
    boosted = _select_dlts_policy_value_supplement(
        [(code, data) for code, data in ranked if code not in selected],
        args,
        target,
    )
    for code in boosted:
        selected.setdefault(code, None)
        if len(selected) >= effective_beam:
            break
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
python -m pytest tests/test_feasibility_signals.py tests/test_qubo.py::test_solver_profile_scheduler_state_v5_wang_dlts_escape_polish_extends_current_best -q --basetemp .pytest_tmp_wang_dlts_selector -p no:cacheprovider
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit**

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py tests\test_qubo.py
git commit -m "feat: add deterministic dlts policy value supplement"
```

---

### Task 5: Add Open Runner Variant And Metrics

**Files:**
- Modify: `scripts/open_cpmp_benchmarks.py`
- Modify: `tests/test_qubo.py`

- [ ] **Step 1: Add failing open-runner alias test**

Append to `tests/test_qubo.py`:

```python
def test_open_runner_accepts_wang_dlts_escape_variant():
    from scripts.open_cpmp_benchmarks import _args_for_variant

    outer = SimpleNamespace(
        time_limit=10.0,
        alpha=10,
        solver_profile=None,
    )
    args = _args_for_variant(outer, "qubo_scheduler_state_v5_wang_dlts_escape_polish")

    assert args.solver_profile == "scheduler_state_v5_wang_dlts_escape_polish"
```

- [ ] **Step 2: Run RED alias test**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_open_runner_accepts_wang_dlts_escape_variant -q --basetemp .pytest_tmp_wang_dlts_alias_red -p no:cacheprovider
```

Expected:

```text
FAILED
```

- [ ] **Step 3: Add variant alias**

In `scripts/open_cpmp_benchmarks.py`, add `qubo_scheduler_state_v5_wang_dlts_escape_polish` wherever explicit variant names are enumerated.

In `_args_for_variant`, add:

```python
if variant == "qubo_scheduler_state_v5_wang_dlts_escape_polish":
    overrides.update({"solver_profile": "scheduler_state_v5_wang_dlts_escape_polish"})
```

- [ ] **Step 4: Add output fields**

In result row serialization, add fields:

```python
"wang_prefix_escape_enabled": getattr(args, "wang_prefix_escape_enabled", False),
"wang_prefix_escape_supplement_width": getattr(args, "wang_prefix_escape_supplement_width", 0),
"dlts_policy_value_enabled": getattr(args, "dlts_policy_value_enabled", False),
"dlts_policy_value_supplement_width": getattr(args, "dlts_policy_value_supplement_width", 0),
```

- [ ] **Step 5: Run alias test**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_open_runner_accepts_wang_dlts_escape_variant -q --basetemp .pytest_tmp_wang_dlts_alias_green -p no:cacheprovider
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Commit**

```powershell
git add scripts\open_cpmp_benchmarks.py tests\test_qubo.py
git commit -m "feat: expose wang dlts escape benchmark variant"
```

---

### Task 6: Residual And Full Open-120 Experiments

**Files:**
- Modify: `docs/experiments/wang2017_dlts_v2_plan_results.md`
- Create: `docs/experiments/scratch_wang2017_dlts_v2_residual.csv`
- Create: `docs/experiments/scratch_wang2017_dlts_v2_open120.csv`

- [ ] **Step 1: Run focused residual cases**

Run:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test --case-ids 4,10 --variant qubo_scheduler_state_v5_wang_dlts_escape_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_wang2017_dlts_v2_residual.csv
```

Expected:

```text
summary table printed
```

The result note must record whether either residual case reaches `business_ok=True`, `goal_validation_ok=True`, and `final_n=0`.

- [ ] **Step 2: Run clean open-120**

Run:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_wang_dlts_escape_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_wang2017_dlts_v2_open120.csv
```

Expected:

```text
summary table printed
```

- [ ] **Step 3: Append result table**

Append to `docs/experiments/wang2017_dlts_v2_plan_results.md`:

```markdown
## V2 Results

| variant | solved | full-quality penalty | solved moves | solved gap sum | avg runtime sec | decision |
|---|---:|---:|---:|---:|---:|---|
| `qubo_scheduler_state_v5_wang_dlts_escape_polish` | TBD | TBD | TBD | TBD | TBD | pending |

## Promotion Check

- Beats single-route penalty `3873`: pending.
- Keeps at least `118/120`: pending.
- Solves `5x5_cv1_test_4`: pending.
- Solves `5x5_cv2_test_10`: pending.
- Adds no new failures: pending.
```

Replace `TBD` values with the actual run summary before committing.

- [ ] **Step 4: Commit**

```powershell
git add docs\experiments\wang2017_dlts_v2_plan_results.md docs\experiments\scratch_wang2017_dlts_v2_residual.csv docs\experiments\scratch_wang2017_dlts_v2_open120.csv
git commit -m "exp: benchmark wang dlts v2 escape profile"
```

---

### Task 7: Cleanup Decision And Route Classification

**Files:**
- Modify: `docs/experiments/wang_qubo_direction_summary.md`
- Modify: `docs/experiments/wang2017_dlts_v2_plan_results.md`

- [ ] **Step 1: Classify the outcome**

Use this exact decision table:

```markdown
## V2 Decision

Classification:

- Promote to new best single route only if open-120 is `>=118/120` and penalty `<3873`.
- Keep as residual ablation if it solves one residual case but adds any new failures.
- Keep as telemetry only if it triggers and changes candidates but does not improve full quality.
- Close the route if it behaves like the previous feasibility escape: nonzero calls but residual cases remain unsolved.
```

- [ ] **Step 2: Update direction summary**

In `docs/experiments/wang_qubo_direction_summary.md`, add a short section:

```markdown
## Wang2017 + DLTS V2 Probe

The V2 probe applied Wang2017 residual-prefix escape signals and deterministic DLTS policy/value scoring on top of `qubo_scheduler_state_v5_wang_rescue_polish`.

Decision: pending.

Baseline remains `qubo_scheduler_state_v5_wang_rescue_polish` unless V2 beats `118/120`, penalty `3873`, solved moves `3692`, and gap sum `350` in a clean open-120 run.
```

Replace `pending` with the actual result after Task 6.

- [ ] **Step 3: Commit**

```powershell
git add docs\experiments\wang_qubo_direction_summary.md docs\experiments\wang2017_dlts_v2_plan_results.md
git commit -m "docs: classify wang dlts v2 experiment"
```

---

## Self-Review

Spec coverage:

- Wang2017 idea is covered by residual teacher extraction and bounded prefix escape.
- DLTS idea is covered by deterministic policy/value scoring and optional feature-style supplement.
- Current best baseline is fixed as the promotion target.
- Failed-route cleanup is included through outcome classification.

Placeholder scan:

- The only `TBD` entries are in the experiment result table and must be replaced after the benchmark run. They are data placeholders, not implementation placeholders.

Type consistency:

- `TransitionFeasibilitySignals`, `DltsPolicyValueWeights`, and `dlts_policy_value_score` are introduced before use.
- `scheduler_state_v5_wang_dlts_escape_polish` and `qubo_scheduler_state_v5_wang_dlts_escape_polish` use distinct internal/external names consistently.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-16-wang2017-dlts-v2-paper-ideas.md`.

Two execution options:

1. Subagent-Driven (recommended) - Dispatch a fresh subagent per task, review between tasks, fast iteration.

2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints.
