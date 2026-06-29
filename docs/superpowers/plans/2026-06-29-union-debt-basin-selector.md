# Union Debt/Basin Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish` profile that improves union beam/QUBO selection with receiver-debt, escape-potential, basin-diversity, and bounded reduction-chain candidate signals.

**Architecture:** Keep the current compact solver and union QUBO selector as the only search path. New signals are computed on candidate states, folded into the existing union QUBO linear and pair terms, and exposed through result/frontier telemetry. Reduction-chain children may enter the ordinary candidate pool, but they never return a terminal fallback solution.

**Tech Stack:** Python 3.13, `argparse`, existing compact CPMP solver in `scripts/benchmark_compact_beam_qubo_n0.py`, open runner in `scripts/open_cpmp_benchmarks.py`, pytest.

---

## File Structure

- Modify `scripts/benchmark_compact_beam_qubo_n0.py`
  - Add profile defaults and CLI arguments.
  - Add `DebtBasinFeatures` and helper scoring functions near existing micro/receiver helpers.
  - Extend `UnionDiversityQuboFeatures`.
  - Integrate linear receiver-debt / escape-potential score into `_union_diversity_qubo_features`.
  - Integrate basin pair penalty into `_union_diversity_objective` and `_select_union_diversity_qubo_supplement`.
  - Track reduction-chain origin/depth using existing FRG frontier-child injection.
  - Add result and frontier trace telemetry fields.

- Modify `scripts/open_cpmp_benchmarks.py`
  - Add the new open-runner variant alias.
  - Forward new profile arguments through `_args_for_variant`.
  - Add debt/basin telemetry fields to output rows.

- Modify `tests/test_qubo.py`
  - Add focused unit tests for profile wiring, debt score, escape score, basin pair penalty, reduction-chain depth, telemetry rows, and frontier trace fields.

- Modify `docs/guides/current-mainline-algorithm.md`
  - Add the new profile under research/ablation profiles only, without changing the default mainline.

---

### Task 1: Profile And Runner Wiring

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `scripts/open_cpmp_benchmarks.py`
- Test: `tests/test_qubo.py`

- [ ] **Step 1: Write the failing profile wiring test**

Append this test near the existing union profile tests in `tests/test_qubo.py`:

```python
def test_solver_profile_for_union_debt_basin_is_opt_in():
    import benchmark_compact_beam_qubo_n0 as bench
    from open_cpmp_benchmarks import _args_for_variant, _variant_names, build_parser

    mainline = _compact_args(solver_profile="scheduler_state_v5_qubo_diversity_union_polish")
    debt_basin = _compact_args(solver_profile="scheduler_state_v5_qubo_diversity_union_debt_basin_polish")

    bench._apply_solver_profile(mainline)
    bench._apply_solver_profile(debt_basin)

    assert getattr(mainline, "debt_basin_enabled", False) is False
    assert debt_basin.solver_profile == "scheduler_state_v5_qubo_diversity_union_debt_basin_polish"
    assert debt_basin.debt_basin_enabled is True
    assert debt_basin.union_diversity_qubo_enabled is True
    assert debt_basin.receiver_safety_enabled is True
    assert debt_basin.micro_rollout_enabled is False
    assert debt_basin.frg_frontier_child_enabled is True
    assert debt_basin.frg_frontier_child_append_only is False
    assert debt_basin.frg_frontier_child_parent_width == 8
    assert debt_basin.frg_frontier_child_limit == 1
    assert debt_basin.frg_frontier_child_max_steps == 2
    assert debt_basin.w_receiver_debt > 0.0
    assert debt_basin.w_escape_potential > 0.0
    assert debt_basin.union_diversity_qubo_basin_similarity_penalty > 0.0

    outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish"])
    open_args = _args_for_variant(outer, "qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish")

    assert open_args.solver_profile == "scheduler_state_v5_qubo_diversity_union_debt_basin_polish"
    assert "qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish" in _variant_names("wang_research_all")
```

- [ ] **Step 2: Run the wiring test and verify it fails**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py::test_solver_profile_for_union_debt_basin_is_opt_in -q --basetemp .tmp_pytest_debt_basin_profile_red
```

Expected: the test fails because `scheduler_state_v5_qubo_diversity_union_debt_basin_polish` is not registered.

- [ ] **Step 3: Add default arguments and profile preset**

In `scripts/benchmark_compact_beam_qubo_n0.py`, add these keys to the base solver defaults near the existing union diversity and micro rollout defaults:

```python
        "debt_basin_enabled": False,
        "w_receiver_debt": 0.0,
        "w_escape_potential": 0.0,
        "w_debt_basin_chain_cost": 0.0,
        "union_diversity_qubo_basin_similarity_penalty": 0.0,
        "debt_basin_trace_window": 96,
        "debt_basin_reduction_chain_enabled": False,
```

In `_apply_solver_profile`, add explicit flag routing:

```python
        "debt_basin_enabled": (
            "--debt-basin-enabled",
            "--no-debt-basin-enabled",
        ),
        "w_receiver_debt": "--w-receiver-debt",
        "w_escape_potential": "--w-escape-potential",
        "w_debt_basin_chain_cost": "--w-debt-basin-chain-cost",
        "union_diversity_qubo_basin_similarity_penalty": "--union-diversity-qubo-basin-similarity-penalty",
        "debt_basin_trace_window": "--debt-basin-trace-window",
        "debt_basin_reduction_chain_enabled": (
            "--debt-basin-reduction-chain-enabled",
            "--no-debt-basin-reduction-chain-enabled",
        ),
```

In `build_parser()`, add CLI arguments next to existing union diversity arguments:

```python
    parser.add_argument("--debt-basin-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--w-receiver-debt", type=float, default=None)
    parser.add_argument("--w-escape-potential", type=float, default=None)
    parser.add_argument("--w-debt-basin-chain-cost", type=float, default=None)
    parser.add_argument("--union-diversity-qubo-basin-similarity-penalty", type=float, default=None)
    parser.add_argument("--debt-basin-trace-window", type=int, default=None)
    parser.add_argument("--debt-basin-reduction-chain-enabled", action=argparse.BooleanOptionalAction, default=None)
```

Add the new profile after `scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish`:

```python
_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_debt_basin_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_polish"],
    "debt_basin_enabled": True,
    "receiver_safety_enabled": True,
    "micro_rollout_enabled": False,
    "micro_rollout_trace_only": False,
    "w_receiver_debt": 0.18,
    "w_escape_potential": 0.10,
    "w_debt_basin_chain_cost": 0.02,
    "union_diversity_qubo_basin_similarity_penalty": 0.14,
    "debt_basin_trace_window": 96,
    "debt_basin_reduction_chain_enabled": True,
    "frg_frontier_child_enabled": True,
    "frg_frontier_child_parent_width": 8,
    "frg_frontier_child_limit": 1,
    "frg_frontier_child_max_steps": 2,
    "frg_frontier_child_append_only": False,
    "frg_frontier_child_adaptive_enabled": False,
}
```

- [ ] **Step 4: Wire the open runner variant**

In `scripts/open_cpmp_benchmarks.py`, add the variant to `_variant_names("wang_research_all")`:

```python
            "qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish",
```

In `_args_for_variant`, add:

```python
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_debt_basin_polish"})
```

Add these argument names to the pass-through list in `_args_for_variant`:

```python
        "debt_basin_enabled",
        "w_receiver_debt",
        "w_escape_potential",
        "w_debt_basin_chain_cost",
        "union_diversity_qubo_basin_similarity_penalty",
        "debt_basin_trace_window",
        "debt_basin_reduction_chain_enabled",
```

- [ ] **Step 5: Run the profile wiring test and commit**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py::test_solver_profile_for_union_debt_basin_is_opt_in -q --basetemp .tmp_pytest_debt_basin_profile_green
```

Expected: `1 passed`.

Commit:

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py tests\test_qubo.py
git commit -m "Add union debt basin profile wiring"
```

---

### Task 2: Receiver-Debt And Escape-Potential Scoring

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Test: `tests/test_qubo.py`

- [ ] **Step 1: Write failing scoring tests**

Append these tests near the existing receiver safety tests in `tests/test_qubo.py`:

```python
def test_debt_basin_features_penalize_slack_consuming_non_progress_move():
    import benchmark_compact_beam_qubo_n0 as bench

    stack_count = 3
    max_height = 3
    parent, parent_n = bench._encode([[5, 4], [3, 1], []], stack_count, max_height)
    risky_child, risky_n = bench._encode([[5, 4], [3], [1]], stack_count, max_height)
    safer_child, safer_n = bench._encode([[5], [3, 1], [4]], stack_count, max_height)
    args = _compact_args(_stack_count=stack_count, _max_height=max_height)

    risky = bench._debt_basin_features(parent, risky_child, parent_n, risky_n, 5, args, origin="beam")
    safer = bench._debt_basin_features(parent, safer_child, parent_n, safer_n, 5, args, origin="beam")

    assert risky.receiver_debt > safer.receiver_debt
    assert risky.receiver_debt > 0.0
    assert risky.basin_label != ()
    assert risky.basin_similarity_group


def test_debt_basin_features_reward_escape_space():
    import benchmark_compact_beam_qubo_n0 as bench

    stack_count = 4
    max_height = 3
    parent, parent_n = bench._encode([[6, 2], [5, 1], [4], []], stack_count, max_height)
    escape_child, escape_n = bench._encode([[6], [5, 1], [4], [2]], stack_count, max_height)
    flat_child, flat_n = bench._encode([[6, 2], [5], [4, 1], []], stack_count, max_height)
    args = _compact_args(_stack_count=stack_count, _max_height=max_height)

    escape = bench._debt_basin_features(parent, escape_child, parent_n, escape_n, 6, args, origin="beam")
    flat = bench._debt_basin_features(parent, flat_child, parent_n, flat_n, 6, args, origin="beam")

    assert escape.escape_potential >= flat.escape_potential
    assert escape.escape_potential > 0.0
```

- [ ] **Step 2: Run scoring tests and verify they fail**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py::test_debt_basin_features_penalize_slack_consuming_non_progress_move tests\test_qubo.py::test_debt_basin_features_reward_escape_space -q --basetemp .tmp_pytest_debt_basin_score_red
```

Expected: both tests fail because `_debt_basin_features` is not defined.

- [ ] **Step 3: Add the feature type and helper functions**

In `scripts/benchmark_compact_beam_qubo_n0.py`, add this near `ReceiverSafetyScore`:

```python
class DebtBasinFeatures(NamedTuple):
    receiver_debt: float
    escape_potential: float
    basin_label: tuple[int, int, int, int, int, int]
    basin_similarity_group: str
    reduction_chain_origin: str
    reduction_chain_depth: int
```

Add these helpers near `_receiver_safety_score`:

```python
def _blocked_stack_count(code: bytes, stack_count: int, max_height: int) -> int:
    return int(sum(1 for chunk in _split(code, stack_count, max_height) if _cached_stack_n(chunk, max_height) > 0))


def _top_group_signature_bucket(code: bytes, stack_count: int, max_height: int) -> int:
    tops: list[int] = []
    for chunk in _split(code, stack_count, max_height):
        height = _cached_height(chunk, max_height)
        if height > 0:
            tops.append(int(chunk[height - 1]))
    if not tops:
        return 0
    return int(sum(tops) // max(1, len(tops) * 3))


def _height_pressure_bucket(code: bytes, stack_count: int, max_height: int) -> int:
    return int(min(5, math.floor(_height_pressure(code, stack_count, max_height) * 6.0)))


def _safe_receiver_bucket(count: int) -> int:
    if count <= 0:
        return 0
    if count == 1:
        return 1
    if count == 2:
        return 2
    return 3


def _debt_basin_features(
    parent_code: bytes,
    child_code: bytes,
    parent_n: int,
    child_n: int,
    depth: int,
    args: argparse.Namespace,
    *,
    origin: str = "beam",
) -> DebtBasinFeatures:
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return DebtBasinFeatures(0.0, 0.0, (), "", str(origin or "beam"), 0)

    receiver = _receiver_safety_score(parent_code, child_code, args)
    n_delta = int(child_n) - int(parent_n)
    blocked_after = _blocked_stack_count(child_code, stack_count, max_height)
    blocked_before = _blocked_stack_count(parent_code, stack_count, max_height)
    pressure_after = _height_pressure(child_code, stack_count, max_height)
    pressure_before = _height_pressure(parent_code, stack_count, max_height)

    debt = 0.0
    debt += max(0, -int(receiver.safe_receiver_delta)) * 1.0
    debt += max(0.0, float(receiver.height_pressure_after) - pressure_before) * 1.5
    debt += 0.75 if receiver.last_buffer_consumed else 0.0
    debt += 0.25 if n_delta >= 0 else 0.0
    debt += 0.20 if blocked_after > blocked_before else 0.0

    escape = 0.0
    escape += max(0, int(receiver.safe_receiver_delta)) * 1.0
    escape += max(0, -n_delta) * 0.5
    escape += max(0, blocked_before - blocked_after) * 0.35
    escape += max(0.0, pressure_before - pressure_after) * 1.0
    if not receiver.last_buffer_consumed and receiver.safe_receivers_after > 0:
        escape += 0.15

    n_band = max(0, int(child_n)) // max(1, int(getattr(args, "union_diversity_qubo_n_band_size", 1) or 1))
    source_stack = _transition_source_stack(parent_code, child_code, stack_count, max_height)
    basin_label = (
        int(n_band),
        min(5, blocked_after),
        _safe_receiver_bucket(int(receiver.safe_receivers_after)),
        _top_group_signature_bucket(child_code, stack_count, max_height),
        source_stack if source_stack >= 0 else stack_count,
        _height_pressure_bucket(child_code, stack_count, max_height),
    )
    chain_depth_by_code = getattr(args, "_reduction_chain_depth_by_code", {}) or {}
    chain_depth = int(chain_depth_by_code.get(child_code, 0) or 0)
    chain_origin = str(origin or "beam")
    return DebtBasinFeatures(
        receiver_debt=float(debt),
        escape_potential=float(escape),
        basin_label=basin_label,
        basin_similarity_group=":".join(str(part) for part in basin_label),
        reduction_chain_origin=chain_origin,
        reduction_chain_depth=chain_depth,
    )
```

- [ ] **Step 4: Run scoring tests and commit**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py::test_debt_basin_features_penalize_slack_consuming_non_progress_move tests\test_qubo.py::test_debt_basin_features_reward_escape_space -q --basetemp .tmp_pytest_debt_basin_score_green
```

Expected: `2 passed`.

Commit:

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py tests\test_qubo.py
git commit -m "Add debt basin feature scoring"
```

---

### Task 3: Linear Selector Integration

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Test: `tests/test_qubo.py`

- [ ] **Step 1: Write failing linear integration test**

Append this test near `test_micro_rollout_trace_only_does_not_change_union_linear_energy`:

```python
def test_debt_basin_linear_adjustment_is_opt_in():
    import benchmark_compact_beam_qubo_n0 as bench

    stack_count = 3
    max_height = 3
    parent, parent_n = bench._encode([[5, 4], [3, 1], []], stack_count, max_height)
    child, child_n = bench._encode([[5, 4], [3], [1]], stack_count, max_height)
    off_args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        debt_basin_enabled=False,
        w_receiver_debt=0.50,
        w_escape_potential=0.25,
    )
    on_args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        debt_basin_enabled=True,
        w_receiver_debt=0.50,
        w_escape_potential=0.25,
        w_debt_basin_chain_cost=0.02,
    )

    off = bench._union_diversity_qubo_features(child, (child_n, 4, parent), 0, off_args, origin="beam")
    on = bench._union_diversity_qubo_features(child, (child_n, 4, parent), 0, on_args, origin="beam")

    assert off.receiver_debt == 0.0
    assert off.escape_potential == 0.0
    assert off.debt_basin_linear_adjustment == 0.0
    assert on.receiver_debt > 0.0
    assert on.debt_basin_linear_adjustment > 0.0
    assert on.energy > off.energy
```

- [ ] **Step 2: Run the linear integration test and verify it fails**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py::test_debt_basin_linear_adjustment_is_opt_in -q --basetemp .tmp_pytest_debt_basin_linear_red
```

Expected: the test fails because `UnionDiversityQuboFeatures` has no debt/basin fields.

- [ ] **Step 3: Extend `UnionDiversityQuboFeatures`**

In `scripts/benchmark_compact_beam_qubo_n0.py`, add fields to `UnionDiversityQuboFeatures`:

```python
    receiver_debt: float = 0.0
    escape_potential: float = 0.0
    basin_label: tuple[int, int, int, int, int, int] = ()
    basin_similarity_group: str = ""
    reduction_chain_origin: str = ""
    reduction_chain_depth: int = 0
    debt_basin_linear_adjustment: float = 0.0
```

- [ ] **Step 4: Apply debt/basin linear adjustment**

In `_union_diversity_qubo_features`, after micro/receiver scoring and before constructing `features`, add:

```python
    debt_basin: DebtBasinFeatures | None = None
    debt_basin_linear_adjustment = 0.0
    debt_basin_window = max(0, int(getattr(args, "debt_basin_trace_window", 96) or 0))
    debt_basin_active = bool(getattr(args, "debt_basin_enabled", False))
    if stack_count > 0 and max_height > 0 and rank < debt_basin_window and debt_basin_active:
        debt_basin = _debt_basin_features(parent_code, code, int(parent_n := _code_n(parent_code, stack_count, max_height)), int(n_value), int(depth), args, origin=origin)
        debt_basin_linear_adjustment = (
            float(getattr(args, "w_receiver_debt", 0.0) or 0.0) * float(debt_basin.receiver_debt)
            - float(getattr(args, "w_escape_potential", 0.0) or 0.0) * float(debt_basin.escape_potential)
            + float(getattr(args, "w_debt_basin_chain_cost", 0.0) or 0.0) * float(debt_basin.reduction_chain_depth)
        )
        energy += debt_basin_linear_adjustment
```

When building `UnionDiversityQuboFeatures`, add:

```python
        receiver_debt=float(debt_basin.receiver_debt) if debt_basin is not None else 0.0,
        escape_potential=float(debt_basin.escape_potential) if debt_basin is not None else 0.0,
        basin_label=debt_basin.basin_label if debt_basin is not None else (),
        basin_similarity_group=str(debt_basin.basin_similarity_group) if debt_basin is not None else "",
        reduction_chain_origin=str(debt_basin.reduction_chain_origin) if debt_basin is not None else "",
        reduction_chain_depth=int(debt_basin.reduction_chain_depth) if debt_basin is not None else 0,
        debt_basin_linear_adjustment=float(debt_basin_linear_adjustment),
```

Also store trace features when debt/basin is active:

```python
    if micro is not None or receiver is not None or debt_basin is not None:
        trace_features = getattr(args, "_micro_rollout_feature_trace_by_code", None)
        if trace_features is None:
            trace_features = {}
            args._micro_rollout_feature_trace_by_code = trace_features
        trace_features[code] = features
```

- [ ] **Step 5: Run the linear integration test and commit**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py::test_debt_basin_linear_adjustment_is_opt_in -q --basetemp .tmp_pytest_debt_basin_linear_green
```

Expected: `1 passed`.

Commit:

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py tests\test_qubo.py
git commit -m "Apply debt basin linear selector signal"
```

---

### Task 4: Basin Pair Penalty And Selection Telemetry

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Test: `tests/test_qubo.py`

- [ ] **Step 1: Write failing basin-pair objective test**

Append this helper and test near the existing QUBO diversity tests:

```python
def _union_feature_for_basin(
    basin_label,
    *,
    energy=0.0,
    parent=b"p",
    source_stack=0,
    n_value=5,
):
    import benchmark_compact_beam_qubo_n0 as bench

    return bench.UnionDiversityQuboFeatures(
        energy=energy,
        n_value=n_value,
        depth=1,
        parent_code=parent,
        source_stack=source_stack,
        signature=(0, 0, 0, 0, 0),
        reduce_signature=(source_stack, 0, 0),
        origin="core",
        basin_label=basin_label,
        basin_similarity_group=":".join(str(part) for part in basin_label),
    )


def test_union_diversity_objective_penalizes_same_basin_pair():
    import benchmark_compact_beam_qubo_n0 as bench

    args = _compact_args(
        union_diversity_qubo_same_parent_penalty=0.0,
        union_diversity_qubo_same_source_penalty=0.0,
        union_diversity_qubo_signature_penalty=0.0,
        union_diversity_qubo_reduce_signature_penalty=0.0,
        union_diversity_qubo_n_band_penalty=0.0,
        union_diversity_qubo_basin_similarity_penalty=0.7,
    )
    features = [
        _union_feature_for_basin((5, 2, 1, 3, 0, 4), parent=b"a", source_stack=0),
        _union_feature_for_basin((5, 2, 1, 3, 0, 4), parent=b"b", source_stack=1),
        _union_feature_for_basin((5, 1, 3, 1, 2, 2), parent=b"c", source_stack=2),
    ]
    linear = [0.0, 0.0, 0.0]

    same_basin = bench._union_diversity_objective({0, 1}, linear, features, args)
    different_basin = bench._union_diversity_objective({0, 2}, linear, features, args)

    assert same_basin > different_basin
```

- [ ] **Step 2: Run the objective test and verify it fails**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py::test_union_diversity_objective_penalizes_same_basin_pair -q --basetemp .tmp_pytest_debt_basin_pair_red
```

Expected: the test fails because basin similarity is not included in the objective.

- [ ] **Step 3: Add basin pair penalty to objective and QUBO model**

In `_union_diversity_objective`, add:

```python
    basin_similarity_penalty = float(getattr(args, "union_diversity_qubo_basin_similarity_penalty", 0.0) or 0.0)
```

Inside the pair loop in `_union_diversity_objective`, add:

```python
            if (
                basin_similarity_penalty > 0.0
                and features[i].basin_label
                and features[i].basin_label == features[j].basin_label
            ):
                value += basin_similarity_penalty
```

In `_select_union_diversity_qubo_supplement`, add the same option near the existing pair penalty reads:

```python
    basin_similarity_penalty = float(getattr(args, "union_diversity_qubo_basin_similarity_penalty", 0.0) or 0.0)
```

Inside the QUBO model pair loop, add:

```python
            if (
                basin_similarity_penalty > 0.0
                and features[i].basin_label
                and features[i].basin_label == features[j].basin_label
            ):
                model.add_quadratic(vars_[i], vars_[j], basin_similarity_penalty)
```

- [ ] **Step 4: Track selected debt/basin telemetry**

In `_select_union_diversity_qubo_supplement`, after `picked = _repair_union_diversity_selection(...)`, add:

```python
    if getattr(args, "debt_basin_enabled", False):
        if not hasattr(args, "_receiver_debt_selected_total"):
            args._receiver_debt_selected_total = 0.0
        if not hasattr(args, "_receiver_debt_selected_count"):
            args._receiver_debt_selected_count = 0
        if not hasattr(args, "_escape_potential_selected_total"):
            args._escape_potential_selected_total = 0.0
        if not hasattr(args, "_escape_potential_selected_count"):
            args._escape_potential_selected_count = 0
        if not hasattr(args, "_basin_pair_penalty_total"):
            args._basin_pair_penalty_total = 0.0
        if not hasattr(args, "_basin_unique_selected_total"):
            args._basin_unique_selected_total = 0
        selected_basin_labels = {features[idx].basin_label for idx in picked if features[idx].basin_label}
        args._basin_unique_selected_total += len(selected_basin_labels)
        for idx in picked:
            if features[idx].receiver_debt or features[idx].escape_potential:
                args._receiver_debt_selected_total += float(features[idx].receiver_debt)
                args._receiver_debt_selected_count += 1
                args._escape_potential_selected_total += float(features[idx].escape_potential)
                args._escape_potential_selected_count += 1
        ordered_picked = sorted(picked)
        for pos, i in enumerate(ordered_picked):
            for j in ordered_picked[pos + 1 :]:
                if features[i].basin_label and features[i].basin_label == features[j].basin_label:
                    args._basin_pair_penalty_total += basin_similarity_penalty
```

- [ ] **Step 5: Run the objective test and commit**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py::test_union_diversity_objective_penalizes_same_basin_pair -q --basetemp .tmp_pytest_debt_basin_pair_green
```

Expected: `1 passed`.

Commit:

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py tests\test_qubo.py
git commit -m "Add debt basin diversity pair penalty"
```

---

### Task 5: Reduction-Chain Depth Tracking

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Test: `tests/test_qubo.py`

- [ ] **Step 1: Write failing reduction-chain depth test**

Append this test near existing FRG frontier-child tests:

```python
def test_frg_frontier_child_records_reduction_chain_depth(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as bench

    stack_count = 3
    max_height = 3
    parent_code, parent_n = bench._encode([[3, 2, 1], [], []], stack_count, max_height)
    one_step_code, one_step_n = bench._encode([[3, 2], [1], []], stack_count, max_height)
    two_step_code, two_step_n = bench._encode([[3], [1], [2]], stack_count, max_height)

    def fake_rollout_children(code, n_value, args, max_steps):
        assert max_steps == 2
        return ((one_step_code, one_step_n, 1), (two_step_code, two_step_n, 2))

    monkeypatch.setattr(bench, "_frg_frontier_child_rollout_children", fake_rollout_children)
    args = _compact_args(
        frg_frontier_child_enabled=True,
        frg_frontier_child_parent_width=1,
        frg_frontier_child_limit=4,
        frg_frontier_child_max_steps=2,
        debt_basin_reduction_chain_enabled=True,
        _stack_count=stack_count,
        _max_height=max_height,
    )
    candidates = {}

    added = bench._inject_frg_frontier_child_candidates(
        [(parent_code, parent_n, 4)],
        candidates,
        args,
        depth_cap=99,
        best_depth={},
        parent_chain={parent_code: None},
    )

    assert two_step_code in added
    assert args._reduction_chain_depth_by_code[two_step_code] == 2
    assert args._reduction_chain_origin_by_code[two_step_code] == "frg_frontier_child"
```

- [ ] **Step 2: Run the reduction-chain test and verify it fails**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py::test_frg_frontier_child_records_reduction_chain_depth -q --basetemp .tmp_pytest_debt_basin_chain_red
```

Expected: the test fails because `_reduction_chain_depth_by_code` is not populated.

- [ ] **Step 3: Populate chain origin/depth maps during FRG child injection**

In `_inject_frg_frontier_child_candidates`, initialize maps when `debt_basin_reduction_chain_enabled` is true:

```python
    if getattr(args, "debt_basin_reduction_chain_enabled", False):
        if not hasattr(args, "_reduction_chain_depth_by_code"):
            args._reduction_chain_depth_by_code = {}
        if not hasattr(args, "_reduction_chain_origin_by_code"):
            args._reduction_chain_origin_by_code = {}
```

When a generated child is accepted into `candidates`, record:

```python
                if getattr(args, "debt_basin_reduction_chain_enabled", False):
                    args._reduction_chain_depth_by_code[child_code] = int(chain_steps)
                    args._reduction_chain_origin_by_code[child_code] = "frg_frontier_child"
```

Use the local variable name that represents generated child depth in the existing loop. If the loop currently receives `(child_code, child_n)` without depth, normalize it before the loop:

```python
        normalized_children = []
        for child in rollout_children:
            if len(child) == 3:
                normalized_children.append(child)
            else:
                child_code, child_n = child
                normalized_children.append((child_code, child_n, 1))
```

- [ ] **Step 4: Run the reduction-chain test and commit**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py::test_frg_frontier_child_records_reduction_chain_depth -q --basetemp .tmp_pytest_debt_basin_chain_green
```

Expected: `1 passed`.

Commit:

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py tests\test_qubo.py
git commit -m "Track reduction chain depth for debt basin selector"
```

---

### Task 6: Result Rows, Frontier Trace, And Documentation

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `scripts/open_cpmp_benchmarks.py`
- Modify: `tests/test_qubo.py`
- Modify: `docs/guides/current-mainline-algorithm.md`

- [ ] **Step 1: Write failing frontier trace test**

Append this test near existing frontier trace tests:

```python
def test_frontier_trace_candidate_rows_include_debt_basin_features():
    import benchmark_compact_beam_qubo_n0 as bench

    parent = b"parent"
    code = b"child"
    args = _compact_args(
        solver_profile="scheduler_state_v5_qubo_diversity_union_debt_basin_polish",
        selection_policy="scheduler_state_qubo",
    )
    feature = bench.UnionDiversityQuboFeatures(
        energy=1.0,
        n_value=4,
        depth=3,
        parent_code=parent,
        source_stack=1,
        signature=(0, 0, 0, 0, 0),
        reduce_signature=(1, 0, 0),
        origin="frg_frontier_child",
        receiver_debt=1.25,
        escape_potential=0.50,
        basin_label=(4, 2, 1, 3, 1, 4),
        basin_similarity_group="4:2:1:3:1:4",
        reduction_chain_origin="frg_frontier_child",
        reduction_chain_depth=2,
        debt_basin_linear_adjustment=0.175,
    )
    args._micro_rollout_feature_trace_by_code = {code: feature}

    rows = bench._frontier_trace_candidate_rows(
        case="case_a",
        layer=2,
        ranked=[(code, (4, 3, parent))],
        frontier=[(parent, 5, 2)],
        selected_codes=[code],
        selected_by_qubo={code},
        selected_source_by_code={code: "union_diversity_qubo"},
        args=args,
        top_k=5,
        candidate_origin_by_code={code: "frg_frontier_child"},
    )

    assert rows[0]["receiver_debt"] == 1.25
    assert rows[0]["escape_potential"] == 0.50
    assert rows[0]["basin_label"] == "4:2:1:3:1:4"
    assert rows[0]["basin_similarity_group"] == "4:2:1:3:1:4"
    assert rows[0]["reduction_chain_origin"] == "frg_frontier_child"
    assert rows[0]["reduction_chain_depth"] == 2
    assert rows[0]["selected_by_debt_basin_qubo"] == 1
```

- [ ] **Step 2: Write failing row-field test**

Append this test near open runner row field tests:

```python
def test_open_runner_rows_include_debt_basin_metrics():
    import argparse
    from types import SimpleNamespace

    from open_cpmp_benchmarks import _micro_rollout_row_fields

    args = argparse.Namespace(
        debt_basin_enabled=True,
        w_receiver_debt=0.18,
        w_escape_potential=0.10,
        w_debt_basin_chain_cost=0.02,
        union_diversity_qubo_basin_similarity_penalty=0.14,
        debt_basin_trace_window=96,
        debt_basin_reduction_chain_enabled=True,
    )
    result = SimpleNamespace(
        receiver_debt_selected_total=3.0,
        receiver_debt_avg_selected=1.5,
        escape_potential_selected_total=2.0,
        escape_potential_avg_selected=1.0,
        basin_pair_penalty_total=0.14,
        basin_unique_selected_total=4,
        reduction_chain_generated_total=7,
        reduction_chain_selected_total=2,
        debt_basin_selected_on_path=1,
    )

    row = _micro_rollout_row_fields(result, args)

    assert row["debt_basin_enabled"] is True
    assert row["w_receiver_debt"] == 0.18
    assert row["union_diversity_qubo_basin_similarity_penalty"] == 0.14
    assert row["receiver_debt_avg_selected"] == 1.5
    assert row["reduction_chain_selected_total"] == 2
```

- [ ] **Step 3: Run telemetry tests and verify they fail**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py::test_frontier_trace_candidate_rows_include_debt_basin_features tests\test_qubo.py::test_open_runner_rows_include_debt_basin_metrics -q --basetemp .tmp_pytest_debt_basin_telemetry_red
```

Expected: both tests fail because trace and row fields do not exist.

- [ ] **Step 4: Add frontier trace fields**

In `FRONTIER_TRACE_FIELDS`, append:

```python
    "receiver_debt",
    "escape_potential",
    "basin_label",
    "basin_similarity_group",
    "reduction_chain_origin",
    "reduction_chain_depth",
    "selected_by_debt_basin_qubo",
```

In `_frontier_trace_candidate_rows`, add row values:

```python
                "receiver_debt": float(getattr(micro_feature, "receiver_debt", 0.0)),
                "escape_potential": float(getattr(micro_feature, "escape_potential", 0.0)),
                "basin_label": ":".join(str(part) for part in getattr(micro_feature, "basin_label", ()) or ()),
                "basin_similarity_group": str(getattr(micro_feature, "basin_similarity_group", "")),
                "reduction_chain_origin": str(getattr(micro_feature, "reduction_chain_origin", "")),
                "reduction_chain_depth": int(getattr(micro_feature, "reduction_chain_depth", 0)),
                "selected_by_debt_basin_qubo": int(
                    code in selected_qubo_set and bool(getattr(micro_feature, "basin_label", ()))
                ),
```

- [ ] **Step 5: Add result dataclass fields and return metrics**

In `CompactBeamResult`, add:

```python
    receiver_debt_selected_total: float = 0.0
    receiver_debt_avg_selected: float = 0.0
    escape_potential_selected_total: float = 0.0
    escape_potential_avg_selected: float = 0.0
    basin_pair_penalty_total: float = 0.0
    basin_unique_selected_total: int = 0
    reduction_chain_generated_total: int = 0
    reduction_chain_selected_total: int = 0
    debt_basin_selected_on_path: int = 0
```

Before returning `CompactBeamResult`, compute:

```python
    receiver_debt_count = max(1, int(getattr(args, "_receiver_debt_selected_count", 0) or 0))
    escape_count = max(1, int(getattr(args, "_escape_potential_selected_count", 0) or 0))
    reduction_chain_selected_total = sum(
        1 for code in search_path_codes[1:] if code in (getattr(args, "_reduction_chain_depth_by_code", {}) or {})
    )
    debt_basin_selected_on_path = sum(
        1 for code in search_path_codes[1:] if code in (getattr(args, "_micro_rollout_feature_trace_by_code", {}) or {})
    )
```

Set result fields:

```python
        receiver_debt_selected_total=float(getattr(args, "_receiver_debt_selected_total", 0.0) or 0.0),
        receiver_debt_avg_selected=float(getattr(args, "_receiver_debt_selected_total", 0.0) or 0.0) / receiver_debt_count,
        escape_potential_selected_total=float(getattr(args, "_escape_potential_selected_total", 0.0) or 0.0),
        escape_potential_avg_selected=float(getattr(args, "_escape_potential_selected_total", 0.0) or 0.0) / escape_count,
        basin_pair_penalty_total=float(getattr(args, "_basin_pair_penalty_total", 0.0) or 0.0),
        basin_unique_selected_total=int(getattr(args, "_basin_unique_selected_total", 0) or 0),
        reduction_chain_generated_total=int(getattr(args, "_frg_frontier_child_generated", 0) or 0),
        reduction_chain_selected_total=int(reduction_chain_selected_total),
        debt_basin_selected_on_path=int(debt_basin_selected_on_path),
```

- [ ] **Step 6: Add open runner row fields**

In `_micro_rollout_row_fields` in `scripts/open_cpmp_benchmarks.py`, add:

```python
        "debt_basin_enabled": getattr(args, "debt_basin_enabled", False),
        "w_receiver_debt": getattr(args, "w_receiver_debt", 0.0),
        "w_escape_potential": getattr(args, "w_escape_potential", 0.0),
        "w_debt_basin_chain_cost": getattr(args, "w_debt_basin_chain_cost", 0.0),
        "union_diversity_qubo_basin_similarity_penalty": getattr(
            args, "union_diversity_qubo_basin_similarity_penalty", 0.0
        ),
        "debt_basin_trace_window": getattr(args, "debt_basin_trace_window", 0),
        "debt_basin_reduction_chain_enabled": getattr(args, "debt_basin_reduction_chain_enabled", False),
        "receiver_debt_selected_total": getattr(result, "receiver_debt_selected_total", 0.0),
        "receiver_debt_avg_selected": getattr(result, "receiver_debt_avg_selected", 0.0),
        "escape_potential_selected_total": getattr(result, "escape_potential_selected_total", 0.0),
        "escape_potential_avg_selected": getattr(result, "escape_potential_avg_selected", 0.0),
        "basin_pair_penalty_total": getattr(result, "basin_pair_penalty_total", 0.0),
        "basin_unique_selected_total": getattr(result, "basin_unique_selected_total", 0),
        "reduction_chain_generated_total": getattr(result, "reduction_chain_generated_total", 0),
        "reduction_chain_selected_total": getattr(result, "reduction_chain_selected_total", 0),
        "debt_basin_selected_on_path": getattr(result, "debt_basin_selected_on_path", 0),
```

- [ ] **Step 7: Update current-mainline guide**

In `docs/guides/current-mainline-algorithm.md`, add one row to the research profiles table:

```markdown
| `qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish` | adds receiver-debt, escape-potential, basin-diversity, and bounded reduction-chain selector signals | experimental; not default |
```

- [ ] **Step 8: Run telemetry tests and commit**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py::test_frontier_trace_candidate_rows_include_debt_basin_features tests\test_qubo.py::test_open_runner_rows_include_debt_basin_metrics -q --basetemp .tmp_pytest_debt_basin_telemetry_green
```

Expected: `2 passed`.

Commit:

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py tests\test_qubo.py docs\guides\current-mainline-algorithm.md
git commit -m "Expose debt basin selector telemetry"
```

---

### Task 7: Focused Verification And Trace Smoke

**Files:**
- No required source edits.
- Output: `docs/experiments/debt_basin_*` CSV/summary artifacts.

- [ ] **Step 1: Run focused unit tests**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py::test_solver_profile_for_union_debt_basin_is_opt_in tests\test_qubo.py::test_debt_basin_features_penalize_slack_consuming_non_progress_move tests\test_qubo.py::test_debt_basin_features_reward_escape_space tests\test_qubo.py::test_debt_basin_linear_adjustment_is_opt_in tests\test_qubo.py::test_union_diversity_objective_penalizes_same_basin_pair tests\test_qubo.py::test_frg_frontier_child_records_reduction_chain_depth tests\test_qubo.py::test_frontier_trace_candidate_rows_include_debt_basin_features tests\test_qubo.py::test_open_runner_rows_include_debt_basin_metrics -q --basetemp .tmp_pytest_debt_basin_all
```

Expected: `8 passed`.

- [ ] **Step 2: Run syntax verification**

Run:

```powershell
python -m py_compile scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py scripts\benchmark_compact_paper_set.py
```

Expected: exit code `0` and no output.

- [ ] **Step 3: Run seven-case trace smoke**

Run this exact-case helper:

```powershell
$script = @'
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_bs_frg_paper import discover_paper_cases
from benchmark_compact_paper_set import _run_paper_variant_row, _write_csv, _write_summary
from benchmark_compact_beam_qubo_n0 import _reset_frontier_trace_output

source = ROOT / "external" / "paper_cpmp"
variant = "qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish"
targets = {
    ("BF", "BF14", "cpmp_16_8_103_21_78_1"),
    ("BF", "BF32", "cpmp_20_8_128_52_96_3"),
    ("CV", "6-6", "data6-6-8"),
    ("CV", "5-8", "data5-8-1"),
    ("CV", "5-8", "data5-8-10"),
    ("CV", "10-6", "data10-6-7"),
    ("CV", "10-10", "data10-10-1"),
}
cases = []
for dataset in ("BF", "CV"):
    for case in discover_paper_cases(source, dataset=dataset, limit_per_family=None):
        key = (case.dataset, case.family, case.case_id)
        if key in targets:
            cases.append(case)
cases.sort(key=lambda case: (case.dataset, case.family, case.case_id))
missing = targets - {(case.dataset, case.family, case.case_id) for case in cases}
if missing:
    raise SystemExit(f"missing hard cases: {sorted(missing)}")

output = ROOT / "docs" / "experiments" / "debt_basin_hard_20260629.csv"
frontier = ROOT / "docs" / "experiments" / "debt_basin_hard_frontier_20260629.csv"
_reset_frontier_trace_output(frontier)
args = SimpleNamespace(
    source=source,
    variant=variant,
    alpha=10.0,
    max_moves=None,
    solver_profile="speed_v2",
    target_final_n=0,
    time_limit=20.0,
    beam_width=None,
    core_width=None,
    parent_core_cap=None,
    dynamic_profile="legacy",
    qubo_profile="stable",
    skip_invalid=True,
    frontier_trace_output=frontier,
    frontier_trace_top_k=48,
    terminal_trace_output=None,
    frg_seed_cache_input=None,
    frg_seed_cache_output=None,
    jobs=1,
)
rows = [_run_paper_variant_row(case, args, variant) for case in cases]
_write_csv(rows, output)
summary = _write_summary(rows, output, source, args)
print(f"WROTE {output}")
print(f"WROTE {frontier}")
print(f"WROTE {summary}")
'@
$script | python -
```

Expected: command completes and writes both CSV files. This smoke is diagnostic; do not promote from it.

- [ ] **Step 4: Run paired guard samples**

Run open120 smoke on a small subset:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test --limit-per-family 5 --variant qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish --time-limit 10 --alpha 10 --jobs 4 --output docs\experiments\debt_basin_open_smoke_20260629.csv
```

Run paper sample10:

```powershell
python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF CV --sample-per-family 10 --variant qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish --time-limit 20 --alpha 10 --jobs 8 --output docs\experiments\debt_basin_paper_sample10_t20_20260629.csv
```

Expected: both commands complete. Interpret results against same-day union baseline before any promotion claim.

- [ ] **Step 5: Commit verification artifacts only if they are intended to be reviewed**

If the generated CSVs are large or ignored, leave them untracked and summarize their paths in the final report. If a small markdown summary is created, commit only that markdown:

```powershell
git add docs\experiments\debt_basin_trace_summary_20260629.md
git commit -m "Summarize debt basin selector smoke results"
```

---

## Self-Review

- Spec coverage:
  - Opt-in profile: Task 1.
  - Receiver debt: Tasks 2 and 3.
  - Escape potential: Tasks 2 and 3.
  - Basin label and pair diversity: Task 4.
  - Reduction-chain candidate supplement inside normal selector: Tasks 1 and 5.
  - Telemetry and trace separation of generated/not-selected/dead-lineage outcomes: Task 6.
  - Promotion gate: Task 7.

- Red-flag scan:
  - The plan contains no unresolved marker text, no undefined task names, and no instruction to add unspecified edge handling.

- Type consistency:
  - `DebtBasinFeatures` fields match the fields added to `UnionDiversityQuboFeatures`.
  - Profile names match the design spec:
    - open variant: `qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish`;
    - solver profile: `scheduler_state_v5_qubo_diversity_union_debt_basin_polish`.
  - Telemetry keys match between compact result fields, open row fields, and frontier trace fields.
