# Unified Wang-QUBO Beam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scheduler_state_v7_wang_qubo_unified`, a single compact beam algorithm that uses Wang fixed-prefix/fix-pair structure inside one QUBO-guided search loop instead of chaining independent solvers.

**Architecture:** Keep compact byte-state parent chains as the only path source of truth. Add unified Wang-QUBO feature extraction and energy scoring helpers, use them in a new QUBO supplement selector inside `_select_enhanced_qubo_frontier(...)`, then expose the behavior through one experimental profile and one open-runner alias.

**Tech Stack:** Python, argparse, pytest, compact byte-state helpers in `scripts/benchmark_compact_beam_qubo_n0.py`, QUBO model/sampler utilities already imported there, open benchmark orchestration in `scripts/open_cpmp_benchmarks.py`.

---

## File Structure

- Modify `scripts/benchmark_compact_beam_qubo_n0.py`
  - Add unified Wang-QUBO metadata fields.
  - Add `UnifiedWangQuboFeatures`.
  - Add `_unified_wang_qubo_features(...)`.
  - Add `_unified_wang_qubo_linear_energy(...)`.
  - Add `_select_unified_wang_qubo_supplement(...)`.
  - Wire the new selector into `_select_enhanced_qubo_frontier(...)`.
  - Add `scheduler_state_v7_wang_qubo_unified`.
  - Add parser/profile defaults for unified knobs.
- Modify `scripts/open_cpmp_benchmarks.py`
  - Add `qubo_scheduler_state_v7_wang_qubo_unified`.
  - Add it to `wang_all`.
- Modify `tests/test_qubo.py`
  - Add TDD coverage for feature extraction, energy ordering, selector behavior, profile wiring, and open-runner alias.
- Modify `docs/experiments/wang_qubo_full_quality_trial_summary.md`
  - Add a short entry describing the unified profile and benchmark command after smoke runs finish.

---

### Task 1: Add Unified Wang-QUBO Feature And Energy Tests

**Files:**
- Modify: `tests/test_qubo.py`
- Later modify: `scripts/benchmark_compact_beam_qubo_n0.py`

- [ ] **Step 1: Write failing feature extraction test**

Add this test near the existing Wang/QUBO tests in `tests/test_qubo.py`:

```python
def test_unified_wang_qubo_features_capture_prefix_pair_and_diversity():
    from benchmark_compact_beam_qubo_n0 import _unified_wang_qubo_features
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    good_code, good_n = _encode([[5, 4], [3, 1, 2], []], stack_count, max_height)
    peer_code, peer_n = _encode([[5], [4, 3], [2, 1]], stack_count, max_height)
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
    )

    features = _unified_wang_qubo_features(
        good_code,
        (good_n, 6, b"p0"),
        [(good_code, (good_n, 6, b"p0")), (peer_code, (peer_n, 7, b"p0"))],
        args,
    )

    assert features.fixed_prefix >= 2
    assert features.fix_pair > 0
    assert features.target_gap >= 0
    assert features.same_parent_count == 1
    assert features.depth == 6
    assert features.n_value == good_n
```

- [ ] **Step 2: Write failing energy ordering test**

Add:

```python
def test_unified_wang_qubo_energy_prefers_prefix_and_pair_gain():
    from benchmark_compact_beam_qubo_n0 import _unified_wang_qubo_linear_energy
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    better_code, better_n = _encode([[5, 4], [3, 1, 2], []], stack_count, max_height)
    worse_code, worse_n = _encode([[4, 5], [3, 1, 2], []], stack_count, max_height)
    rows = [
        (better_code, (better_n, 12, b"p1")),
        (worse_code, (worse_n, 12, b"p2")),
    ]
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        unified_wang_fixed_prefix_weight=20.0,
        unified_wang_fix_pair_weight=8.0,
        unified_wang_target_gap_weight=1.0,
        unified_wang_same_parent_penalty=0.0,
        unified_wang_depth_weight=1.0,
        unified_wang_n_weight=1000.0,
    )

    better_energy = _unified_wang_qubo_linear_energy(better_code, rows[0][1], rows, args)
    worse_energy = _unified_wang_qubo_linear_energy(worse_code, rows[1][1], rows, args)

    assert better_energy < worse_energy
```

- [ ] **Step 3: Run tests to verify RED**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_unified_wang_qubo_features_capture_prefix_pair_and_diversity tests/test_qubo.py::test_unified_wang_qubo_energy_prefers_prefix_and_pair_gain -q --basetemp .pytest_tmp_unified_wang_energy_red
```

Expected:

```text
FAILED ... ImportError: cannot import name '_unified_wang_qubo_features'
```

The failure must be a missing-helper failure for `_unified_wang_qubo_features` or `_unified_wang_qubo_linear_energy`, not a syntax error in the new tests.

- [ ] **Step 4: Add feature dataclass and helpers**

In `scripts/benchmark_compact_beam_qubo_n0.py`, add this dataclass near `WangQuboFeatures`:

```python
@dataclass(frozen=True)
class UnifiedWangQuboFeatures:
    n_value: int
    depth: int
    fixed_prefix: int
    fix_pair: int
    target_gap: int
    same_parent_count: int
```

Add these helpers near `_wang_qubo_feature_values(...)`:

```python
def _unified_wang_qubo_features(
    code: bytes,
    data: tuple[int, int, bytes],
    rows: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
) -> UnifiedWangQuboFeatures:
    n_value, depth, parent = data
    _score, fix_pair, fixed_prefix, target_gap = _candidate_wang_probe_score(code, args)
    same_parent_count = sum(1 for other_code, other_data in rows if other_code != code and other_data[2] == parent)
    return UnifiedWangQuboFeatures(
        n_value=int(n_value),
        depth=int(depth),
        fixed_prefix=int(fixed_prefix),
        fix_pair=int(fix_pair),
        target_gap=int(target_gap),
        same_parent_count=int(same_parent_count),
    )


def _unified_wang_qubo_linear_energy(
    code: bytes,
    data: tuple[int, int, bytes],
    rows: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
) -> float:
    features = _unified_wang_qubo_features(code, data, rows, args)
    return (
        float(getattr(args, "unified_wang_n_weight", 1000.0) or 0.0) * float(features.n_value)
        + float(getattr(args, "unified_wang_depth_weight", 1.0) or 0.0) * float(features.depth)
        - float(getattr(args, "unified_wang_fixed_prefix_weight", 0.0) or 0.0) * float(features.fixed_prefix)
        - float(getattr(args, "unified_wang_fix_pair_weight", 0.0) or 0.0) * float(features.fix_pair)
        + float(getattr(args, "unified_wang_target_gap_weight", 0.0) or 0.0) * float(features.target_gap)
        + float(getattr(args, "unified_wang_same_parent_penalty", 0.0) or 0.0) * float(features.same_parent_count)
    )
```

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_unified_wang_qubo_features_capture_prefix_pair_and_diversity tests/test_qubo.py::test_unified_wang_qubo_energy_prefers_prefix_and_pair_gain -q --basetemp .pytest_tmp_unified_wang_energy_green
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit Task 1**

Run:

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py tests\test_qubo.py
git commit -m "Add unified Wang QUBO energy helpers"
```

---

### Task 2: Add Unified QUBO Selector

**Files:**
- Modify: `tests/test_qubo.py`
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`

- [ ] **Step 1: Write failing selector test**

Add this test near `test_wang_pair_qubo_supplement_selects_high_fix_pair_candidate`:

```python
def test_unified_wang_qubo_supplement_selects_best_energy_candidate():
    from benchmark_compact_beam_qubo_n0 import _select_unified_wang_qubo_supplement
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    preferred_code, preferred_n = _encode([[5, 4], [3, 1, 2], []], stack_count, max_height)
    other_code, other_n = _encode([[4, 5], [3, 1, 2], []], stack_count, max_height)
    rows = [
        (other_code, (other_n, 8, b"p0")),
        (preferred_code, (preferred_n, 8, b"p1")),
    ]
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        unified_wang_qubo_max_vars=8,
        unified_wang_qubo_select=1,
        unified_wang_n_weight=1000.0,
        unified_wang_depth_weight=1.0,
        unified_wang_fixed_prefix_weight=30.0,
        unified_wang_fix_pair_weight=10.0,
        unified_wang_target_gap_weight=1.0,
        unified_wang_same_parent_penalty=0.25,
        unified_wang_signature_penalty=0.0,
        unified_wang_complement_bonus=0.0,
        qubo_penalty_a=30.0,
        qubo_reads=2,
        qubo_iterations=20,
        qubo_time_limit=0.001,
        seed=37,
    )

    selected, qvars, qtime, metrics = _select_unified_wang_qubo_supplement(rows, args, target=1)

    assert selected == [preferred_code]
    assert qvars == 2
    assert qtime >= 0.0
    assert metrics["fixed_prefix_selected"] >= 2
    assert metrics["fix_pair_selected"] >= 1
```

- [ ] **Step 2: Run selector test to verify RED**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_unified_wang_qubo_supplement_selects_best_energy_candidate -q --basetemp .pytest_tmp_unified_selector_red
```

Expected:

```text
FAILED ... ImportError: cannot import name '_select_unified_wang_qubo_supplement'
```

- [ ] **Step 3: Add selector helper**

In `scripts/benchmark_compact_beam_qubo_n0.py`, add this helper near `_select_wang_pair_qubo_supplement(...)`:

```python
def _select_unified_wang_qubo_supplement(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    target: int,
) -> tuple[list[bytes], int, float, dict[str, int]]:
    target = max(0, int(target))
    if target <= 0 or not ranked:
        return [], 0, 0.0, {"fixed_prefix_selected": 0, "fix_pair_selected": 0}
    max_vars = max(1, int(getattr(args, "unified_wang_qubo_max_vars", 64) or 64))
    usable = ranked[:max_vars]
    target = min(target, len(usable))
    if target <= 0:
        return [], 0, 0.0, {"fixed_prefix_selected": 0, "fix_pair_selected": 0}

    energies = [
        _unified_wang_qubo_linear_energy(code, data, usable, args)
        for code, data in usable
    ]
    scale = max(1.0, max(abs(value) for value in energies))
    model = QuboModel()
    vars_ = [model.add_binary_var(f"uw_{idx}") for idx in range(len(usable))]
    for idx, energy in enumerate(energies):
        model.add_linear(vars_[idx], float(energy) / scale)

    same_parent_penalty = float(getattr(args, "unified_wang_same_parent_penalty", 0.0) or 0.0)
    signature_penalty = float(getattr(args, "unified_wang_signature_penalty", 0.0) or 0.0)
    complement_bonus = float(getattr(args, "unified_wang_complement_bonus", 0.0) or 0.0)
    features = [_unified_wang_qubo_features(code, data, usable, args) for code, data in usable]
    for i in range(len(usable)):
        parent_i = usable[i][1][2]
        signature_i = (features[i].n_value, features[i].target_gap)
        for j in range(i + 1, len(usable)):
            if same_parent_penalty > 0 and parent_i == usable[j][1][2]:
                model.add_quadratic(vars_[i], vars_[j], same_parent_penalty)
            signature_j = (features[j].n_value, features[j].target_gap)
            if signature_penalty > 0 and signature_i == signature_j:
                model.add_quadratic(vars_[i], vars_[j], signature_penalty)
            if complement_bonus > 0 and features[i].fix_pair > 0 and features[j].fixed_prefix > 0:
                model.add_quadratic(vars_[i], vars_[j], -complement_bonus)

    model.add_exactly_k_penalty(vars_, target, float(getattr(args, "qubo_penalty_a", 30.0)))
    sample_set = SimulatedAnnealingQuboSampler().sample(
        model,
        QuboSamplerConfig(
            num_reads=max(1, int(getattr(args, "qubo_reads", 2) or 2)),
            random_seed=int(getattr(args, "seed", 37)) + 9001 + int(getattr(args, "_unified_wang_qubo_calls_done", 0)),
            time_limit_sec=max(0.001, float(getattr(args, "qubo_time_limit", 0.001) or 0.001)),
            sa_iterations=max(1, int(getattr(args, "qubo_iterations", 20) or 20)),
        ),
    )
    picked: set[int] = set()
    if sample_set.samples:
        picked = {idx for idx in range(len(usable)) if sample_set.samples[0].binary_values.get(idx, 0) == 1}
    while len(picked) > target:
        picked.remove(max(picked, key=lambda idx: (energies[idx], idx)))
    while len(picked) < target:
        missing = [idx for idx in range(len(usable)) if idx not in picked]
        if not missing:
            break
        picked.add(min(missing, key=lambda idx: (energies[idx], idx)))

    ordered = sorted(picked, key=lambda idx: (energies[idx], usable[idx][1][0], usable[idx][1][1], usable[idx][0]))
    selected_features = [features[idx] for idx in ordered]
    metrics = {
        "fixed_prefix_selected": sum(item.fixed_prefix for item in selected_features),
        "fix_pair_selected": sum(item.fix_pair for item in selected_features),
    }
    return [usable[idx][0] for idx in ordered], len(model.var_index), sample_set.runtime_sec, metrics
```

- [ ] **Step 4: Run selector test to verify GREEN**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_unified_wang_qubo_supplement_selects_best_energy_candidate -q --basetemp .pytest_tmp_unified_selector_green
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py tests\test_qubo.py
git commit -m "Add unified Wang QUBO selector"
```

---

### Task 3: Wire Unified Selector Into Compact Frontier And Result Metadata

**Files:**
- Modify: `tests/test_qubo.py`
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`

- [ ] **Step 1: Write failing frontier wiring test**

Add:

```python
def test_unified_wang_selector_is_used_inside_enhanced_frontier(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as bench
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    chosen_code, chosen_n = _encode([[5, 4], [3, 1, 2], []], stack_count, max_height)
    other_code, other_n = _encode([[4, 5], [3, 1, 2], []], stack_count, max_height)
    ranked = [
        (other_code, (other_n, 8, b"p0")),
        (chosen_code, (chosen_n, 8, b"p1")),
    ]
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        use_qubo=True,
        qubo_min_candidates=1,
        max_qubo_calls=0,
        unified_wang_qubo_enabled=True,
        unified_wang_qubo_supplement_width=1,
        unified_wang_qubo_focus_n=-1,
        unified_wang_qubo_n_window=1,
        unified_wang_qubo_select=1,
    )

    def fake_selector(rows, selector_args, target):
        assert rows
        assert target == 1
        return [chosen_code], 3, 0.002, {"fixed_prefix_selected": 4, "fix_pair_selected": 2}

    monkeypatch.setattr(bench, "_select_unified_wang_qubo_supplement", fake_selector)

    selected, selected_by_qubo, qcalls, qvars, qtime = bench._select_enhanced_qubo_frontier(
        ranked,
        candidate_count=len(ranked),
        args=args,
        beam_width=2,
        core_width=1,
        qubo_supplement_width=0,
    )

    assert chosen_code in selected
    assert chosen_code in selected_by_qubo
    assert qcalls == 1
    assert qvars == 3
    assert qtime == 0.002
    assert args._unified_wang_qubo_calls_done == 1
    assert args._unified_wang_fixed_prefix_selected == 4
    assert args._unified_wang_fix_pair_selected == 2
```

- [ ] **Step 2: Run frontier wiring test to verify RED**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_unified_wang_selector_is_used_inside_enhanced_frontier -q --basetemp .pytest_tmp_unified_frontier_red
```

Expected:

```text
FAILED ... AssertionError
```

because `_select_enhanced_qubo_frontier(...)` does not call the unified selector yet.

- [ ] **Step 3: Add metadata fields and counters**

In `COMPACT_EXTRA_FIELDS`, add after `wang_feature_same_parent_weight`:

```python
    "unified_wang_qubo_enabled",
    "unified_wang_qubo_calls",
    "unified_wang_qubo_vars",
    "unified_wang_qubo_time_sec",
    "unified_wang_fixed_prefix_selected",
    "unified_wang_fix_pair_selected",
```

In `CompactBeamResult`, add fields after `wang_pair_qubo_time_sec`:

```python
    unified_wang_qubo_calls: int = 0
    unified_wang_qubo_vars: int = 0
    unified_wang_qubo_time_sec: float = 0.0
    unified_wang_fixed_prefix_selected: int = 0
    unified_wang_fix_pair_selected: int = 0
```

In `run_compact_beam_instance(...)`, initialize these private counters near existing `_wang_pair_qubo_*` counters:

```python
    args._unified_wang_qubo_calls_done = 0
    args._unified_wang_qubo_vars = 0
    args._unified_wang_qubo_time = 0.0
    args._unified_wang_fixed_prefix_selected = 0
    args._unified_wang_fix_pair_selected = 0
```

When constructing `CompactBeamResult`, pass:

```python
        unified_wang_qubo_calls=args._unified_wang_qubo_calls_done,
        unified_wang_qubo_vars=args._unified_wang_qubo_vars,
        unified_wang_qubo_time_sec=args._unified_wang_qubo_time,
        unified_wang_fixed_prefix_selected=args._unified_wang_fixed_prefix_selected,
        unified_wang_fix_pair_selected=args._unified_wang_fix_pair_selected,
```

- [ ] **Step 4: Wire selector into `_select_enhanced_qubo_frontier(...)`**

Insert this block after the Wang pair-QUBO block and before the generic `qubo_enabled` block:

```python
    unified_width = max(0, int(getattr(args, "unified_wang_qubo_supplement_width", 0) or 0))
    unified_focus_n = int(getattr(args, "unified_wang_qubo_focus_n", 8))
    if (
        getattr(args, "unified_wang_qubo_enabled", False)
        and args.use_qubo
        and unified_width > 0
        and len(selected) < effective_beam
        and (unified_focus_n < 0 or int(best_n) <= unified_focus_n)
    ):
        n_window = max(0, int(getattr(args, "unified_wang_qubo_n_window", 1) or 0))
        remaining = [
            (code, data)
            for code, data in ranked
            if code not in selected and data[0] <= int(best_n) + n_window
        ]
        target = min(
            unified_width,
            effective_beam - len(selected),
            int(getattr(args, "unified_wang_qubo_select", unified_width) or unified_width),
        )
        boosted, add_qvars, add_qtime, metrics = _select_unified_wang_qubo_supplement(remaining, args, target)
        if add_qvars:
            qcalls += 1
            qvars += add_qvars
            qtime += add_qtime
            args._unified_wang_qubo_calls_done = int(getattr(args, "_unified_wang_qubo_calls_done", 0)) + 1
            args._unified_wang_qubo_vars = int(getattr(args, "_unified_wang_qubo_vars", 0)) + add_qvars
            args._unified_wang_qubo_time = float(getattr(args, "_unified_wang_qubo_time", 0.0)) + add_qtime
            args._unified_wang_fixed_prefix_selected = int(getattr(args, "_unified_wang_fixed_prefix_selected", 0)) + int(metrics.get("fixed_prefix_selected", 0))
            args._unified_wang_fix_pair_selected = int(getattr(args, "_unified_wang_fix_pair_selected", 0)) + int(metrics.get("fix_pair_selected", 0))
            for code in boosted:
                if code not in selected:
                    selected_by_qubo.add(code)
                selected.setdefault(code, None)
                if len(selected) >= effective_beam:
                    break
```

- [ ] **Step 5: Add row output fields**

In `_row_dict_from_instance_result(...)`, add after the Wang feature weight fields:

```python
            "unified_wang_qubo_enabled": getattr(args, "unified_wang_qubo_enabled", False),
            "unified_wang_qubo_calls": result.unified_wang_qubo_calls,
            "unified_wang_qubo_vars": result.unified_wang_qubo_vars,
            "unified_wang_qubo_time_sec": result.unified_wang_qubo_time_sec,
            "unified_wang_fixed_prefix_selected": result.unified_wang_fixed_prefix_selected,
            "unified_wang_fix_pair_selected": result.unified_wang_fix_pair_selected,
```

- [ ] **Step 6: Run frontier wiring test to verify GREEN**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_unified_wang_selector_is_used_inside_enhanced_frontier -q --basetemp .pytest_tmp_unified_frontier_green
```

Expected:

```text
1 passed
```

- [ ] **Step 7: Commit Task 3**

Run:

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py tests\test_qubo.py
git commit -m "Wire unified Wang QUBO frontier selector"
```

---

### Task 4: Add Unified Profile And Open Runner Alias

**Files:**
- Modify: `tests/test_qubo.py`
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `scripts/open_cpmp_benchmarks.py`

- [ ] **Step 1: Write failing profile test**

Add near existing profile tests:

```python
def test_solver_profile_scheduler_state_v7_wang_qubo_unified_is_single_loop(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v7_wang_qubo_unified")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v7_wang_qubo_unified"
    assert args.selection_policy == "scheduler_state_qubo"
    assert args.scheduler_state_controller == "state_v2"
    assert args.unified_wang_qubo_enabled is True
    assert args.unified_wang_qubo_supplement_width > 0
    assert args.unified_wang_fixed_prefix_weight > 0.0
    assert args.unified_wang_fix_pair_weight > 0.0
    assert args.wang_incumbent_enabled is False
    assert args.wang_rollout_enabled is False
    assert args.post_polish_qubo is False
```

- [ ] **Step 2: Write failing open runner alias test**

Add near `test_open_runner_accepts_wang_feature_and_rescue_polish_variants`:

```python
def test_open_runner_accepts_unified_wang_qubo_variant():
    from open_cpmp_benchmarks import _args_for_variant, _variant_names, build_parser

    outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v7_wang_qubo_unified"])
    args = _args_for_variant(outer, "qubo_scheduler_state_v7_wang_qubo_unified")

    assert args.solver_profile == "scheduler_state_v7_wang_qubo_unified"
    assert "qubo_scheduler_state_v7_wang_qubo_unified" in _variant_names("wang_all")
```

- [ ] **Step 3: Run profile and alias tests to verify RED**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_solver_profile_scheduler_state_v7_wang_qubo_unified_is_single_loop tests/test_qubo.py::test_open_runner_accepts_unified_wang_qubo_variant -q --basetemp .pytest_tmp_unified_profile_red
```

Expected:

```text
FAILED ... invalid choice: 'scheduler_state_v7_wang_qubo_unified'
```

or alias/profile missing failures.

- [ ] **Step 4: Add default profile fields**

In `_SOLVER_PROFILES["speed_v2"]`, add after Wang feature defaults:

```python
        "unified_wang_qubo_enabled": False,
        "unified_wang_qubo_supplement_width": 0,
        "unified_wang_qubo_focus_n": 8,
        "unified_wang_qubo_n_window": 1,
        "unified_wang_qubo_max_vars": 64,
        "unified_wang_qubo_select": 12,
        "unified_wang_n_weight": 1000.0,
        "unified_wang_depth_weight": 1.0,
        "unified_wang_fixed_prefix_weight": 0.0,
        "unified_wang_fix_pair_weight": 0.0,
        "unified_wang_target_gap_weight": 0.0,
        "unified_wang_same_parent_penalty": 0.0,
        "unified_wang_signature_penalty": 0.0,
        "unified_wang_complement_bonus": 0.0,
```

Add parser flags near the Wang feature parser flags:

```python
    parser.add_argument("--unified-wang-qubo-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--unified-wang-qubo-supplement-width", type=int)
    parser.add_argument("--unified-wang-qubo-focus-n", type=int)
    parser.add_argument("--unified-wang-qubo-n-window", type=int)
    parser.add_argument("--unified-wang-qubo-max-vars", type=int)
    parser.add_argument("--unified-wang-qubo-select", type=int)
    parser.add_argument("--unified-wang-n-weight", type=float)
    parser.add_argument("--unified-wang-depth-weight", type=float)
    parser.add_argument("--unified-wang-fixed-prefix-weight", type=float)
    parser.add_argument("--unified-wang-fix-pair-weight", type=float)
    parser.add_argument("--unified-wang-target-gap-weight", type=float)
    parser.add_argument("--unified-wang-same-parent-penalty", type=float)
    parser.add_argument("--unified-wang-signature-penalty", type=float)
    parser.add_argument("--unified-wang-complement-bonus", type=float)
```

There is no separate `_PROFILE_ARG_MAP` in this file. Do not add one. `make_compact_args(...)` records explicit override keys automatically through `_profile_explicit_keys`; parser defaults of `None` are enough for CLI overrides to remain explicit.

- [ ] **Step 5: Add `scheduler_state_v7_wang_qubo_unified`**

In `_SOLVER_PROFILES`, add after `scheduler_state_v6`/existing Wang profiles:

```python
_SOLVER_PROFILES["scheduler_state_v7_wang_qubo_unified"] = {
    **_SOLVER_PROFILES["scheduler_state_v5"],
    "group_qubo_supplement": True,
    "max_group_qubo_calls": 64,
    "qubo_group_max_vars": 96,
    "qubo_group_select": 18,
    "qubo_group_take": 3,
    "group_qubo_risk_gate": True,
    "group_qubo_risk_min_candidates": 10000,
    "group_qubo_risk_parent_share": 0.18,
    "group_qubo_risk_plateau_window": 3,
    "group_qubo_risk_plateau_delta": 0,
    "unified_wang_qubo_enabled": True,
    "unified_wang_qubo_supplement_width": 12,
    "unified_wang_qubo_focus_n": -1,
    "unified_wang_qubo_n_window": 1,
    "unified_wang_qubo_max_vars": 64,
    "unified_wang_qubo_select": 12,
    "unified_wang_n_weight": 1000.0,
    "unified_wang_depth_weight": 1.0,
    "unified_wang_fixed_prefix_weight": 20.0,
    "unified_wang_fix_pair_weight": 8.0,
    "unified_wang_target_gap_weight": 1.0,
    "unified_wang_same_parent_penalty": 0.25,
    "unified_wang_signature_penalty": 0.10,
    "unified_wang_complement_bonus": 0.05,
    "wang_incumbent_enabled": False,
    "wang_cap_enabled": False,
    "wang_rollout_enabled": False,
    "wang_pair_qubo_enabled": False,
    "post_polish_qubo": False,
}
```

- [ ] **Step 6: Add open runner alias**

In `scripts/open_cpmp_benchmarks.py`, add `"qubo_scheduler_state_v7_wang_qubo_unified"` to:

- `build_parser()` `--variant` choices.
- `_variant_names("wang_all")`.

In `_args_for_variant(...)`, add:

```python
    if variant == "qubo_scheduler_state_v7_wang_qubo_unified":
        overrides.update({"solver_profile": "scheduler_state_v7_wang_qubo_unified"})
```

- [ ] **Step 7: Run profile and alias tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_qubo.py::test_solver_profile_scheduler_state_v7_wang_qubo_unified_is_single_loop tests/test_qubo.py::test_open_runner_accepts_unified_wang_qubo_variant -q --basetemp .pytest_tmp_unified_profile_green
```

Expected:

```text
2 passed
```

- [ ] **Step 8: Commit Task 4**

Run:

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py tests\test_qubo.py
git commit -m "Add unified Wang QUBO profile"
```

---

### Task 5: Smoke, Benchmark Documentation, And Guardrails

**Files:**
- Modify: `docs/experiments/wang_qubo_full_quality_trial_summary.md`

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m pytest tests/test_qubo.py tests/test_wang_failure_diagnostics.py tests/test_classical_cpmp_baselines.py tests/test_classical_cpmp_runner.py -q --basetemp .pytest_tmp_unified_wang_full
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

- [ ] **Step 3: Run tiny open smoke**

Run:

```powershell
python scripts\open_cpmp_benchmarks.py --families 5x5_cv1_test --limit-per-family 2 --variant qubo_scheduler_state_v7_wang_qubo_unified --time-limit 3 --output docs\experiments\scratch_unified_wang_qubo_smoke.csv
```

Expected: exit code `0` and ignored scratch CSV/summary written.

- [ ] **Step 4: Run 20-case open smoke**

Run:

```powershell
python scripts\open_cpmp_benchmarks.py --families 5x5_cv1_test --limit-per-family 20 --variant qubo_scheduler_state_v7_wang_qubo_unified --time-limit 10 --output docs\experiments\scratch_unified_wang_qubo_5x5_cv1_20.csv
```

Expected: exit code `0` and ignored scratch CSV/summary written.

- [ ] **Step 5: Update experiment summary**

Append this section to `docs/experiments/wang_qubo_full_quality_trial_summary.md`:

```markdown
## Unified Wang-QUBO Beam Probe

`scheduler_state_v7_wang_qubo_unified` is a single-loop probe. It uses Wang fixed-prefix and fix-pair signals inside the compact QUBO frontier selector, rather than chaining Wang fallback, rescue, and post-polish profiles.

The first smoke commands are:

```powershell
python scripts\open_cpmp_benchmarks.py --families 5x5_cv1_test --limit-per-family 2 --variant qubo_scheduler_state_v7_wang_qubo_unified --time-limit 3 --output docs\experiments\scratch_unified_wang_qubo_smoke.csv
python scripts\open_cpmp_benchmarks.py --families 5x5_cv1_test --limit-per-family 20 --variant qubo_scheduler_state_v7_wang_qubo_unified --time-limit 10 --output docs\experiments\scratch_unified_wang_qubo_5x5_cv1_20.csv
```

The probe should be promoted only if full-case quality improves against the current Wang-feature or Wang-rescue profiles. It should not be judged by solved-case move totals alone.
```

- [ ] **Step 6: Run whitespace check**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors. Git user-ignore permission warnings and CRLF warnings are non-blocking.

- [ ] **Step 7: Commit Task 5**

Run:

```powershell
git add docs\experiments\wang_qubo_full_quality_trial_summary.md
git commit -m "Document unified Wang QUBO probe"
```

---

## Final Verification

- [ ] **Step 1: Run final tests**

Run:

```powershell
python -m pytest tests/test_qubo.py tests/test_wang_failure_diagnostics.py tests/test_classical_cpmp_baselines.py tests/test_classical_cpmp_runner.py tests/test_dlts_original_runner.py tests/test_paper_bbs_probe.py -q --basetemp .pytest_tmp_unified_wang_final
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

## Execution Notes

- This plan intentionally does not add a sequential portfolio runner.
- This plan intentionally does not call `WangBsBabyMovesSolver` or `WangBsGiantMovesSolver` from the unified profile.
- If the unified profile underperforms, keep it experimental rather than replacing `scheduler_state_v5`.
