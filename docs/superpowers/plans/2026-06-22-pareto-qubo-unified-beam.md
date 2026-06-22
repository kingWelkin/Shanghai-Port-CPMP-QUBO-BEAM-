# Pareto QUBO Unified Beam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `scheduler_state_v5_pareto_union_polish`, a unified Pareto-seeded QUBO beam selector that keeps the current legal child generator and tries to improve both open120 and paper BF/CV results without fallback stacking.

**Architecture:** The implementation extends the existing compact beam/QUBO script. A bounded feature extractor scores legal child states on quality, space, repairability, risk, and structural diversity; a deterministic Pareto seed picks objective representatives; one QUBO fill then selects the remaining exploration states with pair penalties. Runner wiring exposes the profile and telemetry for paired benchmark decisions.

**Tech Stack:** Python, existing compact byte-state beam search, `QuboModel`, `SimulatedAnnealingQuboSampler`, pytest, PowerShell benchmark runners.

---

### Task 1: Add Focused Failing Tests for Pareto Seed

**Files:**
- Modify: `tests/test_qubo.py`

- [ ] **Step 1: Add seed behavior tests**

Append these tests near the existing union diversity tests in `tests/test_qubo.py`:

```python
def _pareto_union_feature(
    *,
    rank: int,
    parent: bytes,
    source: int,
    dest: int,
    signature: tuple[int, ...],
    n_value: int = 4,
    space: float = 0.0,
    repair: float = 0.0,
    risk: float = 0.0,
):
    from benchmark_compact_beam_qubo_n0 import ParetoUnionFeatures

    return ParetoUnionFeatures(
        energy=float(n_value * 1000 + rank),
        n_value=n_value,
        depth=rank,
        parent_code=parent,
        source_stack=source,
        dest_stack=dest,
        source_dest=(source, dest),
        signature=signature,
        reduce_signature=(source, signature[0] if signature else 0, 0),
        n_band=n_value // 2,
        space_score=space,
        repair_score=repair,
        risk_score=risk,
        quality_rank=rank,
    )


def test_pareto_union_seed_keeps_distinct_objective_winners():
    from benchmark_compact_beam_qubo_n0 import _select_pareto_union_seed_indices

    args = _compact_args(pareto_union_seed_width=4)
    features = [
        _pareto_union_feature(rank=0, parent=b"p0", source=0, dest=1, signature=(0, 0), space=0.1, repair=0.1, risk=0.8),
        _pareto_union_feature(rank=1, parent=b"p0", source=0, dest=1, signature=(0, 0), space=0.9, repair=0.1, risk=0.7),
        _pareto_union_feature(rank=2, parent=b"p1", source=1, dest=2, signature=(1, 0), space=0.2, repair=0.9, risk=0.6),
        _pareto_union_feature(rank=3, parent=b"p2", source=2, dest=3, signature=(2, 0), space=0.3, repair=0.2, risk=0.1),
        _pareto_union_feature(rank=4, parent=b"p3", source=3, dest=0, signature=(3, 0), space=0.2, repair=0.2, risk=0.5),
    ]

    picked = _select_pareto_union_seed_indices(features, args, target=4)

    assert len(picked) == 4
    assert 0 in picked
    assert 1 in picked
    assert 2 in picked
    assert 3 in picked


def test_pareto_union_seed_respects_target_and_rank_trim():
    from benchmark_compact_beam_qubo_n0 import _select_pareto_union_seed_indices

    args = _compact_args(pareto_union_seed_width=2)
    features = [
        _pareto_union_feature(rank=idx, parent=bytes([idx]), source=idx, dest=idx + 1, signature=(idx,))
        for idx in range(5)
    ]

    picked = _select_pareto_union_seed_indices(features, args, target=2)

    assert len(picked) == 2
    assert picked == [0, 1]
```

- [ ] **Step 2: Run tests and verify they fail for missing symbols**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py -k "pareto_union_seed" -q --basetemp .tmp_pytest
```

Expected result before implementation:

```text
FAILED ... ImportError: cannot import name 'ParetoUnionFeatures'
FAILED ... ImportError: cannot import name '_select_pareto_union_seed_indices'
```

- [ ] **Step 3: Commit failing tests**

Run:

```powershell
git add tests\test_qubo.py
git commit -m "test: cover pareto union seed selection"
```

### Task 2: Implement Pareto Feature Extraction and Seed Selection

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `tests/test_qubo.py`

- [ ] **Step 1: Add profile fields and defaults**

In `COMPACT_EXTRA_FIELDS`, add these fields immediately after the existing `union_diversity_qubo_selected_total` field:

```python
    "pareto_union_enabled",
    "pareto_union_candidate_window",
    "pareto_union_seed_width",
    "pareto_union_qubo_width",
    "pareto_union_max_vars",
    "pareto_union_n_band_size",
    "pareto_union_quality_weight",
    "pareto_union_repair_weight",
    "pareto_union_space_weight",
    "pareto_union_risk_weight",
    "pareto_union_same_parent_penalty",
    "pareto_union_same_source_penalty",
    "pareto_union_same_dest_penalty",
    "pareto_union_same_source_dest_penalty",
    "pareto_union_signature_penalty",
    "pareto_union_reduce_signature_penalty",
    "pareto_union_n_band_penalty",
    "pareto_union_risk_bucket_penalty",
    "pareto_union_layers",
    "pareto_union_seed_selected_total",
    "pareto_union_qubo_calls",
    "pareto_union_qubo_vars",
    "pareto_union_qubo_time_sec",
    "pareto_union_qubo_selected_total",
    "pareto_union_quality_selected",
    "pareto_union_space_selected",
    "pareto_union_repair_selected",
    "pareto_union_risk_selected",
    "pareto_union_distinct_parent_count_last",
    "pareto_union_distinct_source_count_last",
    "pareto_union_distinct_dest_count_last",
    "pareto_union_distinct_signature_count_last",
```

In `make_compact_args`, add these disabled defaults near the existing union diversity defaults:

```python
        "pareto_union_enabled": False,
        "pareto_union_candidate_window": 128,
        "pareto_union_seed_width": 32,
        "pareto_union_qubo_width": 32,
        "pareto_union_max_vars": 96,
        "pareto_union_n_band_size": 2,
        "pareto_union_quality_weight": 1.00,
        "pareto_union_repair_weight": 0.20,
        "pareto_union_space_weight": 0.12,
        "pareto_union_risk_weight": 0.16,
        "pareto_union_same_parent_penalty": 0.55,
        "pareto_union_same_source_penalty": 0.25,
        "pareto_union_same_dest_penalty": 0.20,
        "pareto_union_same_source_dest_penalty": 0.35,
        "pareto_union_signature_penalty": 0.20,
        "pareto_union_reduce_signature_penalty": 0.15,
        "pareto_union_n_band_penalty": 0.08,
        "pareto_union_risk_bucket_penalty": 0.08,
```

- [ ] **Step 2: Add feature type and feature helpers**

Add this code near `UnionDiversityQuboFeatures`:

```python
class ParetoUnionFeatures(NamedTuple):
    energy: float
    n_value: int
    depth: int
    parent_code: bytes
    source_stack: int
    dest_stack: int
    source_dest: tuple[int, int]
    signature: tuple[int, ...]
    reduce_signature: tuple[int, ...]
    n_band: int
    space_score: float
    repair_score: float
    risk_score: float
    quality_rank: int


def _transition_dest_stack(parent_code: bytes, code: bytes, stack_count: int, max_height: int) -> int:
    if not parent_code or stack_count <= 0 or max_height <= 0:
        return -1
    before = _split(parent_code, stack_count, max_height)
    after = _split(code, stack_count, max_height)
    for idx, (before_chunk, after_chunk) in enumerate(zip(before, after)):
        if _chunk_height_raw(after_chunk) > _chunk_height_raw(before_chunk):
            return idx
    return -1


def _pareto_union_space_score(code: bytes, stack_count: int, max_height: int) -> float:
    if stack_count <= 0 or max_height <= 0:
        return 0.0
    chunks = _split(code, stack_count, max_height)
    heights = [_chunk_height_raw(chunk) for chunk in chunks]
    empty_slots = sum(max_height - height for height in heights)
    avg_pressure = sum(heights) / max(1.0, float(stack_count * max_height))
    max_pressure = max(heights, default=0) / max(1.0, float(max_height))
    return max(0.0, float(empty_slots) / max(1.0, float(stack_count * max_height)) - 0.25 * avg_pressure - 0.25 * max_pressure)


def _pareto_union_risk_score(code: bytes, stack_count: int, max_height: int) -> float:
    if stack_count <= 0 or max_height <= 0:
        return 0.0
    stacks = _code_to_stacks(code, stack_count, max_height)
    disorder = future_disorder_count_stacks(stacks)
    total = sum(len(stack) for stack in stacks)
    return float(disorder) / max(1.0, float(total))
```

- [ ] **Step 3: Add feature builder**

Add this code after `_union_diversity_qubo_features`:

```python
def _pareto_union_features(
    code: bytes,
    data: tuple[int, int, bytes],
    rank: int,
    args: argparse.Namespace,
) -> ParetoUnionFeatures:
    n_value, depth, parent_code = data
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    base = _union_diversity_qubo_features(code, data, rank, args)
    source_stack = base.source_stack
    dest_stack = _transition_dest_stack(parent_code, code, stack_count, max_height)
    fixed = _compact_wang_fixed_prefix_score(code, stack_count, max_height) if stack_count > 0 and max_height > 0 else 0
    pair = _compact_wang_fix_pair_score(code, stack_count, max_height) if stack_count > 0 and max_height > 0 else 0
    repair_score = float(fixed + pair)
    space_score = _pareto_union_space_score(code, stack_count, max_height)
    risk_score = _pareto_union_risk_score(code, stack_count, max_height)
    n_band_size = max(1, int(getattr(args, "pareto_union_n_band_size", 2) or 2))
    quality_weight = float(getattr(args, "pareto_union_quality_weight", 1.0) or 1.0)
    repair_weight = float(getattr(args, "pareto_union_repair_weight", 0.20) or 0.0)
    space_weight = float(getattr(args, "pareto_union_space_weight", 0.12) or 0.0)
    risk_weight = float(getattr(args, "pareto_union_risk_weight", 0.16) or 0.0)
    energy = quality_weight * float(base.energy) - repair_weight * repair_score - space_weight * space_score + risk_weight * risk_score
    return ParetoUnionFeatures(
        energy=float(energy),
        n_value=int(n_value),
        depth=int(depth),
        parent_code=parent_code,
        source_stack=int(source_stack),
        dest_stack=int(dest_stack),
        source_dest=(int(source_stack), int(dest_stack)),
        signature=base.signature,
        reduce_signature=base.reduce_signature,
        n_band=int(n_value) // n_band_size,
        space_score=float(space_score),
        repair_score=float(repair_score),
        risk_score=float(risk_score),
        quality_rank=int(rank),
    )
```

- [ ] **Step 4: Add seed selector**

Add this code after `_pareto_union_features`:

```python
def _select_pareto_union_seed_indices(
    features: list[ParetoUnionFeatures],
    args: argparse.Namespace,
    target: int,
) -> list[int]:
    target = max(0, min(int(target), len(features)))
    if target <= 0:
        return []
    picked: list[int] = []
    seen: set[int] = set()

    def add(idx: int) -> None:
        if idx not in seen and len(picked) < target:
            seen.add(idx)
            picked.append(idx)

    objective_orders = [
        sorted(range(len(features)), key=lambda idx: (features[idx].energy, features[idx].quality_rank, idx)),
        sorted(range(len(features)), key=lambda idx: (-features[idx].space_score, features[idx].quality_rank, idx)),
        sorted(range(len(features)), key=lambda idx: (-features[idx].repair_score, features[idx].quality_rank, idx)),
        sorted(range(len(features)), key=lambda idx: (features[idx].risk_score, features[idx].quality_rank, idx)),
    ]
    for order in objective_orders:
        if order:
            add(order[0])

    for attr in ("source_dest", "parent_code", "signature"):
        representatives: dict[object, int] = {}
        for idx, feature in enumerate(features):
            key = getattr(feature, attr)
            if key not in representatives:
                representatives[key] = idx
        for idx in sorted(representatives.values(), key=lambda value: (features[value].quality_rank, value)):
            add(idx)
            if len(picked) >= target:
                break
        if len(picked) >= target:
            break

    for idx in sorted(range(len(features)), key=lambda value: (features[value].quality_rank, value)):
        add(idx)
        if len(picked) >= target:
            break
    return picked
```

- [ ] **Step 5: Run seed tests and verify they pass**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py -k "pareto_union_seed" -q --basetemp .tmp_pytest
```

Expected result:

```text
2 passed
```

- [ ] **Step 6: Commit feature and seed implementation**

Run:

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py tests\test_qubo.py
git commit -m "feat: add pareto union seed features"
```

### Task 3: Add QUBO Fill and Exact-K Repair

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `tests/test_qubo.py`

- [ ] **Step 1: Add repair and objective tests**

Append these tests near the seed tests:

```python
def test_pareto_union_repair_avoids_same_parent_when_quality_tied():
    from benchmark_compact_beam_qubo_n0 import _repair_pareto_union_selection

    args = _compact_args(
        pareto_union_same_parent_penalty=5.0,
        pareto_union_same_source_penalty=0.0,
        pareto_union_same_dest_penalty=0.0,
        pareto_union_same_source_dest_penalty=0.0,
        pareto_union_signature_penalty=0.0,
        pareto_union_reduce_signature_penalty=0.0,
        pareto_union_n_band_penalty=0.0,
        pareto_union_risk_bucket_penalty=0.0,
    )
    features = [
        _pareto_union_feature(rank=0, parent=b"same", source=0, dest=1, signature=(0,)),
        _pareto_union_feature(rank=1, parent=b"same", source=1, dest=2, signature=(1,)),
        _pareto_union_feature(rank=2, parent=b"diff", source=2, dest=3, signature=(2,)),
    ]
    picked = _repair_pareto_union_selection({0, 1}, 2, [0.0, 0.0, 0.01], features, args)

    assert picked == {0, 2}


def test_pareto_union_qubo_selects_exact_target(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _select_pareto_union_qubo_supplement
    from container_reshuffle.qubo.sampler import QuboSampleSet

    def fake_sample(self, model, config):
        return QuboSampleSet(samples=[QuboSample({0: 1, 1: 1, 2: 1}, 0.0)], sampler_name="fake", runtime_sec=0.0)

    monkeypatch.setattr("benchmark_compact_beam_qubo_n0.SimulatedAnnealingQuboSampler.sample", fake_sample)
    args = _compact_args(
        use_qubo=True,
        deterministic_qubo_sampler=True,
        pareto_union_max_vars=4,
        pareto_union_same_parent_penalty=5.0,
    )
    features = [
        _pareto_union_feature(rank=0, parent=b"same", source=0, dest=1, signature=(0,)),
        _pareto_union_feature(rank=1, parent=b"same", source=1, dest=2, signature=(1,)),
        _pareto_union_feature(rank=2, parent=b"diff", source=2, dest=3, signature=(2,)),
        _pareto_union_feature(rank=3, parent=b"other", source=3, dest=0, signature=(3,)),
    ]
    usable = [(bytes([idx + 1]), (feature.n_value, feature.depth, feature.parent_code)) for idx, feature in enumerate(features)]

    selected, qvars, _qtime = _select_pareto_union_qubo_supplement(usable, features, args, target=2)

    assert len(selected) == 2
    assert qvars == 4
```

- [ ] **Step 2: Run tests and verify they fail for missing functions**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py -k "pareto_union_repair or pareto_union_qubo_selects" -q --basetemp .tmp_pytest
```

Expected result before implementation:

```text
FAILED ... ImportError: cannot import name '_repair_pareto_union_selection'
FAILED ... ImportError: cannot import name '_select_pareto_union_qubo_supplement'
```

- [ ] **Step 3: Add objective, repair, and QUBO supplement**

Add this code after `_select_pareto_union_seed_indices`:

```python
def _risk_bucket(value: float) -> int:
    return max(0, min(4, int(float(value) * 5.0)))


def _pareto_union_objective(
    selected: set[int],
    linear: list[float],
    features: list[ParetoUnionFeatures],
    args: argparse.Namespace,
) -> float:
    value = sum(linear[idx] for idx in selected)
    penalties = {
        "parent": float(getattr(args, "pareto_union_same_parent_penalty", 0.55) or 0.0),
        "source": float(getattr(args, "pareto_union_same_source_penalty", 0.25) or 0.0),
        "dest": float(getattr(args, "pareto_union_same_dest_penalty", 0.20) or 0.0),
        "source_dest": float(getattr(args, "pareto_union_same_source_dest_penalty", 0.35) or 0.0),
        "signature": float(getattr(args, "pareto_union_signature_penalty", 0.20) or 0.0),
        "reduce": float(getattr(args, "pareto_union_reduce_signature_penalty", 0.15) or 0.0),
        "n_band": float(getattr(args, "pareto_union_n_band_penalty", 0.08) or 0.0),
        "risk": float(getattr(args, "pareto_union_risk_bucket_penalty", 0.08) or 0.0),
    }
    ordered = sorted(selected)
    for pos, i in enumerate(ordered):
        for j in ordered[pos + 1 :]:
            if penalties["parent"] > 0.0 and features[i].parent_code == features[j].parent_code:
                value += penalties["parent"]
            if penalties["source"] > 0.0 and features[i].source_stack >= 0 and features[i].source_stack == features[j].source_stack:
                value += penalties["source"]
            if penalties["dest"] > 0.0 and features[i].dest_stack >= 0 and features[i].dest_stack == features[j].dest_stack:
                value += penalties["dest"]
            if penalties["source_dest"] > 0.0 and features[i].source_dest == features[j].source_dest:
                value += penalties["source_dest"]
            if penalties["signature"] > 0.0 and features[i].signature == features[j].signature:
                value += penalties["signature"]
            if penalties["reduce"] > 0.0 and features[i].reduce_signature == features[j].reduce_signature:
                value += penalties["reduce"]
            if penalties["n_band"] > 0.0 and features[i].n_band == features[j].n_band:
                value += penalties["n_band"]
            if penalties["risk"] > 0.0 and _risk_bucket(features[i].risk_score) == _risk_bucket(features[j].risk_score):
                value += penalties["risk"]
    return float(value)


def _repair_pareto_union_selection(
    picked: set[int],
    target: int,
    linear: list[float],
    features: list[ParetoUnionFeatures],
    args: argparse.Namespace,
) -> set[int]:
    picked = {idx for idx in picked if 0 <= idx < len(linear)}
    target = max(0, min(target, len(linear)))
    while len(picked) > target:
        picked.remove(max(picked, key=lambda idx: (linear[idx], idx)))
    while len(picked) < target:
        missing = [idx for idx in range(len(linear)) if idx not in picked]
        if not missing:
            break
        picked.add(min(missing, key=lambda idx: (_pareto_union_objective(picked | {idx}, linear, features, args), linear[idx], idx)))
    improved = True
    while improved and len(picked) == target:
        improved = False
        current = _pareto_union_objective(picked, linear, features, args)
        best_swap: tuple[int, int, float] | None = None
        for out_idx in sorted(picked):
            base = picked - {out_idx}
            for in_idx in range(len(linear)):
                if in_idx in picked:
                    continue
                trial = base | {in_idx}
                value = _pareto_union_objective(trial, linear, features, args)
                if value + 1e-12 < current and (best_swap is None or value < best_swap[2] - 1e-12):
                    best_swap = (out_idx, in_idx, value)
        if best_swap is not None:
            picked.remove(best_swap[0])
            picked.add(best_swap[1])
            improved = True
    return picked
```

Then add the QUBO helper:

```python
def _select_pareto_union_qubo_supplement(
    usable: list[tuple[bytes, tuple[int, int, bytes]]],
    features: list[ParetoUnionFeatures],
    args: argparse.Namespace,
    target: int,
) -> tuple[list[bytes], int, float]:
    target = max(0, min(int(target), len(usable), len(features)))
    if target <= 0:
        return [], 0, 0.0
    max_vars = max(1, int(getattr(args, "pareto_union_max_vars", 96) or 96))
    usable = list(usable[:max_vars])
    features = list(features[: len(usable)])
    target = min(target, len(usable))
    linear = _normalize_values([feature.energy for feature in features])
    model = QuboModel()
    vars_ = [model.add_binary_var(f"pu_{idx}") for idx in range(len(usable))]
    for idx, value in enumerate(linear):
        model.add_linear(vars_[idx], value)
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            pair = _pareto_union_objective({i, j}, [0.0] * len(usable), features, args)
            if pair > 0.0:
                model.add_quadratic(vars_[i], vars_[j], pair)
    model.add_exactly_k_penalty(vars_, target, float(getattr(args, "qubo_penalty_a", 30.0) or 30.0))
    sample_set = SimulatedAnnealingQuboSampler().sample(
        model,
        _qubo_sampler_config(
            args,
            num_reads=int(getattr(args, "qubo_reads", 2) or 2),
            random_seed=int(getattr(args, "seed", 37)) + 17011 + int(getattr(args, "_pareto_union_qubo_calls_done", 0)),
            time_limit_sec=float(getattr(args, "qubo_time_limit", 0.001) or 0.001),
            sa_iterations=int(getattr(args, "qubo_iterations", 20) or 20),
        ),
    )
    picked: set[int] = set()
    if sample_set.samples:
        picked = {idx for idx in range(len(usable)) if sample_set.samples[0].binary_values.get(idx, 0) == 1}
    picked = _repair_pareto_union_selection(picked, target, linear, features, args)
    ordered = sorted(picked, key=lambda idx: (features[idx].energy, features[idx].quality_rank, usable[idx][0]))
    return [usable[idx][0] for idx in ordered], len(model.var_index), sample_set.runtime_sec
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py -k "pareto_union" -q --basetemp .tmp_pytest
```

Expected result:

```text
4 passed
```

- [ ] **Step 5: Commit QUBO fill implementation**

Run:

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py tests\test_qubo.py
git commit -m "feat: add pareto union qubo fill"
```

### Task 4: Wire Pareto Selector into Compact Frontier and Telemetry

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `tests/test_qubo.py`

- [ ] **Step 1: Add compact profile test**

Append this test near other solver profile tests:

```python
def test_solver_profile_scheduler_state_v5_pareto_union_polish_extends_union():
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_pareto_union_polish")

    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v5_pareto_union_polish"
    assert args.pareto_union_enabled is True
    assert args.pareto_union_seed_width == 32
    assert args.pareto_union_qubo_width == 32
    assert args.union_diversity_qubo_enabled is True
    assert args.frg_seed_enabled is True
```

- [ ] **Step 2: Add profile preset**

Add this after `scheduler_state_v5_qubo_diversity_union_rollout_polish`:

```python
_SOLVER_PROFILES["scheduler_state_v5_pareto_union_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_polish"],
    "pareto_union_enabled": True,
    "pareto_union_candidate_window": 128,
    "pareto_union_seed_width": 32,
    "pareto_union_qubo_width": 32,
    "pareto_union_max_vars": 96,
    "pareto_union_n_band_size": 2,
    "pareto_union_quality_weight": 1.00,
    "pareto_union_repair_weight": 0.20,
    "pareto_union_space_weight": 0.12,
    "pareto_union_risk_weight": 0.16,
    "pareto_union_same_parent_penalty": 0.55,
    "pareto_union_same_source_penalty": 0.25,
    "pareto_union_same_dest_penalty": 0.20,
    "pareto_union_same_source_dest_penalty": 0.35,
    "pareto_union_signature_penalty": 0.20,
    "pareto_union_reduce_signature_penalty": 0.15,
    "pareto_union_n_band_penalty": 0.08,
    "pareto_union_risk_bucket_penalty": 0.08,
}
```

- [ ] **Step 3: Add profile override keys**

In `_apply_solver_profile`, add explicit flag mappings for each `pareto_union_*` profile key:

```python
        "pareto_union_enabled": ("--pareto-union-enabled", "--no-pareto-union-enabled"),
        "pareto_union_candidate_window": "--pareto-union-candidate-window",
        "pareto_union_seed_width": "--pareto-union-seed-width",
        "pareto_union_qubo_width": "--pareto-union-qubo-width",
        "pareto_union_max_vars": "--pareto-union-max-vars",
        "pareto_union_n_band_size": "--pareto-union-n-band-size",
        "pareto_union_quality_weight": "--pareto-union-quality-weight",
        "pareto_union_repair_weight": "--pareto-union-repair-weight",
        "pareto_union_space_weight": "--pareto-union-space-weight",
        "pareto_union_risk_weight": "--pareto-union-risk-weight",
        "pareto_union_same_parent_penalty": "--pareto-union-same-parent-penalty",
        "pareto_union_same_source_penalty": "--pareto-union-same-source-penalty",
        "pareto_union_same_dest_penalty": "--pareto-union-same-dest-penalty",
        "pareto_union_same_source_dest_penalty": "--pareto-union-same-source-dest-penalty",
        "pareto_union_signature_penalty": "--pareto-union-signature-penalty",
        "pareto_union_reduce_signature_penalty": "--pareto-union-reduce-signature-penalty",
        "pareto_union_n_band_penalty": "--pareto-union-n-band-penalty",
        "pareto_union_risk_bucket_penalty": "--pareto-union-risk-bucket-penalty",
```

- [ ] **Step 4: Add runtime counters**

In the search initialization block near `_union_diversity_qubo_calls_done`, add:

```python
    args._pareto_union_layers = 0
    args._pareto_union_seed_selected_total = 0
    args._pareto_union_qubo_calls_done = 0
    args._pareto_union_qubo_vars = 0
    args._pareto_union_qubo_time = 0.0
    args._pareto_union_qubo_selected_total = 0
    args._pareto_union_quality_selected = 0
    args._pareto_union_space_selected = 0
    args._pareto_union_repair_selected = 0
    args._pareto_union_risk_selected = 0
    args._pareto_union_distinct_parent_count_last = 0
    args._pareto_union_distinct_source_count_last = 0
    args._pareto_union_distinct_dest_count_last = 0
    args._pareto_union_distinct_signature_count_last = 0
```

- [ ] **Step 5: Add selection helper**

Add this helper near `_select_union_diversity_qubo_supplement`:

```python
def _select_pareto_union_supplement(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    target: int,
) -> tuple[list[bytes], set[bytes], int, int, float]:
    target = max(0, int(target))
    if target <= 0 or not ranked:
        return [], set(), 0, 0, 0.0
    window = max(1, int(getattr(args, "pareto_union_candidate_window", 128) or 128))
    usable = list(ranked[:window])
    features = [_pareto_union_features(code, data, rank, args) for rank, (code, data) in enumerate(usable)]
    seed_target = min(target, max(0, int(getattr(args, "pareto_union_seed_width", 32) or 0)), len(usable))
    seed_indices = _select_pareto_union_seed_indices(features, args, seed_target)
    selected_codes: list[bytes] = [usable[idx][0] for idx in seed_indices]
    selected_set = set(selected_codes)
    qcalls = qvars = 0
    qtime = 0.0
    remaining_target = min(target - len(selected_codes), max(0, int(getattr(args, "pareto_union_qubo_width", 32) or 0)))
    if remaining_target > 0:
        q_usable = [(code, data) for code, data in usable if code not in selected_set]
        q_features = [features[idx] for idx, (code, _data) in enumerate(usable) if code not in selected_set]
        q_codes, qvars, qtime = _select_pareto_union_qubo_supplement(q_usable, q_features, args, remaining_target)
        if qvars:
            qcalls = 1
        for code in q_codes:
            if code not in selected_set:
                selected_set.add(code)
                selected_codes.append(code)
    args._pareto_union_seed_selected_total += len(seed_indices)
    args._pareto_union_qubo_selected_total += max(0, len(selected_codes) - len(seed_indices))
    args._pareto_union_quality_selected += int(0 in seed_indices)
    args._pareto_union_space_selected += sum(1 for idx in seed_indices if features[idx].space_score > 0.0)
    args._pareto_union_repair_selected += sum(1 for idx in seed_indices if features[idx].repair_score > 0.0)
    args._pareto_union_risk_selected += sum(1 for idx in seed_indices if features[idx].risk_score <= 0.25)
    selected_features = [features[idx] for idx, (code, _data) in enumerate(usable) if code in selected_set]
    args._pareto_union_distinct_parent_count_last = len({feature.parent_code for feature in selected_features})
    args._pareto_union_distinct_source_count_last = len({feature.source_stack for feature in selected_features})
    args._pareto_union_distinct_dest_count_last = len({feature.dest_stack for feature in selected_features})
    args._pareto_union_distinct_signature_count_last = len({feature.signature for feature in selected_features})
    return selected_codes[:target], set(selected_codes[len(seed_indices) :]), qcalls, qvars, qtime
```

- [ ] **Step 6: Wire helper into `_select_enhanced_qubo_frontier`**

Insert this block before the existing `union_diversity_qubo_enabled` block:

```python
    pareto_width = max(0, int(getattr(args, "pareto_union_seed_width", 0) or 0)) + max(0, int(getattr(args, "pareto_union_qubo_width", 0) or 0))
    if (
        getattr(args, "pareto_union_enabled", False)
        and pareto_width > 0
        and len(selected) < effective_beam
        and candidate_count >= max(1, int(getattr(args, "union_diversity_qubo_min_candidates", 64) or 64))
    ):
        n_window = max(0, int(getattr(args, "union_diversity_qubo_n_window", 2) or 0))
        remaining = [
            (code, data)
            for code, data in ranked
            if code not in selected and data[0] <= int(best_n) + n_window
        ]
        target = min(pareto_width, effective_beam - len(selected), len(remaining))
        boosted, boosted_by_qubo, add_qcalls, add_qvars, add_qtime = _select_pareto_union_supplement(remaining, args, target)
        if boosted:
            args._pareto_union_layers += 1
            qcalls += add_qcalls
            qvars += add_qvars
            qtime += add_qtime
            args._pareto_union_qubo_calls_done += add_qcalls
            args._pareto_union_qubo_vars += add_qvars
            args._pareto_union_qubo_time += add_qtime
            for code in boosted:
                if code not in selected:
                    if code in boosted_by_qubo:
                        selected_by_qubo.add(code)
                        selected_source_by_code.setdefault(code, "pareto_union_qubo")
                    else:
                        selected_source_by_code.setdefault(code, "pareto_union_seed")
                selected.setdefault(code, None)
                if len(selected) >= effective_beam:
                    break
```

- [ ] **Step 7: Add result dataclass and row telemetry**

In `CompactRunResult`, add:

```python
    pareto_union_layers: int = 0
    pareto_union_seed_selected_total: int = 0
    pareto_union_qubo_calls: int = 0
    pareto_union_qubo_vars: int = 0
    pareto_union_qubo_time_sec: float = 0.0
    pareto_union_qubo_selected_total: int = 0
    pareto_union_quality_selected: int = 0
    pareto_union_space_selected: int = 0
    pareto_union_repair_selected: int = 0
    pareto_union_risk_selected: int = 0
    pareto_union_distinct_parent_count_last: int = 0
    pareto_union_distinct_source_count_last: int = 0
    pareto_union_distinct_dest_count_last: int = 0
    pareto_union_distinct_signature_count_last: int = 0
```

In the `CompactRunResult(...)` construction, pass:

```python
        pareto_union_layers=args._pareto_union_layers,
        pareto_union_seed_selected_total=args._pareto_union_seed_selected_total,
        pareto_union_qubo_calls=args._pareto_union_qubo_calls_done,
        pareto_union_qubo_vars=args._pareto_union_qubo_vars,
        pareto_union_qubo_time_sec=args._pareto_union_qubo_time,
        pareto_union_qubo_selected_total=args._pareto_union_qubo_selected_total,
        pareto_union_quality_selected=args._pareto_union_quality_selected,
        pareto_union_space_selected=args._pareto_union_space_selected,
        pareto_union_repair_selected=args._pareto_union_repair_selected,
        pareto_union_risk_selected=args._pareto_union_risk_selected,
        pareto_union_distinct_parent_count_last=args._pareto_union_distinct_parent_count_last,
        pareto_union_distinct_source_count_last=args._pareto_union_distinct_source_count_last,
        pareto_union_distinct_dest_count_last=args._pareto_union_distinct_dest_count_last,
        pareto_union_distinct_signature_count_last=args._pareto_union_distinct_signature_count_last,
```

In `_row_from_result`, add matching row values:

```python
            "pareto_union_enabled": getattr(args, "pareto_union_enabled", False),
            "pareto_union_candidate_window": getattr(args, "pareto_union_candidate_window", 0),
            "pareto_union_seed_width": getattr(args, "pareto_union_seed_width", 0),
            "pareto_union_qubo_width": getattr(args, "pareto_union_qubo_width", 0),
            "pareto_union_max_vars": getattr(args, "pareto_union_max_vars", 0),
            "pareto_union_n_band_size": getattr(args, "pareto_union_n_band_size", 0),
            "pareto_union_quality_weight": getattr(args, "pareto_union_quality_weight", 0.0),
            "pareto_union_repair_weight": getattr(args, "pareto_union_repair_weight", 0.0),
            "pareto_union_space_weight": getattr(args, "pareto_union_space_weight", 0.0),
            "pareto_union_risk_weight": getattr(args, "pareto_union_risk_weight", 0.0),
            "pareto_union_same_parent_penalty": getattr(args, "pareto_union_same_parent_penalty", 0.0),
            "pareto_union_same_source_penalty": getattr(args, "pareto_union_same_source_penalty", 0.0),
            "pareto_union_same_dest_penalty": getattr(args, "pareto_union_same_dest_penalty", 0.0),
            "pareto_union_same_source_dest_penalty": getattr(args, "pareto_union_same_source_dest_penalty", 0.0),
            "pareto_union_signature_penalty": getattr(args, "pareto_union_signature_penalty", 0.0),
            "pareto_union_reduce_signature_penalty": getattr(args, "pareto_union_reduce_signature_penalty", 0.0),
            "pareto_union_n_band_penalty": getattr(args, "pareto_union_n_band_penalty", 0.0),
            "pareto_union_risk_bucket_penalty": getattr(args, "pareto_union_risk_bucket_penalty", 0.0),
            "pareto_union_layers": result.pareto_union_layers,
            "pareto_union_seed_selected_total": result.pareto_union_seed_selected_total,
            "pareto_union_qubo_calls": result.pareto_union_qubo_calls,
            "pareto_union_qubo_vars": result.pareto_union_qubo_vars,
            "pareto_union_qubo_time_sec": result.pareto_union_qubo_time_sec,
            "pareto_union_qubo_selected_total": result.pareto_union_qubo_selected_total,
            "pareto_union_quality_selected": result.pareto_union_quality_selected,
            "pareto_union_space_selected": result.pareto_union_space_selected,
            "pareto_union_repair_selected": result.pareto_union_repair_selected,
            "pareto_union_risk_selected": result.pareto_union_risk_selected,
            "pareto_union_distinct_parent_count_last": result.pareto_union_distinct_parent_count_last,
            "pareto_union_distinct_source_count_last": result.pareto_union_distinct_source_count_last,
            "pareto_union_distinct_dest_count_last": result.pareto_union_distinct_dest_count_last,
            "pareto_union_distinct_signature_count_last": result.pareto_union_distinct_signature_count_last,
```

- [ ] **Step 8: Run compact focused tests**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py -k "pareto_union or scheduler_state_v5_pareto_union" -q --basetemp .tmp_pytest
```

Expected result:

```text
5 passed
```

- [ ] **Step 9: Commit compact integration**

Run:

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py tests\test_qubo.py
git commit -m "feat: wire pareto union beam profile"
```

### Task 5: Wire Open and Paper Runners

**Files:**
- Modify: `scripts/open_cpmp_benchmarks.py`
- Modify: `scripts/benchmark_compact_paper_set.py`
- Modify: `tests/test_qubo.py`

- [ ] **Step 1: Add runner acceptance tests**

Append these tests near existing open runner variant tests:

```python
def test_open_runner_accepts_pareto_union_variant():
    from open_cpmp_benchmarks import _args_for_variant, build_parser

    parsed = build_parser().parse_args(["--variant", "qubo_scheduler_state_v5_pareto_union_polish"])
    args = _args_for_variant(parsed, "qubo_scheduler_state_v5_pareto_union_polish")

    assert args.solver_profile == "scheduler_state_v5_pareto_union_polish"


def test_paper_runner_accepts_pareto_union_passthrough():
    from benchmark_compact_paper_set import build_parser

    args = build_parser().parse_args([
        "--variant",
        "qubo_scheduler_state_v5_pareto_union_polish",
        "--pareto-union-seed-width",
        "16",
        "--pareto-union-qubo-width",
        "8",
    ])

    assert args.variant == "qubo_scheduler_state_v5_pareto_union_polish"
    assert args.pareto_union_seed_width == 16
    assert args.pareto_union_qubo_width == 8
```

- [ ] **Step 2: Add open runner fields**

In `OPEN_EXTRA_FIELDS`, add the Pareto field names from Task 2. Add this helper near `_union_diversity_qubo_row_fields`:

```python
def _pareto_union_row_fields(result, args: argparse.Namespace) -> dict[str, object]:
    return {
        "pareto_union_enabled": getattr(args, "pareto_union_enabled", False),
        "pareto_union_candidate_window": getattr(args, "pareto_union_candidate_window", 0),
        "pareto_union_seed_width": getattr(args, "pareto_union_seed_width", 0),
        "pareto_union_qubo_width": getattr(args, "pareto_union_qubo_width", 0),
        "pareto_union_max_vars": getattr(args, "pareto_union_max_vars", 0),
        "pareto_union_n_band_size": getattr(args, "pareto_union_n_band_size", 0),
        "pareto_union_quality_weight": getattr(args, "pareto_union_quality_weight", 0.0),
        "pareto_union_repair_weight": getattr(args, "pareto_union_repair_weight", 0.0),
        "pareto_union_space_weight": getattr(args, "pareto_union_space_weight", 0.0),
        "pareto_union_risk_weight": getattr(args, "pareto_union_risk_weight", 0.0),
        "pareto_union_layers": getattr(result, "pareto_union_layers", 0),
        "pareto_union_seed_selected_total": getattr(result, "pareto_union_seed_selected_total", 0),
        "pareto_union_qubo_calls": getattr(result, "pareto_union_qubo_calls", 0),
        "pareto_union_qubo_vars": getattr(result, "pareto_union_qubo_vars", 0),
        "pareto_union_qubo_time_sec": getattr(result, "pareto_union_qubo_time_sec", 0.0),
        "pareto_union_qubo_selected_total": getattr(result, "pareto_union_qubo_selected_total", 0),
        "pareto_union_distinct_parent_count_last": getattr(result, "pareto_union_distinct_parent_count_last", 0),
        "pareto_union_distinct_source_count_last": getattr(result, "pareto_union_distinct_source_count_last", 0),
        "pareto_union_distinct_dest_count_last": getattr(result, "pareto_union_distinct_dest_count_last", 0),
        "pareto_union_distinct_signature_count_last": getattr(result, "pareto_union_distinct_signature_count_last", 0),
    }
```

Call it in the row builder:

```python
    row.update(_pareto_union_row_fields(result, args))
```

- [ ] **Step 3: Add variant mapping**

Add the new variant string to `_variant_names`, the parser choices list, and `_args_for_variant`:

```python
    if variant == "qubo_scheduler_state_v5_pareto_union_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_pareto_union_polish"})
```

- [ ] **Step 4: Add passthrough keys**

In `_args_for_variant`, add `pareto_union_*` keys to the passthrough list. In `build_parser`, add CLI flags:

```python
    parser.add_argument("--pareto-union-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pareto-union-candidate-window", type=int)
    parser.add_argument("--pareto-union-seed-width", type=int)
    parser.add_argument("--pareto-union-qubo-width", type=int)
    parser.add_argument("--pareto-union-max-vars", type=int)
    parser.add_argument("--pareto-union-n-band-size", type=int)
    parser.add_argument("--pareto-union-quality-weight", type=float)
    parser.add_argument("--pareto-union-repair-weight", type=float)
    parser.add_argument("--pareto-union-space-weight", type=float)
    parser.add_argument("--pareto-union-risk-weight", type=float)
    parser.add_argument("--pareto-union-same-parent-penalty", type=float)
    parser.add_argument("--pareto-union-same-source-penalty", type=float)
    parser.add_argument("--pareto-union-same-dest-penalty", type=float)
    parser.add_argument("--pareto-union-same-source-dest-penalty", type=float)
    parser.add_argument("--pareto-union-signature-penalty", type=float)
    parser.add_argument("--pareto-union-reduce-signature-penalty", type=float)
    parser.add_argument("--pareto-union-n-band-penalty", type=float)
    parser.add_argument("--pareto-union-risk-bucket-penalty", type=float)
```

Add the same parser arguments to `scripts/benchmark_compact_paper_set.py`; it reuses `_args_for_variant`, so the parsed args only need to exist.

- [ ] **Step 5: Run runner tests**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py -k "pareto_union_variant or pareto_union_passthrough" -q --basetemp .tmp_pytest
```

Expected result:

```text
2 passed
```

- [ ] **Step 6: Commit runner wiring**

Run:

```powershell
git add scripts\open_cpmp_benchmarks.py scripts\benchmark_compact_paper_set.py tests\test_qubo.py
git commit -m "feat: expose pareto union benchmark profile"
```

### Task 6: Smoke, Paired Benchmark, and Decision Report

**Files:**
- Create: `docs/experiments/pareto_union_experiment_20260622.md`

- [ ] **Step 1: Run compile check**

Run:

```powershell
$env:PYTHONPATH='.'; python -m py_compile scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py scripts\benchmark_compact_paper_set.py
```

Expected result: command exits with code `0`.

- [ ] **Step 2: Run focused pytest slice**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py -k "pareto_union or qubo_diversity_union" -q --basetemp .tmp_pytest
```

Expected result: all selected tests pass.

- [ ] **Step 3: Run paper sample-per-family 1 smoke**

Run:

```powershell
$env:PYTHONPATH='.'; python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF CV --sample-per-family 1 --variant qubo_scheduler_state_v5_pareto_union_polish --time-limit 5 --alpha 10 --jobs 8 --output docs\experiments\paper_sample1_pareto_union_t5_20260622.csv
```

Expected result: CSV and summary are written; at least one row has `pareto_union_layers > 0`.

- [ ] **Step 4: Run paper sample10 paired benchmark**

Run baseline:

```powershell
$env:PYTHONPATH='.'; python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF CV --sample-per-family 10 --variant qubo_scheduler_state_v5_qubo_diversity_union_polish --time-limit 20 --alpha 10 --jobs 8 --output docs\experiments\paper_sample10_union_paired_t20_20260622.csv
```

Run Pareto:

```powershell
$env:PYTHONPATH='.'; python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF CV --sample-per-family 10 --variant qubo_scheduler_state_v5_pareto_union_polish --time-limit 20 --alpha 10 --jobs 8 --output docs\experiments\paper_sample10_pareto_union_t20_20260622.csv
```

Compute paired diff:

```powershell
$u = Import-Csv docs\experiments\paper_sample10_union_paired_t20_20260622.csv
$p = Import-Csv docs\experiments\paper_sample10_pareto_union_t20_20260622.csv
$key = { param($row) "$($row.dataset)|$($row.family)|$($row.case)" }
$umap=@{}; foreach($row in $u){ $umap[(& $key $row)]=$row }
$gained=@(); $lost=@(); $moveDelta=0; $common=0
foreach($row in $p){
  $k=& $key $row; $b=$umap[$k]
  if($null -eq $b){ continue }
  $bs=($b.goal_validation_ok -eq 'True' -and $b.final_n -eq '0' -and $b.moves -ne '')
  $ps=($row.goal_validation_ok -eq 'True' -and $row.final_n -eq '0' -and $row.moves -ne '')
  if($ps -and -not $bs){$gained+=$k}
  if($bs -and -not $ps){$lost+=$k}
  if($bs -and $ps){$common++; $moveDelta += [int]$row.moves - [int]$b.moves}
}
[pscustomobject]@{
  union_valid=($u|?{$_.goal_validation_ok -eq 'True' -and $_.final_n -eq '0' -and $_.moves -ne ''}).Count
  pareto_valid=($p|?{$_.goal_validation_ok -eq 'True' -and $_.final_n -eq '0' -and $_.moves -ne ''}).Count
  gained_count=$gained.Count
  lost_count=$lost.Count
  common=$common
  common_move_delta=$moveDelta
  union_penalty=(($u|%{[int]$_.full_quality_penalty})|Measure-Object -Sum).Sum
  pareto_penalty=(($p|%{[int]$_.full_quality_penalty})|Measure-Object -Sum).Sum
  pareto_layers=(($p|%{[int]$_.pareto_union_layers})|Measure-Object -Sum).Sum
  pareto_seed=(($p|%{[int]$_.pareto_union_seed_selected_total})|Measure-Object -Sum).Sum
  pareto_qubo=(($p|%{[int]$_.pareto_union_qubo_selected_total})|Measure-Object -Sum).Sum
} | ConvertTo-Json -Depth 4
```

- [ ] **Step 5: Run open120 only if paper gate is not worse**

Run this only when paper solved count does not drop below baseline and full penalty is not worse by more than `0`:

```powershell
$env:PYTHONPATH='.'; python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_pareto_union_polish --time-limit 10 --alpha 10 --jobs 8 --output docs\experiments\open120_pareto_union_t10_20260622.csv
```

- [ ] **Step 6: Write decision report**

Create `docs/experiments/pareto_union_experiment_20260622.md` with this structure:

```markdown
# Pareto Union Experiment 2026-06-22

## Artifacts

- Paper baseline: `docs/experiments/paper_sample10_union_paired_t20_20260622.csv`
- Paper Pareto: `docs/experiments/paper_sample10_pareto_union_t20_20260622.csv`
- Open120 Pareto: `docs/experiments/open120_pareto_union_t10_20260622.csv`

## Paper Paired Result

| metric | union | pareto | delta |
|---|---:|---:|---:|
| solved | 0 | 0 | 0 |
| solved moves | 0 | 0 | 0 |
| full penalty | 0 | 0 | 0 |

## Telemetry

| metric | value |
|---|---:|
| pareto layers | 0 |
| seed selected | 0 |
| qubo selected | 0 |

## Decision

Keep as ablation until paired benchmark metrics justify promotion.
```

Replace the zero values with measured values from the paired diff.

- [ ] **Step 7: Commit report**

Run:

```powershell
git add docs\experiments\pareto_union_experiment_20260622.md
git commit -m "docs: report pareto union benchmark"
```

## Self-Review

- Spec coverage: Tasks cover feature extraction, Pareto seed, QUBO fill, compact profile, telemetry, open runner, paper runner, smoke tests, paired paper benchmark, and open120 guard.
- Type consistency: `ParetoUnionFeatures`, `_select_pareto_union_seed_indices`, `_repair_pareto_union_selection`, `_select_pareto_union_qubo_supplement`, and `_select_pareto_union_supplement` are introduced before integration steps use them.
- Scope control: The plan changes only selection, profiles, telemetry, runner wiring, tests, and experiment docs. It does not add fallback solvers or change legal move generation.
