# Quality Reservoir Union Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an opt-in `quality_reservoir` Union + QUBO profile that moves strong-gated FRG benefits into continuous candidate quality scoring across open120, BF, and CV.

**Architecture:** Add a bounded reservoir between candidate generation and QUBO selection. The reservoir enriches core, Wang, and FRG candidates with continuous quality features, keeps a quality/diversity-balanced pool, and passes the filtered candidates into the existing Union QUBO selector. QUBO remains the only beam selector and FRG remains candidate supply only.

**Tech Stack:** Python, existing compact CPMP solver in `scripts/benchmark_compact_beam_qubo_n0.py`, open benchmark runner in `scripts/open_cpmp_benchmarks.py`, existing pytest suite in `tests/test_qubo.py`, CSV/Markdown benchmark outputs under `docs/experiments`.

---

## File Structure

- Modify `scripts/benchmark_compact_beam_qubo_n0.py`
  - Add `quality_reservoir_*` parameters to default args, extra CSV fields, CLI parser, profile maps, and solver result telemetry.
  - Add `QualityReservoirCandidate` and helper functions for feature enrichment, scoring, bounded keep selection, and selected-origin accounting.
  - Apply the reservoir after Wang/FRG candidate injection and before `_select_next_frontier`.
  - Add QUBO linear and pair energy terms using quality reservoir fields exposed through `UnionDiversityQuboFeatures`.
- Modify `scripts/open_cpmp_benchmarks.py`
  - Add the new open runner variant and `wang_research_all` wiring.
  - Serialize `quality_reservoir_*` telemetry for normal and priority-group rows.
- Modify `tests/test_qubo.py`
  - Add focused unit tests for profile wiring, feature defaults, reservoir filtering, QUBO energy, pair penalty, selected telemetry, and CSV serialization.
- Create benchmark artifacts only when executing validation:
  - `docs/experiments/quality_reservoir_open120_t10_<date>.csv`
  - `docs/experiments/quality_reservoir_bf_full_t10_<date>.csv`
  - `docs/experiments/quality_reservoir_cv_full_t10_<date>.csv`
  - `docs/experiments/quality_reservoir_scorecard_<date>.md`

## Task 1: Profile Parameters And Runner Wiring

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `scripts/open_cpmp_benchmarks.py`
- Test: `tests/test_qubo.py`

- [ ] **Step 1: Write the failing profile wiring test**

Add this test near the existing `test_open_and_paper_runners_accept_qubo_diversity_union_frg_soft_supply_qubo_variant` tests:

```python
def test_open_and_paper_runners_accept_qubo_diversity_union_quality_reservoir_variant():
    from benchmark_compact_beam_qubo_n0 import build_parser as compact_build_parser
    from benchmark_compact_paper_set import _args_for_variant as paper_args_for_variant
    from open_cpmp_benchmarks import _args_for_variant, _variant_names, build_parser

    variant = "qubo_scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish"
    outer = build_parser().parse_args(["--variant", variant])
    args = _args_for_variant(outer, variant)
    paper_args = paper_args_for_variant(variant, alpha=10, time_limit=10)
    compact_args = compact_build_parser().parse_args(["--solver-profile", args.solver_profile])

    assert args.solver_profile == "scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish"
    assert paper_args.solver_profile == "scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish"
    assert compact_args.quality_reservoir_enabled is True
    assert compact_args.frg_frontier_child_enabled is True
    assert compact_args.frg_frontier_child_append_only is False
    assert compact_args.union_diversity_qubo_adaptive_macro_quota_enabled is True
    assert compact_args.quality_reservoir_max_candidates > compact_args.beam_width
    assert variant in _variant_names("wang_research_all")
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .).Path
python -m pytest tests/test_qubo.py::test_open_and_paper_runners_accept_qubo_diversity_union_quality_reservoir_variant -q
```

Expected: FAIL because the variant and `quality_reservoir_*` args do not exist.

- [ ] **Step 3: Add compact defaults and CLI flags**

In `scripts/benchmark_compact_beam_qubo_n0.py`, add these names to `COMPACT_EXTRA_FIELDS` near existing Union/FRG fields:

```python
    "quality_reservoir_enabled",
    "quality_reservoir_max_candidates",
    "quality_reservoir_top_quality",
    "quality_reservoir_origin_reserve",
    "quality_reservoir_witness_reserve",
    "quality_reservoir_progress_reserve",
    "quality_reservoir_witness_weight",
    "quality_reservoir_progress_weight",
    "quality_reservoir_source_scarcity_weight",
    "quality_reservoir_receiver_debt_weight",
    "quality_reservoir_height_risk_weight",
    "quality_reservoir_chain_cost_weight",
    "quality_reservoir_pair_receiver_debt_penalty",
    "quality_reservoir_pair_macro_signature_penalty",
    "quality_reservoir_generated_total",
    "quality_reservoir_kept_total",
    "quality_reservoir_selected_total",
    "quality_reservoir_witness_found_total",
    "quality_reservoir_avg_witness_quality",
    "quality_reservoir_avg_receiver_debt",
    "quality_reservoir_avg_height_risk",
    "quality_reservoir_pair_penalty_total",
```

In the default config dictionary near other `union_diversity_qubo_*` defaults, add:

```python
        "quality_reservoir_enabled": False,
        "quality_reservoir_max_candidates": 0,
        "quality_reservoir_top_quality": 0,
        "quality_reservoir_origin_reserve": 0,
        "quality_reservoir_witness_reserve": 0,
        "quality_reservoir_progress_reserve": 0,
        "quality_reservoir_witness_weight": 0.0,
        "quality_reservoir_progress_weight": 0.0,
        "quality_reservoir_source_scarcity_weight": 0.0,
        "quality_reservoir_receiver_debt_weight": 0.0,
        "quality_reservoir_height_risk_weight": 0.0,
        "quality_reservoir_chain_cost_weight": 0.0,
        "quality_reservoir_pair_receiver_debt_penalty": 0.0,
        "quality_reservoir_pair_macro_signature_penalty": 0.0,
```

In the profile map, add:

```python
SCHEDULER_STATE_V5_QUBO_DIVERSITY_UNION_QUALITY_RESERVOIR_POLISH_PROFILE = {
    **SCHEDULER_STATE_V5_QUBO_DIVERSITY_UNION_FRG_SOFT_SUPPLY_QUBO_POLISH_PROFILE,
    "quality_reservoir_enabled": True,
    "quality_reservoir_max_candidates": 1600,
    "quality_reservoir_top_quality": 900,
    "quality_reservoir_origin_reserve": 48,
    "quality_reservoir_witness_reserve": 96,
    "quality_reservoir_progress_reserve": 96,
    "quality_reservoir_witness_weight": 0.42,
    "quality_reservoir_progress_weight": 0.28,
    "quality_reservoir_source_scarcity_weight": 0.18,
    "quality_reservoir_receiver_debt_weight": 0.26,
    "quality_reservoir_height_risk_weight": 0.18,
    "quality_reservoir_chain_cost_weight": 0.10,
    "quality_reservoir_pair_receiver_debt_penalty": 0.12,
    "quality_reservoir_pair_macro_signature_penalty": 0.10,
    "union_diversity_qubo_macro_quality_bonus": 0.18,
    "union_diversity_qubo_frg_witness_signature_penalty": 0.10,
}
```

Register the key `"scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish"` in the same solver-profile dictionary that contains the soft-supply profile.

Add flag-map entries in `_apply_solver_profile`:

```python
        "quality_reservoir_enabled": "--quality-reservoir-enabled",
        "quality_reservoir_max_candidates": "--quality-reservoir-max-candidates",
        "quality_reservoir_top_quality": "--quality-reservoir-top-quality",
        "quality_reservoir_origin_reserve": "--quality-reservoir-origin-reserve",
        "quality_reservoir_witness_reserve": "--quality-reservoir-witness-reserve",
        "quality_reservoir_progress_reserve": "--quality-reservoir-progress-reserve",
        "quality_reservoir_witness_weight": "--quality-reservoir-witness-weight",
        "quality_reservoir_progress_weight": "--quality-reservoir-progress-weight",
        "quality_reservoir_source_scarcity_weight": "--quality-reservoir-source-scarcity-weight",
        "quality_reservoir_receiver_debt_weight": "--quality-reservoir-receiver-debt-weight",
        "quality_reservoir_height_risk_weight": "--quality-reservoir-height-risk-weight",
        "quality_reservoir_chain_cost_weight": "--quality-reservoir-chain-cost-weight",
        "quality_reservoir_pair_receiver_debt_penalty": "--quality-reservoir-pair-receiver-debt-penalty",
        "quality_reservoir_pair_macro_signature_penalty": "--quality-reservoir-pair-macro-signature-penalty",
```

Add parser arguments near related Union parser arguments:

```python
    parser.add_argument("--quality-reservoir-enabled", action="store_true")
    parser.add_argument("--quality-reservoir-max-candidates", type=int, default=0)
    parser.add_argument("--quality-reservoir-top-quality", type=int, default=0)
    parser.add_argument("--quality-reservoir-origin-reserve", type=int, default=0)
    parser.add_argument("--quality-reservoir-witness-reserve", type=int, default=0)
    parser.add_argument("--quality-reservoir-progress-reserve", type=int, default=0)
    parser.add_argument("--quality-reservoir-witness-weight", type=float, default=0.0)
    parser.add_argument("--quality-reservoir-progress-weight", type=float, default=0.0)
    parser.add_argument("--quality-reservoir-source-scarcity-weight", type=float, default=0.0)
    parser.add_argument("--quality-reservoir-receiver-debt-weight", type=float, default=0.0)
    parser.add_argument("--quality-reservoir-height-risk-weight", type=float, default=0.0)
    parser.add_argument("--quality-reservoir-chain-cost-weight", type=float, default=0.0)
    parser.add_argument("--quality-reservoir-pair-receiver-debt-penalty", type=float, default=0.0)
    parser.add_argument("--quality-reservoir-pair-macro-signature-penalty", type=float, default=0.0)
```

- [ ] **Step 4: Add open and paper runner variant wiring**

In `scripts/open_cpmp_benchmarks.py`, add `"qubo_scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish"` to `wang_research_all`, parser choices, and `_args_for_variant`:

```python
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish"})
```

In `scripts/benchmark_compact_paper_set.py`, if it has an explicit variant map, add the same mapping:

```python
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish"})
```

- [ ] **Step 5: Run the profile wiring test**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .).Path
python -m pytest tests/test_qubo.py::test_open_and_paper_runners_accept_qubo_diversity_union_quality_reservoir_variant -q
```

Expected: PASS.

- [ ] **Step 6: Commit profile wiring**

Run:

```powershell
git add scripts/benchmark_compact_beam_qubo_n0.py scripts/open_cpmp_benchmarks.py scripts/benchmark_compact_paper_set.py tests/test_qubo.py
git commit -m "Add quality reservoir union profile wiring"
```

Expected: commit succeeds. If `scripts/benchmark_compact_paper_set.py` has no diff, omit it from `git add`.

## Task 2: Quality Feature Enrichment

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Test: `tests/test_qubo.py`

- [ ] **Step 1: Write failing neutral/default feature test**

Add this test near existing Union diversity feature tests:

```python
def test_quality_reservoir_feature_defaults_are_neutral_for_core_candidate():
    import argparse
    import benchmark_compact_beam_qubo_n0 as bench

    parent_stacks = [[3, 1], [], []]
    child_stacks = [[3], [1], []]
    parent_code, parent_n = bench._encode(parent_stacks, 3, 3)
    child_code, child_n = bench._encode(child_stacks, 3, 3)
    args = argparse.Namespace(
        _stack_count=3,
        _max_height=3,
        _candidate_origin_by_code={},
        _union_rollout_probe_current_results={},
        _reduction_chain_depth_by_code={},
        quality_reservoir_witness_weight=0.42,
        quality_reservoir_progress_weight=0.28,
        quality_reservoir_source_scarcity_weight=0.18,
        quality_reservoir_receiver_debt_weight=0.26,
        quality_reservoir_height_risk_weight=0.18,
        quality_reservoir_chain_cost_weight=0.10,
    )

    feature = bench._quality_reservoir_feature(
        child_code,
        (child_n, 1, parent_code),
        args,
    )

    assert feature.origin == "core"
    assert feature.n_delta == parent_n - child_n
    assert feature.witness_quality == 0.0
    assert 0.0 <= feature.receiver_debt <= 1.0
    assert 0.0 <= feature.height_risk <= 1.0
    assert feature.reservoir_quality == bench._quality_reservoir_score(feature, args)
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .).Path
python -m pytest tests/test_qubo.py::test_quality_reservoir_feature_defaults_are_neutral_for_core_candidate -q
```

Expected: FAIL because `_quality_reservoir_feature` is not defined.

- [ ] **Step 3: Add the feature type and helper functions**

In `scripts/benchmark_compact_beam_qubo_n0.py`, add this near `UnionDiversityQuboFeatures`:

```python
class QualityReservoirCandidate(NamedTuple):
    code: bytes
    n_value: int
    depth: int
    parent_code: bytes
    origin: str
    n_delta: int
    witness_quality: float
    source_scarcity_gain: float
    receiver_debt: float
    height_risk: float
    chain_length_penalty: float
    source_signature: int
    receiver_signature: int
    macro_signature: tuple[int, ...]
    n_band: int
    reservoir_quality: float = 0.0
```

Add these helpers near the existing `_union_diversity_completion_witness_quality` helpers:

```python
def _quality_reservoir_score(feature: QualityReservoirCandidate, args: argparse.Namespace) -> float:
    return float(
        float(getattr(args, "quality_reservoir_witness_weight", 0.0) or 0.0) * float(feature.witness_quality)
        + float(getattr(args, "quality_reservoir_progress_weight", 0.0) or 0.0) * max(0.0, float(feature.n_delta))
        + float(getattr(args, "quality_reservoir_source_scarcity_weight", 0.0) or 0.0) * float(feature.source_scarcity_gain)
        - float(getattr(args, "quality_reservoir_receiver_debt_weight", 0.0) or 0.0) * float(feature.receiver_debt)
        - float(getattr(args, "quality_reservoir_height_risk_weight", 0.0) or 0.0) * float(feature.height_risk)
        - float(getattr(args, "quality_reservoir_chain_cost_weight", 0.0) or 0.0) * float(feature.chain_length_penalty)
    )


def _quality_reservoir_feature(
    code: bytes,
    data: tuple[int, int, bytes],
    args: argparse.Namespace,
) -> QualityReservoirCandidate:
    n_value, depth, parent_code = data
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    origin = _union_diversity_candidate_origin(code, args)
    parent_n = _code_n(parent_code, stack_count, max_height) if parent_code and stack_count > 0 and max_height > 0 else int(n_value)
    n_delta = int(parent_n) - int(n_value)
    source_stack = _transition_source_stack(parent_code, code, stack_count, max_height) if stack_count > 0 and max_height > 0 else -1
    witness = (getattr(args, "_union_rollout_probe_current_results", {}) or {}).get(code)
    witness_quality = _union_diversity_completion_witness_quality(witness, args, n_value=int(n_value))
    source_scarcity = _union_diversity_source_scarcity(parent_code, source_stack, args)
    receiver_debt = 0.0
    height_risk = 0.0
    receiver_signature = -1
    if stack_count > 0 and max_height > 0 and parent_code:
        receiver = _receiver_safety_score(parent_code, code, args)
        receiver_debt = _clamp01(
            0.45 * max(0.0, -float(receiver.safe_receiver_delta))
            + 0.35 * float(receiver.height_pressure_after)
            + (0.20 if receiver.last_buffer_consumed else 0.0)
        )
        height_risk = _clamp01(float(receiver.height_pressure_after))
        receiver_signature = min(15, max(0, int(round(float(receiver.height_pressure_after) * 10.0))))
    chain_depth = int((getattr(args, "_reduction_chain_depth_by_code", {}) or {}).get(code, 1) or 1)
    chain_penalty = _clamp01(float(max(0, chain_depth - 1)) / 8.0)
    macro_signature = (
        min(15, max(-1, int(source_stack))),
        min(15, max(0, int(chain_depth))),
        min(15, max(0, int(n_value))),
        1 if str(origin) == "frg_frontier_child" else 0,
    )
    feature = QualityReservoirCandidate(
        code=code,
        n_value=int(n_value),
        depth=int(depth),
        parent_code=parent_code,
        origin=str(origin),
        n_delta=int(n_delta),
        witness_quality=float(witness_quality),
        source_scarcity_gain=float(source_scarcity if str(origin) == "frg_frontier_child" else 0.0),
        receiver_debt=float(receiver_debt),
        height_risk=float(height_risk),
        chain_length_penalty=float(chain_penalty),
        source_signature=int(source_stack),
        receiver_signature=int(receiver_signature),
        macro_signature=tuple(macro_signature),
        n_band=int(n_value) // max(1, int(getattr(args, "union_diversity_qubo_n_band_size", 1) or 1)),
    )
    return feature._replace(reservoir_quality=_quality_reservoir_score(feature, args))
```

- [ ] **Step 4: Run the feature test**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .).Path
python -m pytest tests/test_qubo.py::test_quality_reservoir_feature_defaults_are_neutral_for_core_candidate -q
```

Expected: PASS.

- [ ] **Step 5: Commit feature enrichment**

Run:

```powershell
git add scripts/benchmark_compact_beam_qubo_n0.py tests/test_qubo.py
git commit -m "Add quality reservoir candidate features"
```

Expected: commit succeeds.

## Task 3: Bounded Reservoir Filtering

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Test: `tests/test_qubo.py`

- [ ] **Step 1: Write failing reservoir keep test**

Add:

```python
def test_quality_reservoir_keeps_quality_and_origin_reserve():
    import argparse
    import benchmark_compact_beam_qubo_n0 as bench

    parent_code = b"parent"
    features = [
        bench.QualityReservoirCandidate(b"core_low", 8, 1, parent_code, "core", 0, 0.0, 0.0, 0.3, 0.2, 0.0, -1, 2, (), 8, -0.13),
        bench.QualityReservoirCandidate(b"core_high", 7, 1, parent_code, "core", 1, 0.0, 0.0, 0.1, 0.1, 0.0, -1, 1, (), 7, 0.23),
        bench.QualityReservoirCandidate(b"frg_high", 6, 2, parent_code, "frg_frontier_child", 2, 0.9, 0.7, 0.1, 0.2, 0.1, 0, 2, (0, 2, 6, 1), 6, 0.91),
        bench.QualityReservoirCandidate(b"wang_mid", 7, 1, parent_code, "wang_rollout", 1, 0.0, 0.0, 0.1, 0.2, 0.0, 1, 2, (), 7, 0.20),
    ]
    args = argparse.Namespace(
        quality_reservoir_top_quality=1,
        quality_reservoir_origin_reserve=1,
        quality_reservoir_witness_reserve=1,
        quality_reservoir_progress_reserve=1,
        quality_reservoir_max_candidates=3,
    )

    kept = bench._quality_reservoir_keep_codes(features, args)

    assert b"frg_high" in kept
    assert b"core_high" in kept
    assert len(kept) == 3
    assert b"core_low" not in kept
```

- [ ] **Step 2: Run the failing reservoir keep test**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .).Path
python -m pytest tests/test_qubo.py::test_quality_reservoir_keeps_quality_and_origin_reserve -q
```

Expected: FAIL because `_quality_reservoir_keep_codes` is not defined.

- [ ] **Step 3: Add keep and apply helpers**

Add near quality feature helpers:

```python
def _quality_reservoir_keep_codes(
    features: list[QualityReservoirCandidate],
    args: argparse.Namespace,
) -> set[bytes]:
    if not features:
        return set()
    max_candidates = max(0, int(getattr(args, "quality_reservoir_max_candidates", 0) or 0))
    if max_candidates <= 0 or len(features) <= max_candidates:
        return {feature.code for feature in features}
    keep: set[bytes] = set()

    def add_ranked(rows: list[QualityReservoirCandidate], limit: int, key) -> None:
        for feature in sorted(rows, key=key)[: max(0, int(limit))]:
            keep.add(feature.code)

    add_ranked(
        features,
        int(getattr(args, "quality_reservoir_top_quality", 0) or 0),
        lambda feature: (-float(feature.reservoir_quality), int(feature.n_value), int(feature.depth), feature.code),
    )
    origin_reserve = max(0, int(getattr(args, "quality_reservoir_origin_reserve", 0) or 0))
    if origin_reserve > 0:
        origins = sorted({feature.origin for feature in features})
        for origin in origins:
            rows = [feature for feature in features if feature.origin == origin]
            add_ranked(rows, origin_reserve, lambda feature: (-float(feature.reservoir_quality), int(feature.n_value), int(feature.depth), feature.code))
    add_ranked(
        features,
        int(getattr(args, "quality_reservoir_witness_reserve", 0) or 0),
        lambda feature: (-float(feature.witness_quality), int(feature.n_value), int(feature.depth), feature.code),
    )
    add_ranked(
        features,
        int(getattr(args, "quality_reservoir_progress_reserve", 0) or 0),
        lambda feature: (-int(feature.n_delta), int(feature.n_value), int(feature.depth), feature.code),
    )
    if len(keep) > max_candidates:
        ranked_keep = sorted(
            (feature for feature in features if feature.code in keep),
            key=lambda feature: (-float(feature.reservoir_quality), int(feature.n_value), int(feature.depth), feature.code),
        )
        keep = {feature.code for feature in ranked_keep[:max_candidates]}
    if len(keep) < max_candidates:
        add_ranked(
            [feature for feature in features if feature.code not in keep],
            max_candidates - len(keep),
            lambda feature: (-float(feature.reservoir_quality), int(feature.n_value), int(feature.depth), feature.code),
        )
    return keep


def _apply_quality_reservoir_candidates(
    candidates: dict[bytes, tuple[int, int, bytes]],
    args: argparse.Namespace,
) -> set[bytes]:
    if not bool(getattr(args, "quality_reservoir_enabled", False)):
        return set(candidates)
    features = [_quality_reservoir_feature(code, data, args) for code, data in candidates.items()]
    keep = _quality_reservoir_keep_codes(features, args)
    generated = len(features)
    kept = len(keep)
    if not hasattr(args, "_quality_reservoir_generated_total"):
        args._quality_reservoir_generated_total = 0
        args._quality_reservoir_kept_total = 0
        args._quality_reservoir_witness_found_total = 0
        args._quality_reservoir_witness_quality_total = 0.0
        args._quality_reservoir_receiver_debt_total = 0.0
        args._quality_reservoir_height_risk_total = 0.0
    args._quality_reservoir_generated_total += generated
    args._quality_reservoir_kept_total += kept
    kept_features = [feature for feature in features if feature.code in keep]
    args._quality_reservoir_witness_found_total += sum(1 for feature in kept_features if feature.witness_quality > 0.0)
    args._quality_reservoir_witness_quality_total += sum(float(feature.witness_quality) for feature in kept_features)
    args._quality_reservoir_receiver_debt_total += sum(float(feature.receiver_debt) for feature in kept_features)
    args._quality_reservoir_height_risk_total += sum(float(feature.height_risk) for feature in kept_features)
    if len(keep) < len(candidates):
        for code in list(candidates):
            if code not in keep:
                candidates.pop(code, None)
    args._quality_reservoir_last_features_by_code = {feature.code: feature for feature in kept_features}
    return keep
```

- [ ] **Step 4: Run the reservoir keep test**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .).Path
python -m pytest tests/test_qubo.py::test_quality_reservoir_keeps_quality_and_origin_reserve -q
```

Expected: PASS.

- [ ] **Step 5: Hook the reservoir into the search loop**

In `scripts/benchmark_compact_beam_qubo_n0.py`, inside the main layer loop after Wang and non-append FRG candidate injection and before `_select_next_frontier(candidates, args)`, add:

```python
        if getattr(args, "quality_reservoir_enabled", False):
            kept_codes = _apply_quality_reservoir_candidates(candidates, args)
            candidate_origin_by_code = {
                code: origin
                for code, origin in candidate_origin_by_code.items()
                if code in kept_codes
            }
            args._candidate_origin_by_code = candidate_origin_by_code
            frg_frontier_child_codes &= kept_codes
            args._frg_frontier_child_codes = set(frg_frontier_child_codes)
            peak_candidates = max(peak_candidates, len(candidates))
            _update_layer_candidate_metrics(candidates, args, cap)
```

- [ ] **Step 6: Commit reservoir filtering**

Run:

```powershell
git add scripts/benchmark_compact_beam_qubo_n0.py tests/test_qubo.py
git commit -m "Add quality reservoir candidate filter"
```

Expected: commit succeeds.

## Task 4: QUBO Energy And Selection Telemetry

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Test: `tests/test_qubo.py`

- [ ] **Step 1: Write failing QUBO energy test**

Add:

```python
def test_quality_reservoir_qubo_energy_rewards_quality_and_penalizes_debt():
    import argparse
    import benchmark_compact_beam_qubo_n0 as bench

    parent = b"parent"
    good = bench.UnionDiversityQuboFeatures(
        energy=10.0,
        n_value=5,
        depth=3,
        parent_code=parent,
        source_stack=0,
        signature=(1, 1, 1, 1, 0),
        reduce_signature=(0, 1, 1),
        origin="frg_frontier_child",
        receiver_debt=0.1,
        source_scarcity=0.8,
        completion_witness_quality=0.9,
        quality_reservoir_quality=0.85,
        quality_reservoir_height_risk=0.1,
        quality_reservoir_chain_penalty=0.1,
        is_frg_macro=True,
    )
    bad = good._replace(
        energy=10.0,
        receiver_debt=0.9,
        completion_witness_quality=0.0,
        quality_reservoir_quality=-0.2,
        quality_reservoir_height_risk=0.8,
    )
    args = argparse.Namespace(
        quality_reservoir_enabled=True,
        quality_reservoir_witness_weight=0.42,
        quality_reservoir_progress_weight=0.28,
        quality_reservoir_source_scarcity_weight=0.18,
        quality_reservoir_receiver_debt_weight=0.26,
        quality_reservoir_height_risk_weight=0.18,
        quality_reservoir_chain_cost_weight=0.10,
    )

    assert bench._quality_reservoir_qubo_linear_energy(good, args) < bench._quality_reservoir_qubo_linear_energy(bad, args)
```

- [ ] **Step 2: Run the failing energy test**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .).Path
python -m pytest tests/test_qubo.py::test_quality_reservoir_qubo_energy_rewards_quality_and_penalizes_debt -q
```

Expected: FAIL because `quality_reservoir_*` fields and helper are not defined.

- [ ] **Step 3: Extend `UnionDiversityQuboFeatures`**

Add these defaults to `UnionDiversityQuboFeatures`:

```python
    quality_reservoir_quality: float = 0.0
    quality_reservoir_witness_quality: float = 0.0
    quality_reservoir_receiver_debt: float = 0.0
    quality_reservoir_height_risk: float = 0.0
    quality_reservoir_chain_penalty: float = 0.0
    quality_reservoir_macro_signature: tuple[int, ...] = ()
```

In `_union_diversity_qubo_features`, read the last reservoir feature:

```python
    reservoir_feature = (getattr(args, "_quality_reservoir_last_features_by_code", {}) or {}).get(code)
```

Add these values when constructing `UnionDiversityQuboFeatures`:

```python
        quality_reservoir_quality=float(reservoir_feature.reservoir_quality) if reservoir_feature is not None else 0.0,
        quality_reservoir_witness_quality=float(reservoir_feature.witness_quality) if reservoir_feature is not None else 0.0,
        quality_reservoir_receiver_debt=float(reservoir_feature.receiver_debt) if reservoir_feature is not None else 0.0,
        quality_reservoir_height_risk=float(reservoir_feature.height_risk) if reservoir_feature is not None else 0.0,
        quality_reservoir_chain_penalty=float(reservoir_feature.chain_length_penalty) if reservoir_feature is not None else 0.0,
        quality_reservoir_macro_signature=reservoir_feature.macro_signature if reservoir_feature is not None else (),
```

Add helper:

```python
def _quality_reservoir_qubo_linear_energy(
    feature: UnionDiversityQuboFeatures,
    args: argparse.Namespace,
) -> float:
    if not bool(getattr(args, "quality_reservoir_enabled", False)):
        return float(feature.energy)
    value = float(feature.energy)
    value -= float(getattr(args, "quality_reservoir_witness_weight", 0.0) or 0.0) * float(feature.quality_reservoir_witness_quality)
    value -= float(getattr(args, "quality_reservoir_progress_weight", 0.0) or 0.0) * max(0.0, float(feature.n_value))
    value -= float(getattr(args, "quality_reservoir_source_scarcity_weight", 0.0) or 0.0) * float(feature.source_scarcity)
    value += float(getattr(args, "quality_reservoir_receiver_debt_weight", 0.0) or 0.0) * float(feature.quality_reservoir_receiver_debt)
    value += float(getattr(args, "quality_reservoir_height_risk_weight", 0.0) or 0.0) * float(feature.quality_reservoir_height_risk)
    value += float(getattr(args, "quality_reservoir_chain_cost_weight", 0.0) or 0.0) * float(feature.quality_reservoir_chain_penalty)
    value -= float(feature.quality_reservoir_quality)
    return float(value)
```

In the selector that builds `linear = [feature.energy ...]`, replace with:

```python
linear = [_quality_reservoir_qubo_linear_energy(feature, args) for feature in features]
```

- [ ] **Step 4: Add pair penalty test**

Add:

```python
def test_quality_reservoir_pair_penalty_discourages_duplicate_macro_signature():
    import argparse
    import benchmark_compact_beam_qubo_n0 as bench

    parent = b"parent"
    f0 = bench.UnionDiversityQuboFeatures(
        energy=1.0,
        n_value=4,
        depth=2,
        parent_code=parent,
        source_stack=0,
        signature=(1, 1, 1, 1, 0),
        reduce_signature=(0, 1, 1),
        origin="frg_frontier_child",
        quality_reservoir_macro_signature=(0, 2, 4, 1),
    )
    f1 = f0._replace(energy=1.1)
    f2 = f0._replace(energy=1.2, quality_reservoir_macro_signature=(1, 1, 4, 1), source_stack=1)
    args = argparse.Namespace(
        union_diversity_qubo_same_parent_penalty=0.0,
        union_diversity_qubo_same_source_penalty=0.0,
        union_diversity_qubo_signature_penalty=0.0,
        union_diversity_qubo_reduce_signature_penalty=0.0,
        union_diversity_qubo_n_band_penalty=0.0,
        union_diversity_qubo_n_band_size=1,
        union_diversity_qubo_origin_similarity_penalty=0.0,
        union_diversity_qubo_same_origin_parent_n_band_penalty=0.0,
        union_diversity_qubo_frg_witness_signature_penalty=0.0,
        union_diversity_qubo_basin_similarity_penalty=0.0,
        union_diversity_qubo_adaptive_macro_quota_enabled=False,
        quality_reservoir_enabled=True,
        quality_reservoir_pair_receiver_debt_penalty=0.0,
        quality_reservoir_pair_macro_signature_penalty=2.0,
    )

    duplicate = bench._union_diversity_objective({0, 1}, [1.0, 1.1, 1.2], [f0, f1, f2], args)
    diverse = bench._union_diversity_objective({0, 2}, [1.0, 1.1, 1.2], [f0, f1, f2], args)

    assert duplicate > diverse
```

- [ ] **Step 5: Add pair penalty implementation**

In `_union_diversity_objective`, add:

```python
    reservoir_receiver_debt_penalty = float(getattr(args, "quality_reservoir_pair_receiver_debt_penalty", 0.0) or 0.0)
    reservoir_macro_signature_penalty = float(getattr(args, "quality_reservoir_pair_macro_signature_penalty", 0.0) or 0.0)
```

Inside the pair loop, add:

```python
            if (
                reservoir_receiver_debt_penalty > 0.0
                and int(round(features[i].quality_reservoir_receiver_debt * 10.0))
                == int(round(features[j].quality_reservoir_receiver_debt * 10.0))
                and (features[i].quality_reservoir_receiver_debt > 0.0 or features[j].quality_reservoir_receiver_debt > 0.0)
            ):
                value += reservoir_receiver_debt_penalty
            if (
                reservoir_macro_signature_penalty > 0.0
                and features[i].quality_reservoir_macro_signature
                and features[i].quality_reservoir_macro_signature == features[j].quality_reservoir_macro_signature
            ):
                value += reservoir_macro_signature_penalty
```

- [ ] **Step 6: Run QUBO energy tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .).Path
python -m pytest tests/test_qubo.py::test_quality_reservoir_qubo_energy_rewards_quality_and_penalizes_debt tests/test_qubo.py::test_quality_reservoir_pair_penalty_discourages_duplicate_macro_signature -q
```

Expected: PASS.

- [ ] **Step 7: Commit QUBO energy changes**

Run:

```powershell
git add scripts/benchmark_compact_beam_qubo_n0.py tests/test_qubo.py
git commit -m "Use quality reservoir signals in Union QUBO"
```

Expected: commit succeeds.

## Task 5: Telemetry And CSV Rows

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `scripts/open_cpmp_benchmarks.py`
- Test: `tests/test_qubo.py`

- [ ] **Step 1: Write failing telemetry row test**

Add near existing row serialization tests:

```python
def test_open_priority_group_row_serializes_quality_reservoir_fields():
    import argparse
    from benchmark_compact_beam_qubo_n0 import BeamRunResult
    from open_cpmp_benchmarks import Instance, ProblemVariant, _priority_group_row_dict_from_instance_result

    instance = Instance(
        "tiny_quality_reservoir_row",
        ProblemVariant.PREMARSHALLING_CPMP,
        3,
        3,
        3,
        [[3], [2], [1]],
    )
    result = BeamRunResult(
        moves=[],
        final_n=0,
        runtime_sec=0.1,
        quality_reservoir_generated_total=20,
        quality_reservoir_kept_total=9,
        quality_reservoir_selected_total=4,
        quality_reservoir_witness_found_total=3,
        quality_reservoir_avg_witness_quality=0.7,
        quality_reservoir_avg_receiver_debt=0.2,
        quality_reservoir_avg_height_risk=0.3,
        quality_reservoir_pair_penalty_total=1.5,
    )
    args = argparse.Namespace(
        solver_profile="scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish",
        selection_policy="scheduler_state_qubo",
        quality_reservoir_enabled=True,
        quality_reservoir_max_candidates=1600,
    )

    row = _priority_group_row_dict_from_instance_result(result, instance, args)

    assert row["quality_reservoir_enabled"] is True
    assert row["quality_reservoir_max_candidates"] == 1600
    assert row["quality_reservoir_generated_total"] == 20
    assert row["quality_reservoir_kept_total"] == 9
    assert row["quality_reservoir_selected_total"] == 4
    assert row["quality_reservoir_avg_witness_quality"] == 0.7
```

- [ ] **Step 2: Run the failing row test**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .).Path
python -m pytest tests/test_qubo.py::test_open_priority_group_row_serializes_quality_reservoir_fields -q
```

Expected: FAIL because `BeamRunResult` and row serialization do not expose the fields.

- [ ] **Step 3: Add result fields and final result aggregation**

Add these fields to the solver result dataclass:

```python
    quality_reservoir_generated_total: int = 0
    quality_reservoir_kept_total: int = 0
    quality_reservoir_selected_total: int = 0
    quality_reservoir_witness_found_total: int = 0
    quality_reservoir_avg_witness_quality: float = 0.0
    quality_reservoir_avg_receiver_debt: float = 0.0
    quality_reservoir_avg_height_risk: float = 0.0
    quality_reservoir_pair_penalty_total: float = 0.0
```

Before returning the result, compute:

```python
    quality_kept = max(1, int(getattr(args, "_quality_reservoir_kept_total", 0) or 0))
    quality_reservoir_avg_witness_quality = float(getattr(args, "_quality_reservoir_witness_quality_total", 0.0) or 0.0) / float(quality_kept)
    quality_reservoir_avg_receiver_debt = float(getattr(args, "_quality_reservoir_receiver_debt_total", 0.0) or 0.0) / float(quality_kept)
    quality_reservoir_avg_height_risk = float(getattr(args, "_quality_reservoir_height_risk_total", 0.0) or 0.0) / float(quality_kept)
```

Set result fields from `args._quality_reservoir_*`.

Inside the selected-code accounting block after `_select_next_frontier`, add:

```python
        if getattr(args, "quality_reservoir_enabled", False):
            quality_features = getattr(args, "_quality_reservoir_last_features_by_code", {}) or {}
            if not hasattr(args, "_quality_reservoir_selected_total"):
                args._quality_reservoir_selected_total = 0
            args._quality_reservoir_selected_total += sum(1 for code in selected_codes if code in quality_features)
```

When `_union_diversity_objective` adds reservoir pair penalties, accumulate into a local `reservoir_pair_penalty`; after the loop, if enabled:

```python
    if bool(getattr(args, "quality_reservoir_enabled", False)):
        if not hasattr(args, "_quality_reservoir_pair_penalty_total"):
            args._quality_reservoir_pair_penalty_total = 0.0
        args._quality_reservoir_pair_penalty_total += float(reservoir_pair_penalty)
```

- [ ] **Step 4: Add open row serialization helper**

In `scripts/open_cpmp_benchmarks.py`, add:

```python
def _quality_reservoir_row_fields(result, args: argparse.Namespace) -> dict[str, object]:
    return {
        "quality_reservoir_enabled": bool(getattr(args, "quality_reservoir_enabled", False)),
        "quality_reservoir_max_candidates": int(getattr(args, "quality_reservoir_max_candidates", 0) or 0),
        "quality_reservoir_generated_total": int(getattr(result, "quality_reservoir_generated_total", 0) or 0),
        "quality_reservoir_kept_total": int(getattr(result, "quality_reservoir_kept_total", 0) or 0),
        "quality_reservoir_selected_total": int(getattr(result, "quality_reservoir_selected_total", 0) or 0),
        "quality_reservoir_witness_found_total": int(getattr(result, "quality_reservoir_witness_found_total", 0) or 0),
        "quality_reservoir_avg_witness_quality": float(getattr(result, "quality_reservoir_avg_witness_quality", 0.0) or 0.0),
        "quality_reservoir_avg_receiver_debt": float(getattr(result, "quality_reservoir_avg_receiver_debt", 0.0) or 0.0),
        "quality_reservoir_avg_height_risk": float(getattr(result, "quality_reservoir_avg_height_risk", 0.0) or 0.0),
        "quality_reservoir_pair_penalty_total": float(getattr(result, "quality_reservoir_pair_penalty_total", 0.0) or 0.0),
    }
```

Call `row.update(_quality_reservoir_row_fields(result, args))` in both normal and priority-group row builders, next to `_frg_frontier_child_row_fields`.

- [ ] **Step 5: Run telemetry tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .).Path
python -m pytest tests/test_qubo.py::test_open_priority_group_row_serializes_quality_reservoir_fields -q
```

Expected: PASS.

- [ ] **Step 6: Commit telemetry**

Run:

```powershell
git add scripts/benchmark_compact_beam_qubo_n0.py scripts/open_cpmp_benchmarks.py tests/test_qubo.py
git commit -m "Expose quality reservoir telemetry"
```

Expected: commit succeeds.

## Task 6: Focused Smoke And Full Scorecard

**Files:**
- Modify only if a test reveals a bug: `scripts/benchmark_compact_beam_qubo_n0.py`, `scripts/open_cpmp_benchmarks.py`, `tests/test_qubo.py`
- Create benchmark outputs: `docs/experiments/*.csv`, `docs/experiments/*_summary.md`

- [ ] **Step 1: Run focused test subset**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .).Path
python -m pytest tests/test_qubo.py -q
```

Expected: all tests in `tests/test_qubo.py` pass.

- [ ] **Step 2: Run open120 focused residual smoke**

Run:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv2_test --case-ids 17 --variant qubo_scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish --time-limit 10 --alpha 10 --output docs\experiments\quality_reservoir_focus_cv2_17_t10_20260630.csv
```

Expected: CSV exists; row contains `quality_reservoir_generated_total`, `quality_reservoir_kept_total`, and `quality_reservoir_selected_total`.

- [ ] **Step 3: Run BF/CV hard subset smoke**

Run:

```powershell
python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF CV --families BF14 BF32 5-5 10-6 --sample-per-family 3 --variant qubo_scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish --time-limit 10 --alpha 10 --output docs\experiments\quality_reservoir_paper12_t10_smoke_20260630.csv
```

Expected: CSV and summary files exist; BF and CV rows include quality reservoir telemetry.

- [ ] **Step 4: Run full open120**

Run:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish --time-limit 10 --alpha 10 --jobs 8 --output docs\experiments\quality_reservoir_open120_t10_20260630.csv
```

Expected: summary reports open120 solved count, moves, penalty, runtime, and quality reservoir totals.

- [ ] **Step 5: Run full paper BF and CV**

Run BF:

```powershell
python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF --variant qubo_scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish --time-limit 10 --alpha 10 --output docs\experiments\quality_reservoir_bf_full_t10_20260630.csv
```

Run CV:

```powershell
python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets CV --variant qubo_scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish --time-limit 10 --alpha 10 --output docs\experiments\quality_reservoir_cv_full_t10_20260630.csv
```

Expected: both summary files exist and parse without blank `final_n`, `moves`, or `runtime` fields.

- [ ] **Step 6: Create the equal-weight scorecard**

Run this Python command from the repo root to create `docs/experiments/quality_reservoir_scorecard_20260630.md` from the generated CSVs:

```powershell
python - <<'PY'
import csv
from pathlib import Path

ROOT = Path("docs/experiments")
FILES = {
    "open120": ROOT / "quality_reservoir_open120_t10_20260630.csv",
    "BF": ROOT / "quality_reservoir_bf_full_t10_20260630.csv",
    "CV": ROOT / "quality_reservoir_cv_full_t10_20260630.csv",
}

def as_float(row, *names):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                return 0.0
    return 0.0

def as_int(row, *names):
    return int(round(as_float(row, *names)))

def solved(row):
    final_n = row.get("final_n")
    if final_n not in (None, ""):
        try:
            return int(float(final_n)) == 0 and row.get("moves") not in (None, "")
        except ValueError:
            return False
    status = str(row.get("status", "")).upper()
    return status in {"OK", "SOLVED", "SUCCESS", "PRIMARY_SOLVED"}

def feasible(row):
    if solved(row):
        return True
    value = row.get("feasible")
    if value not in (None, ""):
        return str(value).lower() in {"1", "true", "yes"}
    status = str(row.get("status", "")).upper()
    return "FEASIBLE" in status

rows = []
for group, path in FILES.items():
    with path.open(newline="", encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    solved_count = sum(1 for row in data if solved(row))
    feasible_count = sum(1 for row in data if feasible(row))
    penalty = sum(as_float(row, "full_penalty", "penalty") for row in data)
    moves = sum(as_int(row, "moves") for row in data if solved(row))
    runtime_values = [as_float(row, "runtime_sec", "runtime") for row in data]
    avg_runtime = sum(runtime_values) / len(runtime_values) if runtime_values else 0.0
    rows.append((group, len(data), solved_count, feasible_count, penalty, moves, avg_runtime))

lines = [
    "# Quality Reservoir Equal-Weight Scorecard",
    "",
    "## Promotion Rule",
    "",
    "open120, BF, and CV each count for one third of the decision. A result cannot be promoted if it materially regresses any group.",
    "",
    "## Compared Variants",
    "",
    "- current Union mainline: `qubo_scheduler_state_v5_qubo_diversity_union_polish`",
    "- strong gated FRG Union: `qubo_scheduler_state_v5_qubo_diversity_union_frg_gated_polish`",
    "- quality reservoir: `qubo_scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish`",
    "",
    "## Results",
    "",
    "| group | cases | solved | feasible | full penalty | solved moves | avg runtime |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for group, cases, solved_count, feasible_count, penalty, moves, avg_runtime in rows:
    lines.append(f"| {group} | {cases} | {solved_count} | {feasible_count} | {penalty:.0f} | {moves} | {avg_runtime:.3f} |")
lines.extend([
    "",
    "## Decision",
    "",
    "Promotion status: keep experimental until all three groups are compared against same-day Union and strong-gated baselines.",
    "",
])
(ROOT / "quality_reservoir_scorecard_20260630.md").write_text("\n".join(lines), encoding="utf-8")
PY
```

Expected: `docs/experiments/quality_reservoir_scorecard_20260630.md` exists and contains numeric rows for open120, BF, and CV.

- [ ] **Step 7: Commit benchmark report**

Run:

```powershell
git add docs/experiments/quality_reservoir_* scripts/benchmark_compact_beam_qubo_n0.py scripts/open_cpmp_benchmarks.py scripts/benchmark_compact_paper_set.py tests/test_qubo.py
git commit -m "Evaluate quality reservoir union profile"
```

Expected: commit succeeds. If benchmark CSVs are intentionally not committed in this repo, commit only the summary markdown and code/test changes, and state which CSVs remain untracked.

## Self-Review

- Spec coverage:
  - Opt-in profile: Task 1.
  - Continuous reservoir instead of hard gate: Tasks 2 and 3.
  - QUBO remains selector: Task 4.
  - Telemetry: Task 5.
  - open120, BF, CV equal-weight evidence: Task 6.
- Placeholder scan:
  - No unresolved marker strings are used for implementation content.
- Type consistency:
  - `QualityReservoirCandidate` feeds `_quality_reservoir_last_features_by_code`.
  - `UnionDiversityQuboFeatures` receives `quality_reservoir_*` fields from that map.
  - `_quality_reservoir_qubo_linear_energy` and `_union_diversity_objective` consume the same field names.
- Scope:
  - The plan touches only the solver, runners, tests, and experiment artifacts. It does not refactor the solver or add fallback behavior.
