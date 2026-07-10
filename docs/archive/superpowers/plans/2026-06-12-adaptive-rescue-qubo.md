# Adaptive Rescue QUBO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable `adaptive_rescue_v1` profile that keeps `qubo_adaptive` speed on 7x5 cv1 while allowing trigger-style QUBO rescue on plateau layers.

**Architecture:** Reuse the existing `adaptive_qubo` frontier selector and the existing rescue gate. The new profile sets `selection_policy=adaptive_qubo` plus `rescue_enabled=True`; `_select_adaptive_qubo_frontier` checks the rescue gate first and delegates to `_select_rescue_qubo_frontier` only on risky plateau layers. Open benchmark gets a new `qubo_adaptive_rescue` variant that applies the profile while preserving the 5x5 adaptive protection.

**Tech Stack:** Python, argparse namespaces, pytest, existing compact beam QUBO scripts.

---

### Task 1: Red Tests

**Files:**
- Modify: `tests/test_qubo.py`

- [x] **Step 1: Add a profile test**

Add a test near existing solver profile tests:

```python
def test_solver_profile_adaptive_rescue_v1_combines_adaptive_with_light_rescue(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="adaptive_rescue_v1")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.selection_policy == "adaptive_qubo"
    assert args.rescue_enabled is True
    assert args.adaptive_high_frontier == 800
    assert args.adaptive_mid_frontier == 1000
    assert args.adaptive_low_frontier == 1200
    assert args.rescue_max_layers == 8
    assert args.rescue_qubo_supplement_width >= 32
    assert args.rescue_max_qubo_vars >= 80
```

- [x] **Step 2: Add a selector delegation test**

Add:

```python
def test_adaptive_qubo_uses_rescue_when_plateau_is_active(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as compact

    ranked = [(b"a", (18, 7, b"p"))]
    args = _compact_args(
        selection_policy="adaptive_qubo",
        rescue_enabled=True,
        rescue_plateau_window=2,
        rescue_plateau_delta=0,
        rescue_cap_slack=30,
        rescue_min_n=12,
        rescue_min_candidates=100,
        rescue_max_layers=4,
        _best_n_history=[18, 18],
        _layer_candidate_count=1000,
        _layer_depth_slack=10,
        _rescue_layers_used=0,
    )
    calls: list[str] = []

    monkeypatch.setattr(compact, "_select_rescue_qubo_frontier", lambda *_args: calls.append("rescue") or ([b"r"], {b"r"}, 1, 3, 0.1))
    monkeypatch.setattr(compact, "_select_enhanced_qubo_frontier", lambda *_args, **_kwargs: calls.append("enhanced") or ([b"e"], set(), 0, 0, 0.0))

    selected = compact._select_adaptive_qubo_frontier(ranked, len(ranked), args)

    assert selected[0] == [b"r"]
    assert calls == ["rescue"]
```

- [x] **Step 3: Add an open runner variant test**

Extend the open runner variant test:

```python
adaptive_rescue_outer = build_parser().parse_args(["--variant", "qubo_adaptive_rescue"])
adaptive_rescue_args = _args_for_variant(adaptive_rescue_outer, "qubo_adaptive_rescue")

_apply_solver_profile(adaptive_rescue_args)

assert adaptive_rescue_args.solver_profile == "adaptive_rescue_v1"
assert adaptive_rescue_args.selection_policy == "adaptive_qubo"
assert adaptive_rescue_args.rescue_enabled is True
```

- [x] **Step 4: Run red tests**

Run:

```powershell
python -m pytest -q tests\test_qubo.py -k "adaptive_rescue"
```

Expected before implementation: fail because `adaptive_rescue_v1` and `qubo_adaptive_rescue` do not exist, and adaptive selector ignores rescue.

### Task 2: Implementation

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `scripts/open_cpmp_benchmarks.py`

- [x] **Step 1: Add solver profile**

Add after `light_rescue_qubo_v1`:

```python
_SOLVER_PROFILES["adaptive_rescue_v1"] = {
    **_SOLVER_PROFILES["light_rescue_qubo_v1"],
    "selection_policy": "adaptive_qubo",
    "adaptive_high_n": 16,
    "adaptive_low_n": 8,
    "adaptive_high_frontier": 800,
    "adaptive_mid_frontier": 1000,
    "adaptive_low_frontier": 1200,
    "adaptive_core_ratio": 0.90,
    "adaptive_candidate_pool": 0,
}
```

- [x] **Step 2: Route adaptive selector through rescue**

At the top of `_select_adaptive_qubo_frontier`, add:

```python
best_n = int(ranked[0][1][0]) if ranked else math.inf
if _rescue_layer_active(args, best_n):
    return _select_rescue_qubo_frontier(ranked, candidate_count, args)
```

Keep existing adaptive width logic unchanged when rescue is inactive.

- [x] **Step 3: Add open runner variant**

Add `qubo_adaptive_rescue` to parser choices and map it in `_args_for_variant`:

```python
if variant == "qubo_adaptive_rescue":
    overrides.update({"solver_profile": "adaptive_rescue_v1"})
```

Then reuse the same adaptive frontier override block as `qubo_adaptive`, including the 5x5 protection.

### Task 3: Verification And Matrix

**Files:**
- Test: `tests/test_qubo.py`
- Generate scratch outputs under `docs/experiments/scratch_adaptive_rescue_*`

- [x] **Step 1: Run targeted tests**

```powershell
python -m pytest -q tests\test_qubo.py -k "adaptive_rescue or open_runner_scheduler"
```

- [x] **Step 2: Run full local tests and compile**

```powershell
python -m pytest -q -p no:cacheprovider
python -m py_compile scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py
```

- [x] **Step 3: Run 7x5 cv1 40-case comparison**

```powershell
python scripts\open_cpmp_benchmarks.py --families 7x5_cv1_test 7x5_cv1_validation --limit-per-family 20 --variant qubo_adaptive_rescue --time-limit 3 --alpha 10 --no-auto-clone --output docs\experiments\scratch_adaptive_rescue_7x5_cv1_40cases.csv
```

Compare against `open_cpmp_dlts_7x5_cv1_40cases_adaptive.csv`: target is at least 22/40 solved without a large runtime regression; any improvement above 22/40 is a candidate promotion.

### Task 4: Documentation And Cleanup

**Files:**
- Modify: `docs/experiments/open_cpmp_adaptive_frontier_experiment_summary.md`
- Modify: `docs/guides/current-qubo-beam-algorithm.md`

- [x] **Step 1: Record results**

Add a short note with solved count, moves, average runtime, and average reference gap for `adaptive_rescue_v1`.

- [x] **Step 2: Clean scratch files after extracting numbers**

Remove only files matching:

```powershell
docs\experiments\scratch_adaptive_rescue_*
```

- [x] **Step 3: Final verification**

Run:

```powershell
python -m pytest -q -p no:cacheprovider
```
