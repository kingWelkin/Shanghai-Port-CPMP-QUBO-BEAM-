# Signal-Aware Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the scheduler's dataset-name gate with instance and layer signals so adaptive/rescue decisions are case-aware without reading `cv1/cv2/cv3` labels.

**Architecture:** Keep the current compact beam search file and add small helper functions for structural instance features and adaptive gate signals. `scheduler_v1` should default to signal-aware behavior; the old case-id flag remains for reproducibility but is no longer enabled by the main scheduler profile.

**Tech Stack:** Python, argparse namespaces, pytest.

---

### Task 1: Red Tests For Signal-Aware Adaptive Gate

**Files:**
- Modify: `tests/test_qubo.py`

- [x] **Step 1: Add tests that fail before implementation**

Add tests near existing scheduler tests:

```python
def test_scheduler_adaptive_gate_uses_priority_duplication_not_case_id():
    import benchmark_compact_beam_qubo_n0 as compact

    args = _compact_args(
        scheduler_adaptive_cv1_only=False,
        scheduler_adaptive_max_duplication_ratio=0.0,
        scheduler_adaptive_min_best_n_ratio=0.45,
        scheduler_adaptive_high_n=16,
        _stack_count=7,
        _initial_n=32,
        _priority_duplication_ratio=0.25,
        _layer_candidate_count=12000,
        _layer_parent_top_share=0.05,
        _layer_depth_slack=99,
        _best_n_history=[24],
        _case_id="unlabeled_priority_group_case",
    )

    assert not compact._scheduler_uses_adaptive(args, best_n=24)

    args._priority_duplication_ratio = 0.0
    args._case_id = "7x5_cv2_test_2"

    assert compact._scheduler_uses_adaptive(args, best_n=24)
```

```python
def test_scheduler_adaptive_gate_blocks_plateau_when_slack_is_tight():
    import benchmark_compact_beam_qubo_n0 as compact

    args = _compact_args(
        scheduler_adaptive_cv1_only=False,
        scheduler_adaptive_max_duplication_ratio=1.0,
        scheduler_adaptive_plateau_window=3,
        scheduler_adaptive_cap_slack=12,
        scheduler_adaptive_high_n=16,
        _stack_count=7,
        _initial_n=30,
        _priority_duplication_ratio=0.0,
        _layer_candidate_count=12000,
        _layer_parent_top_share=0.05,
        _layer_depth_slack=8,
        _best_n_history=[24, 24, 24],
    )

    assert not compact._scheduler_uses_adaptive(args, best_n=24)

    args._best_n_history = [26, 25, 24]

    assert compact._scheduler_uses_adaptive(args, best_n=24)
```

- [x] **Step 2: Run tests and verify red**

Run:

```powershell
python -m pytest -q tests\test_qubo.py -k "scheduler_adaptive_gate"
```

Expected: fail because the new scheduler fields are not implemented.

### Task 2: Implement Minimal Signal Helpers

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`

- [x] **Step 1: Add structural feature helper**

Add:

```python
def _priority_duplication_ratio_for_instance(instance: Instance) -> float:
    values = [box for stack in instance.stacks for box in stack]
    if not values:
        return 0.0
    return 1.0 - (len(set(values)) / len(values))
```

- [x] **Step 2: Record instance features in `run_compact_beam_instance`**

Set:

```python
args._initial_n = n0
args._priority_duplication_ratio = _priority_duplication_ratio_for_instance(instance)
args._fill_ratio = instance.container_count / max(1, instance.stack_count * instance.max_height)
```

- [x] **Step 3: Update `_scheduler_uses_adaptive`**

Add gates for:

- `scheduler_adaptive_max_duplication_ratio`
- `scheduler_adaptive_min_best_n_ratio`
- `scheduler_adaptive_max_candidate_pressure`
- `scheduler_adaptive_max_parent_top_share`
- `scheduler_adaptive_plateau_window`
- `scheduler_adaptive_cap_slack`

### Task 3: Profiles, CLI, Rows, Docs

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `scripts/open_cpmp_benchmarks.py`
- Modify: `docs/guides/instance-aware-scheduler-probes.md`

- [x] **Step 1: Add defaults and profile values**

`scheduler_v1` should use:

```python
"scheduler_adaptive_cv1_only": False,
"scheduler_adaptive_max_duplication_ratio": 0.0,
"scheduler_adaptive_min_best_n_ratio": 0.45,
"scheduler_adaptive_max_candidate_pressure": 0.0,
"scheduler_adaptive_max_parent_top_share": 0.0,
"scheduler_adaptive_plateau_window": 3,
"scheduler_adaptive_cap_slack": 12,
```

- [x] **Step 2: Expose fields in CLI and CSV rows**

Add matching parser options and row fields in compact and open benchmark outputs.

- [x] **Step 3: Update docs**

Replace wording that says `cv2/cv3` are manually disabled with structural explanation based on repeated priorities and hard-layer signals.

### Task 4: Verification

**Files:**
- Test: `tests/test_qubo.py`

- [x] **Step 1: Run unit tests**

```powershell
python -m pytest -q tests\test_qubo.py -k "scheduler"
python -m pytest -q tests\test_qubo.py
```

- [x] **Step 2: Run compile check**

```powershell
python -m py_compile scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py
```

- [x] **Step 3: Run smoke/matrix**

```powershell
python scripts\benchmark_compact_beam_qubo_n0.py --cases A1_01 A1_26 --solver-profile scheduler_v1 --time-limit 8 --output docs\experiments\scratch_signal_scheduler_smoke.csv
python scripts\open_cpmp_benchmarks.py --families 7x5_cv2_test --limit-per-family 10 --variant qubo_scheduler --time-limit 5 --no-auto-clone --output docs\experiments\scratch_signal_7x5_cv2_scheduler.csv
```

Expected: compact smoke reaches `final_n=0`; open matrix should be compared against the previous 10/10 case-aware result.
