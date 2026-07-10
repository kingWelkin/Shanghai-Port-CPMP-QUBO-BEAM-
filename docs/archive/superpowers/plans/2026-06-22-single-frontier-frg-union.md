# Single-Frontier FRG Union Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ablation profile that merges FRG descendants into the existing union beam/QUBO candidate pool before selection while preserving the configured beam width.

**Architecture:** The profile derives from the rollout-graft union baseline and enables the existing pre-selection `_inject_frg_frontier_child_candidates` path with `frg_frontier_child_append_only=False`. Frontier trace rows gain a separate `candidate_origin` field, so origin and selection mechanism are not conflated. No selector score, QUBO energy, candidate legality rule, adaptive schedule, or post-selection append behavior changes.

**Tech Stack:** Python, pytest, existing compact beam/QUBO runner, existing paper-set and open120 benchmark runners.

---

### Task 1: Specify the New Profile and Trace Contract

**Files:**
- Modify: `tests/test_qubo.py`

- [ ] **Step 1: Write the failing profile test**

Add this test beside the existing frontier-profile tests:

```python
def test_solver_profile_scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish_merges_before_selection():
    import benchmark_compact_beam_qubo_n0 as bench

    args = _compact_args(solver_profile="scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish")
    bench._apply_solver_profile(args)

    assert args.frg_frontier_child_enabled is True
    assert args.frg_frontier_child_parent_width == 16
    assert args.frg_frontier_child_limit == 2
    assert args.frg_frontier_child_max_steps == 3
    assert args.frg_frontier_child_append_only is False
    assert args.frg_frontier_child_adaptive_enabled is False
```

- [ ] **Step 2: Write the failing trace-origin test**

Extend the `test_frontier_trace_candidate_rows_mark_sources_and_parent_reuse` call with `candidate_origin_by_code={c2: "frg_frontier_child"}` and append:

```python
    assert rows[0]["candidate_origin"] == "frg_frontier_child"
    assert rows[1]["candidate_origin"] == "beam"
    assert rows[2]["candidate_origin"] == "beam"
```

- [ ] **Step 3: Verify RED**

Run:

```powershell
$env:PYTHONPATH='.'
python -m pytest tests\test_qubo.py -k "single_frontier or frontier_trace_candidate_rows_mark_sources" -q --basetemp .tmp_pytest_single_frontier_red
```

Expected: the profile is unknown and `_frontier_trace_candidate_rows` does not accept `candidate_origin_by_code` or does not emit `candidate_origin`.

### Task 2: Implement the Merged-Pool Profile and Trace Origin

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py:2060-2085,832-912,9666-9785`

- [ ] **Step 1: Add the profile**

Add this definition immediately after the existing frontier profile:

```python
_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_rollout_graft_polish"],
    "frg_frontier_child_enabled": True,
    "frg_frontier_child_parent_width": 16,
    "frg_frontier_child_limit": 2,
    "frg_frontier_child_max_steps": 3,
    "frg_frontier_child_append_only": False,
}
```

- [ ] **Step 2: Add a distinct trace origin**

Add `candidate_origin` to `FRONTIER_TRACE_FIELDS`. Extend `_frontier_trace_candidate_rows` with an optional `candidate_origin_by_code` mapping and emit:

```python
origin_by_code = candidate_origin_by_code or {}
...
"candidate_origin": origin_by_code.get(code, "beam"),
```

Keep `selected_source` unchanged: it identifies how a candidate was selected (`core`, `tail_fill`, or QUBO), while `candidate_origin` identifies how it entered the pool.

- [ ] **Step 3: Wire FRG origin into the per-layer trace**

Before `_frontier_trace_candidate_rows`, construct and pass:

```python
trace_candidate_origins = {
    code: "frg_frontier_child"
    for code in frg_frontier_child_codes
    if code in candidates
}
```

This set is populated before `_select_next_frontier` for the new profile, so all merged FRG descendants are ranked and can be selected by the ordinary beam/QUBO logic. Do not call `_append_frg_frontier_child_frontier` differently; the profile disables it by configuration.

- [ ] **Step 4: Verify GREEN**

Run the RED command again. Expected: both tests pass.

### Task 3: Wire the Public Variant

**Files:**
- Modify: `scripts/open_cpmp_benchmarks.py:988-995,1442-1451,1711-1718`
- Modify: `tests/test_qubo.py`

- [ ] **Step 1: Write the failing runner test**

Add:

```python
def test_open_runner_accepts_qubo_diversity_union_frg_single_frontier_variant():
    from open_cpmp_benchmarks import _args_for_variant, _variant_names, build_parser

    variant = "qubo_scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish"
    outer = build_parser().parse_args(["--variant", variant])
    args = _args_for_variant(outer, variant)

    assert args.solver_profile == "scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish"
    assert variant in _variant_names("wang_research_all")
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
$env:PYTHONPATH='.'
python -m pytest tests\test_qubo.py::test_open_runner_accepts_qubo_diversity_union_frg_single_frontier_variant -q --basetemp .tmp_pytest_single_frontier_runner_red
```

Expected: assertion failure because the variant is not listed or mapped.

- [ ] **Step 3: Add variant list entries and mapping**

In each maintained union/frontier variant list, add:

```python
"qubo_scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish",
```

In `_args_for_variant`, add:

```python
if variant == "qubo_scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish":
    overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish"})
```

- [ ] **Step 4: Verify GREEN and compile**

Run:

```powershell
$env:PYTHONPATH='.'
python -m pytest tests\test_qubo.py -k "single_frontier or frontier_trace_candidate_rows_mark_sources" -q --basetemp .tmp_pytest_single_frontier_green
python -m py_compile scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py
```

Expected: focused tests pass and both scripts compile.

- [ ] **Step 5: Commit implementation and tests**

```powershell
git add scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py tests\test_qubo.py
git commit -m "feat: add single-frontier FRG union ablation"
```

### Task 4: Diagnose and Benchmark the Ablation

**Files:**
- Create: `docs/experiments/single_frontier_frg_union_decision_20260622.md`

- [ ] **Step 1: Run focused paper-CV trace diagnostics**

Run direct `run_compact_beam_instance` diagnostics for `data10-10-1` and `data10-10-2` with `frontier_trace_top_k=128`, `time_limit=20.0`, and profile `scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish`. Confirm that the trace contains `candidate_origin=frg_frontier_child`, that `selected_count` never exceeds the configured beam width, and that selected FRG children retain their normal `selected_source`.

- [ ] **Step 2: Run paired paper CV sample2**

```powershell
python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets CV --sample-per-family 2 --variant qubo_scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish --time-limit 20 --alpha 10 --jobs 2 --output docs\experiments\paper_cv_sample2_union_frg_single_frontier_t20_jobs2_20260622.csv
```

- [ ] **Step 3: Run paired open120**

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish --time-limit 10 --alpha 10 --jobs 8 --output docs\experiments\union_frg_single_frontier_open120_t10_jobs8_20260622.csv
```

- [ ] **Step 4: Record the decision**

Compare against `union_frg_frontier_append_light_open120_t10_jobs8_20260622.csv`, `paper_cv_sample2_union_graft_t20_jobs2_20260622.csv`, and `paper_cv_sample2_union_frg_frontier_adaptive_v2_t20_jobs2_20260622.csv`. Record solved count, moves, full penalty, runtime, FRG generated/merged/selected telemetry, and the promotion decision. Do not promote unless the design document's dual-corpus rule passes.

### Plan Self-Review

- The only behavioral change is selecting FRG children inside the existing fixed-width beam rather than appending them after selection.
- Trace origin is a separate field, so QUBO/core selection provenance remains observable.
- Every production change has a prior focused failing test and a green verification command.
- Benchmark results remain evidence, not an automatic mainline promotion.
