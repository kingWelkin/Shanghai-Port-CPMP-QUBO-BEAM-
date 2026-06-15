# Wang2017 Task-Guided Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and benchmark an experimental Wang2017 task-guided continuation selector for the compact CPMP solver.

**Architecture:** Keep compact byte-state search as the owner. Add reusable task-feature helpers in `scripts/benchmark_compact_beam_qubo_n0.py`, expose the profile through `scripts/open_cpmp_benchmarks.py`, and test via existing `tests/test_qubo.py` patterns.

**Tech Stack:** Python dataclasses, argparse profiles, pytest, DLTS open benchmark CSV/markdown outputs.

---

## Task 1: Task Feature Tests

**Files:**
- Modify: `tests/test_qubo.py`

- [ ] Add a failing test for `_wang2017_task_features_for_code(parent_code, child_code, data, args)`.
- [ ] Add a failing test for `_select_wang2017_task_supplement(...)` preferring lower tier-protection and higher stable target height.
- [ ] Run the two tests and confirm they fail because helpers are missing.

## Task 2: Task Feature Implementation

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`

- [ ] Add `Wang2017TaskFeatures` dataclass.
- [ ] Add compact parent-child move inference.
- [ ] Add task tuple scoring using legal child state signals.
- [ ] Add `_select_wang2017_task_supplement(...)`.
- [ ] Run Task 1 tests and confirm they pass.

## Task 3: Profile And Telemetry Surface

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `scripts/open_cpmp_benchmarks.py`
- Modify: `tests/test_qubo.py`

- [ ] Add failing tests for `scheduler_state_v5_wang2017_task_polish` and `qubo_scheduler_state_v5_wang2017_task_polish`.
- [ ] Add profile defaults and CLI flags.
- [ ] Add result fields for calls, added candidates, active layers, selected tier-protection total, and selected stable-height total.
- [ ] Add open-runner variant and pass-through fields.
- [ ] Run profile/open-runner tests and confirm they pass.

## Task 4: Frontier Wiring

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `tests/test_qubo.py`

- [ ] Add a failing test proving the task selector is called inside `_select_enhanced_qubo_frontier`.
- [ ] Wire the selector after feasibility escape and before generic QUBO.
- [ ] Initialize counters in `run_compact_beam_instance`.
- [ ] Run focused selector tests and confirm they pass.

## Task 5: Experiments And Documentation

**Files:**
- Modify: `docs/experiments/wang_qubo_full_quality_trial_summary.md`

- [ ] Run focused pytest and py_compile.
- [ ] Run full open-120 new profile benchmark.
- [ ] Run full open-120 rescue-polish baseline benchmark in the same turn.
- [ ] Append a concise evidence table and promotion decision.
- [ ] Commit the implementation and documentation.
