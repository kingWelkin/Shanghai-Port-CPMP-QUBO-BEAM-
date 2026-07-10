# Wang2017 Task-Guided Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and benchmark an experimental Wang2017 task-guided continuation selector for the compact CPMP solver.

**Architecture:** Keep compact byte-state search as the owner. Add reusable task-feature helpers in `scripts/benchmark_compact_beam_qubo_n0.py`, expose the profile through `scripts/open_cpmp_benchmarks.py`, and test via existing `tests/test_qubo.py` patterns.

**Tech Stack:** Python dataclasses, argparse profiles, pytest, DLTS open benchmark CSV/markdown outputs.

---

## Task 1: Task Feature Tests

**Files:**
- Modify: `tests/test_qubo.py`

- [x] Add a failing test for `_wang2017_task_features_for_code(parent_code, child_code, data, args)`.
- [x] Add a failing test for `_select_wang2017_task_supplement(...)` preferring lower tier-protection and higher stable target height.
- [x] Run the two tests and confirm they fail because helpers are missing.

## Task 2: Task Feature Implementation

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`

- [x] Add `Wang2017TaskFeatures` dataclass.
- [x] Add compact parent-child move inference.
- [x] Add task tuple scoring using legal child state signals.
- [x] Add `_select_wang2017_task_supplement(...)`.
- [x] Run Task 1 tests and confirm they pass.

## Task 3: Profile And Telemetry Surface

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `scripts/open_cpmp_benchmarks.py`
- Modify: `tests/test_qubo.py`

- [x] Add failing tests for `scheduler_state_v5_wang2017_task_polish` and `qubo_scheduler_state_v5_wang2017_task_polish`.
- [x] Add profile defaults and CLI flags.
- [x] Add result fields for calls, added candidates, active layers, selected tier-protection total, and selected stable-height total.
- [x] Add open-runner variant and pass-through fields.
- [x] Run profile/open-runner tests and confirm they pass.

## Task 4: Frontier Wiring

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `tests/test_qubo.py`

- [x] Add a failing test proving the task selector is called inside `_select_enhanced_qubo_frontier`.
- [x] Wire the selector after feasibility escape and before generic QUBO.
- [x] Initialize counters in `run_compact_beam_instance`.
- [x] Run focused selector tests and confirm they pass.

## Task 5: Experiments And Documentation

**Files:**
- Modify: `docs/experiments/wang_qubo_full_quality_trial_summary.md`

- [x] Run focused pytest and py_compile.
- [x] Run full open-120 new profile benchmark.
- [x] Run full open-120 rescue-polish baseline benchmark in the same turn.
- [x] Append a concise evidence table and promotion decision.
- [x] Commit the implementation and documentation.
