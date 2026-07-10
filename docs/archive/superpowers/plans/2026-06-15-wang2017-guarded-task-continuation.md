# Wang2017 Guarded Task Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and benchmark a conservative guarded Wang2017 task continuation profile, then keep any gain behind a no-regret portfolio if the single route still regresses.

**Architecture:** Reuse the existing task-feature selector but add a small gate helper in `scripts/benchmark_compact_beam_qubo_n0.py`. Expose a separate solver profile and open-runner variant so the aggressive profile remains available as a negative ablation.

**Tech Stack:** Python argparse profiles, deterministic compact beam selection, pytest, DLTS open benchmark CSV summaries.

---

## Task 1: RED Tests

**Files:**
- Modify: `tests/test_qubo.py`

- [x] Add a failing test for `_wang2017_task_gate_reason(...)` showing `active`, `not_plateau`, `slack_too_loose`, and `disabled`.
- [x] Add a failing profile test for `scheduler_state_v5_wang2017_task_guarded_polish` with width 1 and n-window 0.
- [x] Add a failing open-runner test for `qubo_scheduler_state_v5_wang2017_task_guarded_polish`.
- [x] Run the focused tests and confirm they fail because the guarded profile and gate do not exist.

## Task 2: GREEN Implementation

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `scripts/open_cpmp_benchmarks.py`

- [x] Add task gate defaults: enabled false, guarded plateau window 7, plateau delta 0.
- [x] Add `_wang2017_task_gate_reason(args, best_n)` and `_wang2017_task_layer_active(args, best_n)`.
- [x] Wire the existing task supplement through the gate.
- [x] Add `scheduler_state_v5_wang2017_task_guarded_polish`.
- [x] Add open-runner variant and pass-through fields.
- [x] Add `qubo_scheduler_state_v5_wang2017_task_guarded_portfolio` as a no-regret open-runner portfolio.
- [x] Run the focused tests and confirm they pass.

## Task 3: Experiments

**Files:**
- Modify: `docs/experiments/wang_qubo_full_quality_trial_summary.md`

- [x] Run the five-case residual benchmark for the guarded variant.
- [x] Run open-120 for the guarded variant.
- [x] Compare against `docs/experiments/scratch_wang_rescue_polish_open120_t10.csv`.
- [x] Compare the no-regret portfolio against the current clean best CSV.
- [x] Append a short evidence table and promotion decision.

## Task 4: Verification And Commit

**Files:**
- Modify: `tests/test_qubo.py`
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `scripts/open_cpmp_benchmarks.py`
- Modify: `docs/experiments/wang_qubo_full_quality_trial_summary.md`
- Modify: `docs/superpowers/plans/2026-06-15-wang2017-guarded-task-continuation.md`

- [x] Run focused pytest.
- [x] Run broad related pytest.
- [x] Run `python -m py_compile scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py`.
- [x] Run `git diff --check`.
- [x] Commit only the guarded-task files and documentation.
