# FRG Compound Guarded Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a guarded loose FRG compound profile that keeps the residual escape signal while blocking known high-risk structures.

**Architecture:** Keep `qubo_scheduler_state_v5_wang_rescue_polish` as the baseline. Add a small gate stack around the existing FRG compound append path: structure, low-residual band, candidate pressure, and parent-diversity veto. Do not use case-id rules.

**Tech Stack:** Python, pytest, compact beam runner, open CPMP runner.

---

### Task 1: Guarded Profile

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `scripts/open_cpmp_benchmarks.py`
- Modify: `tests/test_qubo.py`

- [ ] Add tests for structural, residual, candidate, and parent-diversity gates.
- [ ] Add `scheduler_state_v5_frg_compound_guarded_polish`.
- [ ] Add open-runner alias `qubo_scheduler_state_v5_frg_compound_guarded_polish`.
- [ ] Run unit tests, residual smoke, and open120 comparison against the curated v5 baseline.
