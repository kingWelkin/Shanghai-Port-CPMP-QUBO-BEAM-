# Union Rollout Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Keep checkboxes updated as each step is completed.

**Goal:** Add `scheduler_state_v5_qubo_diversity_union_rollout_polish`, an opt-in profile that adds a small FRG-style child rollout probe to the existing union beam+QUBO selector. The probe is only a selection signal. It must not generate moves, return FRG solutions, or become a fallback.

**Baseline:** Keep `scheduler_state_v5_qubo_diversity_union_polish` unchanged. Promotion is only possible after paired paper-set and open120 verification.

## Task 1: Red Tests for Rollout Probe

**Files:**
- Modify: `tests/test_qubo.py`

- [x] Add a cache test that probes two candidate wrappers with the same compact layout code and asserts the injected completion function is called once.
- [x] Add scoring tests for rollout energy:
  - feasible rollout lowers the union linear energy,
  - dead-end rollout raises the union linear energy.
- [x] Add profile wiring tests:
  - the new rollout profile enables `union_rollout_probe_enabled`,
  - the old union profile leaves it disabled,
  - `open_cpmp_benchmarks.py` accepts the new variant.
- [x] Run the focused tests and confirm they fail before implementation.

## Task 2: Implement Probe Helpers

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`

- [x] Add disabled-by-default config fields and CSV fields:
  - `union_rollout_probe_enabled`
  - `union_rollout_probe_max_candidates`
  - `union_rollout_probe_time_limit`
  - `union_rollout_probe_max_steps`
  - `union_rollout_probe_feasible_bonus`
  - `union_rollout_probe_final_n_weight`
  - `union_rollout_probe_moves_weight`
  - `union_rollout_probe_dead_end_penalty`
  - telemetry counters for calls, candidates, feasible, time, cache hits, selected total.
- [x] Add a small `UnionRolloutProbeResult` structure and cache helper keyed by compact layout code.
- [x] Add a rollout energy helper used by `UnionDiversityQuboFeatures`.
- [x] Keep pairwise diversity terms unchanged in the first version.

## Task 3: Wire Into Union Selector

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`

- [x] Sample a bounded set from the existing ranked child list before `_select_union_diversity_qubo_supplement`.
- [x] Reuse the current union focus window (`best_n + n_window`) and `union_rollout_probe_max_candidates`.
- [x] Run the FRG completion probe only for sampled candidates and only when the profile enables it.
- [x] Add rollout fields to union feature construction and apply only to the QUBO linear term.
- [x] Increment telemetry when probed candidates are selected by the union QUBO selector.

## Task 4: Add Profile and Runners

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `scripts/open_cpmp_benchmarks.py`
- Modify as needed: `scripts/benchmark_compact_paper_set.py`

- [x] Add `scheduler_state_v5_qubo_diversity_union_rollout_polish` extending `scheduler_state_v5_qubo_diversity_union_polish`.
- [x] Add open runner variant `qubo_scheduler_state_v5_qubo_diversity_union_rollout_polish`.
- [x] Ensure paper-set runner can pass the new profile by variant name.
- [x] Ensure all new telemetry fields appear in CSV output for compact, open120, and paper-set runs.

## Task 5: Verification and Decision

**Commands:**

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py -k "union_rollout or qubo_diversity_union" -q --basetemp .tmp_pytest
$env:PYTHONPATH='.'; python -m py_compile scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py scripts\benchmark_compact_paper_set.py
```

- [x] Run focused tests and compile checks.
- [x] Run a tiny smoke benchmark for the new profile.
- [x] Run paper sample paired against current union if smoke is healthy.
  - Note: completed as `sample-per-family=1, t5` precheck, not the full `sample10, t20` promotion gate.
- [ ] Run open120 guard if paper sample improves.
- [ ] Commit only after tests pass and results are recorded.

## Non-Goals

- No BS-FRG fallback.
- No case-family-specific gates.
- No promotion based on single-case rescue.
- No default change to the current strongest baseline.
