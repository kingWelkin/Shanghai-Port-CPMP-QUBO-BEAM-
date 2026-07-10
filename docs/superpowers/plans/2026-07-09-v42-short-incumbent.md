# v4.2 Short Incumbent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a v4.2 experimental profile that lowers path length by validating and accepting shorter BS-FRG incumbents while preserving v4.0 solved coverage.

**Architecture:** Keep the v4.0 chain-aware solver as the primary search. Pass its validated move list as an incumbent bound into BS-FRG postcompress, then accept only a shorter replay-valid path. Expose the profile through both Open120 and paper runners.

**Tech Stack:** Python, pytest, existing compact CPMP solver scripts, existing BS-FRG solver.

---

### Task 1: Incumbent-Aware BS-FRG Postcompress

**Files:**
- Modify: `container_reshuffle/solvers/base.py`
- Modify: `container_reshuffle/solvers/bs_frg.py`
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Test: `tests/test_bs_frg_solver.py`

- [ ] **Step 1: Add failing tests**

Add tests proving `_bs_frg_postcompress_run_solver` passes the current v4 move list as `SolverConfig.exact_incumbent_moves`, and `BsFrgSolver` respects an incumbent bound without returning the incumbent as a new solution.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/test_bs_frg_solver.py -k "incumbent or postcompress" -q`

- [ ] **Step 3: Implement incumbent-bound plumbing**

Add a small config flag for stop-on-incumbent-improvement and wire current postcompress moves into BS-FRG solver config. Use the bound only for pruning and early return; do not return the incumbent path as a BS-FRG solution.

- [ ] **Step 4: Run tests and confirm pass**

Run: `python -m pytest tests/test_bs_frg_solver.py -k "incumbent or postcompress" -q`

### Task 2: v4.2 Profile and Runner Wiring

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `scripts/open_cpmp_benchmarks.py`
- Modify: `scripts/benchmark_compact_paper_set.py`
- Test: `tests/test_qubo.py`

- [ ] **Step 1: Add failing profile wiring tests**

Assert that the new Open120 and paper variant names map to the v4.2 solver profile and that applying the profile enables long-path BS-FRG postcompress with incumbent bound.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/test_qubo.py -k "v42 or short_incumbent" -q`

- [ ] **Step 3: Implement profile and variant mappings**

Add profile `scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_short_incumbent_v42_polish` and matching runner variant. Set postcompress gates to long-path mode (`min_moves=120`, `time_limit=4s`) and keep validation-only acceptance.

- [ ] **Step 4: Run tests and confirm pass**

Run: `python -m pytest tests/test_qubo.py -k "v42 or short_incumbent" -q`

### Task 3: Focused Benchmark Gate

**Files:**
- Output: `docs/experiments/v42_short_incumbent_probe_20260709/*.csv`

- [ ] **Step 1: Run Open120 hard sample**

Run the new v4.2 profile on Open cases with known v4.0 excess moves, including `5x5_cv2_test_12`, `5x5_cv1_test_12`, and `7x5_cv2_test_2`.

- [ ] **Step 2: Run Paper hard sample**

Run the new v4.2 profile on Paper cases including `CV/10-6/data10-6-12`, `CV/5-4/data5-4-28`, and one BF case where v4.0 is already shorter than BS-FRG.

- [ ] **Step 3: Compare to v4.0 and BS-FRG reference**

Report solved count, moves, runtime, postcompress accepted count, and moves saved.
