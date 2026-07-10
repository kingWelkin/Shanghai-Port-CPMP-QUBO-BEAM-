# Deterministic QUBO Sampler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in fixed-iteration QUBO sampler mode and test whether it removes Wang/QUBO quality drift under the current v5 benchmark.

**Architecture:** Keep `QuboSamplerConfig` backward compatible by adding a boolean flag. Compact benchmark code reads `args.deterministic_qubo_sampler` and passes the inverse as `respect_time_limit` to every local simulated-annealing QUBO call.

**Tech Stack:** Python, pytest, existing compact CPMP benchmark scripts.

---

### Task 1: Sampler Fixed-Iteration Flag

**Files:**
- Modify: `container_reshuffle/qubo/sampler.py`
- Test: `tests/test_qubo.py`

- [ ] Write `test_simulated_annealing_can_ignore_tiny_time_limit`.
- [ ] Run the test and confirm it fails because `QuboSamplerConfig` lacks `respect_time_limit`.
- [ ] Add `respect_time_limit: bool = True` and make `SimulatedAnnealingQuboSampler.sample` skip deadline breaks when false.
- [ ] Rerun the sampler test.

### Task 2: Compact Runner Wiring

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `scripts/open_cpmp_benchmarks.py`
- Test: `tests/test_qubo.py`

- [ ] Add `deterministic_qubo_sampler` to compact result fields and solver profile defaults.
- [ ] Add `--deterministic-qubo-sampler/--no-deterministic-qubo-sampler` to both parsers.
- [ ] Replace local `QuboSamplerConfig(...)` construction in compact benchmark QUBO calls with a helper that sets `respect_time_limit=not args.deterministic_qubo_sampler`.
- [ ] Add tests for parser pass-through and row serialization.

### Task 3: Verification and Benchmark

**Files:**
- Read: `docs/experiments/scratch_wang_baseline_clean_open120_t20_20260616.csv`
- Write: `docs/experiments/scratch_wang_feature_cache_deterministic_qubo_open120_t20_20260617.csv`

- [ ] Run focused pytest for sampler/open-runner deterministic QUBO tests.
- [ ] Run full `tests/test_qubo.py`.
- [ ] Run open120 at 20s with `--deterministic-qubo-sampler`.
- [ ] Compare valid count, invalid cases, solved moves, full penalty, and runtime against the clean 20s baseline and previous cache run.
