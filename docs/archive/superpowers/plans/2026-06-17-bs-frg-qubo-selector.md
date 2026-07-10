# BS-FRG QUBO Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in BS-FRG QUBO beam supplement based on `scheduler_state_v5_wang_rescue_polish`.

**Architecture:** Reuse compact beam's legal child generation and insert a small QUBO supplement into `_select_enhanced_qubo_frontier`. The QUBO combines FRG transition quality with code-stable diversity penalties and exactly-k selection, then falls back to ranked fill as before.

**Tech Stack:** Python, pytest, existing `QuboModel`, `SimulatedAnnealingQuboSampler`, compact benchmark scripts.

---

### Task 1: Selector Unit Behavior

**Files:**
- Modify: `tests/test_qubo.py`
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`

- [x] **Step 1: Write the failing selector test**

Add a test that calls `_select_bs_frg_qubo_supplement` with four legal encoded child layouts. The test expects the selector to return two candidates and diversify away from duplicate parent/signature choices.

- [x] **Step 2: Run the focused test and confirm failure**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py::test_bs_frg_qubo_supplement_selects_quality_diverse_candidates --basetemp .tmp_pytest -p no:cacheprovider -q
```

Expected: import or attribute failure because `_select_bs_frg_qubo_supplement` does not exist yet.

- [x] **Step 3: Implement the selector**

Add helper functions in `scripts/benchmark_compact_beam_qubo_n0.py`:

- `_bs_frg_candidate_features(...)`;
- `_bs_frg_qubo_linear_energy(...)`;
- `_bs_frg_signature(...)`;
- `_select_bs_frg_qubo_supplement(...)`.

The implementation must use `_qubo_sampler_config(...)` and repair the sample to exact target cardinality.

- [x] **Step 4: Run the focused test and confirm pass**

Run the same pytest command and confirm one passing test.

### Task 2: Frontier Integration and Telemetry

**Files:**
- Modify: `tests/test_qubo.py`
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`

- [x] **Step 1: Write the failing frontier integration test**

Add a test that enables `bs_frg_qubo_enabled`, calls `_select_enhanced_qubo_frontier`, and asserts:

- selected BS-FRG candidate is included;
- `selected_by_qubo` includes it;
- QUBO call/var/time counters increment;
- `_bs_frg_qubo_selected_total` increments.

- [x] **Step 2: Run the focused test and confirm failure**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py::test_bs_frg_qubo_selector_is_used_inside_enhanced_frontier --basetemp .tmp_pytest -p no:cacheprovider -q
```

Expected: candidate is not selected because the hook is not wired.

- [x] **Step 3: Wire selector into `_select_enhanced_qubo_frontier`**

Insert the BS-FRG QUBO supplement after existing deterministic FRG/DLTS supplements and before generic QUBO fill. Update `CompactBeamResult`, row serialization, and runtime counters.

- [x] **Step 4: Run the focused tests and confirm pass**

Run both BS-FRG focused tests.

### Task 3: Profile and Open Runner Plumbing

**Files:**
- Modify: `tests/test_qubo.py`
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `scripts/open_cpmp_benchmarks.py`

- [x] **Step 1: Write failing plumbing tests**

Add tests that assert:

- `_apply_solver_profile` maps `scheduler_state_v5_bs_frg_qubo_polish` to enabled BS-FRG QUBO settings;
- `open_cpmp_benchmarks` accepts and forwards BS-FRG QUBO CLI overrides.

- [x] **Step 2: Run focused tests and confirm failure**

Run the new plumbing tests.

- [x] **Step 3: Add profile, CLI flags, variant alias, and output fields**

Add a new profile based on `scheduler_state_v5_wang_rescue_polish`. Add compact CLI args and open runner pass-through args for BS-FRG QUBO fields.

- [x] **Step 4: Run focused tests and full `tests/test_qubo.py`**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py --basetemp .tmp_pytest_full -p no:cacheprovider -q
```

### Task 4: Benchmark Smoke

**Files:**
- Create: `docs/experiments/scratch_bs_frg_qubo_small_20260617.csv`
- Create: `docs/experiments/scratch_bs_frg_qubo_small_20260617_summary.md`

- [x] **Step 1: Run a small DLTS slice**

Run:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test --limit-per-family 5 --variant qubo_scheduler_state_v5_bs_frg_qubo_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_bs_frg_qubo_small_20260617.csv
```

- [x] **Step 2: Compare with v5 baseline**

Report valid count, moves, penalty, runtime, and whether the new selector deserves open120.
