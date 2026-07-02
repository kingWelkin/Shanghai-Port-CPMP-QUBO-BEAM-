# Promotion Paper Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repeatable BF/CV sample10 paper-set benchmark gate for compact v5 solver profiles.

**Architecture:** Create a small compact-solver paper runner that reuses the existing BF/CV discovery and loader code from `scripts/benchmark_bs_frg_paper.py`. Keep BS-FRG reproduction separate; this runner only executes compact beam/QUBO profiles and writes CSV plus markdown summary.

**Tech Stack:** Python 3.13, pytest, existing compact CPMP benchmark scripts.

---

### Task 1: Add Paper Sample Discovery Tests

**Files:**
- Modify: `tests/test_bs_frg_paper_benchmarks.py`

- [ ] **Step 1: Write failing tests**

Add tests that import `discover_sampled_paper_cases` from `benchmark_compact_paper_set.py`. The tests should create tiny BF and CV directory trees under `tmp_path`, then verify:

- each family is capped at 10 cases
- case order is natural sort order
- CV families group by the `H-S` part of `dataH-S-i.dat`

- [ ] **Step 2: Run red tests**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_bs_frg_paper_benchmarks.py -q --basetemp .tmp_pytest
```

Expected: tests fail with `ModuleNotFoundError` or missing `discover_sampled_paper_cases`.

### Task 2: Add Compact Paper Runner

**Files:**
- Create: `scripts/benchmark_compact_paper_set.py`

- [ ] **Step 1: Implement minimal discovery API**

Create `discover_sampled_paper_cases(source, sample_per_family=10, datasets=("BF", "CV"))` using existing `discover_paper_cases` behavior and natural sorting from `benchmark_bs_frg_paper.py`.

- [ ] **Step 2: Implement compact run path**

Add a CLI that accepts:

```text
--source
--datasets
--sample-per-family
--variant
--time-limit
--alpha
--output
```

For each case, load the instance with the existing paper loader, run `run_compact_beam_instance`, validate with the open-row wrapper, and write rows.

- [ ] **Step 3: Write markdown summary**

The summary should report total solved, total rows, solved moves, average runtime, per-dataset totals, per-family totals, and failures.

- [ ] **Step 4: Run green tests**

Run the same pytest command from Task 1 and expect pass.

### Task 3: Add Smoke and Full Sample Verification

**Files:**
- Generated: `docs/experiments/*promotion_paper_sample*.csv`
- Generated: `docs/experiments/*promotion_paper_sample*_summary.md`

- [ ] **Step 1: Run a small smoke**

Run BF/CV with `--sample-per-family 1` on `scheduler_state_v5_qubo_diversity_union_polish` at a short limit to verify wiring.

- [ ] **Step 2: Run the intended sample10 gate**

Run BF/CV with `--sample-per-family 10`, `--time-limit 20`, `--alpha 10`, and variant `scheduler_state_v5_qubo_diversity_union_polish`.

- [ ] **Step 3: Report results**

Record solved count, moves, runtime, and failure list. Compare only after a same-day baseline paper sample exists.

### Task 4: Final Checks

**Files:**
- Modify: only runner/test/docs created by this plan

- [ ] **Step 1: Compile scripts**

Run:

```powershell
$env:PYTHONPATH='.'; python -m py_compile scripts\benchmark_compact_paper_set.py
```

- [ ] **Step 2: Check diff hygiene**

Run:

```powershell
git diff --check -- scripts\benchmark_compact_paper_set.py tests\test_bs_frg_paper_benchmarks.py docs\superpowers\plans\2026-06-21-promotion-paper-set.md
```

- [ ] **Step 3: Commit implementation if tests and sample run complete**

Commit only the files from this plan and the already approved spec.
