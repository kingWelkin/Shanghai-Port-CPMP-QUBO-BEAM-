# Macro Chain Witness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a QUBO-selectable macro-chain witness profile for large CV recovery without using BS-FRG as a final fallback.

**Architecture:** The implementation extends the existing compact beam/QUBO runner. A bounded BS-FRG rollout from selected frontier parents creates macro-edge candidates, and normal QUBO survivor selection decides whether those candidates enter the next frontier.

**Tech Stack:** Python, pytest, existing compact byte-state CPMP runner, existing BS-FRG helper functions, existing open/paper benchmark scripts.

---

### Task 1: Routing and Profile Tests

**Files:**
- Modify: `tests/test_qubo.py`
- Modify: `scripts/open_cpmp_benchmarks.py`
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`

- [ ] **Step 1: Write failing tests**

Add tests asserting that `qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish` routes to `scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish`, and that paper-set unknown variants raise a clear error instead of silently using `speed_v2`.

- [ ] **Step 2: Verify red**

Run:

```powershell
pytest tests/test_qubo.py -q -k "macro_chain_witness or unknown_paper_variant"
```

Expected: tests fail because the profile and routing do not exist yet.

- [ ] **Step 3: Implement routing**

Add the solver profile and open-runner variant mapping. Add a paper-runner guard that raises `ValueError` when `_args_for_variant` leaves `solver_profile` unchanged for an unknown `qubo_scheduler_state_v5...` variant.

- [ ] **Step 4: Verify green**

Run:

```powershell
pytest tests/test_qubo.py -q -k "macro_chain_witness or unknown_paper_variant"
```

Expected: selected tests pass.

### Task 2: Macro-Chain Witness Candidate Injection

**Files:**
- Modify: `tests/test_qubo.py`
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`

- [ ] **Step 1: Write failing candidate test**

Add a focused unit test that monkeypatches the BS-FRG completion helper to return a feasible move path. The test calls the new injection helper and asserts:

- one candidate is added;
- `_frg_macro_edge_path_by_code` contains the child path;
- `_frg_macro_edge_parent_by_code` binds the child to the source parent;
- terminal graft counters remain zero.

- [ ] **Step 2: Verify red**

Run:

```powershell
pytest tests/test_qubo.py -q -k "macro_chain_witness_candidate"
```

Expected: test fails because the helper does not exist.

- [ ] **Step 3: Implement helper**

Add `_inject_macro_chain_witness_candidates` beside the existing FRG frontier injection helpers. Reuse `_bs_frg_completion_cached`, `_layout_path_from_moves`, `_encode`, `_parent_chain_would_cycle`, and macro-edge state dictionaries. The helper must add candidates to the normal `candidates` dict only.

- [ ] **Step 4: Verify green**

Run:

```powershell
pytest tests/test_qubo.py -q -k "macro_chain_witness_candidate"
```

Expected: candidate injection test passes.

### Task 3: Selection Integration

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`
- Modify: `tests/test_qubo.py`

- [ ] **Step 1: Write failing selection test**

Add a test showing a macro-chain witness candidate can survive through `_select_next_frontier` and be counted as selected macro-edge telemetry, without calling terminal graft.

- [ ] **Step 2: Verify red**

Run:

```powershell
pytest tests/test_qubo.py -q -k "macro_chain_witness_selected"
```

Expected: test fails until integration is wired into the search loop.

- [ ] **Step 3: Wire helper into the layer loop**

Call `_inject_macro_chain_witness_candidates` after shallow FRG child injection and before `_select_next_frontier`. Record origin as `frg_macro_chain_witness`. Keep the candidate in the same pool as all other candidates.

- [ ] **Step 4: Verify green**

Run:

```powershell
pytest tests/test_qubo.py -q -k "macro_chain_witness"
```

Expected: all macro-chain tests pass.

### Task 4: Benchmark Smoke

**Files:**
- No source edits unless tests expose a defect.

- [ ] **Step 1: Run focused unit suite**

```powershell
pytest tests/test_qubo.py -q -k "macro_chain_witness or unknown_paper_variant"
```

- [ ] **Step 2: Run py_compile**

```powershell
python -m py_compile scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py scripts\benchmark_compact_paper_set.py
```

- [ ] **Step 3: Run CV smoke**

```powershell
python scripts\benchmark_compact_paper_set.py --source "C:\Users\Donghao\Desktop\上港堆叠装箱\external\paper_cpmp" --datasets CV --families 10-10 --sample-per-family 1 --jobs 1 --variant qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish --time-limit 60 --alpha 10 --output docs\experiments\probe_cv_10_10_1_macro_chain_witness_t60_20260701.csv
```

- [ ] **Step 4: Run open smoke**

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test --limit-per-family 3 --variant qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish --time-limit 10 --alpha 10 --output docs\experiments\probe_open6_macro_chain_witness_t10_20260701.csv
```

Expected: no crashes; collect whether the new profile improves CV without damaging open smoke.
