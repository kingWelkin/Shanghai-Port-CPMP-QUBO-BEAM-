# QUBO Diversity Union Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ablation-friendly `scheduler_state_v5_qubo_diversity_union_polish` profile that keeps the v5 deterministic core and uses one QUBO selector to pick diverse exploration states from combined Wang/FRG/rollout signals.

**Architecture:** The new profile extends `scheduler_state_v5_frg_seed_qubo_polish`. Candidate generation remains unchanged. A new union selector runs inside `_select_enhanced_qubo_frontier` before the generic QUBO supplement, scores a bounded top-ranked candidate window, and uses QUBO pair penalties to avoid same-parent, same-source, same-signature, and same n-band collapse.

**Tech Stack:** Python, existing `QuboModel`, `SimulatedAnnealingQuboSampler`, `compute_frg_transition_signals`, compact Wang feature helpers, open benchmark runner.

## Status

Completed and promoted as the current mainline candidate. The profile landed as `scheduler_state_v5_qubo_diversity_union_polish`, with the open runner variant `qubo_scheduler_state_v5_qubo_diversity_union_polish`.

The implementation was later extended with opt-in rollout graft and FRG frontier ablations, but those descendants remain experimental unless a paired full run beats the diversity-union mainline on both open120 and the paper-set gate.

---

### Task 1: Unit-Test Union QUBO Selection

**Files:**
- Modify: `tests/test_qubo.py`

- [x] **Step 1: Write selector behavior tests**

Add tests that build several same-quality child states and assert the union selector prefers different parents/sources/signatures when pair penalties are active.

- [x] **Step 2: Run focused failing tests**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py::test_union_diversity_qubo_selects_quality_diverse_candidates -q --basetemp .tmp_pytest
```

Expected before implementation: import or attribute failure for the new selector.

### Task 2: Implement Selector and Profile

**Files:**
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`

- [x] **Step 1: Add config fields**

Add disabled-by-default `union_diversity_qubo_*` fields to `COMPACT_EXTRA_FIELDS` and `speed_v2`.

- [x] **Step 2: Add feature and selector helpers**

Implement `UnionDiversityQuboFeatures`, `_union_diversity_qubo_features`, `_union_diversity_objective`, `_repair_union_diversity_selection`, and `_select_union_diversity_qubo_supplement`.

- [x] **Step 3: Wire into enhanced frontier**

Call the union selector before generic `focused_multi` QUBO so it acts as one exploration slot source and does not replace core beam selection.

- [x] **Step 4: Add profile**

Add `scheduler_state_v5_qubo_diversity_union_polish` extending the recorded FRG seed struct-skip baseline with union selector enabled and existing ledger rank disabled.

### Task 3: Wire Open Runner

**Files:**
- Modify: `scripts/open_cpmp_benchmarks.py`
- Modify: `tests/test_qubo.py`

- [x] **Step 1: Add variant mapping**

Map `qubo_scheduler_state_v5_qubo_diversity_union_polish` to the new solver profile and include it in CLI choices.

- [x] **Step 2: Add parser passthrough**

Expose union selector weights as optional CLI flags and pass them into compact args.

- [x] **Step 3: Run focused tests**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py::test_open_runner_accepts_qubo_diversity_union_variant -q --basetemp .tmp_pytest
```

Expected after implementation: pass.

### Task 4: Verify and Benchmark

**Files:**
- Create: `docs/experiments/qubo_diversity_union_experiment_20260621.md`

- [x] **Step 1: Run unit suite slice**

Run focused union tests plus existing FRG seed tests.

- [x] **Step 2: Run compile check**

Run:

```powershell
$env:PYTHONPATH='.'; python -m py_compile scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py
```

- [x] **Step 3: Run smoke and focused residual**

Run first 10 open cases and the current residual cases against the new variant.

- [x] **Step 4: Run full open120 paired with recorded baseline**

Run open120 t10 for the new profile and compare against `frg_seed_struct_skip_open120_t10_jobs8_20260621.csv`.

- [x] **Step 5: Record decision**

Promote only if the new profile does not drop below `115/120` and improves penalty or moves. Otherwise keep it as an ablation.

### Self-Review

- The plan only changes beam selection and benchmark wiring.
- There is no new fallback solver and no case-specific gate.
- The recorded struct-skip baseline remains the promotion benchmark.

### Outcome

The selected `w4` diversity-union setting met the promotion thresholds on open120:

| profile | valid | full penalty | solved moves | gap sum | avg runtime sec |
|---|---:|---:|---:|---:|---:|
| `scheduler_state_v5_qubo_diversity_union_polish` | 118/120 | 3858 | 3677 | 335 | 4.442 |

The paper sample10 gate improved sharply over the old `wang_rescue_polish` baseline:

| profile | cases | solved | full penalty | avg runtime sec |
|---|---:|---:|---:|---:|
| `wang_rescue_polish` | 611 | 224 | 1114273 | 14.995 |
| `union_diversity` | 611 | 572 | 1058300 | 12.567 |

Decision: keep `scheduler_state_v5_qubo_diversity_union_polish` as the current mainline candidate and keep `scheduler_state_v5_wang_rescue_polish` as the explicit paired baseline.
