# Wang Paper Audit Documentation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the interrupted Wang-Jin-Lim / bounded-beam audit into durable project documentation and clear promotion boundaries.

**Architecture:** Keep implementation changes out of scope unless the audit exposes a concrete correctness bug. Add one focused audit guide, link it from the docs index, and verify the existing Wang/classical probes with targeted tests and small smoke benchmarks. Treat `target_guided_v1` and `wang_probe_v1` as inspired search signals, `WangBsG/B` as adapted classical baselines, and `paper_bounded_beam_cpmp_probe` as a negative adapter probe.

**Tech Stack:** Markdown docs, Python benchmark scripts, pytest, existing DLTS external benchmark runner.

---

### Task 1: Write The Audit Guide

**Files:**
- Create: `docs/guides/wang-target-guided-paper-audit.md`

- [ ] **Step 1: Create the audit document**

Create `docs/guides/wang-target-guided-paper-audit.md` with this structure:

```markdown
# Wang-Jin-Lim Target-Guided Paper Audit

## Paper Identity

The paper under audit is Wang, Jin, and Lim, "Target-guided algorithms for the container pre-marshalling problem", Omega 53 (2015), pp. 67-77, DOI `10.1016/j.omega.2014.12.002`.

It should not be referred to as a Zhang paper in project documentation. If another Zhang paper is later added, it must get a separate audit page and separate implementation mapping.

## Implementation Mapping

| Project item | Closest paper concept | Current status |
|---|---|---|
| `target_guided_v1` in `scripts/benchmark_compact_beam_qubo_n0.py` | Target layout preference | Inspired signal only; not a TGH reproduction |
| `wang_probe_v1` | Fixed-prefix / target-pair intuition | Inspired probe only; not a full BS-G/BS-B reproduction |
| `WangBsGiantMovesSolver` | BS-G giant-move beam search | Adapted classical baseline |
| `WangBsBabyMovesSolver` | BS-B baby-move beam search | Adapted classical baseline |
| `paper_bounded_beam_cpmp_probe` | Bacci-Mattia-Ventura bounded beam BRP skeleton | Negative CPMP adapter probe |

## Main Audit Findings

1. `target_guided_v1` does not implement Wang-Jin-Lim TGH. It only adds target-layout gap ranking and protected target-guided supplements inside the compact QUBO beam.
2. `wang_probe_v1` is not full BS-G or BS-B. It scores compact states by fixed-prefix and repair-pair signals, then injects a bounded supplement.
3. `WangBsGiantMovesSolver` and `WangBsBabyMovesSolver` are the closest classical-paper implementations, but they are still adapted to the repository's legal top-move CPMP state engine.
4. The bounded-beam probe should remain a negative result unless larger tests show it can solve external CPMP instances.

## Evidence From Current Audit

- Crossref metadata for DOI `10.1016/j.omega.2014.12.002` identifies the authors as Ning Wang, Bo Jin, and Andrew Lim.
- `python -m pytest -q tests\test_classical_cpmp_baselines.py tests\test_classical_cpmp_runner.py tests\test_paper_bbs_probe.py -p no:cacheprovider` returned `14 passed`.
- `python -m py_compile container_reshuffle\solvers\classical_cpmp.py scripts\benchmark_classical_cpmp_baselines.py scripts\benchmark_paper_bounded_beam_cpmp.py` returned exit code 0.
- Smoke benchmark `scratch_wang_baseline_5x5_cv2_2_audit.csv`: `wang_bs_g` solved 2/2 with 80 total moves; `wang_bs_b` solved 2/2 with 67 total moves.
- Smoke benchmark `scratch_paper_bbs_audit_5x5_cv2_2.csv`: `paper_bounded_beam_cpmp_probe` solved 0/2.

## Promotion Boundary

Only `WangBsG/B` should be described as adapted classical Wang-Jin-Lim baselines. `target_guided_v1`, `wang_probe_v1`, and scheduler Wang variants should be described as inspired signals or probes. The bounded-beam adapter should not be promoted beyond a negative evidence probe.
```

- [ ] **Step 2: Verify wording**

Run:

```powershell
rg -n -i "zhang|full reproduction|exact reproduction|paper reproduction" docs/guides/wang-target-guided-paper-audit.md
```

Expected: either no output, or only a sentence explicitly saying this is not a Zhang paper.

### Task 2: Link The Audit Page

**Files:**
- Modify: `docs/index.md`

- [ ] **Step 1: Add one docs index entry**

Add this bullet near the existing QUBO / external CPMP guide links:

```markdown
- [Wang-Jin-Lim target-guided paper audit](../../guides/wang-target-guided-paper-audit.md)：区分 Wang-Jin-Lim 2015、项目内 target-guided 信号、Wang classical baseline、以及 bounded-beam 负结果 probe。
```

- [ ] **Step 2: Verify the link target exists**

Run:

```powershell
Test-Path docs\guides\wang-target-guided-paper-audit.md
```

Expected: `True`.

### Task 3: Keep Code Labels Conservative

**Files:**
- Inspect: `scripts/benchmark_compact_beam_qubo_n0.py`
- Inspect: `container_reshuffle/solvers/classical_cpmp.py`
- Inspect: `scripts/benchmark_paper_bounded_beam_cpmp.py`

- [ ] **Step 1: Check labels**

Run:

```powershell
rg -n -i "Zhang|full reproduction|exact reproduction|paper reproduction|Wang2015|probe|inspired|adapted" scripts container_reshuffle tests docs
```

Expected:
- No `Zhang` label for the Wang-Jin-Lim work.
- `target_guided_v1` and `wang_probe_v1` remain probe/signal language.
- `WangBsG/B` may keep `paper=Wang2015` status strings because they are the adapted classical baselines.
- `paper_bounded_beam_cpmp_probe` remains marked as a probe.

- [ ] **Step 2: Patch only misleading labels**

If Step 1 finds misleading wording, patch only the affected documentation or status text. Do not change algorithm behavior in this task.

### Task 4: Verification

**Files:**
- Test: `tests/test_classical_cpmp_baselines.py`
- Test: `tests/test_classical_cpmp_runner.py`
- Test: `tests/test_paper_bbs_probe.py`

- [ ] **Step 1: Run targeted tests**

Run:

```powershell
python -m pytest -q tests\test_classical_cpmp_baselines.py tests\test_classical_cpmp_runner.py tests\test_paper_bbs_probe.py -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 2: Run compile check**

Run:

```powershell
python -m py_compile container_reshuffle\solvers\classical_cpmp.py scripts\benchmark_classical_cpmp_baselines.py scripts\benchmark_paper_bounded_beam_cpmp.py
```

Expected: exit code 0.

- [ ] **Step 3: Optionally refresh smoke evidence**

Run only if the current scratch files are stale or deleted:

```powershell
python scripts\benchmark_classical_cpmp_baselines.py --families 5x5_cv2_test --limit-per-family 2 --solvers wang_bs_g wang_bs_b --time-limit 1 --max-depth 80 --beam-width 64 --max-candidates-per-layer 64 --output docs\experiments\scratch_wang_baseline_5x5_cv2_2_audit.csv
python scripts\benchmark_paper_bounded_beam_cpmp.py --families 5x5_cv2_test --limit-per-family 2 --beta 100 --time-limit 1 --branching incumbent --output docs\experiments\scratch_paper_bbs_audit_5x5_cv2_2.csv
```

Expected:
- Wang baseline smoke solves both cases, with explicit move gaps.
- Bounded-beam adapter remains recorded as probe evidence, not a promoted solver.

### Task 5: Final Report

**Files:**
- Read: `git diff --stat`
- Read: `git status --short`

- [ ] **Step 1: Summarize final state**

Report:
- The paper identity correction.
- Which code paths are paper-inspired signals versus adapted baselines.
- Test and smoke evidence.
- Any scratch files left under `docs/experiments/scratch_*_audit*`.

- [ ] **Step 2: Ask before committing**

Do not commit until the user explicitly asks.
