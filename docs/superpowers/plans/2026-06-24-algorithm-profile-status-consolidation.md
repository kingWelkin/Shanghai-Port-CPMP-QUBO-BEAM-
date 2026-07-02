# Algorithm Profile Status Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the current CPMP algorithm conclusions, profile roles, and next-step guidance into repository documentation and one focused commit.

**Architecture:** This is a documentation-only consolidation. The canonical status table lives in `docs/guides/algorithm-profile-status.md`, while `README.md` and `docs/guides/current-mainline-algorithm.md` point readers to that table.

**Tech Stack:** Markdown documentation, existing Python benchmark/test commands for verification, git for the final commit.

---

### Task 1: Add The Canonical Profile Status Guide

**Files:**
- Create: `docs/guides/algorithm-profile-status.md`

- [x] **Step 1: Create a guide that separates defaults from experiments**

Create `docs/guides/algorithm-profile-status.md` with these sections:

```markdown
# Algorithm Profile Status

## Decision Snapshot

## Promotion Rule

## Stable Defaults And Baselines

## Opt-In Ablations

## Reproduction And Reference Assets

## Rejected Or Diagnostic Conclusions

## Recommended Next Work

## Source Evidence
```

- [x] **Step 2: Fill each profile row from current evidence**

Include the current mainline, paired baseline, internal-speed baseline, QUBO diversity variants, micro-rollout variants, FRG frontier variants, origin-aware variants, Wang BS-B, and BS-FRG pure reproduction.

- [x] **Step 3: Make the next-step guidance conservative**

State that the next algorithm work should use trace-first evidence, near-tie guards, and paired open120 plus paper sample10 gates instead of increasing QUBO size or turning BS-FRG into a fallback.

### Task 2: Add Entry Points From Existing Docs

**Files:**
- Modify: `README.md`
- Modify: `docs/guides/current-mainline-algorithm.md`

- [x] **Step 1: Link the status guide from README**

Add `docs/guides/algorithm-profile-status.md` to the Documentation section and keep the default mainline unchanged.

- [x] **Step 2: Link the status guide from the mainline guide**

Add a short note near the current ablation table explaining that the new guide is the canonical profile status table.

### Task 3: Verify And Commit

**Files:**
- Verify: Markdown and git state only

- [x] **Step 1: Run diff hygiene**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors.

- [x] **Step 2: Run focused docs-safe regression**

Run:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py -k "micro or receiver or qubo_diversity_union or frontier_trace_candidate_rows or wang_all" -q --basetemp .tmp_pytest_profile_status
```

Expected: the focused QUBO/profile wiring tests pass.

- [x] **Step 3: Review staged changes**

Run:

```powershell
git diff --stat
git status --short
```

Expected: only the documentation consolidation files are staged for commit. Existing local zip handoff artifacts remain untracked unless explicitly requested.

- [x] **Step 4: Commit**

Run:

```powershell
git add docs\guides\algorithm-profile-status.md README.md docs\guides\current-mainline-algorithm.md docs\superpowers\plans\2026-06-24-algorithm-profile-status-consolidation.md
git commit -m "docs: consolidate algorithm profile status"
```

Expected: one focused documentation commit.
