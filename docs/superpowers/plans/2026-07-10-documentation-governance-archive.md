# Documentation Governance And Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize all Markdown documentation around the promoted v4.3 mainline without deleting historical benchmark evidence.

**Architecture:** Keep six active source-of-truth documents and operational guides in place. Move superseded guides, reports, brainstorm briefs, legacy notes, and completed plans/specs to `docs/archive/`; keep `docs/experiments/` in place as an immutable evidence vault to avoid mass link churn.

**Tech Stack:** Git, Markdown, PowerShell, ripgrep

---

### Task 1: Create The Archive Structure

**Files:**
- Create: `docs/archive/README.md`
- Create: `docs/archive/guides/README.md`
- Create: `docs/archive/reports/README.md`
- Create: `docs/archive/brainstorm/README.md`
- Create: `docs/archive/superpowers/README.md`

- [x] Create lifecycle indexes that state v4.3 is current and archived content is historical.
- [x] Create `docs/reports/README.md` as the active report entrypoint.
- [x] Create `docs/guides/README.md` as the active guide entrypoint.

### Task 2: Move Superseded Documents

**Files:**
- Move: 12 superseded files from `docs/guides/` to `docs/archive/guides/`
- Move: 3 v4.0/v4.2 reports from `docs/reports/` to `docs/archive/reports/`
- Move: `CHATGPT_PROJECT_SUMMARY.md` and `CHATGPT_NEXT_STEP_BRAINSTORM.md` to `docs/archive/brainstorm/`
- Move: `docs/legacy/greedy_readme.md` to `docs/archive/legacy/greedy_readme.md`
- Move: completed files in `docs/superpowers/plans/` and `docs/superpowers/specs/` to `docs/archive/superpowers/`, excluding this governance plan and design

- [x] Use `git mv` for every tracked file.
- [x] Add a historical banner to archived guides, reports, and brainstorm briefs.

### Task 3: Rewrite Active Navigation

**Files:**
- Modify: `README.md`
- Modify: `docs/index.md`
- Modify: `docs/experiments/README.md`
- Modify: `docs/guides/algorithm-profile-status.md`
- Modify: `docs/guides/current-qubo-beam-algorithm.md`
- Modify: `docs/guides/current-best-algorithm.md`
- Modify: `docs/guides/current-mainline-algorithm.md`

- [x] Put the complete v4.3 profile and `1681/1681`, `105530` headline first.
- [x] Remove active claims that v4.0 or the early union profile is current.
- [x] Point baseline and research links to `docs/archive/`.
- [x] Define `docs/experiments/` as evidence, not recommendation.

### Task 4: Verify Documentation Integrity

**Files:**
- Verify: every tracked `*.md`

- [x] Run `rg` for stale current/default/promoted v4.0 and old-union claims.
- [x] Parse relative Markdown links and require zero missing local targets.
- [x] Run `git diff --check`.
- [x] Run `python -m pytest -q` to ensure documentation moves did not break repository tests.

### Task 5: Publish

**Files:**
- Commit: all documentation governance and archive changes

- [ ] Commit with message `Reorganize documentation around v4.3`.
- [ ] Push `main` to `origin` and `github` without force.
- [ ] Verify both remote `main` refs point to the new commit.
