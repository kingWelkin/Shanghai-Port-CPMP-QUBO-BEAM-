# Canonical Algorithm Scorecard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Organize the existing Real30, Open120, BF721, and CV840 historical comparison into one immediately readable, repository-tracked experiment folder without rerunning any solver.

**Architecture:** Keep one canonical Markdown table as the human entry point, one wide CSV as the machine-readable scorecard, and one source manifest that points each cell to its historical evidence. Copy only the ignored pure BS-FRG paper full-run source needed to audit the BF/CV split.

**Tech Stack:** Markdown, CSV, PowerShell/Git verification (no solver reruns)

---

### Task 1: Verify Historical Inputs

- [x] Aggregate the existing source CSVs and confirm the fixed scorecard values.
- [x] Confirm that v4.3 Real30 is `30/30, 525` and that `30/30, 516` belongs to pure BS-FRG.
- [x] Copy `docs/experiments/bs_frg_pure_paper_full_20260618.csv` and its summary into the canonical folder's `sources/` directory.

### Task 2: Build the Lightweight Scorecard

- [x] Create `docs/experiments/canonical_algorithm_scorecard_v43/README.md` with the four-suite table at the top, the v4.3 row highlighted, and the solved-first metric contract stated directly below it.
- [x] Create `canonical_algorithm_scorecard.csv` with solved, total, and solved-only move sums for all five algorithms.
- [x] Create `source_manifest.csv` mapping every measured or not-applicable cell to its exact evidence path and benchmark scope.
- [x] Add a precise `.gitignore` exception for this folder and a prominent link near the top of `docs/experiments/README.md`.

### Task 3: Verify and Publish

- [x] Check Markdown and CSV values agree exactly.
- [x] Check every manifest path exists and is tracked, and verify the copied pure BS-FRG files match their originals by SHA-256.
- [x] Stage the focused artifacts, run `git diff --cached --check`, commit, and push `main` to both configured remotes.
