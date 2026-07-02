# Promotion Paper Set Design

## Goal

Make paper-set evidence a required part of future solver promotion decisions. A candidate profile should no longer be judged on open120 alone.

## Scope

This design adds a repeatable benchmark gate named `paper_bfcv_sample10_t20` alongside the existing open120 and paper-selected checks.

The first paper gate uses the already downloaded BF/CV corpora in `external/paper_cpmp`:

- BF: each discovered BF family contributes the first 10 `.bay` instances in natural sort order.
- CV: each discovered CV `H-S` family contributes the first 10 `.dat` instances in natural sort order.
- The default solver time limit is 20 seconds per case.
- The default alpha remains 10, matching current v5 comparison runs.

The full BF/CV corpus stays as a heavier final verification gate, not the default day-to-day tuning run.

## Promotion Suite

Future promotion summaries should include three suites:

1. `open120_t10`
   - Source: `external/DLTS`
   - Families: the six open120 families already used in `scripts/open_cpmp_benchmarks.py`
   - Metrics: valid count, full penalty, solved moves, gap sum, runtime, QUBO telemetry

2. `paper_selected14_t20`
   - Source: curated BF/CV representative cases already used by current paper-selected experiments
   - Metrics: solved count, solved moves, failures, runtime, relevant selector telemetry

3. `paper_bfcv_sample10_t20`
   - Source: `external/paper_cpmp`
   - Datasets: BF and CV
   - Sampling: first 10 cases per family
   - Metrics: solved count, solved moves, final_n penalty, failures, runtime, relevant selector telemetry

## Decision Rules

The baseline remains the current strongest accepted profile until a candidate beats it in a same-day paired run.

A candidate can be promoted only if all of these hold:

- `open120_t10` does not regress below the accepted valid-count floor.
- `open120_t10` does not exceed the accepted full-penalty and solved-move thresholds.
- `paper_selected14_t20` does not lose solved cases versus the paired baseline.
- `paper_bfcv_sample10_t20` does not lose solved cases versus the paired baseline.
- Any move-sum improvement that comes with new failures is rejected.
- A win isolated to one paper family is treated as diagnostic evidence, not a promotion reason.

## Implementation Shape

Add a small paper-set runner for the compact v5 solver family instead of overloading the BS-FRG reproduction runner.

The runner should reuse:

- BF/CV discovery and loaders from `scripts/benchmark_bs_frg_paper.py`
- compact solver construction from `scripts/benchmark_compact_beam_qubo_n0.py`
- open-style row validation from `scripts/open_cpmp_benchmarks.py` when priority-group labels require relaxed validation

The runner should emit:

- one CSV with all sampled BF/CV rows
- one markdown summary with dataset/family totals and failure list
- optional paired diff when both baseline and candidate CSVs are present

## Testing

Focused tests should cover:

- BF/CV sample10 discovery uses natural ordering and caps each family at 10.
- CV family grouping uses the `H-S` portion of `dataH-S-i.dat`.
- The compact paper runner accepts the current v5-like variants.
- Summary generation reports solved count, moves, failures, and average runtime.

Smoke testing should run a very small BF/CV subset first, then the full sample10 gate.

## Non-Goals

- Do not replace the current open120 gate.
- Do not promote a solver from paper-set evidence alone.
- Do not run the full 1561-case BF/CV corpus on every tuning loop.
- Do not add BS-FRG as a fallback path to the v5 solver.
