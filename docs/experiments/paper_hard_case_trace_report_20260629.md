# Paper BF/CV Hard-Case Trace Report - 2026-06-29

## Decision

Do not tune another weight yet.

The hard BF/CV failures split into different mechanisms:

- BF14/BF32: current compact union does not reproduce the short BS-FRG recovery
  behavior. When it succeeds, it is often through a long FRG seed path, not the
  beam path.
- CV 5-8 / 6-6: micro rollout can perturb a good compact path into a much
  longer feasible route. This is a path-debt problem.
- CV 10-6: the search can lower `n` substantially but gets trapped in a dead
  low-n basin.
- CV 10-10: the search barely lowers `n`; useful recovery candidates are
  effectively absent from the retained frontier.

The next algorithm idea should be receiver-debt / basin-aware selection, not a
new broad fallback.

## Artifacts

Compact union trace:

- `docs/experiments/paper_hard_trace_union_20260629.csv`
- `docs/experiments/paper_hard_trace_union_frontier_20260629.csv`

Compact micro-tiebreak trace:

- `docs/experiments/paper_hard_trace_micro_tiebreak_20260629.csv`
- `docs/experiments/paper_hard_trace_micro_tiebreak_frontier_20260629.csv`

BS-FRG paper-pure reference:

- `docs/experiments/paper_hard_trace_bs_frg_pure_20260629.csv`

Instrumentation added:

- `scripts/benchmark_compact_paper_set.py` now accepts
  `--frontier-trace-output`, `--frontier-trace-top-k`, and
  `--terminal-trace-output`, forwarding them to the compact solver.

## Method

Seven exact hard/regression cases were traced sequentially with `time_limit=20`
and frontier `top_k=48`:

| dataset | family | case | why selected |
|---|---|---|---|
| BF | BF14 | `cpmp_16_8_103_21_78_1` | BF deep-recovery stress case |
| BF | BF32 | `cpmp_20_8_128_52_96_3` | micro-tiebreak can expose a long feasible path |
| CV | 6-6 | `data6-6-8` | large micro-vs-union path-quality swing |
| CV | 5-8 | `data5-8-1` | micro path-debt regression |
| CV | 5-8 | `data5-8-10` | CV 5-8 control case |
| CV | 10-6 | `data10-6-7` | low-n dead-basin failure |
| CV | 10-10 | `data10-10-1` | very hard CV structural failure |

The trace is diagnostic only. It does not replace paired open120 and paper
sample10 promotion runs.

## Outcome Table

`primary` means feasible, business-cap accepted, and within the benchmark time
scorecard. `feasible` means the path reaches `final_n=0` but may be over cap or
over time.

| case | union result | micro-tiebreak result | BS-FRG paper-pure | classification |
|---|---:|---:|---:|---|
| BF14 / `cpmp_16_8_103_21_78_1` | feasible, 946 moves, not primary | feasible, 946 moves, not primary | 246 moves | long FRG seed, compact beam does not find short route |
| BF32 / `cpmp_20_8_128_52_96_3` | failed, `final_n=96` | feasible, 747 moves, not primary | 145 moves | micro exposes long seed path, not good recovery |
| CV 6-6 / `data6-6-8` | primary, 55 moves | feasible, 404 moves, not primary | 64 moves | micro path-debt regression |
| CV 5-8 / `data5-8-1` | primary, 46 moves | feasible, 53 moves, over time | 47 moves | micro path-debt regression |
| CV 5-8 / `data5-8-10` | primary, 44 moves | primary, 44 moves | 53 moves | compact path already good |
| CV 10-6 / `data10-6-7` | failed, `final_n=50` | failed, `final_n=50` | 224 moves | low-n dead basin |
| CV 10-10 / `data10-10-1` | failed, `final_n=82` | failed, `final_n=82` | 489 moves | useful candidate universe missing |

## Trace Signals

### BF14/BF32

The compact trace does not show a short beam route.

- BF14 compact beam frontier only moved from `n=78` to layer-best `n=68`
  across 15 traced layers.
- BF14 final compact path came from `frg_seed_used=True`, `frg_seed_moves=946`.
- BF32 union beam moved only from `n=96` to `n=91` across 5 traced layers and
  produced no feasible compact path.
- BF32 micro-tiebreak also did not solve through the beam; it accepted an FRG
  seed with 747 moves.
- BS-FRG paper-pure solved the same two cases in 246 and 145 moves.

Interpretation: the current FRG seed is not the same quality as the paper-style
BS-FRG search. The compact beam/QUBO layer sees very large candidate pools, but
the retained frontier does not enter the short recovery basin.

### CV 5-8 And CV 6-6

The compact union route is already good here; micro rollout can make it worse.

- `data6-6-8`: union primary path is 55 moves. Micro-tiebreak path is 404
  moves and not primary.
- `data5-8-1`: union primary path is 46 moves. Micro-tiebreak path is 53 moves
  and over time.
- Micro-tiebreak selected many micro-scored rows in these cases:
  - `data6-6-8`: 166 micro-selected candidates in result telemetry.
  - `data5-8-1`: 168 micro-selected candidates in result telemetry.
- Frontier trace shows many selected rows with `receiver_last_buffer_consumed`
  after micro is enabled:
  - `data6-6-8`: 42 selected top-trace rows.
  - `data5-8-1`: 50 selected top-trace rows.

Interpretation: micro rollout is not merely finding "safer" states. It can
reward local continuations that consume receiver slack and create a later cleanup
debt. This explains why solve count can improve while moves and full penalty get
worse.

### CV 10-6

This is not a simple "no progress" case.

- Union trace: `n=50` reaches minimum layer-best `n=9` across 91 layers.
- Micro trace: `n=50` reaches minimum layer-best `n=8` across 92 layers.
- Both variants still finish with `final_n=50`, no valid compact path.
- BS-FRG paper-pure reaches 224 moves.

Interpretation: useful reducing candidates do appear, and the beam can enter a
low-n region, but that region does not contain a valid completion under the
current retained lineage. The issue is basin quality, not just candidate
generation count.

### CV 10-10

This is closer to candidate-universe failure.

- Union trace: `n=82` only reaches minimum layer-best `n=76` across 17 layers.
- Micro trace: `n=82` only reaches minimum layer-best `n=79` across 14 layers.
- Candidate pools are large, around 54k-58k average candidates per traced layer,
  but the search still barely reduces `n`.
- BS-FRG paper-pure reaches 489 moves.

Interpretation: large candidate count is not enough. The current one-step
successor universe plus fixed beam selection is not producing or retaining the
right long-horizon recovery states.

## What This Says About Current Union

The current mainline remains:

```text
qubo_scheduler_state_v5_qubo_diversity_union_polish
```

This trace does not invalidate it. It explains its boundary.

Strengths:

- Good on CV 5-8 style cases when the compact route is already in the right
  basin.
- QUBO selection contributes sparse frontier diversity without breaking the
  open120 level.
- The mainline can use FRG seed paths as a safety signal.

Weaknesses:

- FRG seed quality can be much worse than full BS-FRG paper-pure recovery.
- Micro rollout is not a safe global score; it needs a debt guard.
- QUBO diversity over parent/source/signature is not enough to distinguish
  recovery basins.
- High candidate count does not imply useful candidate coverage.

## Next Algorithm Direction

Build one diagnostic/ablation profile, not a stack of gates:

```text
scheduler_state_v5_qubo_diversity_union_debt_basin_polish
```

Proposed signals:

- `receiver_debt`: penalty for consuming scarce receiver slack, especially when
  the candidate does not immediately reduce final-n risk.
- `escape_potential`: bonus when the move creates future legal reduction space,
  not only a one-step safe continuation.
- `basin_label`: coarse route class used as a QUBO pair term, so the beam keeps
  genuinely different recovery plans rather than many variants of the same
  basin.

Promotion rule remains unchanged:

- no open120 regression;
- no paper sample10 solved-count or full-penalty regression;
- trace evidence must show whether the improvement came from useful-child
  generation, useful-child selection, or escaping a dead parent basin.
