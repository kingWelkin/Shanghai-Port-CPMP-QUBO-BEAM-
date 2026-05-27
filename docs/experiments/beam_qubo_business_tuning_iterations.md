# Beam / QUBO Beam Business Tuning Iterations

Scope: 15 full paired real cases (`01+26`, `W=20,H=6,N about 60`). Objective is lower final `N(Future)` under `G<=floor(1.3*N0)`.

## Full Runs

| Iteration | Solver | Avg final N | Avg reduction | Avg moves | Avg runtime(s) | Avg QUBO calls | Source |
|---|---|---:|---:|---:|---:|---:|---|
| I1 narrow10 plain/qubo, no elite | beam | 10.67 | 18.40 | 24.13 | 3.634 | 0.00 | `docs/experiments/full_real_business_beam_qubo_narrow10.csv` |
| I1 narrow10 plain/qubo, no elite | qubo_beam | 11.20 | 17.87 | 23.40 | 3.746 | 5.60 | `docs/experiments/full_real_business_beam_qubo_narrow10.csv` |
| I2 narrow10 + QUBO elite80, diversity0.5 | beam | 10.67 | 18.40 | 24.13 | 3.645 | 0.00 | `docs/experiments/full_real_business_beam_qubo_narrow10_elite.csv` |
| I2 narrow10 + QUBO elite80, diversity0.5 | qubo_beam | 11.27 | 17.80 | 23.47 | 3.792 | 5.60 | `docs/experiments/full_real_business_beam_qubo_narrow10_elite.csv` |
| I3 narrow10 + QUBO elite80, diversity0.1 | beam | 10.67 | 18.40 | 24.13 | 3.689 | 0.00 | `docs/experiments/full_real_business_beam_qubo_vs_greedy.csv` |
| I3 narrow10 + QUBO elite80, diversity0.1 | qubo_beam | 10.47 | 18.60 | 24.60 | 3.972 | 5.73 | `docs/experiments/full_real_business_beam_qubo_vs_greedy.csv` |

## Sample Sweep

The sample sweep used five representative cases (`A1,A5,B3,C4,C5`) with 4 seconds per solver. It showed that narrower beams are better for the business objective because they reach deeper trajectories before the time limit. Raw sweep: `docs/experiments/beam_qubo_business_tuning_sweep_sample.csv`.

| Config | Solver | Avg final N | Avg reduction | Avg runtime(s) |
|---|---|---:|---:|---:|
| narrow_10x3 | beam | 14.40 | 15.20 | 3.167 |
| narrow_10x3 | qubo | 14.60 | 15.00 | 3.551 |
| narrow_20x3 | beam | 16.40 | 13.20 | 4.007 |
| narrow_20x3 | qubo | 17.00 | 12.60 | 4.018 |
| current_30x3 | beam | 18.60 | 11.00 | 4.013 |
| current_30x3 | qubo | 18.40 | 11.20 | 4.016 |
| wider_50x5 | beam | 21.80 | 7.80 | 4.007 |
| wider_50x5 | qubo | 20.80 | 8.80 | 4.008 |
| wider_80x5 | beam | 23.60 | 6.00 | 4.011 |
| wider_80x5 | qubo | 23.60 | 6.00 | 4.056 |
| deep_20x2 | beam | 16.00 | 13.60 | 4.012 |
| deep_20x2 | qubo | 16.40 | 13.20 | 4.004 |
| branch_30x4 | beam | 18.20 | 11.40 | 4.005 |
| branch_30x4 | qubo | 19.00 | 10.60 | 4.016 |

## Current Best Configuration

- `beam`: `beam_width=10`, `max_candidates_per_layer=100`, `candidate_limit_per_state=3`, current `N(Future)` first scoring.
- `qubo_beam`: same beam settings plus QUBO selection every 5 layers, `diversity_weight=0.1`, 80% elite retention.
- Current best average final `N(Future)` is `10.47` from `qubo_beam`; plain beam is `10.67` and much simpler.
