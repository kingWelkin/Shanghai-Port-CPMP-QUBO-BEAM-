# Macro Credit V2 Parameter Tuning - 2026-07-01

## Decision

Keep `qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish` unchanged as the mainline profile.

The current default parameter balance remains the best verified trade-off across the required benchmark axes:

- open120: full coverage, `120/120`, moves/full penalty `4105`, ref gap sum `710`.
- paper sample1 BF/CV: best primary solved and best full penalty among the full-sample tuning candidates, `56/62`, full penalty `4437`.
- No tested parameter override improved the combined open120 + BF + CV promotion gate.

Lower move totals in worse candidates are not treated as improvements when primary solved drops, because unsolved cases do not contribute solved moves and must be judged through full penalty.

## Mainline Parameters

Profile: `qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish`

```text
union_diversity_qubo_completion_witness_credit_weight = 48.0
union_diversity_qubo_receiver_debt_credit_weight = 10.0
union_diversity_qubo_source_scarcity_credit_weight = 8.0
union_diversity_qubo_macro_path_credit_weight = 10.0
union_diversity_qubo_macro_future_credit_min_witness_quality = 0.35
union_diversity_qubo_receiver_debt_penalty_weight = 18.0
union_diversity_qubo_macro_future_credit_cap = 24.0
union_diversity_qubo_macro_edge_chain_cost_weight = 200.0
quality_reservoir_enabled = False
```

## Full Paper Sample1 Results

Command shape:

```powershell
python scripts\benchmark_compact_paper_set.py --source "C:\Users\Donghao\Desktop\上港堆叠装箱\external\paper_cpmp" --datasets BF CV --sample-per-family 1 --jobs 8 --variant qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish --time-limit 20 --alpha 10 <overrides> --output docs\experiments\<name>.csv
```

| Candidate | Override | Primary solved | Feasibility | Moves | Full penalty | Avg runtime sec | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| default v2 | none | 56/62 | 56/62 | 2638 | 4437 | 8.154 | Keep |
| witness56 | completion witness weight `56` | 55/62 | 58/62 | 2513 | 4963 | 8.088 | Reject: fewer primary solves, worse penalty |
| cap20 | macro future credit cap `20` | 49/62 | 55/62 | 2114 | 6157 | 8.233 | Reject: large BF and CV regression |
| witness025 | min witness quality `0.25` | 54/62 | 56/62 | 2446 | 5034 | 7.600 | Reject: fewer primary solves, worse penalty |
| debt12 | receiver debt credit weight `12` | 52/62 | 56/62 | 2319 | 5452 | 8.357 | Reject: fewer primary solves, worse penalty |

Key full-sample failures:

- `witness56` improved CV primary from `17/21` to `18/21`, but BF primary dropped from `39/41` to `37/41` and introduced three feasibility-only-over-time cases.
- `cap20` looked promising on the residual edge set, but full paper sample1 dropped to `49/62` and lost additional medium CV cases such as `CV/5-8` and `CV/6-6`.
- `witness025` and `debt12` both confirmed that loosening the witness gate or increasing receiver-debt credit hurts the BF/CV balance.

## Residual Edge Screen

Edge set:

```text
BF14, BF30, BF32, CV/5-4, CV/5-5, CV/5-9, CV/6-6, CV/10-6, CV/10-10
```

Command shape:

```powershell
python scripts\benchmark_compact_paper_set.py --source "C:\Users\Donghao\Desktop\上港堆叠装箱\external\paper_cpmp" --datasets BF CV --families BF14 BF30 BF32 5-4 5-5 5-9 6-6 10-6 10-10 --sample-per-family 1 --jobs 6 --variant qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish --time-limit 20 --alpha 10 <overrides> --output docs\experiments\<name>.csv
```

| Candidate | Primary solved | Feasibility | Moves | Full penalty | Avg runtime sec |
|---|---:|---:|---:|---:|---:|
| witness56 | 6/9 | 6/9 | 518 | 1285 | 11.540 |
| cap20 | 5/9 | 5/9 | 510 | 1388 | 11.636 |
| witness56_path8 | 5/9 | 5/9 | 491 | 1369 | 12.422 |
| witness56_scarcity10 | 5/9 | 5/9 | 399 | 1579 | 12.579 |
| default v2 | 3/9 | 4/9 | 214 | 2013 | 13.731 |

The edge screen is useful for finding stress points, but it over-rewards parameters that aggressively favor macro witnesses on a few residual cases. Full paper sample1 showed that those same settings destabilize broader BF/CV behavior.

## Interpretation

Macro credit v2 works because it gives QUBO enough cross-layer information to select recovery chains with future evidence, while still keeping the beam's local ranking and Wang pair QUBO stabilizers in control.

The rejected overrides failed for different reasons:

- Higher witness weight (`56`) over-trusts completion witnesses. It rescues `CV/6-6` and improves feasibility, but turns some BF cases into over-time feasibility-only hits and loses primary solves.
- Lower credit cap (`20`) under-trusts long macro futures. It can help a residual edge slice, but the full run loses both BF and medium CV cases.
- Lower witness threshold (`0.25`) admits noisier witnesses. It does not add enough new primary solves to pay for the ranking drift.
- Higher receiver-debt credit (`12`) biases toward debt-heavy states too early. It preserves some feasibility but worsens primary solve rate and penalty.

Current default v2 is therefore a conservative Pareto point: it does not solve every residual hard case, but it gives the strongest combined primary solved count and full penalty across the paper sample while preserving open120 full coverage.

## Next Tuning Boundary

Further progress should avoid one-dimensional weight tuning. The remaining hard cases are structural:

- `BF14`, `BF30`: deep recovery chains are still hard to preserve under a primary beam ranking.
- `CV/5-4`, `CV/10-6`, `CV/10-10`: candidate generation and receiver feasibility remain the bottleneck, not just QUBO selection weights.

The next useful experiment should improve witness quality or candidate generation before QUBO selection, rather than further increasing macro credit weights.
