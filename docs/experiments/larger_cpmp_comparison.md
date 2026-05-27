# Larger CPMP API+Repair And MIP Trial

Instance: `larger_cpmp_6x5_n25_seed5`

- Variant: `premarshalling_cpmp`
- Stacks: `[[5,24,7,12], [11,13,23,19,4], [18,16,22,8,1], [3,25,6,2,17], [10,14,9,20], [15,21]]`
- m=6, H=5, n=25, occupancy=83.3%
- MIP was run with `max_steps=33`.
- Cold MIP runs used `TimeLimit=60s`.
- Warm-start MIP runs used the validated 32-move beam solution as a MIP start and `TimeLimit=600s`.
- Sparse API runs call the external Ising API only every `k` layers; skipped layers fall back to classical top-B beam selection.

## Result Summary

| Solver | Feasible | Obj | Bound | Gap | Runtime sec | Vars | Constraints | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| greedy_cpmp | False | - | - | - | 0.0001 | - | - | NO_FEASIBLE_FOUND |
| beam | True | 32 | - | - | 1.4701 | - | - | FEASIBLE |
| rolling_horizon | False | - | - | - | 4.1552 | - | - | NO_FEASIBLE_FOUND |
| qubo_beam_api_raw | False | - | - | - | 133.7557 | - | - | NO_FEASIBLE_FOUND; samplers=ising_matrix_api,ising_matrix_api_fallback_simulated_annealing |
| qubo_beam_api_elite_repair_every_layer | True | 33 | - | - | 93.4226 | - | - | FEASIBLE; samplers=ising_matrix_api |
| qubo_beam_api_sparse_k4 | True | 32 | - | - | 23.7024 | - | - | FEASIBLE; samplers=ising_matrix_api |
| qubo_beam_api_sparse_k8 | True | 32 | - | - | 12.3466 | - | - | FEASIBLE; samplers=ising_matrix_api |
| qubo_beam_api_sparse_k16 | True | 32 | - | - | 6.8282 | - | - | FEASIBLE; samplers=ising_matrix_api |
| mip_3d_cold_60s | False | - | 5 | - | 60.8260 | 27,962 | 72,383 | TIME_LIMIT |
| mip_4d_cold_60s | False | - | 6 | - | 61.6296 | 75,858 | 131,184 | TIME_LIMIT |
| mip_3d_warm_10min | True | 32 | 6 | 81.25% | 600.8779 | 27,962 | 72,383 | TIME_LIMIT |
| mip_4d_warm_10min | True | 32 | 0 | 100.00% | 601.7844 | 75,858 | 131,184 | TIME_LIMIT |

## Interpretation

- The larger near-full case is meaningfully harder than the previous 12-container benchmark: greedy fails immediately and rolling horizon does not finish a valid trajectory.
- Plain beam finds a 32-relocation feasible solution quickly.
- Raw API QUBO selection can lose important high-score beam candidates and failed on this instance.
- Adding elite-anchor repair makes the API route robust enough to recover a validated 33-relocation solution. It is close to beam quality but slower because API latency dominates runtime.
- Sparse API calls fix the runtime issue on this instance. Calling the API every 4, 8, or 16 layers all found the 32-relocation beam-quality solution.
- The best sparse run here used only one API call (`k=16`), found 32 relocations, and took 6.83 seconds. This confirms the earlier overhead was call frequency, not single-call API difficulty.
- Cold MIP did not find a feasible incumbent in 60 seconds for either model.
- With the beam MIP start, both MIP models kept the 32-relocation incumbent for 10 minutes but did not improve or prove optimality.
- The 3D integer-position model is substantially smaller than the 4D slot model here: 27,962 variables and 72,383 constraints versus 75,858 variables and 131,184 constraints.
- The 3D model produced the more useful 10-minute bound in this run: bound 6 and gap 81.25%. The 4D model retained the incumbent but its reported bound remained 0, giving a 100% gap.
- Current API+repair is useful as a hybrid candidate selector, but for this size it still needs stronger quality repair or adaptive fallback to beat plain beam consistently.
