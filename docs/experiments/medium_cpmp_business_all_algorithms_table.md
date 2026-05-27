# Medium CPMP Business Case: All Solver Table

Instance: `medium_cpmp_business_5x4_n12_seed3`, `m=5`, `H=4`, `n=12`, initial `N(Future)=7`, business cap `G<=9`.

Best feasible move count in this table: `8`. `mip_3d` is archived and kept only as a comparison row.

| Solver | Feasible | Optimal | Moves | Gap | Runtime(s) | Max G | Total N | QUBO calls | QUBO time(s) | MIP nodes | MIP vars | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| random | False | False |  |  | 0.000 |  |  | 0 | 0 |  |  | NO_FEASIBLE_FOUND |
| greedy_retrieval | False | False |  |  | 0 |  |  | 0 | 0 |  |  | NOT_APPLICABLE |
| greedy_cpmp | False | False |  |  | 0.000 |  |  | 0 | 0 |  |  | NO_FEASIBLE_FOUND |
| beam | True | False | 8 | 0 | 0.197 | 8 | 29 | 0 | 0 |  |  | FEASIBLE |
| diverse_beam | True | False | 8 | 0 | 117.258 | 8 | 29 | 0 | 0 |  |  | FEASIBLE |
| rolling_horizon | False | False |  |  | 0.069 |  |  | 0 | 0 |  |  | NO_FEASIBLE_FOUND |
| qubo_beam | True | False | 8 | 0 | 0.698 | 8 | 29 | 2 | 0.400 |  |  | FEASIBLE; samplers=simulated_annealing |
| qubo_beam_api | True | False | 8 | 0 | 1.356 | 8 | 29 | 2 | 0.400 |  |  | FEASIBLE; samplers=ising_matrix_api_fallback_simulated_annealing |
| qubo_rolling_horizon | False | False |  |  | 5.030 |  |  | 1 | 5.000 |  |  | NO_FEASIBLE_FOUND |
| qubo_batch_relocation | False | False |  |  | 0 |  |  | 0 | 0 |  |  | NOT_APPLICABLE |
| stack_assignment_guided_cpmp | True | False | 8 | 0 | 5.103 | 8 | 29 | 1 | 5.000 |  |  | FEASIBLE |
| mip_4d | True | True | 8 | 0 | 8.755 | 8 | 29 | 0 | 0 | 37 | 7388 | OPTIMAL |
| mip_3d | True | True | 8 | 0 | 24.157 | 8 | 29 | 0 | 0 | 510 | 4236 | OPTIMAL |
| rolling_horizon_mip | True | False | 9 | 1 | 3.292 | 9 | 32 | 0 | 0 | 7 | 26295 | FEASIBLE; mip_calls=9; window_statuses=OPTIMAL |
| final_assignment_mip | True | False | 8 | 0 | 11.516 | 8 | 29 | 0 | 0 | 1 | 7493 | OPTIMAL; first_target=INFEASIBLE; repair=beam_target; realization_optimal=True; master=OPTIMAL |
| pattern_column_mip | True | False | 8 | 0 | 11.901 | 8 | 29 | 0 | 0 | 1 | 11358 | OPTIMAL; first_target=INFEASIBLE; repair=beam_target; realization_optimal=True; master=OPTIMAL |

## Notes

- `greedy_retrieval` and `qubo_batch_relocation` are CRP-only, so they are `NOT_APPLICABLE` on this CPMP instance.
- `qubo_beam` and `qubo_beam_api` use the current fast QUBO settings: `max_qubo_vars=80`, `qubo_call_interval=4`, `qubo_time_limit_sec=0.2`, `qubo_sa_iterations=1000`.
- `mip_4d` is the recommended exact benchmark; `mip_3d` is archived.
- `final_assignment_mip` and `pattern_column_mip` are hybrid workflows. Their fixed-target realization was optimal, but the whole workflow is not a global proof, so `optimal=False`.
- Raw CSV: `docs/experiments/medium_cpmp_business_all_algorithms_table.csv`.
