# Medium CPMP Algorithm Comparison

Instance: `medium_cpmp_5x4_n12_seed1`

- Variant: `premarshalling_cpmp`
- Stacks: `[[4,11,2], [9,1,12], [7,3], [10,5,8], [6]]`
- m=5, H=4, n=12, occupancy=60%
- Goal: every stack strictly decreasing bottom -> top

Benchmark files:

- `docs/experiments/medium_cpmp_algorithms.csv`: default heuristic/QUBO run
- `docs/experiments/medium_cpmp_mip.csv`: exact MIP run with `max_steps=7`, `TimeLimit=30s`
- `docs/experiments/medium_cpmp_wide_beam.csv`: wider plain beam supplement
- `docs/experiments/medium_cpmp_qubo_beam_api.csv`: QUBO beam selection using the sibling Ising matrix API plus decode/repair
- `docs/experiments/medium_cpmp_comparison.csv`: merged comparison

## Summary

- Exact optimum: `6` relocations, proven by both MIP-3D and MIP-4D.
- Default beam / diverse beam / rolling horizon / QUBO beam / stack-assignment-guided CPMP all found 7 relocations, i.e. absolute gap `+1`, relative gap `16.7%`.
- A wider plain beam found the 6-relocation optimum in under 0.4s on this instance.
- QUBO beam with the external Ising matrix API plus repair found the 6-relocation optimum; the returned trajectory passed simulation validation.
- Greedy CPMP and random did not complete this instance under the tested depth.
- QUBO rolling horizon did not complete in this configuration; it generated QUBO calls but failed to assemble a validated full trajectory.

## Results

| Solver | Feasible | Optimal | Obj | Gap | Runtime sec | Expanded | Generated | QUBO vars | MIP vars | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| random | False | False | - | - | 0.0002 | 0 | 0 | 0 | - | NO_FEASIBLE_FOUND |
| greedy_cpmp | False | False | - | - | 0.0001 | 0 | 0 | 0 | - | NO_FEASIBLE_FOUND |
| beam | True | False | 7.0 | 1 / 16.7% | 0.0899 | 421 | 4598 | 0 | - | FEASIBLE |
| beam_wide | True | False | 6.0 | 0 / 0.0% | 0.3748 | 1720 | 15557 | 0 | - | FEASIBLE |
| diverse_beam | True | False | 7.0 | 1 / 16.7% | 24.6638 | 346 | 3765 | 0 | - | FEASIBLE |
| rolling_horizon | True | False | 7.0 | 1 / 16.7% | 0.0831 | 421 | 7123 | 0 | - | FEASIBLE |
| qubo_beam | True | False | 7.0 | 1 / 16.7% | 30.2579 | 345 | 3798 | 1019 | - | FEASIBLE |
| qubo_beam_api | True | False | 6.0 | 0 / 0.0% | 12.6071 | 328 | 3661 | 819 | - | FEASIBLE; samplers=ising_matrix_api |
| qubo_rolling_horizon | False | False | - | - | 20.1019 | 0 | 0 | 4723 | - | NO_FEASIBLE_FOUND |
| stack_assignment_guided_cpmp | True | False | 7.0 | 1 / 16.7% | 4.5703 | 343 | 3696 | 60 | - | FEASIBLE |
| mip_3d | True | True | 6.0 | 0 / 0.0% | 2.6955 | 0 | 0 | 0 | 2171 | OPTIMAL |
| mip_4d | True | True | 6.0 | 0 / 0.0% | 1.5954 | 0 | 0 | 0 | 5371 | OPTIMAL |

## Interpretation

- This instance is useful because it separates one-step greedy from search: Greedy gets stuck, but beam-style search succeeds.
- The default beam settings are fast but not exhaustive enough to prove or find the best 6-step path.
- Increasing plain beam width improves quality substantially here, with much less overhead than diverse/QUBO selection.
- The API-backed QUBO selection improved this run relative to the local-SA QUBO beam: fewer QUBO layers were needed before a 6-step validated solution was reached, although API call latency still dominates runtime.
- MIP remains practical on this moderate case and provides the benchmark optimum, but it should still be treated as a small/medium benchmark tool rather than the large-scale main route.
