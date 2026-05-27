# Larger Real-Data Greedy vs MIP Benchmark

Cases are derived from current `data/input` by selecting four real global stacks from A1 and re-ranking their original seq IDs to `1..n`.
These are still local subproblems, not the full 60-container production instances.

- MIP time limit: 30 seconds per model per case.
- `Wall sec` includes model construction, solve, extraction, and validation.
- `Solver sec` is Gurobi runtime only.
- Greedy uses the current project greedy rule with the same `maxSteps` as MIP.

CSV: `docs/experiments/greedy_vs_mip_larger_real.csv`

## Summary Table

| Case | Method | m | H | n | maxSteps | Feasible | Optimal | Moves/Obj | Gap vs best | Wall sec | Solver sec | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| larger_real_A1_1_2_4_7 | Greedy | 4 | 6 | 8 | 8 | False | - | 0 | - | 0.0001 | - | NOT_SORTED |
| larger_real_A1_1_2_4_7 | MIP-3D | 4 | 6 | 8 | 8 | True | True | 3 | 0 | 0.2857 | 0.1470 | OPTIMAL |
| larger_real_A1_1_2_4_7 | MIP-4D | 4 | 6 | 8 | 8 | True | True | 3 | 0 | 19.4123 | 17.5840 | OPTIMAL |
| larger_real_A1_1_2_3_6 | Greedy | 4 | 6 | 8 | 8 | False | - | 1 | - | 0.0002 | - | NOT_SORTED |
| larger_real_A1_1_2_3_6 | MIP-3D | 4 | 6 | 8 | 8 | True | True | 3 | 0 | 2.0893 | 1.9600 | OPTIMAL |
| larger_real_A1_1_2_3_6 | MIP-4D | 4 | 6 | 8 | 8 | True | True | 3 | 0 | 25.0089 | 23.1800 | OPTIMAL |
| larger_real_A1_1_2_4_6_n9 | Greedy | 4 | 6 | 9 | 9 | False | - | 1 | - | 0.0002 | - | NOT_SORTED |
| larger_real_A1_1_2_4_6_n9 | MIP-3D | 4 | 6 | 9 | 9 | True | True | 5 | 0 | 8.8725 | 8.6650 | OPTIMAL |
| larger_real_A1_1_2_4_6_n9 | MIP-4D | 4 | 6 | 9 | 9 | False | False | - | - | 32.4686 | 30.0980 | TIME_LIMIT |

## Move Sequences

- `larger_real_A1_1_2_4_7` / `Greedy`: -
- `larger_real_A1_1_2_4_7` / `MIP-3D`: 1:1 2->4; 2:2 2->3; 3:8 1->2
- `larger_real_A1_1_2_4_7` / `MIP-4D`: 1:1 2->4; 2:2 2->3; 3:8 1->2
- `larger_real_A1_1_2_3_6` / `Greedy`: 1:6 4->3
- `larger_real_A1_1_2_3_6` / `MIP-3D`: 1:1 3->2; 2:8 1->3; 3:6 4->3
- `larger_real_A1_1_2_3_6` / `MIP-4D`: 1:1 3->2; 2:8 1->3; 3:6 4->3
- `larger_real_A1_1_2_4_6_n9` / `Greedy`: 1:7 4->3
- `larger_real_A1_1_2_4_6_n9` / `MIP-3D`: 1:9 1->2; 2:4 1->3; 3:9 2->1; 4:7 4->1; 5:4 3->1
- `larger_real_A1_1_2_4_6_n9` / `MIP-4D`: -

## Observations

- MIP-3D proves optimality on all three larger cases within 30 seconds.
- MIP-4D proves optimality on the two `n=8` cases but hits the 30-second time limit without a validated incumbent on the `n=9` case in this run.
- Greedy is much faster but does not produce sorted final layouts on these larger local cases.
