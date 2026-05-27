# Greedy vs Exact MIP Benchmark

Generated: 2026-05-26T11:22:37
Python: 3.13.11
Gurobi time limit per MIP solve: 20 seconds

This benchmark uses small pre-marshalling instances plus small cases derived from `data/input` by selecting a few real stacks and re-ranking their original seq IDs to `1..n`.
The exact MIP models are not intended for the full 60-container production cases.

CSV: `docs/experiments/greedy_vs_mip.csv`

## Summary Table

| Case | Method | m | H | n | maxSteps | Feasible | Optimal | Moves/Obj | Gap vs best | Wall sec | Solver sec | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sorted_2x2 | Greedy | 2 | 2 | 2 | 2 | True | - | 0 | 0 | 0.0000 | - | SORTED |
| sorted_2x2 | MIP-3D | 2 | 2 | 2 | 2 | True | True | 0 | 0 | 0.0041 | 0.0010 | OPTIMAL |
| sorted_2x2 | MIP-4D | 2 | 2 | 2 | 2 | True | True | 0 | 0 | 0.0016 | 0.0000 | OPTIMAL |
| one_move_2x2 | Greedy | 2 | 2 | 2 | 3 | True | - | 1 | 0 | 0.0000 | - | SORTED |
| one_move_2x2 | MIP-3D | 2 | 2 | 2 | 3 | True | True | 1 | 0 | 0.0028 | 0.0010 | OPTIMAL |
| one_move_2x2 | MIP-4D | 2 | 2 | 2 | 3 | True | True | 1 | 0 | 0.0024 | 0.0020 | OPTIMAL |
| two_bad_stacks_3x2 | Greedy | 3 | 2 | 4 | 4 | True | - | 2 | 0 | 0.0001 | - | SORTED |
| two_bad_stacks_3x2 | MIP-3D | 3 | 2 | 4 | 4 | True | True | 2 | 0 | 0.0342 | 0.0210 | OPTIMAL |
| two_bad_stacks_3x2 | MIP-4D | 3 | 2 | 4 | 4 | True | True | 2 | 0 | 0.0207 | 0.0100 | OPTIMAL |
| buffer_needed_3x3 | Greedy | 3 | 3 | 5 | 5 | False | - | 1 | - | 0.0001 | - | NOT_SORTED |
| buffer_needed_3x3 | MIP-3D | 3 | 3 | 5 | 5 | True | True | 4 | 0 | 0.1314 | 0.1060 | OPTIMAL |
| buffer_needed_3x3 | MIP-4D | 3 | 3 | 5 | 5 | True | True | 4 | 0 | 0.1391 | 0.0940 | OPTIMAL |
| mixed_4x3 | Greedy | 4 | 3 | 6 | 6 | False | - | 2 | - | 0.0002 | - | NOT_SORTED |
| mixed_4x3 | MIP-3D | 4 | 3 | 6 | 6 | True | True | 3 | 0 | 0.2862 | 0.2260 | OPTIMAL |
| mixed_4x3 | MIP-4D | 4 | 3 | 6 | 6 | True | True | 3 | 0 | 0.8996 | 0.7680 | OPTIMAL |
| real_A1_stacks_1_2_3 | Greedy | 3 | 6 | 5 | 5 | False | - | 0 | - | 0.0002 | - | NOT_SORTED |
| real_A1_stacks_1_2_3 | MIP-3D | 3 | 6 | 5 | 5 | True | True | 2 | 0 | 0.0649 | 0.0430 | OPTIMAL |
| real_A1_stacks_1_2_3 | MIP-4D | 3 | 6 | 5 | 5 | True | True | 2 | 0 | 1.4072 | 1.0980 | OPTIMAL |
| real_A1_stacks_1_2_4 | Greedy | 3 | 6 | 6 | 6 | False | - | 0 | - | 0.0001 | - | NOT_SORTED |
| real_A1_stacks_1_2_4 | MIP-3D | 3 | 6 | 6 | 6 | True | True | 4 | 0 | 0.2094 | 0.1710 | OPTIMAL |
| real_A1_stacks_1_2_4 | MIP-4D | 3 | 6 | 6 | 6 | True | True | 4 | 0 | 4.9471 | 4.4720 | OPTIMAL |
| real_A2_stacks_1_2_8 | Greedy | 3 | 6 | 7 | 6 | False | - | 1 | - | 0.0001 | - | NOT_SORTED |
| real_A2_stacks_1_2_8 | MIP-3D | 3 | 6 | 7 | 6 | True | True | 4 | 0 | 0.3912 | 0.3370 | OPTIMAL |
| real_A2_stacks_1_2_8 | MIP-4D | 3 | 6 | 7 | 6 | True | True | 4 | 0 | 4.4751 | 3.8830 | OPTIMAL |
| real_B1_stacks_1_2_3 | Greedy | 3 | 6 | 6 | 5 | False | - | 2 | - | 0.0001 | - | NOT_SORTED |
| real_B1_stacks_1_2_3 | MIP-3D | 3 | 6 | 6 | 5 | True | True | 3 | 0 | 0.1268 | 0.0950 | OPTIMAL |
| real_B1_stacks_1_2_3 | MIP-4D | 3 | 6 | 6 | 5 | True | True | 3 | 0 | 1.8761 | 1.4830 | OPTIMAL |
| real_C1_stacks_1_2_3 | Greedy | 3 | 6 | 6 | 5 | False | - | 0 | - | 0.0001 | - | NOT_SORTED |
| real_C1_stacks_1_2_3 | MIP-3D | 3 | 6 | 6 | 5 | True | True | 2 | 0 | 0.0842 | 0.0500 | OPTIMAL |
| real_C1_stacks_1_2_3 | MIP-4D | 3 | 6 | 6 | 5 | True | True | 2 | 0 | 1.3294 | 0.9420 | OPTIMAL |

## Move Sequences

- `sorted_2x2` / `Greedy`: -
- `sorted_2x2` / `MIP-3D`: -
- `sorted_2x2` / `MIP-4D`: -
- `one_move_2x2` / `Greedy`: 1:2 1->2
- `one_move_2x2` / `MIP-3D`: 1:2 1->2
- `one_move_2x2` / `MIP-4D`: 1:2 1->2
- `two_bad_stacks_3x2` / `Greedy`: 1:4 1->3; 2:3 2->3
- `two_bad_stacks_3x2` / `MIP-3D`: 1:4 1->3; 2:3 2->3
- `two_bad_stacks_3x2` / `MIP-4D`: 1:4 1->3; 2:3 2->3
- `buffer_needed_3x3` / `Greedy`: 1:5 1->2
- `buffer_needed_3x3` / `MIP-3D`: 1:5 1->2; 2:1 1->3; 3:5 2->1; 4:4 2->1
- `buffer_needed_3x3` / `MIP-4D`: 1:5 1->2; 2:1 1->3; 3:5 2->1; 4:4 2->1
- `mixed_4x3` / `Greedy`: 1:4 3->4; 2:6 1->2
- `mixed_4x3` / `MIP-3D`: 1:6 1->4; 2:5 2->4; 3:4 3->4
- `mixed_4x3` / `MIP-4D`: 1:6 1->4; 2:5 2->4; 3:4 3->4
- `real_A1_stacks_1_2_3` / `Greedy`: -
- `real_A1_stacks_1_2_3` / `MIP-3D`: 1:1 3->2; 2:5 1->3
- `real_A1_stacks_1_2_3` / `MIP-4D`: 1:1 3->2; 2:5 1->3
- `real_A1_stacks_1_2_4` / `Greedy`: -
- `real_A1_stacks_1_2_4` / `MIP-3D`: 1:6 1->2; 2:4 1->3; 3:6 2->1; 4:4 3->1
- `real_A1_stacks_1_2_4` / `MIP-4D`: 1:1 2->3; 2:2 2->3; 3:6 1->2; 4:2 3->1
- `real_A2_stacks_1_2_8` / `Greedy`: 1:4 3->1
- `real_A2_stacks_1_2_8` / `MIP-3D`: 1:1 2->1; 2:4 3->1; 3:5 3->2; 4:4 1->2
- `real_A2_stacks_1_2_8` / `MIP-4D`: 1:1 2->1; 2:4 3->1; 3:5 3->2; 4:4 1->2
- `real_B1_stacks_1_2_3` / `Greedy`: 1:3 3->1; 2:6 3->1
- `real_B1_stacks_1_2_3` / `MIP-3D`: 1:3 3->1; 2:1 2->1; 3:6 3->2
- `real_B1_stacks_1_2_3` / `MIP-4D`: 1:3 3->1; 2:1 2->1; 3:6 3->2
- `real_C1_stacks_1_2_3` / `Greedy`: -
- `real_C1_stacks_1_2_3` / `MIP-3D`: 1:2 2->3; 2:5 1->2
- `real_C1_stacks_1_2_3` / `MIP-4D`: 1:2 2->3; 2:5 1->2

## Notes

- `Wall sec` includes Python model construction, Gurobi solve, extraction, and validation.
- `Solver sec` is Gurobi's own runtime and is blank for the greedy planner.
- `Gap vs best` uses the best proven optimal MIP objective for each case.
- Greedy rows marked `NOT_SORTED` are partial heuristic trajectories, not feasible final pre-marshalling solutions.
