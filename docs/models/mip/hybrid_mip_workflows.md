# Hybrid MIP Workflows

This file documents three experimental MIP workflows added after archiving the 3D model. They are not global exact solvers unless explicitly stated; every returned move sequence is still validated by simulation and by the business `G=M+N(Future)` cap.

## Rolling Horizon MIP

Solver name: `rolling_horizon_mip`

The solver repeatedly builds a short 4D slot-occupancy MIP from the current state:

- horizon length: `rolling_depth`
- objective: prioritize lower terminal `N(Future)`, then lower cumulative window `N(Future)`, then fewer moves
- business cap: `current_move_count + local_moves + N(Future) <= floor(alpha*N0)`
- execution: commit the first `prefix_commit_length` moves, then rebuild the next window

This is a heuristic. It is faster and more scalable than one full-horizon exact MIP, but it does not prove global optimality.

## Final Stack Assignment + Realization MIP

Solver name: `final_assignment_mip`

The workflow has two stages:

1. Master MIP assigns each box to a final physical stack, respecting stack capacity.
2. A 4D realization MIP tries to realize that assignment within `max_depth`.

The master objective is a proxy: keep boxes near their current stacks, penalize moving deeply buried boxes, and penalize grouping boxes that are already in an opposite order. Because this master does not fully know dynamic top-move feasibility, the selected target can be infeasible under the business cap.

If the first target is infeasible, the solver repairs the target using a validated beam-search terminal layout and reruns the 4D realization MIP with that target. The status records this as `repair=beam_target`.

## Pattern / Column Generation Prototype

Solver name: `pattern_column_mip`

This is an upfront-column prototype, not a full branch-and-price implementation.

1. Generate legal decreasing stack patterns, i.e. subsets sorted bottom -> top decreasing.
2. Master MIP selects one pattern per physical stack and covers each box exactly once.
3. A 4D realization MIP tries to realize the selected pattern assignment.

As with final assignment, the selected pattern target can be dynamically infeasible, so the same beam-target repair is used.

## Medium Benchmark

On `medium_cpmp_business_5x4_n12_seed3` with `max_depth=9`, `alpha=1.3`:

| Solver | Feasible | Moves | Runtime(s) | Notes |
|---|---:|---:|---:|---|
| `rolling_horizon_mip` | true | 9 | 3.34 | fast, not globally optimal |
| `final_assignment_mip` | true | 8 | 11.55 | first target infeasible, repaired by beam target |
| `pattern_column_mip` | true | 8 | 11.62 | first target infeasible, repaired by beam target |

Raw result: `docs/experiments/medium_cpmp_business_hybrid_mip.csv`.
