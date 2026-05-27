# Larger CPMP Business-Constraint Run

Instance: `larger_cpmp_6x5_n25_seed5`.

Business rule enabled:

- `N(Future)` is counted from the first adjacent violation in each stack upward.
- `G = move_count + N(Future)` must satisfy `G <= floor(1.3 * N(0))` at every intermediate state.
- Non-MIP search ranks CPMP states primarily by cumulative `N(Future)`, then current `N(Future)`, then move count.

For this instance, `N(0)=18`, so the hard limit is `floor(1.3*18)=23`.

| Solver | Feasible | Objective | Runtime sec | Expanded | Generated | API calls | Entropy max | Entropy limit | Validation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy_cpmp | False | inf | 0.0002 | 0 | 0 | 0 |  |  | False |
| beam | False | inf | 0.5346 | 1676 | 12179 | 0 |  |  | False |
| rolling_horizon | False | inf | 0.3579 | 1320 | 15777 | 0 |  |  | False |
| qubo_beam_api_sparse_k16 | False | inf | 1.1721 | 1676 | 12179 | 0 |  |  | False |

## Interpretation

The previously reported 32-move beam/API solutions violate the business cap because their maximum `G` reaches 32 while the limit is 23. After moving the cap into the search expansion logic, these solvers no longer return those paths as feasible.
