# Full Real Focused Route Legality Audit

This audit reruns the selected direct component for each focused route and independently replays every move from the original stacks.

CSV: `docs\experiments\full_real_business_focused_legality_audit.csv`

## Summary

- Checked rows: `60`
- Passed rows: `60`
- Failed rows: `0`

| route | rows | passed | avg final N | avg moves | max observed G |
|---|---:|---:|---:|---:|---:|
| beam_focused_with_qubo_portfolio | 15 | 15 | 10.93 | 22.40 | 39 |
| beam_focused_with_tuned_qubo | 15 | 15 | 11.67 | 21.40 | 39 |
| beam_focused_with_dense_qubo | 15 | 15 | 11.87 | 21.00 | 39 |
| beam_focused_with_qubo | 15 | 15 | 12.00 | 20.60 | 39 |

## Checks

- source and destination stack indices are in range
- source and destination differ
- source stack is non-empty
- moved box equals the current source top
- destination stack is not full
- `from_level` and `to_level` match replayed stack heights
- container multiset is unchanged after every move
- every intermediate `G = moves + N(Future)` satisfies `G <= floor(1.3*N0)`
- final replayed stacks equal the solver state
