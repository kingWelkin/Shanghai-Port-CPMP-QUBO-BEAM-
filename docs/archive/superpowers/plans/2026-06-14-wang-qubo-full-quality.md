# Wang/QUBO Full-Case Quality Trial Plan

## Goal

Try the Wang-inspired improvement paths against the current best compact QUBO/scheduler line and rank by full-case quality, not solved-case averages.

## Scope

- Add full-case quality metrics that penalize unsolved and invalid rows.
- Add Wang/QUBO portfolio candidates so Wang incumbent, Wang pair-QUBO, Wang hybrid, and current scheduler variants can be evaluated together.
- Keep compact code/parent reconstruction intact; Wang modules may add validated incumbents or frontier supplements, not splice arbitrary move sequences into the compact path.
- Produce benchmark CSV/summary artifacts for the internal 30-case set and open DLTS subsets.

## Implementation Steps

1. Add failing tests for full-case quality scoring and portfolio profile parsing.
2. Extend `benchmark_qubo_no_regret_portfolio_n0.py` with full-quality fields, Wang candidate profiles, and summary tables.
3. Add Wang-aware rescue/pair profile aliases only where the existing benchmark runner can consume them cleanly.
4. Run targeted unit tests.
5. Run smoke benchmarks first, then full internal 30 and open DLTS evaluation batches.
6. Write a concise experiment summary with full-case quality rankings and next recommendations.

## Quality Metric

Selection remains no-regret per case among valid candidates. Summary ranking uses:

- `full_quality_key`: invalid count, unsolved count, total penalized moves, total runtime.
- Internal business cases require both `final_n == 0` and `business_ok`.
- Penalized moves use actual valid moves; invalid/unsolved rows receive `max(reference, initial_n) + penalty_factor * initial_n` when a reference exists, otherwise `cap_13 + penalty_factor * n0`.

## Verification

- `python -m pytest tests/test_qubo.py -q`
- `python -m pytest tests/test_classical_cpmp_baselines.py tests/test_classical_cpmp_runner.py -q`
- Benchmark smoke with a small case set before full runs.
