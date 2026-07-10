# Wang-Jin-Lim Target-Guided Paper Audit

> [!IMPORTANT]
> Historical document. It does not define the current default. The promoted mainline is [v4.3](../../guides/v43-mainline-algorithm.md), validated at `1681/1681` with `105530` moves.

## Paper Identity

The paper under audit is Wang, Jin, and Lim, "Target-guided algorithms for the container pre-marshalling problem", Omega 53 (2015), pp. 67-77, DOI `10.1016/j.omega.2014.12.002`.

It should not be referred to as a Zhang paper in project documentation. If another Zhang paper is later added, it must get a separate audit page and separate implementation mapping.

## Implementation Mapping

| Project item | Closest paper concept | Current status |
|---|---|---|
| `target_guided_v1` in `scripts/benchmark_compact_beam_qubo_n0.py` | Target layout preference | Inspired signal only; not a TGH reproduction |
| `wang_probe_v1` | Fixed-prefix / target-pair intuition | Inspired probe only; not a full BS-G/BS-B reproduction |
| `WangBsGiantMovesSolver` | BS-G giant-move beam search | Adapted classical baseline |
| `WangBsBabyMovesSolver` | BS-B baby-move beam search | Adapted classical baseline |
| `paper_bounded_beam_cpmp_probe` | Bacci-Mattia-Ventura bounded beam BRP skeleton | Negative CPMP adapter probe |

## Main Audit Findings

1. `target_guided_v1` does not implement Wang-Jin-Lim TGH. It only adds target-layout gap ranking and protected target-guided supplements inside the compact QUBO beam.
2. `wang_probe_v1` is not full BS-G or BS-B. It scores compact states by fixed-prefix and repair-pair signals, then injects a bounded supplement.
3. `WangBsGiantMovesSolver` and `WangBsBabyMovesSolver` are the closest classical-paper implementations, but they are still adapted to the repository's legal top-move CPMP state engine.
4. The bounded-beam probe should remain a negative result unless larger tests show it can solve external CPMP instances.

## Evidence From Current Audit

- Crossref metadata for DOI `10.1016/j.omega.2014.12.002` identifies the authors as Ning Wang, Bo Jin, and Andrew Lim.
- `python -m pytest -q tests\test_classical_cpmp_baselines.py tests\test_classical_cpmp_runner.py tests\test_paper_bbs_probe.py -p no:cacheprovider` returned `14 passed`.
- `python -m py_compile container_reshuffle\solvers\classical_cpmp.py scripts\benchmark_classical_cpmp_baselines.py scripts\benchmark_paper_bounded_beam_cpmp.py` returned exit code 0.
- Smoke benchmark `scratch_wang_baseline_5x5_cv2_2_audit.csv`: `wang_bs_g` solved 2/2 with 80 total moves; `wang_bs_b` solved 2/2 with 67 total moves.
- Smoke benchmark `scratch_paper_bbs_audit_5x5_cv2_2.csv`: `paper_bounded_beam_cpmp_probe` solved 0/2.

## Promotion Boundary

Only `WangBsG/B` should be described as adapted classical Wang-Jin-Lim baselines. `target_guided_v1`, `wang_probe_v1`, and scheduler Wang variants should be described as inspired signals or probes. The bounded-beam adapter should not be promoted beyond a negative evidence probe.
