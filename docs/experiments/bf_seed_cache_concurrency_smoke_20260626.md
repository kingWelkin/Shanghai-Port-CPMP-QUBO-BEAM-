# BF Seed Cache Concurrency Smoke, 2026-06-26

Scope:
- Variant: `qubo_scheduler_state_v5_qubo_diversity_union_polish`
- Families: `BF14`, `BF32`
- Sample: first case per family from `external\paper_cpmp`
- Purpose: verify that parent-process FRG seed materialization can be consumed by a parallel paper runner.

Commands:
- Focused tests: `python -m pytest tests\test_qubo.py::test_run_compact_beam_uses_materialized_frg_seed_before_live tests\test_qubo.py::test_run_compact_beam_reports_materialized_frg_seed_cache_miss tests\test_bs_frg_paper_benchmarks.py::test_paper_runner_materializes_frg_seed_cache_before_collecting_rows -q --basetemp .tmp_pytest_bf_seed_cache_focus`
- Regression slice: `python -m pytest tests\test_qubo.py -k "frg_seed or open_runner_accepts_frg_seed or union_diversity" tests\test_bs_frg_paper_benchmarks.py -q --basetemp .tmp_pytest_bf_seed_cache_regression`
- Priority-row regression after smoke: `python -m pytest tests\test_qubo.py -k "frg_seed or priority_group_row_serializes_frg_seed_cache or open_runner_accepts_frg_seed or union_diversity" tests\test_bs_frg_paper_benchmarks.py -q --basetemp .tmp_pytest_bf_seed_cache_regression_after_priority_fix`
- Compile check: `python -m py_compile scripts\benchmark_compact_beam_qubo_n0.py scripts\benchmark_compact_paper_set.py scripts\open_cpmp_benchmarks.py`
- Jobs1 materialization: `python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF --families BF14 BF32 --sample-per-family 1 --variant qubo_scheduler_state_v5_qubo_diversity_union_polish --alpha 10 --time-limit 20 --jobs 1 --frg-seed-cache-output docs\experiments\bf_seed_cache_smoke_20260626.jsonl --output docs\experiments\bf_seed_cache_smoke_jobs1_20260626.csv`
- Jobs2 consumption: `python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF --families BF14 BF32 --sample-per-family 1 --variant qubo_scheduler_state_v5_qubo_diversity_union_polish --alpha 10 --time-limit 20 --jobs 2 --frg-seed-cache-input docs\experiments\bf_seed_cache_smoke_20260626.jsonl --output docs\experiments\bf_seed_cache_smoke_jobs2_20260626.csv`

Outcome:
- Focused tests: PASS, `3 passed`.
- Regression slice: PASS, `21 passed, 345 deselected`.
- Priority-row regression after smoke: PASS, `22 passed, 345 deselected`.
- Compile check: PASS.
- BF smoke: PASS for cache-consumption mechanics.

Smoke rows:

| run | family | case | primary_ok | feasibility_ok | business_ok | time_limit_ok | final_n | moves | runtime_sec | frg_seed_cache_hit | frg_seed_cache_source | frg_seed_used | frg_seed_cap_ok |
|---|---|---|---|---|---|---|---:|---:|---:|---|---|---|---|
| jobs1 | BF14 | `cpmp_16_8_103_21_78_1` | False | True | False | False | 0 | 946 | 20.309 | True | artifact | True | False |
| jobs1 | BF32 | `cpmp_20_8_128_52_96_1` | False | True | False | False | 0 | 627 | 24.213 | True | artifact | True | False |
| jobs2 | BF14 | `cpmp_16_8_103_21_78_1` | False | True | False | False | 0 | 946 | 20.844 | True | artifact | True | False |
| jobs2 | BF32 | `cpmp_20_8_128_52_96_1` | False | True | False | False | 0 | 627 | 22.535 | True | artifact | True | False |

Artifacts:
- `docs\experiments\bf_seed_cache_smoke_20260626.jsonl`
- `docs\experiments\bf_seed_cache_smoke_jobs1_20260626.csv`
- `docs\experiments\bf_seed_cache_smoke_jobs1_20260626_summary.md`
- `docs\experiments\bf_seed_cache_smoke_jobs2_20260626.csv`
- `docs\experiments\bf_seed_cache_smoke_jobs2_20260626_summary.md`

Interpretation:
- Jobs1 and jobs2 both consumed materialized FRG seeds from the artifact path: `frg_seed_cache_hit=True`, `frg_seed_cache_source=artifact`.
- The two BF rows reproduced identical `N=0` move counts across jobs1 and jobs2, confirming the seed materialization/consumption mechanism for this smoke scope.
- These are feasibility-only rows under the strict paper scorecard. Both are `primary_ok=False` because they are over time and over the business cap.
- This artifact path is experimental and does not promote lazy portfolio or CV receiver-debt work.
