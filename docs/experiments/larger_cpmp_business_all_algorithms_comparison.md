# 大算例算法对比（业务约束版）

实例：`larger_cpmp_6x5_n25_seed5`，`m=6, H=5, n=25`，占用率 `25/30=0.8333`。

业务口径：初始 `N(Future)=18`，`alpha=1.3`，因此硬约束为 `G=M+N(Future)<=23`。CPMP 终局 `N(Future)=0`，所以本轮 MIP 步数上限设为 `23`。

## 汇总

| 算法 | feasible | objective/moves | runtime(s) | expanded | generated | QUBO calls | QUBO time(s) | repair | MIP vars | MIP constrs | MIP nodes | best bound | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| random | False | inf | 0.001 | 0 | 0 | 0 | 0.000 | 0 |  |  |  |  | NO_FEASIBLE_FOUND |
| greedy_cpmp | False | inf | 0.000 | 0 | 0 | 0 | 0.000 | 0 |  |  |  |  | NO_FEASIBLE_FOUND |
| beam | False | inf | 0.534 | 1676 | 12179 | 0 | 0.000 | 0 |  |  |  |  | NO_FEASIBLE_FOUND |
| diverse_beam | False | inf | 30.291 | 589 | 4306 | 0 | 0.000 | 0 |  |  |  |  | NO_FEASIBLE_FOUND |
| rolling_horizon | False | inf | 0.341 | 1320 | 15777 | 0 | 0.000 | 0 |  |  |  |  | NO_FEASIBLE_FOUND |
| qubo_beam | False | inf | 2.454 | 774 | 5812 | 2 | 2.001 | 0 |  |  |  |  | NO_FEASIBLE_FOUND; simulated_annealing |
| qubo_beam_api | False | inf | 8.958 | 795 | 5894 | 2 | 5.369 | 2 |  |  |  |  | NO_FEASIBLE_FOUND; ising_matrix_api,ising_matrix_api_fallback_simulated_annealing |
| qubo_rolling_horizon | False | inf | 1.005 | 0 | 0 | 1 | 1.000 | 1 |  |  |  |  | NO_FEASIBLE_FOUND |
| stack_assignment_guided_cpmp | False | inf | 5.194 | 977 | 7334 | 1 | 5.000 | 2 |  |  |  |  | NO_FEASIBLE_FOUND |
| mip_3d | False | inf | 623.445 | 0 | 0 | 0 | 0.000 | 0 | 39659 | 61733 | 1 | 1940.000 | TIME_LIMIT |
| mip_4d | False | inf | 601.201 | 0 | 0 | 0 | 0.000 | 0 | 55115 | 99384 | 1 | 1062.000 | TIME_LIMIT |

## 结论

- 本轮业务约束版大算例中，所有非 MIP 算法都未找到满足 `G<=23` 的完整 CPMP 解。
- `beam` 和 `rolling_horizon` 很快耗尽业务可行前沿；`diverse_beam` 的相似度选择成本较高，30 秒内仍无解。
- `qubo_beam` 和 `qubo_beam_api` 本轮都实际调用了 QUBO 选择器；API 版本部分调用成功、部分回退到模拟退火，但仍未找到可行终局。
- `mip_3d` 和 `mip_4d` 在 10 分钟限时内均未找到 incumbent，说明加入 `G` 硬约束和累计 `N(Future)` 目标后，这个 25 箱高占用算例对精确模型仍很难。
- 旧口径下曾找到的 `32` 到 `33` 步方案在新业务口径下不合格；因为终局 `N(Future)=0`，仅搬移数一项就有 `G=32` 或 `33`，超过本实例上限 `23`。
- MIP 表中的 `best bound` 是内部目标 `(T+1)*sum_t N_t + move_count` 的界，不是搬移步数界。

## 配置说明

- 快速非 MIP：`max_depth=45, beam_width=180, max_candidates_per_layer=2500, candidate_limit_per_state=35`。
- `diverse_beam` 单独限成 `beam_width=60, max_candidates_per_layer=600, candidate_limit_per_state=20`，否则相似度二次选择会拖住整批。
- `qubo_beam`/`qubo_beam_api`：`beam_width=80, max_qubo_vars=180, qubo_call_interval=8, qubo_min_candidates_for_call=1`。
- `qubo_rolling_horizon`：`max_depth=23, rolling_depth=2, beam_width=40`。
- MIP：`max_steps=23, TimeLimit=600s, MIPGap=0.0, alpha=1.3`。

原始明细：`docs/experiments/larger_cpmp_business_all_algorithms_comparison.csv`。
