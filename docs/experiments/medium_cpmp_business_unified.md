# 中型算例业务约束统一对比

实例：`medium_cpmp_business_5x4_n12_seed3`

- `m=5, H=4, n=12`
- 初始 stacks：`[[], [1, 7, 10, 4], [8], [2, 12, 3, 9], [11, 5, 6]]`
- 初始 `N(Future)=7`
- `alpha=1.3`
- 业务上限：`G=M+N(Future)<=floor(1.3*7)=9`
- MIP/搜索步数上限：`max_depth=max_steps=9`

## 汇总

| 算法 | feasible | optimal | moves | runtime(s) | max G | total N | QUBO calls | QUBO vars | repair | MIP vars | MIP constrs | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| greedy_cpmp | False | False | inf | 0.000 |  |  | 0 | 0 | 0 |  |  | NO_FEASIBLE_FOUND |
| beam | True | False | 8 | 0.197 | 8 | 29 | 0 | 0 | 0 |  |  | FEASIBLE |
| diverse_beam | True | False | 8 | 117.258 | 8 | 29 | 0 | 0 | 0 |  |  | FEASIBLE |
| rolling_horizon | False | False | inf | 0.069 |  |  | 0 | 0 | 0 |  |  | NO_FEASIBLE_FOUND |
| qubo_beam | True | False | 8 | 17.020 | 8 | 29 | 4 | 730 | 0 |  |  | FEASIBLE; simulated_annealing |
| qubo_beam_api | True | False | 8 | 37.710 | 8 | 29 | 4 | 730 | 6 |  |  | FEASIBLE; ising_matrix_api_fallback_simulated_annealing |
| qubo_rolling_horizon | False | False | inf | 5.030 |  |  | 1 | 1164 | 1 |  |  | NO_FEASIBLE_FOUND |
| stack_assignment_guided_cpmp | True | False | 8 | 5.103 | 8 | 29 | 1 | 60 | 0 |  |  | FEASIBLE |
| mip_3d | True | True | 8 | 50.435 | 8 | 29 | 0 | 0 | 0 | 5698 | 9785 | OPTIMAL |
| mip_4d | True | True | 8 | 7.425 | 8 | 29 | 0 | 0 | 0 | 7388 | 13672 | OPTIMAL |

## 代表性最优路径

这是 `mip_3d` 和 `stack_assignment_guided_cpmp` 找到的 8 步路径。栈下标为 0-based，level 为 bottom -> top 的 1-based 层号。

| step | box | from | to | from level | to level |
|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 1 | 2 | 4 | 2 |
| 1 | 10 | 1 | 0 | 3 | 1 |
| 2 | 9 | 3 | 0 | 4 | 2 |
| 3 | 7 | 1 | 0 | 2 | 3 |
| 4 | 6 | 4 | 0 | 3 | 4 |
| 5 | 3 | 3 | 2 | 3 | 3 |
| 6 | 1 | 1 | 2 | 1 | 4 |
| 7 | 12 | 3 | 1 | 2 | 1 |

最终 stacks：`[[10, 9, 7, 6], [12], [8, 4, 3, 1], [2], [11, 5]]`。

`N(Future)` 轨迹：`[7, 6, 5, 4, 3, 2, 1, 1, 0]`。

## 结论

- 这个中型算例证明了业务约束下仍可存在非平凡可行解：8 步整理完成，且最大 `G=8`，低于上限 `9`。
- 3D 和 4D MIP 都证明 optimal，最优搬移数为 8；4D 在该实例上更快，约 7.4 秒。
- 3D MIP 已做变量精简：去掉独立 `pos[p,t]`、`height[i,t]`、`z[p,t]`，中间时刻用 `entropy_slot` 兼任防重位，只在最终时刻保留 `ord[p,q,T]`。
- MIP 已增加加速约束和 warm start：`sum active >= N0`，`N_{t+1} >= N_t - active_t`，以及 runner 自动用 beam 找一条业务合规路径作为 Gurobi MIP start。
- 中型算例 3D 从最初 `5698` vars / `9785` constrs / `50.4s` 降到增强后的 `4236` vars / `7976` constrs / `24.2s`，节点数从 `1440` 降到 `510`。4D 节点数从 `169` 降到 `37`。
- `beam` 很快找到同最优步数解；`qubo_beam` 和 `stack_assignment_guided_cpmp` 也达到同样质量，但 QUBO 调用带来额外时间。
- `qubo_beam_api` 本轮状态显示 API sampler 回退到了模拟退火，说明这次没有拿到稳定 API 直接解。
- `greedy_cpmp` 和两个 rolling horizon 版本在当前参数下失败，主要是局部决策太短，容易进入没有业务可行后继的路径。

## QUBO Beam 加速设置

原始统一配置中 `qubo_beam` 每次 QUBO 调用最多使用 5 秒，4 次调用累计 QUBO 时间约 16.4 秒。新增独立配置 `qubo_time_limit_sec` 和 `qubo_sa_iterations` 后，可以在不影响 MIP/整体实验限时的情况下压低单次 QUBO 采样时间。

快速配置：`max_qubo_vars=80, qubo_call_interval=4, qubo_time_limit_sec=0.2, qubo_sa_iterations=1000`。

| 配置 | feasible | moves | runtime(s) | QUBO calls | QUBO vars | QUBO time(s) |
|---|---:|---:|---:|---:|---:|---:|
| 原始 QUBO beam | True | 8 | 17.020 | 4 | 730 | 16.424 |
| 快速 QUBO beam | True | 8 | 0.698 | 2 | 90 | 0.400 |

快速配置原始明细：`docs/experiments/medium_cpmp_business_qubo_beam_fast.csv`。

原始明细：`docs/experiments/medium_cpmp_business_unified.csv`。

增强后 MIP 复测明细：`docs/experiments/medium_cpmp_business_mip_enhanced.csv`。

Hybrid MIP 三条实验路线的结果见 `docs/experiments/medium_cpmp_business_hybrid_mip.csv`：

| Hybrid solver | feasible | moves | runtime(s) | status |
|---|---:|---:|---:|---|
| `rolling_horizon_mip` | True | 9 | 3.335 | `FEASIBLE; mip_calls=9; window_statuses=OPTIMAL` |
| `final_assignment_mip` | True | 8 | 11.555 | `first_target=INFEASIBLE; repair=beam_target` |
| `pattern_column_mip` | True | 8 | 11.620 | `first_target=INFEASIBLE; repair=beam_target` |
