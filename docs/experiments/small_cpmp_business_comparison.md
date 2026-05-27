# 小算例业务约束对比

实例：`small_cpmp_business_4x3_n8_seed1`

- `m=4, H=3, n=8`
- 初始 stacks：`[[6, 1], [2, 5, 3], [], [4, 7, 8]]`
- 初始 `N(Future)=4`
- `alpha=1.3`
- 业务上限：`G=M+N(Future)<=floor(1.3*4)=5`

## 结果

| 算法 | feasible | optimal | moves | runtime(s) | max G | limit | total N | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| greedy_cpmp | False | False | inf | 0.000 |  |  |  | NO_FEASIBLE_FOUND |
| beam | True | False | 4 | 0.003 | 4 | 5 | 10 | FEASIBLE |
| diverse_beam | True | False | 4 | 0.002 | 4 | 5 | 10 | FEASIBLE |
| rolling_horizon | True | False | 4 | 0.002 | 4 | 5 | 10 | FEASIBLE |
| qubo_beam | True | False | 4 | 8.474 | 4 | 5 | 10 | FEASIBLE; simulated_annealing |
| qubo_beam_api | True | False | 4 | 6.859 | 4 | 5 | 10 | FEASIBLE; ising_matrix_api |
| qubo_rolling_horizon | False | False | inf | 4.944 |  |  |  | NO_FEASIBLE_FOUND |
| stack_assignment_guided_cpmp | True | False | 4 | 2.663 | 4 | 5 | 10 | FEASIBLE |
| mip_3d | True | True | 4 | 0.112 | 4 | 5 | 10 | OPTIMAL |
| mip_4d | True | True | 4 | 0.090 | 4 | 5 | 10 | OPTIMAL |

## 最优 move sequence

输入和输出栈下标均为 0-based，level 为 bottom -> top 的 1-based 层号。

| step | box | from | to | from level | to level |
|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 3 | 2 | 3 | 1 |
| 1 | 7 | 3 | 2 | 2 | 2 |
| 2 | 3 | 1 | 3 | 3 | 2 |
| 3 | 5 | 1 | 2 | 2 | 3 |

最终 stacks：`[[6, 1], [2], [8, 7, 5], [4, 3]]`。

## 结论

- 小算例说明业务约束不是逻辑上不可行；只要 `N0` 和空位结构允许，启发式和 MIP 都能找到满足 `G<=1.3N0` 的解。
- 这个例子中 3D 和 4D MIP 都证明内部目标最优，并且返回同一条 4 步解。
- `greedy_cpmp` 仍失败，说明当前 chen greedy 规则在新业务约束下过于局部，容易走到没有业务可行后继的状态。
- `qubo_rolling_horizon` 在这个配置下失败，是局部路径 QUBO/horizon 太短的问题，不代表实例不可行。

原始明细：`docs/experiments/small_cpmp_business_comparison.csv`。
