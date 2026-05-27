# Medium CPMP Business Classic Model Check

算例：`medium_cpmp_business_5x4_n12_seed3`

口径：

- 业务约束：`G = M(Now) + N(Future) <= floor(1.3 * N(0))`
- `N(0)=7`，约束上限 `9`
- 目标：在约束内完成良序，并比较经典 MIP/混合 MIP 模型表现

CSV：`docs/experiments/medium_cpmp_business_classic_model_check.csv`

## 结果

| solver | feasible | optimal | objective | runtime s | entropy max | final N | vars | constraints | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mip_4d` | true | true | 8 | 9.043 | 8 | 0 | 7,388 | 13,683 | `OPTIMAL` |
| `rolling_horizon_mip` | true | false | 9 | 3.515 | 9 | 0 | 26,295 | 44,667 | `FEASIBLE; mip_calls=9; window_statuses=OPTIMAL` |
| `final_assignment_mip` | true | false | 8 | 11.602 | 8 | 0 | 7,493 | 13,847 | `OPTIMAL; first_target=INFEASIBLE; repair=beam_target; realization_optimal=True; master=OPTIMAL` |
| `pattern_column_mip` | true | false | 8 | 11.504 | 8 | 0 | 11,358 | 13,712 | `OPTIMAL; first_target=INFEASIBLE; repair=beam_target; realization_optimal=True; master=OPTIMAL` |

## 结论

- 在 12 箱中型业务算例上，经典 4D slot-occupancy MIP 可以直接给出全局最优证明，最优搬移数为 `8`。
- `rolling_horizon_mip` 更快，但多 1 步；它适合更大算例的可行解搜索，不适合证明全局最优。
- `final_assignment_mip` 和 `pattern_column_mip` 在这个算例上达到与 4D MIP 相同的 8 步，但由于 master/realization 是分解流程，只能作为 hybrid feasible route，不声明全局最优。
- 这次运行进一步支持当前口径：小/中型算例用 `mip_4d` 做精确基准；大算例优先用 beam / QUBO beam / focused portfolio。
