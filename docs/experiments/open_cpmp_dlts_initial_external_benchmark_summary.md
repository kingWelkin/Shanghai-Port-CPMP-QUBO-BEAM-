# Open CPMP 外部算例初测

## 数据来源

本轮接入 DLTS 开源 CPMP 数据集：https://github.com/ahottung/DLTS

DLTS 仓库包含 CPMP validation/test instances 以及对应 optimal solutions。当前 runner 不把外部数据直接提交进仓库，默认从 `external/DLTS` clone，或用 `--source` 指向本地 DLTS 目录。

## 接入方式

- 新脚本：`scripts/open_cpmp_benchmarks.py`
- 主算法入口仍然复用 `scripts/benchmark_compact_beam_qubo_n0.py` 的 `speed_v2` 版本。
- 开放算例不使用上港的 `alpha=1.3` 业务裁剪，本轮使用 `--alpha 10`。
- DLTS `*.dat` 第一行是 `stack_count container_count`，后续每行是 `height + bottom-to-top priorities`。
- 容量优先从 reference `.out` 的 `AA tiers-stacks-time` 读取；没有 reference 时用 `initial_height + 2`。
- 当前核心 `Instance` 要求箱号唯一，DLTS cv2/cv3 含重复优先级组，因此本轮只报告唯一优先级的 cv1 分支。重复优先级需要后续做 group-priority 支持后再严肃比较。

## 运行命令

```powershell
python scripts\open_cpmp_benchmarks.py --source $env:TEMP\dlts_cpmp_source --families 5x5_cv1_test 5x5_cv1_validation --limit-per-family 100 --variant both --time-limit 3 --alpha 10 --output docs\experiments\open_cpmp_dlts_5x5_cv1_200cases_both.csv
```

```powershell
python scripts\open_cpmp_benchmarks.py --source $env:TEMP\dlts_cpmp_source --families 7x5_cv1_test 7x5_cv1_validation --limit-per-family 20 --variant both --time-limit 3 --alpha 10 --output docs\experiments\open_cpmp_dlts_7x5_cv1_40cases_both.csv
```

## 结果

| scope | variant | cases | N=0 solved | avg runtime sec | avg ref gap among solved | qubo calls | qubo time sec |
|---|---|---:|---:|---:|---:|---:|---:|
| 5x5_cv1 | QUBO Beam | 200 | 101 | 1.212 | 2.149 | 3624 | 3.987 |
| 5x5_cv1 | classic no-QUBO | 200 | 84 | 0.766 | 3.000 | 0 | 0.000 |
| 7x5_cv1 | QUBO Beam | 40 | 13 | 2.431 | 4.231 | 565 | 0.634 |
| 7x5_cv1 | classic no-QUBO | 40 | 15 | 2.236 | 6.733 | 0 | 0.000 |

注意：moves total 不能跨不同 solved 集合直接比较，因为 solved 数量不同。本表保留 avg reference gap 作为质量指标。

## 初步判断

在 5x5_cv1 的 200 个外部实例上，QUBO Beam 相比 classic no-QUBO 多解出 17 个实例，并且已解实例的平均参考 gap 从 3.000 降到 2.149；代价是平均时间从 0.766 秒升到 1.212 秒。

在 7x5_cv1 的 40 个样本上，3 秒预算下 QUBO Beam 的成功数略低于 classic no-QUBO，但已解实例的平均参考 gap 更低。这个结果说明上港 30 例 tuned profile 可以迁移出一定 QUBO 质量优势，但还不是开放 CPMP 的强 profile。

10x5_cv1 的单例 smoke 在 5 秒内没有解出，说明大规模 10x5 需要专门的开放算例 profile，而不是直接复用当前 `speed_v2`。

## 后续优先级

1. 增加重复优先级 group-priority 表达，覆盖 DLTS cv2/cv3。
2. 为 7x5/10x5 单独调开放 benchmark profile，避免 3 秒预算下大量卡在中间 N。
3. 在开放数据上重新做 QUBO/no-QUBO 时间匹配，而不是只复用上港 30 例的 classic baseline。
