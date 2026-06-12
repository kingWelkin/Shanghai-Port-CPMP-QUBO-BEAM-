# Open CPMP 重复优先级扩展基准

本轮把 DLTS `cv2/cv3` 的重复优先级算例接入当前 compact QUBO Beam runner。这里不能把重复优先级强行改成唯一箱号，否则会变成另一个问题。

## 语义

- `cv1`：仍按项目原有唯一箱号语义校验，箱号必须是 `1..N`。
- `cv2/cv3`：按 priority-group 语义校验，同一优先级的箱子不可区分。
- 合法动作仍然只允许移动栈顶箱，不能移出堆场，目标是每个栈自底向顶非递增，等价于 `N(Future)=0`。
- 成功行都通过独立模拟验解，`goal_validation_ok=True` 且 `business_ok=True`。

## 结果

5x5 cv2/cv3 各 20 例，三种变体同跑：

| scope | variant | cases | N=0 solved | moves total | avg runtime sec | avg ref gap |
|---|---|---:|---:|---:|---:|---:|
| 5x5_cv2_test | QUBO Beam | 20 | 14 | 368 | 1.205 | 2.000 |
| 5x5_cv2_test | adaptive QUBO | 20 | 14 | 368 | 1.161 | 2.000 |
| 5x5_cv2_test | classic no-QUBO | 20 | 13 | 357* | 0.720 | 3.077 |
| 5x5_cv3_test | QUBO Beam | 20 | 18 | 460 | 1.153 | 2.056 |
| 5x5_cv3_test | adaptive QUBO | 20 | 18 | 460 | 1.170 | 2.056 |
| 5x5_cv3_test | classic no-QUBO | 20 | 20 | 522 | 0.763 | 2.200 |

7x5 cv2/cv3 各 10 例，三种变体同跑：

| scope | variant | cases | N=0 solved | moves total | avg runtime sec | avg ref gap |
|---|---|---:|---:|---:|---:|---:|
| 7x5_cv2_test | QUBO Beam | 10 | 9 | 330 | 3.252 | 6.222 |
| 7x5_cv2_test | adaptive QUBO | 10 | 8 | 273* | 2.052 | 4.500 |
| 7x5_cv2_test | classic no-QUBO | 10 | 9 | 353 | 2.407 | 8.778 |
| 7x5_cv3_test | QUBO Beam | 10 | 10 | 356 | 3.214 | 5.500 |
| 7x5_cv3_test | adaptive QUBO | 10 | 10 | 359 | 2.783 | 5.800 |
| 7x5_cv3_test | classic no-QUBO | 10 | 10 | 366 | 2.380 | 6.500 |

`*` 总步数只统计成功算例，不能直接和更高成功率的行比较总量。

## 读法

重复优先级分支上，QUBO 的优势更像质量改进，不是稳定成功率优势。

- 5x5 cv2：QUBO 比 classic 多解 1 例，成功解平均 gap 更小。
- 5x5 cv3：classic 全解，但 QUBO 成功解总步数和平均 gap 更低。
- 7x5 cv2：QUBO 与 classic 同为 9/10，但 QUBO 少 23 步，平均 gap 从 8.778 降到 6.222。
- 7x5 cv3：三者都 10/10，QUBO 少 10 步，平均 gap 从 6.500 降到 5.500。
- adaptive QUBO 在 7x5 上通常更快，但可能牺牲成功率或少量质量，暂不替代 `speed_v2` 主线。

因此当前更稳妥的表述是：QUBO 在重复优先级开放算例中提供了可见的质量改善证据，但成功率优势依赖算例族，不能写成单向压制 classic Beam。

## 复现命令

```powershell
python scripts\open_cpmp_benchmarks.py --source $env:TEMP\dlts_cpmp_source --families 5x5_cv2_test 5x5_cv3_test --limit-per-family 20 --variant all --time-limit 3 --alpha 10 --output docs\experiments\open_cpmp_dlts_5x5_cv23_test20_priority_groups_all.csv
```

```powershell
python scripts\open_cpmp_benchmarks.py --source $env:TEMP\dlts_cpmp_source --families 7x5_cv2_test 7x5_cv3_test --limit-per-family 10 --variant all --time-limit 5 --alpha 10 --output docs\experiments\open_cpmp_dlts_7x5_cv23_test10_priority_groups_all.csv
```

明细文件：

- `open_cpmp_dlts_5x5_cv23_test20_priority_groups_all.csv`
- `open_cpmp_dlts_5x5_cv23_test20_priority_groups_all_summary.md`
- `open_cpmp_dlts_7x5_cv23_test10_priority_groups_all.csv`
- `open_cpmp_dlts_7x5_cv23_test10_priority_groups_all_summary.md`
