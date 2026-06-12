# 上港堆叠装箱

这是一个单贝位 container pre-marshalling / CPMP 实验项目。核心约束是：

- 每一步只能移动某个非空栈的栈顶箱。
- 只能把箱子放到另一个未满栈的栈顶。
- 不能移动非栈顶箱。
- 不能把箱子移出堆场。
- 最终验收目标是 `N=0`，即所有箱子都不再处于逆序影响范围内。

当前主线是 **QUBO 增强 Beam**。QUBO 不直接输出最终动作，也不负责可行性证明；所有动作都由 classical state engine 生成，并由 validator 重新模拟验收。

## 快速运行

推荐先安装为 editable 包：

```powershell
python -m pip install -e .
python -m greedy --zone A1
```

也可以直接运行脚本：

```powershell
.\run.bat --zone A1
.\run.bat
```

当前主推 `N=0` benchmark：

```powershell
python scripts\benchmark_compact_beam_qubo_n0.py --profile speed_v2 --output docs\experiments\micro_optimized_speed_v2_full30.csv
```

严格 QUBO / no-QUBO 时间匹配消融：

```powershell
python scripts\benchmark_compact_qubo_ablation_time_aligned.py --solver-profile speed_v2 --sweep-all-no-qubo --output docs\experiments\micro_optimized_strict_ablation_full30.csv --no-qubo-widths 1300 1400 1600 2000 2400 3200 4800 6400
```

## 当前结论

当前只把 30 个单贝位 `N=0` 分支作为主结论。

| 算法 | 定位 | 解出 | 总步数 | 平均时间 |
|---|---|---:|---:|---:|
| QUBO Beam `speed_v2` 微优化版 | 当前主推 | 30/30 | 509 | 2.237 秒 |
| enhanced no-QUBO | 强工程消融 | 30/30 | 510 | 2.258 秒 |
| classic no-QUBO time-matched | 传统 Beam 严格消融 | 25/30 | 364* | 2.409 秒 |
| classic no-QUBO budget-best | 传统 Beam 严格消融 | 25/30 | 364* | 2.409 秒 |
| dynamic QUBO quality | 质量增强变体 | 30/30 | 507 | 4.902 秒 |
| A* no-proof | 速度强基线，不证明最优 | 30/30 | 514 | 2.829 秒 |

`*` classic no-QUBO 的步数只统计成功算例，不能直接与 30/30 算法的总步数对比。

26 号贝位：

| 算法 | 解出 | 总步数 | 平均时间 |
|---|---:|---:|---:|
| QUBO Beam `speed_v2` 微优化版 | 15/15 | 414 | 4.031 秒 |
| enhanced no-QUBO | 15/15 | 415 | 4.004 秒 |
| classic no-QUBO time-matched | 10/15 | 269* | 4.396 秒 |

## QUBO 的实际作用

当前版本中，QUBO 的作用要分开看：

- **Frontier QUBO**：在 enhanced Beam 的 deterministic core 之外，小比例补充候选状态。它不是逐步主导最终路径，而是稀疏改变搜索前沿结构。
- **Post-polish QUBO**：主搜索已经找到 `N=0` 后，选择 suffix/window 做局部路径压缩。它不能把 `N>0` 失败路径变成 `N=0`。
- **相对 classic Beam**：同时间预算下，QUBO Beam 为 30/30，classic no-QUBO 为 25/30。
- **相对 enhanced no-QUBO**：enhanced no-QUBO 也能 30/30，说明 deterministic parent-diversity 机制很强；QUBO 当前可见贡献主要是 1 步压缩和少量路径祖先影响。

## 文档入口

- [文档导航](docs/index.md)
- [当前 QUBO Beam 主算法完整描述](docs/guides/current-qubo-beam-algorithm.md)
- [实验结论汇总](docs/experiments/README.md)
- [Task 04：QUBO 贡献拆解](docs/experiments/frontier_qubo_contribution_task04_summary.md)
- [Task 08：`N=0` 高效算法矩阵](docs/experiments/n0_efficient_algorithm_matrix_task08_summary.md)
- [Task 09：严格时间匹配消融](docs/experiments/time_aligned_ablation_task09_summary.md)
- [v3 任务收口记录](docs/experiments/v3_task_closure_summary.md)
- [数据格式](docs/guides/data-format.md)
- [运行说明](docs/guides/usage.md)
- [开发说明](docs/guides/development.md)

## 项目结构

```text
.
├── config/          # 运行默认参数记录
├── data/
│   ├── input/       # 原始输入数据
│   └── results/     # 历史/生成结果
├── docs/            # 文档入口、模型说明、使用指南和实验结论
├── notebooks/       # 探索分析预留目录
├── scripts/         # benchmark、消融和分析脚本
├── src/greedy/      # Python 源码包
├── tests/           # pytest 回归测试
├── pyproject.toml
└── run.bat
```

## 验证

```powershell
python -m pytest -q
```

如果 `gurobipy` 不可用，MIP 相关用例会按测试配置跳过或返回不可用状态。当前 QUBO Beam 主线不依赖 Gurobi。

## 开源 CPMP 外部算例

已经新增 DLTS 开源 CPMP 数据集 runner：

```powershell
python scripts\open_cpmp_benchmarks.py --families 5x5_cv1_test 5x5_cv1_validation --limit-per-family 100 --variant both --time-limit 3 --alpha 10 --output docs\experiments\open_cpmp_dlts_5x5_cv1_200cases_both.csv
```

cv1 唯一优先级结果见 [Open CPMP 外部算例初测](docs/experiments/open_cpmp_dlts_initial_external_benchmark_summary.md)。当前 runner 也已支持 cv2/cv3 重复优先级分支，按 priority-group 语义验解；扩展结果见 [Open CPMP 重复优先级扩展基准](docs/experiments/open_cpmp_priority_group_benchmark_summary.md)。
