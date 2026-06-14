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

QUBO 算法簇 no-regret 择优：

```powershell
python scripts\benchmark_qubo_no_regret_portfolio_n0.py --portfolio-profiles stable,qubo_bucket --output docs\experiments\no_regret_stable_bucket_full30.csv
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

补充：`stable+qubo_bucket` no-regret portfolio 为 30/30、508 步，但总组合时间 115.057 秒。它是安全择优外壳，不是单 profile 加速结果。

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
- **Bucket QUBO**：在近似等价候选桶内选择补充状态，内部 30 可作为 no-regret 挑战者，但外部 7x5 hard case 不替代 rescue。
- **实例感知 scheduler**：新增 `scheduler_v1`，默认继承 `speed_v2`，在平台硬例触发轻量 rescue；adaptive 起步由重复优先级比例、`best_n` 比例和 plateau/slack 信号决定，不再依赖算例名。
- **相对 classic Beam**：同时间预算下，QUBO Beam 为 30/30，classic no-QUBO 为 25/30。
- **相对 enhanced no-QUBO**：enhanced no-QUBO 也能 30/30，说明 deterministic parent-diversity 机制很强；QUBO 当前可见贡献主要是 1 步压缩和少量路径祖先影响。
- **扩大 QUBO 影响的边界**：`window_polish_v2` 已实现为只接受更短路径的安全后处理，但内部 30 复跑为 30 次窗口尝试、0 次接受、总步数不变，因此只保留为实验 profile。

## 文档入口

- [文档导航](docs/index.md)
- [当前 QUBO Beam 主算法完整描述](docs/guides/current-qubo-beam-algorithm.md)
- [QUBO 机制归因记录](docs/guides/qubo-mechanism-attribution.md)
- [QUBO 影响增强 profile 试验记录](docs/guides/qubo-quality-profile-probes.md)
- [实例感知 scheduler / light rescue / top-k 试验记录](docs/guides/instance-aware-scheduler-probes.md)
- [外部 7x5 触发式 rescue QUBO](docs/guides/current-qubo-beam-algorithm.md#9-trigger-rescue-qubo-变体)
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

cv1 唯一优先级结果见 [Open CPMP 外部算例初测](docs/experiments/open_cpmp_dlts_initial_external_benchmark_summary.md)。当前 runner 也已支持 cv2/cv3 重复优先级分支，按 priority-group 语义验解；扩展结果见 [Open CPMP 重复优先级扩展基准](docs/experiments/open_cpmp_priority_group_benchmark_summary.md)。严格时间匹配 no-QUBO 消融见 [Open CPMP 时间匹配消融](docs/experiments/open_cpmp_time_matched_ablation_summary.md)。
