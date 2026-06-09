# 上港堆叠装箱

这是一个堆场搬箱路径规划项目。核心约束是只能搬移栈顶箱，且不能把箱子移出堆场；
当前实验主线是用 QUBO 增强 Beam 在单贝 CPMP 算例上达到完全良序 `N=0`。
仓库同时保留早期 `01+26` 配对业务改善流程、A* 对照、classic Beam 消融和
小规模 Python + Gurobi 精确 MIP 求解器。

## 项目结构

```text
.
├── config/          # 运行默认参数记录
├── data/
│   ├── input/       # 原始输入数据
│   └── results/     # 历史/生成结果
├── docs/            # 文档入口、模型说明、使用指南和试验结论
├── notebooks/       # 探索分析预留目录
├── src/greedy/      # Python 源码包
├── pyproject.toml   # 包元数据和依赖
└── run.bat          # Windows 快捷入口
```

## 快速运行

推荐先安装为 editable 包：

```powershell
python -m pip install -e .
python -m greedy --zone A1
```

也可以不安装，直接用批处理脚本：

```powershell
.\run.bat --zone A1
.\run.bat
```

默认读取 `data/input`，输出写入 `data/results`。

## 常用命令

```powershell
# 跑全部 01+26 配对箱区
python -m greedy

# 只跑 A1
python -m greedy --zone A1

# 指定贝位组合
python -m greedy --zone B2 --bays 01,26

# 单文件旧模式，不合并贝位
python -m greedy --per-file --glob "A1_*_input.txt"
```

## 文档

- [文档导航](docs/index.md)
- [数据格式](docs/guides/data-format.md)
- [运行说明](docs/guides/usage.md)
- [chen_greedy_algorithm 说明](docs/models/greedy/chen_greedy_algorithm.md)
- [精确 MIP 求解](docs/models/mip/mip.md)
- [3D/4D MIP LaTeX 模型](docs/models/mip/上港翻箱堆叠问题模型.tex)
- [试验结论汇总](docs/experiments/README.md)
- [QUBO Beam 与 A* 速度对齐结论](docs/experiments/qubo_astar_speed_alignment.md)
- [稳定版 QUBO Beam speed 结果](docs/experiments/compact_beam_qubo_speed_optimized_v2_full30_repeat.md)
- [完整真实配对算例当前推荐口径](docs/experiments/full_real_business_current_recommendation.md)
- [贪心 vs MIP benchmark](docs/experiments/greedy_vs_mip.md)
- [更大真实局部 benchmark](docs/experiments/greedy_vs_mip_larger_real.md)
- [结果解读](docs/experiments/results.md)
- [开发说明](docs/guides/development.md)

## 当前状态

当前主推结果是稳定版非 dynamic QUBO Beam speed，用于 30 个单贝 CPMP 算例：

- `N=0` 解出率：30/30；
- 总步数：509；
- 平均运行时间：2.274 秒；
- 最大运行时间：4.752 秒；
- 26 号位：15/15 解出，总步数 414，平均运行时间 4.107 秒。

对照结果为：A* no-proof 同样 30/30，但总步数 514、平均时间 2.829 秒；
classic Beam 严格消融在时间匹配口径下为 21/30，budget-best 口径为 28/30。
因此当前推荐写法应以 QUBO Beam 为主算法：在保持完全良序和动作合法性的同时，
比 A* no-proof 少 5 步且平均更快。

dynamic QUBO 作为质量增强变体保留：`balanced2200` 为 508 步、4.137 秒；
质量档为 507 步、4.902 秒。它们适合强调步数进一步下降，但不作为速度主线。

早期 15 个 `01+26` 配对结果属于业务改善口径：在
`G <= floor(1.3 * N(0))` 内尽量降低最终 `N(Future)`，不是全场严格良序
CPMP 终局最优解。相关结果仍保留在历史试验文档中，不再作为 README 主结果。

精确 MIP 求解器位于 `greedy.premarshalling_mip`，适合小规模 pre-marshalling
实例验证最优搬箱步数。

## container_reshuffle 实验框架

新增 `container_reshuffle` Python 包，用于统一实验 container reshuffling / retrieval CRP / pre-marshalling CPMP。

### 问题定义

- `retrieval_crp`：按 `1,2,...,n` 顺序取箱。目标箱在栈顶时免费取出；非目标箱只能从一个栈顶搬到另一个未满栈顶，目标是最小化搬移次数。
- `premarshalling_cpmp`：不允许取箱。每步只能搬一个栈顶箱到另一个未满栈顶；最终每个栈必须 bottom -> top 严格递减，例如 `[9,6,3,1]`。

所有 stack 均用 bottom -> top 的 list 表示。`container_reshuffle` 内部 `Move.from_stack` / `Move.to_stack` 使用 Python 风格 0-based 栈索引，层号为 bottom -> top 的 1-based level。

### 输入格式

```json
{
  "instance_id": "tiny_cpmp_1",
  "variant": "premarshalling_cpmp",
  "stack_count": 3,
  "max_height": 2,
  "container_count": 3,
  "stacks": [[1, 3], [2], []]
}
```

### 已实现算法

- `random`
- `greedy_retrieval`
- `greedy_cpmp`
- `beam`
- `diverse_beam`
- `rolling_horizon`
- `qubo_beam`
- `qubo_rolling_horizon`
- `qubo_batch_relocation`
- `stack_assignment_guided_cpmp`
- `mip_4d`
- `mip_3d`（archived，仅用于历史对照）
- `rolling_horizon_mip`
- `final_assignment_mip`
- `pattern_column_mip`

QUBO/Ising 只用于候选 beam 选择、rolling path 选择、CRP blocker destination batch assignment、CPMP target stack assignment guidance。QUBO 样本不会直接被声明为可行解，必须经过 decode、repair，并由 classical state engine 重新模拟验证。

MIP 只用于 tiny/small CPMP benchmark。当前推荐使用 `mip_4d` 作为业务约束下的 exact benchmark；`mip_3d` 已封存为历史对照模型，因为业务 `N(Future)` 依赖相邻槽位关系，3D 需要额外辅助 slot 变量且 relaxation 较弱。只有 Gurobi 返回 `OPTIMAL` 且验证通过时才标记 optimal。

新增的 hybrid MIP 路线包括 rolling-horizon window MIP、final stack assignment + realization MIP、pattern/column master + realization MIP。它们主要用于探索较大算例，不默认声明全局最优。

### 运行

```powershell
python -m pytest tests
python examples/run_experiment.py --config examples/example_config.json
```

示例输出：

```text
examples/results/results.csv
examples/results/results.json
```

如果 `gurobipy` 不可用，MIP solver 会返回 `GUROBI_NOT_AVAILABLE`；测试中对应 MIP 用例会 skip。

完整真实配对算例的推荐对照可复现：

```powershell
python scripts/benchmark_full_real_beam_improvements.py
python scripts/build_full_real_focused_beam_report.py
python scripts/benchmark_full_real_adaptive_beam.py
```
