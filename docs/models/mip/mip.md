# 精确 MIP 求解

项目新增了 Python + Gurobi 的精确 pre-marshalling MIP 求解器，适合小规模实例验证业务约束下的最优路径质量。

当前推荐主模型是 `mip_4d` / `Solve4DSlotMip`。`mip_3d` / `Solve3DIntegerPositionMip` 已封存为 legacy comparison model，仅用于历史复现和模型对照，不再作为推荐 exact benchmark。

较大算例的实验型 MIP workflow 见 [hybrid_mip_workflows.md](hybrid_mip_workflows.md)，包括 rolling-horizon MIP、final assignment + realization MIP 和 pattern/column master + realization MIP。

## 入口

```python
from greedy import (
    PremarshallingInstance,
    Solve3DIntegerPositionMip,
    Solve4DSlotMip,
)

inst = PremarshallingInstance(
    StackCount=2,
    MaxHeight=2,
    ContainerCount=2,
    Stacks=[[1, 2], []],  # bottom -> top
)

sol = Solve4DSlotMip(inst, maxSteps=3)
print(sol.Objective, sol.Moves)
```

两个求解器：

- `Solve4DSlotMip(instance, maxSteps, timeLimitSec=None, diagnosticsDir=None)`：推荐模型。
- `Solve3DIntegerPositionMip(instance, maxSteps, timeLimitSec=None, diagnosticsDir=None)`：封存模型，仅保留对照用途。

## 语义

- 箱号必须是 `1..n`。
- 箱号越小，提取优先级越高。
- 输入堆栈按 bottom -> top 给出。
- 一步移动只能从一个堆顶取箱，放到另一个未满堆的顶部。
- 不允许 retrieval。
- 业务定义的 `N(Future)`：每个堆从第 2 层开始向上寻找第一个“上层箱号 > 下层箱号”的位置，并将该层及其上方所有箱计入逆序箱数。
- 业务熵约束：任意时刻 `G = M(Now) + N(Future) <= floor(alpha * N(0))`，默认 `alpha=1.3`。
- MIP 主目标是最小化全路径累计 `N(Future)`，移动次数只作为同等 `N(Future)` 下的 tie-break。
- 允许有 idle 后缀；模型加入了 no-late-idle 对称约束。
- 最终每个堆必须 bottom -> top 严格递减。

## 模型

`Solve3DIntegerPositionMip` 使用：

- container-stack-time assignment
- integer stack level
- 辅助槽位变量计算业务 `N(Future)`

该模型在当前业务目标下表现弱于 4D。原因是 `N(Future)` 依赖相邻层关系，3D 为了精确表达它必须引入辅助 slot 结构，同时仍保留 big-M 层位/位置约束，LP 松弛较弱。因此它已归档，不作为后续实验默认模型。

`Solve4DSlotMip` 使用：

- explicit slot occupancy `x[p,i,l,t]`
- explicit move variable `a[p,i,l,k,r,s]`

两个模型都会设置 `MIPGap = 0.0`。如提供 `timeLimitSec`，会设置 Gurobi `TimeLimit`。

当前实现中，MIP 内部目标为：

```text
min (K + 1) * sum_t N_t + sum_s active_s
```

其中 `K` 是最大步数。该写法保证先优化累计 `N(Future)`，再优化搬箱次数。

## 诊断

若模型 infeasible，会写出：

```text
mip_diagnostics/*.lp
mip_diagnostics/*.ilp
```

也可以指定目录：

```python
Solve4DSlotMip(inst, maxSteps=0, diagnosticsDir="tmp/mip_diagnostics")
```

## 结果校验

求解器会对提取出的移动序列做独立模拟校验：

- 源堆非空
- 搬走的是源堆顶箱
- 目标堆不同且未满
- from/to level 与模拟状态一致
- 最终所有堆严格递减
- 输出 `objective` 字段等于提取移动数；MIP 内部业务目标另由 `disorder_total` 等字段报告
- CPMP 解还会检查 `max_t G_t <= floor(alpha * N(0))`

报告中会额外输出 `entropy_max`、`entropy_limit`、`entropy_alpha`、`disorder_total`、`disorder_initial` 和 `disorder_final`。
