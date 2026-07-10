# QUBO 机制归因记录

> [!IMPORTANT]
> Historical document. It does not define the current default. The promoted mainline is [v4.3](../../guides/v43-mainline-algorithm.md), validated at `1681/1681` with `105530` moves.

本文只回答一个问题：当前代码里哪些 QUBO 介入真正产生了可见收益，哪些只是扩大了 QUBO 参与度但没有改善结果。

结论先写在前面：`speed_v2` 仍是默认主线。QUBO 当前最稳的证据不是“越大越好”，而是两个较窄作用：

1. 在 enhanced Beam 已经很强的前提下，frontier QUBO 作为稀疏补充，帮助主算法相对传统 classic Beam 保持更好的归零稳定性。
2. post-polish suffix QUBO 在已找到 `N=0` 后做安全路径压缩，内部 30 算例可见收益为 1 步。

## 机制总表

| 机制 | QUBO 介入位置 | 观察结果 | 归因结论 |
|---|---|---|---|
| frontier QUBO supplement | 主搜索每层 deterministic core 之后，少量补充 frontier | `speed_v2` 30/30；strict classic no-QUBO time-matched 25/30 | 相对传统 Beam 有稳定性贡献，但不是全程主导最终路径 |
| enhanced no-QUBO | 保留 parent diversity 等工程机制，禁用 QUBO | 30/30、510 步 | 说明 30/30 不是纯 QUBO 贡献，deterministic diversity 很强 |
| suffix post-polish QUBO | 找到 `N=0` 后选择尾部重搜起点 | `speed_v2` 总步数 509，其中 post-polish saving 为 1 步 | 当前最清楚的 QUBO 质量收益 |
| `window_polish_v2` | 找到 `N=0` 后选择局部窗口 bridge 替换 | 30/30、509 步；窗口 30 次尝试、0 次接受；额外约 8.09 秒 | 安全但没有收益，不进入主线 |
| `frontier_quality_v1` | 加大 frontier QUBO 变量、调用和一步未来 `N` 偏置 | 30/30、510 步，QUBO vars 明显增加 | 简单放大 QUBO 没有证明质量更好 |
| `qubo_bucket` | 在近似等价候选桶内让 QUBO 选补充状态 | 内部 portfolio 可从 509 择优到 508，但总组合时间 115.057 秒 | 适合做 QUBO 算法簇挑战者，不是单 profile 加速主线 |
| `rescue_qubo_v1` | 硬例平台层触发早 QUBO 和局部加宽 | 外部 `7x5_cv2_test` 前 10 从 9/10 到 10/10 | 对开放硬例的解出率有价值，但会增加时间和 gap |
| protected group QUBO | 分组保护式补充选择 | `A2_26` smoke 同步数但更慢 | 诊断价值大于当前质量收益 |

## Window Polish v2 结果

Window polish v2 的设计目标是把 QUBO 的作用扩大到“已找到可行解后的路径压缩”，且只接受更短替换，因此不会劣化步数。

实现入口：

```powershell
python scripts\benchmark_compact_beam_qubo_n0.py --solver-profile window_polish_v2 --output docs\experiments\scratch_window_polish_v2_full30.csv
```

与同代码复跑的 `speed_v2` 对照：

| profile | 解出 | 总步数 | 总时间 sec | 平均时间 sec | QUBO calls | QUBO vars | post saved | window attempts | window successes | window saved | window time sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `speed_v2` current rerun | 30/30 | 509 | 63.758 | 2.125 | 366 | 22555 | 1 | 0 | 0 | 0 | 0.000 |
| `window_polish_v2` | 30/30 | 509 | 71.844 | 2.395 | 396 | 23819 | 1 | 30 | 0 | 0 | 5.675 |

解释：

- v2 没有改变解出率，也没有减少总步数。
- 30 个窗口桥接尝试全部没有找到可接受的严格更短替换。
- 额外时间主要来自 bridge 验证，不是 QUBO 采样本身。
- 因为它只接受更短路径，所以质量不劣化；但当前证据不足以把它写成有效增强。

因此 `window_polish_v2` 保留为可复现实验 profile，而不是默认主线。

## 可写成正贡献的部分

当前最稳妥的论文式写法是：

- QUBO Beam 相对传统 classic Beam 的优势体现在同时间预算下的归零稳定性：30/30 对 25/30。
- 相对 enhanced no-QUBO，QUBO 的可见边际收益较小：当前内部 30 的明确收益是 1 步路径压缩。
- QUBO 不直接生成动作，不绕过栈顶移动约束，也不把箱子移出堆场；所有路径都由 state engine 生成并由 validator 重放。
- 扩大 QUBO 影响的负结果同样重要：简单增加变量、调用、窗口压缩和分组保护，并没有稳定提升主线。

## 不建议继续投入的方向

- 全局放大 frontier QUBO：已有 `frontier_quality_v1` 显示更慢且步数不优。
- 重型 window bridge：成本集中在 bridge 搜索，30 算例没有接受替换。
- 复杂 pairwise similarity：初步 smoke 没有显示直接收益。

## 更值得保留的方向

- `speed_v2`：默认主线。
- `rescue_qubo_v1`：外部硬例平台救援。
- `qubo_bucket` + no-regret portfolio：报告 QUBO 算法簇择优时可用，但必须报告组合时间。
- suffix post-polish：小而稳，当前唯一清楚减少步数的 QUBO 后处理机制。
