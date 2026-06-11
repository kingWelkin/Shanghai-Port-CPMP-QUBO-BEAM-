# v3 任务收口记录

本文记录本轮 v3 task 的最终状态。

| Task | 状态 | 结论 |
|---|---|---|
| 01 instrumentation | 已完成 | 增加 compact CSV 字段、summary、QUBO 路径统计、post-polish 事件追踪 |
| 02 QUBO influence profiles | 已完成 | simple 放大 QUBO 影响不稳定，不进主线 |
| 03 group QUBO | 已完成 | group supplement/majority 已实验；不提升主线，保留为反例证据 |
| 04 pairwise / contribution ablation | 已完成 | 拆清 frontier QUBO 与 post-polish QUBO；frontier direct path 很少，post 主要节省 1 步 |
| 05 rank-pool | 已完成 | `rank_pool=8000` 可作为工程加速候选，但弱化 QUBO 证据，不进当前主线 |
| 06 stream top-k / bounded heap | 已完成实验版 | Python heap 版不划算，不进主线 |
| 07 micro optimizations | 已完成 | 当前主线 30/30、509 步、平均 2.237 秒 |
| 08 experiment matrix | 已完成 | 当前 `N=0` 高效算法矩阵已整理 |
| 09 time-aligned ablation | 已完成 | 严格 classic no-QUBO 为 25/30，QUBO 主线为 30/30 |
| 文档收口 | 已完成 | README、docs/index、experiments README、当前算法说明已更新 |

## 当前主线

当前推荐主线：

```text
QUBO Beam speed_v2 微优化版
```

核心指标：

- 30/30 `N=0`
- 总步数 509
- 平均时间 2.237 秒
- 26 号贝位 15/15，总步数 414

## 当前 baseline 口径

必须区分两种 no-QUBO：

1. **classic no-QUBO**
   - 传统 Beam 消融。
   - 无 parent cap / deterministic core 多样性保护。
   - 严格时间匹配结果：25/30。

2. **enhanced no-QUBO**
   - 强工程消融。
   - 保留 compact state、剪枝、deterministic core、`parent_core_cap=4`。
   - 关闭 frontier QUBO 和 post QUBO。
   - 当前结果：30/30、510 步。

主文档中不能把 enhanced no-QUBO 当作传统 Beam baseline。

## QUBO 模块最终解释

当前最稳妥的解释：

- QUBO 不负责动作可行性。
- QUBO 不直接把 `N>0` 路径变成 `N=0`。
- Frontier QUBO 是小比例候选补充，稀疏影响搜索前沿。
- Post-polish QUBO 主要负责局部路径压缩，当前明确贡献是 1 步。
- 相对 classic Beam，QUBO Beam 框架有 30/30 对 25/30 的成功率优势。
- 相对 enhanced no-QUBO，QUBO 的可见独立贡献主要是 509 对 510 的步数差。
