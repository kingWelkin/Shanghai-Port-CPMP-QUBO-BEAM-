# 完整真实配对算例当前推荐口径

本文记录 `data/input` 中 15 个 `01+26` 配对真实算例在业务约束
`G = M(Now) + N(Future) <= floor(1.3 * N(0))` 下的当前推荐口径。

这些结果不是全场严格良序 CPMP 的终局最优解，而是在 1.3 倍业务预算内尽量降低
`N(Future)` 的 best-effort 轨迹。因此本组报告应按业务改善效果比较，不应按
MIP optimality gap 解读。

## 推荐接入

| 路线 | 平均最终 N | 平均搬移数 | 平均运行时间 | 口径 |
|---|---:|---:|---:|---|
| `beam_focused_with_qubo_portfolio` | 8.53 | 26.80 | 37.313s | 当前最高质量综合 route |
| `beam_focused_with_tuned_qubo` | 8.80 | 26.47 | 21.491s | 原算法 + dense QUBO 增强 |
| `beam_focused_with_dense_qubo` | 8.93 | 26.33 | 18.559s | 原算法替换为 dense QUBO 分支 |
| `qbo_beam_standalone_portfolio` | 9.00 | 26.87 | 26.027s | 当前最高质量 QUBO-only route |
| `beam_focused_with_qubo` | 9.33 | 25.53 | 14.217s | 质量优先推荐 |
| `beam_focused_plain` | 9.47 | 25.47 | 11.286s | 不使用 QUBO/API 的质量路线 |
| `feature_adaptive_beam` | 9.93 | 24.80 | 5.568s | 速度/质量折中路线 |
| `qbo_beam_dense` | 10.07 | 25.20 | 7.273s | 单配置 QUBO 增强 |
| `beam` | 10.67 | 24.13 | 2.460s | 稳定 baseline |
| `qubo_beam` | 10.60 | 24.27 | 2.932s | QUBO selection baseline |
| `chen_greedy_algorithm_1p3` | 17.13 | 20.20 | 0.115s | 生产快速 baseline |

当前推荐：

- 需要最好 `N(Future)` 时，使用 `beam_focused_with_qubo_portfolio`。
- 需要 20 秒级质量路线时，使用 `beam_focused_with_dense_qubo` 或 `beam_focused_with_tuned_qubo`。
- 需要 15 秒级稳定路线时，使用原 `beam_focused_with_qubo`。
- 需要更短运行时间且结果明显优于贪心时，使用 `feature_adaptive_beam`。
- `chen_greedy_algorithm_1p3` 保留为最快、最稳的业务 baseline。

## 增强 QUBO 合入原 Focused 算法

已把 standalone QUBO 增强分支合入原来的 focused portfolio 口径：

| 路线 | 组件 | 平均最终 N | 平均搬移数 | 平均运行时间 | 结论 |
|---|---|---:|---:|---:|---|
| `beam_focused_with_qubo_portfolio` | `beam` + `enhanced_beam_fast` + QUBO-only mini-portfolio | 8.53 | 26.80 | 37.313s | 当前最好 |
| `beam_focused_with_tuned_qubo` | `beam` + `enhanced_beam_fast` + `qubo_beam` + `qbo_beam_dense` | 8.80 | 26.47 | 21.491s | 性价比较好 |
| `beam_focused_with_dense_qubo` | `beam` + `enhanced_beam_fast` + `qbo_beam_dense` | 8.93 | 26.33 | 18.559s | 比原 QUBO route 更强 |
| `beam_focused_with_qubo` | `beam` + `enhanced_beam_fast` + `qubo_beam` | 9.33 | 25.53 | 14.217s | 原推荐口径 |

结论：

- 增强后的 QUBO 分支合入原 focused 算法后有效；平均最终 `N(Future)` 从 `9.33` 降到 `8.53`。
- 如果不想跑完整 QUBO-only mini-portfolio，`beam_focused_with_dense_qubo` 仍能把平均最终 `N(Future)` 降到 `8.93`。
- 运行时间与 QUBO 调用数同步上升，因此建议按场景分层使用：生产快速用 `beam` / `feature_adaptive_beam`，质量优先用 tuned QUBO focused route。

## QUBO Standalone 增强试验

已补充一轮只增强 QUBO 分支自身的试验，不使用 plain/enhanced beam 作为 portfolio 组件：

| 路线 | 平均最终 N | 平均搬移数 | 平均运行时间 | 结论 |
|---|---:|---:|---:|---|
| `qbo_beam_standalone_portfolio` | 9.00 | 26.87 | 26.027s | 当前最好，但运行更久 |
| `qbo_beam_dense` | 10.07 | 25.20 | 7.273s | 单配置有效增强 |
| `qbo_beam_base_rerun` | 10.60 | 24.27 | 4.205s | 当前 baseline 复跑 |
| `qbo_beam_low_div` | 10.67 | 24.73 | 5.628s | 无明显收益 |
| `qbo_beam_wide` | 11.87 | 21.67 | 8.921s | 不推荐 |

关键结论：

- QUBO 单分支可以增强；`qbo_beam_dense` 通过更频繁的小 QUBO 选择，把单配置平均最终 `N(Future)` 从 `10.60` 降到 `10.07`。
- QUBO-only mini-portfolio 进一步把平均最终 `N(Future)` 降到 `9.00`，超过 `beam_focused_with_qubo` 的 `9.33`。
- 这个提升来自 QUBO 参数/搜索节奏的互补性，不依赖 plain beam 或 enhanced beam 结果。
- 代价是运行时间和 QUBO 调用数显著增加；平均约 `26s`、`46.47` 次 QUBO 调用。

## Best Beam 后接 QUBO 的试验

已补充一轮“enhanced beam 多规则候选生成 + QUBO 层选择”的直接接入试验：

| 路线 | 平均最终 N | 平均搬移数 | 平均运行时间 | 结论 |
|---|---:|---:|---:|---|
| `enhanced_qubo_beam_elite` | 12.33 | 20.13 | 9.838s | 不推荐 |
| `enhanced_qubo_beam_ranked` | 13.33 | 17.33 | 10.301s | 不推荐 |

直接把 QUBO 用作 enhanced beam 的层选择器会过早偏向“多样但短路径”的状态，导致
`N(Future)` 降幅不足。当前更稳的方式是保留 `beam` / `enhanced_beam_fast` 的经典 beam
分支，再把 `qubo_beam` 作为 focused portfolio 的独立候选分支，即 `beam_focused_with_qubo`。

## LNS / Weighted-A* 对照试验

已补充更强但仍快于完整 MIP 的 benchmark，包括 weighted A*、beam suffix repair LNS、
QUBO suffix repair LNS。所有路线都只从合法状态扩展，且全程满足 `G <= floor(1.3 * N(0))`：

| 路线 | 平均最终 N | 平均搬移数 | 平均运行时间 | 结论 |
|---|---:|---:|---:|---|
| `weighted_astar_business_portfolio` | 10.13 | 25.80 | 30.053s | 当前最强 A* 对照，但仍弱于 focused beam |
| `weighted_astar_business_previous` | 11.60 | 23.93 | 10.004s | 旧单队列 A* |
| `lns_beam_repair` | 13.87 | 16.80 | 24.015s | 搬移更少，但 `N(Future)` 降幅不足 |
| `lns_qubo_repair` | 14.13 | 16.20 | 30.283s | QUBO-LNS 未超过 beam-LNS |

关键结论：

- 这三类方法可作为“比 MIP 快、比简单启发式严谨”的对照 benchmark，但当前不适合作为主算法。
- 增强后的 weighted A* 使用 legacy、aggressive multi-queue、hybrid multi-queue 三个分支做 portfolio，平均最终 `N(Future)` 从 `11.60` 降到 `10.13`。
- 这个结果已经明显强于贪心和多数 metaheuristics，但仍弱于 `beam_focused_plain` 的 `9.47`，更弱于 `beam_focused_with_qubo_portfolio` 的 `8.53`。
- LNS 从已有解回滚 suffix 后重规划，倾向保守缩短搬移数，反而不如 focused beam/QUBO 在业务预算内充分降低 `N(Future)`。
- Weighted A* portfolio 已缓解单路径 best-first 的问题，但多分支只在 open-list 层面互补，仍不如 beam/QUBO 对整层候选集的系统保留和选择。

## 不作为推荐接入

以下路线已保留实验记录，但当前不进入推荐主口径：

- `fused_beam`：平均最终 N 为 11.73，低于 focused beam 组合。
- `beam_post_repair`：平均最终 N 为 10.73，只在少数单例改善，整体不稳定。
- `enhanced_qubo_beam_elite` / `enhanced_qubo_beam_ranked`：验证了 best beam 后直接接 QUBO 的方案，但整体弱于 focused portfolio。
- `weighted_astar_business_portfolio`：平均最终 N 为 10.13，是当前最强 A* benchmark，但整体弱于 focused beam/QUBO。
- `lns_beam_repair` / `lns_qubo_repair`：局部 suffix repair 没有超过当前 focused portfolio。
- `adaptive_beam_top1/top2`：短探针选择会错过部分 enhanced/QUBO 有优势的案例。
- `enhanced_beam_wide`、`hybrid_enhanced_beam`：运行时间增加后平均质量没有稳定收益。

## 复现实验

```powershell
python scripts/benchmark_full_real_beam_improvements.py
python scripts/build_full_real_focused_beam_report.py
python scripts/benchmark_full_real_adaptive_beam.py
python scripts/benchmark_full_real_enhanced_qubo_beam.py
python scripts/benchmark_full_real_qubo_standalone_tuning.py
python scripts/benchmark_full_real_lns_astar.py
python scripts/benchmark_full_real_weighted_astar_tuning.py
```

核心报告：

- [focused beam 对照](full_real_business_focused_beam_comparison.md)
- [beam 改进路线对照](full_real_business_beam_improvements_comparison.md)
- [enhanced beam 后接 QUBO 对照](full_real_business_enhanced_qubo_beam_comparison.md)
- [standalone QUBO tuning 对照](full_real_business_qubo_standalone_tuning_comparison.md)
- [weighted-A* tuning 对照](full_real_business_weighted_astar_tuning.md)
- [LNS / weighted-A* 对照](full_real_business_lns_astar_comparison.md)
- [adaptive beam 对照](full_real_business_adaptive_beam_comparison.md)
- [metaheuristics 对照](full_real_business_metaheuristics_comparison.md)
- [genetic + tabu 对照](full_real_business_genetic_tabu_comparison.md)
