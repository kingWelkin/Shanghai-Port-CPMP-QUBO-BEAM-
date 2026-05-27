# 完整真实配对算例当前推荐口径

本文记录 `data/input` 中 15 个 `01+26` 配对真实算例在业务约束
`G = M(Now) + N(Future) <= floor(1.3 * N(0))` 下的当前推荐口径。

这些结果不是全场严格良序 CPMP 的终局最优解，而是在 1.3 倍业务预算内尽量降低
`N(Future)` 的 best-effort 轨迹。因此本组报告应按业务改善效果比较，不应按
MIP optimality gap 解读。

## 推荐接入

| 路线 | 平均最终 N | 平均搬移数 | 平均运行时间 | 口径 |
|---|---:|---:|---:|---|
| `beam_focused_with_qubo` | 9.33 | 25.53 | 14.217s | 质量优先推荐 |
| `beam_focused_plain` | 9.47 | 25.47 | 11.286s | 不使用 QUBO/API 的质量路线 |
| `feature_adaptive_beam` | 9.93 | 24.80 | 5.568s | 速度/质量折中路线 |
| `beam` | 10.67 | 24.13 | 2.460s | 稳定 baseline |
| `qubo_beam` | 10.60 | 24.27 | 2.932s | QUBO selection baseline |
| `chen_greedy_algorithm_1p3` | 17.13 | 20.20 | 0.115s | 生产快速 baseline |

当前推荐：

- 需要最好 `N(Future)` 时，使用 `beam_focused_with_qubo`。
- 需要更短运行时间且结果明显优于贪心时，使用 `feature_adaptive_beam`。
- `chen_greedy_algorithm_1p3` 保留为最快、最稳的业务 baseline。

## 不作为推荐接入

以下路线已保留实验记录，但当前不进入推荐主口径：

- `fused_beam`：平均最终 N 为 11.73，低于 focused beam 组合。
- `beam_post_repair`：平均最终 N 为 10.73，只在少数单例改善，整体不稳定。
- `adaptive_beam_top1/top2`：短探针选择会错过部分 enhanced/QUBO 有优势的案例。
- `enhanced_beam_wide`、`hybrid_enhanced_beam`：运行时间增加后平均质量没有稳定收益。

## 复现实验

```powershell
python scripts/benchmark_full_real_beam_improvements.py
python scripts/build_full_real_focused_beam_report.py
python scripts/benchmark_full_real_adaptive_beam.py
```

核心报告：

- [focused beam 对照](full_real_business_focused_beam_comparison.md)
- [beam 改进路线对照](full_real_business_beam_improvements_comparison.md)
- [adaptive beam 对照](full_real_business_adaptive_beam_comparison.md)
- [metaheuristics 对照](full_real_business_metaheuristics_comparison.md)
- [genetic + tabu 对照](full_real_business_genetic_tabu_comparison.md)
