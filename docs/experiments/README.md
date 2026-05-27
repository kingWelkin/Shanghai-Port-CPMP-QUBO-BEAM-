# 试验结论

本目录统一存放当前项目的试验报告、历史结果解读和 CSV 明细。

## 文件说明

- [results.md](results.md)：解释 `data/results` 中贪心输出字段、轨迹和验收结果。
- [greedy_vs_mip.md](greedy_vs_mip.md)：小规模人工与真实局部算例的贪心/MIP 对比。
- [greedy_vs_mip.csv](greedy_vs_mip.csv)：对应明细数据。
- [greedy_vs_mip_larger_real.md](greedy_vs_mip_larger_real.md)：较大真实局部算例的贪心/MIP 对比。
- [greedy_vs_mip_larger_real.csv](greedy_vs_mip_larger_real.csv)：对应明细数据。
- [greedy_gap_analysis.md](greedy_gap_analysis.md)：基于 Greedy 结果和 MIP 最优值的 gap/缺口分析。
- [medium_cpmp_comparison.md](medium_cpmp_comparison.md)：一个中等 CPMP 算例上的多算法对比。
- [full_real_business_current_recommendation.md](full_real_business_current_recommendation.md)：15 个真实配对算例在 `1.3N0` 业务约束下的当前推荐算法口径。
- [full_real_business_focused_beam_comparison.md](full_real_business_focused_beam_comparison.md)：focused beam / QUBO beam / metaheuristics 的完整拉表对比。

## 当前结论

- 当前完整真实配对案例采用业务改善口径：在 `G = M(Now) + N(Future) <= floor(1.3 * N(0))` 内尽量降低 `N(Future)`。
- `chen_greedy_algorithm_1p3` 是最快 baseline；完整 15 个配对案例平均最终 `N(Future)=17.13`。
- 当前推荐的质量路线是 `beam_focused_with_qubo`，平均最终 `N(Future)=9.33`；推荐的速度/质量折中路线是 `feature_adaptive_beam`，平均最终 `N(Future)=9.93`。
- `fused_beam`、`beam_post_repair` 等增强试验保留为对照记录，但不进入推荐主口径。
- 小规模 pre-marshalling 子问题上，当前推荐 4D MIP 作为精确最优基准；3D MIP 已封存为历史对照。
- 当前 benchmark 中，Greedy 完整良序的 3 个算例 gap 均为 `0`；其余算例 Greedy 未良序，不能计算正式 optimality gap。
- 早期搬移数口径下 3D MIP 在部分局部试验中更小；加入业务 `N(Future)` 后，3D 需要辅助槽位变量，当前中型业务算例上弱于 4D。
- 4D MIP 的槽位表达更直观，且更适合直接表达业务相邻层逆序指标，但变量规模仍会随 `n * m * H * K` 和 move 变量快速增长。
