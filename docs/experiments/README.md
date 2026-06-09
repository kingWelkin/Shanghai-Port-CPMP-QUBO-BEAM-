# 试验结论

本目录统一存放当前项目的试验报告、历史结果解读和 CSV 明细。

## 文件说明

- [results.md](results.md)：解释 `data/results` 中贪心输出字段、轨迹和验收结果。
- [qubo_astar_speed_alignment.md](qubo_astar_speed_alignment.md)：当前 QUBO Beam 主结果和 A* no-proof 速度对齐结论。
- [compact_beam_qubo_speed_optimized_v2_full30_repeat.md](compact_beam_qubo_speed_optimized_v2_full30_repeat.md)：稳定版 QUBO Beam speed 的 30 个单贝算例结果。
- [astar_qubo_baseline_beam_comparison.md](astar_qubo_baseline_beam_comparison.md)：A* no-proof、QUBO Beam、classic Beam 消融的历史对比表。
- [greedy_vs_mip.md](greedy_vs_mip.md)：小规模人工与真实局部算例的贪心/MIP 对比。
- [greedy_vs_mip.csv](greedy_vs_mip.csv)：对应明细数据。
- [greedy_vs_mip_larger_real.md](greedy_vs_mip_larger_real.md)：较大真实局部算例的贪心/MIP 对比。
- [greedy_vs_mip_larger_real.csv](greedy_vs_mip_larger_real.csv)：对应明细数据。
- [greedy_gap_analysis.md](greedy_gap_analysis.md)：基于 Greedy 结果和 MIP 最优值的 gap/缺口分析。
- [medium_cpmp_comparison.md](medium_cpmp_comparison.md)：一个中等 CPMP 算例上的多算法对比。
- [full_real_business_all_algorithms_extended_comparison.md](full_real_business_all_algorithms_extended_comparison.md)：15 个真实配对算例按最终 `N(Future)` 汇总的扩展算法总表，覆盖 beam、QUBO、tabu、GRASP、VNS、SA、genetic、LNS、A* 等路线。
- [full_real_business_current_recommendation.md](full_real_business_current_recommendation.md)：15 个真实配对算例在 `1.3N0` 业务约束下的当前推荐算法口径。
- [full_real_business_focused_beam_comparison.md](full_real_business_focused_beam_comparison.md)：focused beam / QUBO beam / metaheuristics 的完整拉表对比。

## 当前结论

- 当前主推结果是稳定版非 dynamic QUBO Beam speed：30 个单贝 CPMP 算例全部达到 `N=0`，总步数 509，平均时间 2.274 秒，最大时间 4.752 秒。
- 相比 A* no-proof，稳定版 QUBO Beam 在同样 30/30 解出下少 5 步，并且平均时间从 2.829 秒降到 2.274 秒。
- classic Beam 作为严格消融：时间匹配口径为 21/30，budget-best 口径为 28/30。
- dynamic QUBO 作为质量增强变体：`balanced2200` 为 508 步、4.137 秒；质量档为 507 步、4.902 秒。
- 早期完整真实配对案例采用业务改善口径：在 `G = M(Now) + N(Future) <= floor(1.3 * N(0))` 内尽量降低 `N(Future)`。这些结果保留为历史业务路线，不再作为当前主结果。
- `fused_beam`、`beam_post_repair` 等增强试验保留为对照记录，但不进入推荐主口径。
- 小规模 pre-marshalling 子问题上，当前推荐 4D MIP 作为精确最优基准；3D MIP 已封存为历史对照。
- 当前 benchmark 中，Greedy 完整良序的 3 个算例 gap 均为 `0`；其余算例 Greedy 未良序，不能计算正式 optimality gap。
- 早期搬移数口径下 3D MIP 在部分局部试验中更小；加入业务 `N(Future)` 后，3D 需要辅助槽位变量，当前中型业务算例上弱于 4D。
- 4D MIP 的槽位表达更直观，且更适合直接表达业务相邻层逆序指标，但变量规模仍会随 `n * m * H * K` 和 move 变量快速增长。
