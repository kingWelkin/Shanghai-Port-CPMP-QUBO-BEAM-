# 文档导航

## 读者入口

- 想跑程序：看 [运行说明](usage.md)
- 想了解目录：看 [项目结构](project-structure.md)
- 想准备输入：看 [数据格式](data-format.md)
- 想理解算法：看 [算法说明](algorithm.md)
- 想看输出含义：看 [结果解读](results.md)
- 想改代码：看 [开发说明](development.md)

## 关键结论

当前项目的稳定主流程是贪心搬箱：

1. 读取 `data/input` 中同一箱区的 `{ZONE}_01_input.txt` 与 `{ZONE}_26_input.txt`。
2. 合并为一个 `W=20, H=6` 的堆场。
3. 反复选择一个合法跨堆/跨贝顶箱搬移动作。
4. 全程约束 `G = 搬箱步数 + N(Future) <= floor(1.5 * N(0))`。
5. 输出每步搬箱轨迹、末态堆表和验收报告。

QUBO 精炼代码已保留，但目前不建议作为默认路径使用。
