# CPMP QUBO–Beam v5.0

面向集装箱预翻箱问题（CPMP）的研究版本：**Beam 构造合法完整路径，QUBO 联合优化多个箱子的整条路线，FRG 循环精炼，独立 Pair 轨迹提供补充搜索。**

主运行配置为 `qubo_pair`。本仓库提供可运行源码、公开基准输入、测试、结果摘要、客户说明和完整流程图。版本标记为 **v5.0 研究发布**；历史生产默认 v4.3 的全量验收结论不自动转移到本版本。

## 算法与流程图

- [客户算法说明](docs/engineering_delivery/qubo_beam_v50_20261013/CPMP_QUBO_Beam_v5.0_客户算法说明.md)
- [完整模型、系数、流程与验证说明](docs/guides/v50-qubo-beam-algorithm.md)
- 完整流程图：[PNG](docs/engineering_delivery/qubo_beam_v50_20261013/CPMP_QUBO_Beam_v5.0_完整算法流程图.png) · [SVG](docs/engineering_delivery/qubo_beam_v50_20261013/CPMP_QUBO_Beam_v5.0_完整算法流程图.svg) · [Mermaid 源码](docs/engineering_delivery/qubo_beam_v50_20261013/CPMP_QUBO_Beam_v5.0_完整算法流程图.mmd)
- [扩预算与长尾诊断](docs/experiments/v50_budget_tail_20261013/研究结论_20261013.md)

![完整算法流程图](docs/engineering_delivery/qubo_beam_v50_20261013/CPMP_QUBO_Beam_v5.0_完整算法流程图.png)

## 快速运行

参考环境为 Python 3.13 / Windows；项目要求 Python 3.10 及以上。先在仓库根目录安装依赖：

```powershell
python -m pip install -e ".[test]"
python scripts/run_cpmp_v50.py --list-cases
```

从上述列表选一个原始算例 ID，运行完整 v5.0 Q+Pair：

```powershell
python scripts/run_cpmp_v50.py --case "CASE_ID_FROM_LIST" --budget-sec 20 --output outputs/result.json
```

`--case` 使用 [hardset169 清单](configs/hardset169.json) 中的精确名称；`--output` 必须为尚不存在的文件。输入来源和第三方许可见 [benchmarks/README.md](benchmarks/README.md)。169 例为 BF 73、CV 84、Open120 12。仓库同时保留上游来源明确的完整基准输入，方便核查。

输出 JSON 包含 `feasible`、实际 `moves`、基本移箱 `path`、回放后的 `final_stacks`、截止前就绪/接收时间、工作进程状态和校验结果。栈下标从 0 开始，层位从底部起按 1 编号。`global_cpmp_optimal` 为 `false`。

可选模式：

| 模式 | 含义 |
|---|---|
| `qubo_pair` | 默认主方案：Q 与 Pair 独立轨迹按完整轮次交替运行 |
| `qubo_cycle` | 单轨联合 QUBO + FRG |
| `single_pair` | 匹配的经典控制：Single 与 Pair 两轨 |

此入口调用研究中使用的前端、后端和截止时间协议，省去对历史日志及实验机器 CPU 编号的依赖。它用于执行算法，不重建原多方法实验的 CPU 轮换、并发负载或统计设计。

**预算口径：**模块导入与输入身份检查在 `READY` 前预热；原始实例载入、Beam、路线建模、暖启动、QUBO、FRG、输出账本校验及消息接收共享同一 `START`。只接受就绪和接收时间均未超过预算的完整路径。工作进程清理、交付前额外复核及 JSON 写盘可能使命令总耗时超过该预算，均不产生新的计分候选。

QUBO 后端为经典结构化搜索；主方案不调用远程硬件服务。`gurobipy` 是继承的研究对照依赖，MILP 对照需有效许可证。可选硬件接口必须显式设置 `ISING_API_BASE_URL`。

## QUBO 优化的对象

一个二进制变量表示某个物理箱的一条完整候选路线。联合模型最小化真实移箱次数，同时编码每箱恰选一条路线、跨箱路线兼容和时序容量约束。每个邻域最多释放 8 个箱子，每轮最多处理 12 个接触邻域；超过 20000 条路线或约化后 4096 个二进制变量时跳过整个模型。

单模型联合搜索请求上限为 50 毫秒或 50000 节点，同时服从总预算。三种顺序的 Single 暖启动实际计算并计时。路线选择恢复为真实动作后，必须通过栈顶、容量、箱子守恒和终态顺序回放。

[28 变量数值模型](docs/experiments/action_coupling_qubo_20261011/final_assessment_20261013/example_qubo_28_variables.json) 保存了完整系数与赋值：100 步输入，经 Single 种子得到 99 步，联合 QUBO 得到 98 步，因子能量、矩阵能量和真实路径长度一致。

## 已有证据

| 范围 | 结论 |
|---|---|
| hardset169，20 秒，两次运行 | Q+Pair 与 v4.3 均为 325/338 次可行；成功样本不完全相同 |
| 双方两次均成功的 150 个原始算例 | Q+Pair 平均 169.63 步，v4.3 为 201.68 步，整流程减少 15.89% |
| 与匹配 Single+Pair 比较 | 面板 A/B 平均分别减少约 1.08% / 0.92%；这是新增 QUBO 的净增量 |
| 120 秒扩预算诊断原主组 12 例 | Q+Pair 平均 128.08 步，v4.3 为 159.42 步，减少 19.65% |

失败、单例退化、不同分母和比较区间均保留在文档中。27 例扩预算诊断刻意包含长尾，不能外推为全 169 例结果。Q→Single→Pair 长尾保护源码作为实验扩展保留，不计入上述主方案成绩。

## 验证与发布范围

```powershell
python -m pytest -q
```

[发布说明](docs/releases/v5.0.md) 记录本次源码范围与实际验证。此仓库是独立、整理后的 v5.0 快照，包含必要传递依赖、测试和精简证据。内部业务原始数据、个人工作指令、临时缓存与海量逐次日志未纳入。结果摘要仍保留原始实验身份及校验信息，不把本次发布检查写成新的性能实验。

历史 `research_*.py` 入口供方法和协议查阅；部分冻结/汇总命令依赖完整历史归档，不能仅凭本快照执行全部原始研究流程。新实例求解使用 `run_cpmp_v50.py`；重新开展性能实验应建立新的冻结协议。
