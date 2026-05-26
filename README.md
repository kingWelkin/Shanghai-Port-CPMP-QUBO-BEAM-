# 上港堆叠装箱

这是一个堆场搬箱路径规划项目。当前主流程把同一箱区的多个贝位
（默认 `01` 与 `26`）合并为一个堆场，只允许从某堆顶部取箱并放到另一堆顶部，
通过确定性贪心算法在约束内降低逆序箱数；可选 QUBO 精炼仍处于实验状态。

## 项目结构

```text
.
├── config/          # 运行默认参数记录
├── data/
│   ├── input/       # 原始输入数据
│   └── results/     # 历史/生成结果
├── docs/            # 项目说明、算法、数据格式和结果解读
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
- [数据格式](docs/data-format.md)
- [运行说明](docs/usage.md)
- [算法说明](docs/algorithm.md)
- [结果解读](docs/results.md)
- [开发说明](docs/development.md)

## 当前状态

现有 15 个配对案例都能生成满足 `G <= floor(1.5 * N(0))` 的贪心轨迹，
但历史结果显示这些案例均未达到全场良序。也就是说，当前主算法适合在约束内快速降低乱度，
不应理解为一定能完成全场排序。
