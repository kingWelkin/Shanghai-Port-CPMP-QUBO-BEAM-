# 项目结构

```text
.
├── config/
│   ├── README.md
│   └── defaults.toml
├── data/
│   ├── README.md
│   ├── input/
│   └── results/
├── docs/
│   ├── assets/
│   ├── experiments/
│   │   ├── README.md
│   │   ├── greedy_vs_mip.md
│   │   ├── greedy_vs_mip_larger_real.md
│   │   └── results.md
│   ├── guides/
│   │   ├── usage.md
│   │   ├── data-format.md
│   │   ├── project-structure.md
│   │   └── development.md
│   ├── legacy/
│   ├── models/
│   │   ├── greedy/
│   │   │   ├── chen_greedy_algorithm.md
│   │   │   ├── chen_greedy_algorithm.tex
│   │   │   └── chen_greedy_algorithm.pdf
│   │   └── mip/
│   │       ├── mip.md
│   │       ├── 上港翻箱堆叠问题模型.tex
│   │       └── 上港翻箱堆叠问题模型.pdf
│   └── index.md
├── notebooks/
│   └── README.md
├── src/
│   └── greedy/
├── README.md
├── pyproject.toml
└── run.bat
```

## 目录职责

- `src/greedy`：可复用 Python 包代码。
- `data/input`：原始输入数据，不放运行脚本。
- `data/results`：运行输出和基线结果。
- `docs`：面向使用者和开发者的说明文档。
- `docs/assets`：文档图片素材。
- `docs/experiments`：试验结论、benchmark 报告和 CSV 明细。
- `docs/guides`：运行、数据格式、工程结构和开发说明。
- `docs/legacy`：迁移前的旧说明，保留作对照。
- `docs/models/greedy`：`chen_greedy_algorithm` 说明和历史 PDF。
- `docs/models/mip`：MIP 求解说明和 3D/4D 数学模型 tex/pdf。
- `notebooks`：探索分析，不承载核心逻辑。
- `config`：参数约定和未来配置入口。
