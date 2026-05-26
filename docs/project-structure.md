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
│   ├── legacy/
│   ├── index.md
│   ├── usage.md
│   ├── data-format.md
│   ├── algorithm.md
│   ├── results.md
│   └── development.md
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
- `docs/legacy`：迁移前的旧说明，保留作对照。
- `notebooks`：探索分析，不承载核心逻辑。
- `config`：参数约定和未来配置入口。
