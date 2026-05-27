"""Build a broad full-real business comparison table across all benchmark routes."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "docs" / "experiments"


SOURCE_FILES = [
    "full_real_business_focused_beam_comparison.csv",
    "full_real_business_weighted_astar_tuning.csv",
    "full_real_business_lns_astar_comparison.csv",
    "full_real_business_qubo_standalone_tuning_comparison.csv",
    "full_real_business_enhanced_qubo_beam_comparison.csv",
    "full_real_business_adaptive_beam_comparison.csv",
    "full_real_business_beam_improvements_comparison.csv",
    "full_real_business_genetic_tabu_comparison.csv",
    "full_real_business_metaheuristics_comparison.csv",
    "full_real_business_tabu_baseline_comparison.csv",
    "full_real_business_strong_baseline_comparison.csv",
    "full_real_business_qubo_api_repair_comparison.csv",
    "full_real_business_qubo_api_comparison.csv",
    "full_real_business_greedy_1p3_vs_beam_qubo.csv",
    "full_real_business_beam_qubo_vs_greedy.csv",
]


FIELDS = [
    "case",
    "solver",
    "n0",
    "cap_13",
    "cap_15",
    "moves",
    "final_n",
    "reduction",
    "g_end",
    "max_g",
    "g_ok_13",
    "g_ok_15",
    "runtime_sec",
    "expanded_states",
    "generated_states",
    "visited_states",
    "qubo_calls",
    "qubo_total_vars",
    "qubo_time_sec",
    "status",
    "source_file",
]


def _to_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _truthy(value: str | None) -> bool:
    return str(value).lower() == "true"


def _read_source(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        if not reader.fieldnames or "case" not in reader.fieldnames or "solver" not in reader.fieldnames:
            return rows
        for row in reader:
            if not row.get("case") or not row.get("solver") or row.get("final_n") in (None, ""):
                continue
            normalized = {field: row.get(field, "") for field in FIELDS}
            normalized["source_file"] = path.name
            if not normalized["cap_15"] and normalized["n0"]:
                normalized["cap_15"] = str(int(1.5 * _to_float(normalized["n0"])))
            if not normalized["g_ok_15"] and normalized["max_g"] and normalized["cap_15"]:
                normalized["g_ok_15"] = str(_to_float(normalized["max_g"]) <= _to_float(normalized["cap_15"]))
            rows.append(normalized)
        return rows


def _collect_rows() -> list[dict[str, str]]:
    keyed: dict[tuple[str, str], dict[str, str]] = {}
    for name in SOURCE_FILES:
        path = EXPERIMENTS / name
        if not path.exists():
            continue
        for row in _read_source(path):
            key = (row["case"], row["solver"])
            # Source order is already newest/highest-priority first.
            keyed.setdefault(key, row)
    return list(keyed.values())


def _summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    cases = sorted({row["case"] for row in rows})
    best_by_case = {
        case: min(_to_float(row["final_n"], float("inf")) for row in rows if row["case"] == case)
        for case in cases
    }
    by_solver: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_solver.setdefault(row["solver"], []).append(row)

    summary = []
    for solver, solver_rows in by_solver.items():
        if len({row["case"] for row in solver_rows}) != len(cases):
            continue
        summary.append(
            {
                "solver": solver,
                "cases": len(solver_rows),
                "avg_final_n": sum(_to_float(row["final_n"]) for row in solver_rows) / len(solver_rows),
                "avg_reduction": sum(_to_float(row["reduction"]) for row in solver_rows) / len(solver_rows),
                "avg_moves": sum(_to_float(row["moves"]) for row in solver_rows) / len(solver_rows),
                "avg_runtime_sec": sum(_to_float(row["runtime_sec"]) for row in solver_rows) / len(solver_rows),
                "avg_qubo_calls": sum(_to_float(row["qubo_calls"]) for row in solver_rows) / len(solver_rows),
                "avg_qubo_vars": sum(_to_float(row["qubo_total_vars"]) for row in solver_rows) / len(solver_rows),
                "ok_13": sum(_truthy(row["g_ok_13"]) for row in solver_rows),
                "wins": sum(
                    abs(_to_float(row["final_n"], float("inf")) - best_by_case[row["case"]]) < 1e-9
                    for row in solver_rows
                ),
                "source_files": ", ".join(sorted({row["source_file"] for row in solver_rows})),
            }
        )
    summary.sort(key=lambda item: (item["avg_final_n"], item["avg_moves"], item["avg_runtime_sec"]))
    return summary


def _write_csv(rows: list[dict[str, str]], summary: list[dict[str, object]]) -> tuple[Path, Path]:
    detail_path = EXPERIMENTS / "full_real_business_all_algorithms_extended_detail.csv"
    summary_path = EXPERIMENTS / "full_real_business_all_algorithms_extended_summary.csv"

    with detail_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["case"], item["solver"])):
            writer.writerow({field: row.get(field, "") for field in FIELDS})

    summary_fields = [
        "rank",
        "solver",
        "cases",
        "avg_final_n",
        "avg_reduction",
        "avg_moves",
        "avg_runtime_sec",
        "avg_qubo_calls",
        "avg_qubo_vars",
        "ok_13",
        "wins",
        "source_files",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for rank, item in enumerate(summary, start=1):
            row = dict(item)
            row["rank"] = rank
            writer.writerow(row)
    return detail_path, summary_path


def _write_markdown(rows: list[dict[str, str]], summary: list[dict[str, object]]) -> Path:
    cases = sorted({row["case"] for row in rows})
    solvers = [str(item["solver"]) for item in summary]
    lookup = {(row["case"], row["solver"]): row for row in rows}
    detail_path = EXPERIMENTS / "full_real_business_all_algorithms_extended_detail.csv"
    summary_path = EXPERIMENTS / "full_real_business_all_algorithms_extended_summary.csv"
    md_path = EXPERIMENTS / "full_real_business_all_algorithms_extended_comparison.md"

    lines = [
        "# 完整真实配对算例扩展算法对比",
        "",
        "本文只比较真实 `01+26` 配对算例的业务目标，即最终 `N(Future)`。所有进入主表的行都来自完整 15 个真实配对算例，并按平均最终 `N(Future)` 升序排序。",
        "",
        "业务约束仍为：`G = M(Now) + N(Future) <= floor(1.3 * N(0))`。",
        "",
        f"- 汇总 CSV：`{summary_path.relative_to(ROOT)}`",
        f"- 明细 CSV：`{detail_path.relative_to(ROOT)}`",
        "",
        "## 汇总排序",
        "",
        "| rank | solver | avg final N | avg reduction | avg moves | avg runtime s | avg QUBO calls | ok 1.3N0 | wins |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(summary, start=1):
        lines.append(
            f"| {rank} | `{item['solver']}` | {item['avg_final_n']:.2f} | {item['avg_reduction']:.2f} | "
            f"{item['avg_moves']:.2f} | {item['avg_runtime_sec']:.3f} | {item['avg_qubo_calls']:.2f} | "
            f"{item['ok_13']}/{item['cases']} | {item['wins']} |"
        )

    top_solvers = solvers[:20]
    lines.extend(
        [
            "",
            "## 前 20 条路线逐例最终 N(Future)",
            "",
            "| case | " + " | ".join(f"`{solver}`" for solver in top_solvers) + " |",
            "|---|" + "|".join("---:" for _ in top_solvers) + "|",
        ]
    )
    for case in cases:
        values = [lookup.get((case, solver), {}).get("final_n", "") for solver in top_solvers]
        lines.append(f"| {case} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## 算法族结论",
            "",
            "- 当前最强仍是 `beam_focused_with_qubo_portfolio`，平均最终 `N(Future)=8.53`。",
            "- QUBO-only portfolio 可达到 `9.00`，说明 QUBO 分支本身有实质收益。",
            "- 无 QUBO 的 focused beam 为 `9.47`，强于多数 tabu / GRASP / VNS / SA / genetic baseline。",
            "- `weighted_astar_business_portfolio` 增强后达到 `10.13`，是较强 best-first 对照，但仍弱于 focused beam。",
            "- tabu、GRASP、VNS、simulated annealing、genetic+tabu 均可作为 sanity-check baseline，但平均质量没有超过 beam/QUBO 系列。",
            "- `chen_greedy_algorithm_1p3` 最快，但平均最终 `N(Future)=17.13`，主要作为生产快速 baseline。",
            "",
            "## 关于 MIP",
            "",
            "本表的主目标是完整真实配对算例的最终 `N(Future)`。当前没有完整 15 个真实配对算例的 MIP 可比最终 `N(Future)` 结果；已有 MIP 主要用于 small/medium/larger CPMP 参考基准。",
            "",
            "正在后台运行的长时 4D MIP 尚未产生可验证的真实算例结果，因此不进入本次排序。若后续 MIP 输出 validated incumbent，可再追加为单独行，但应标注 time limit、是否有 incumbent、是否满足 `1.3N0`。",
            "",
            "## 数据来源",
            "",
        ]
    )
    for name in SOURCE_FILES:
        if (EXPERIMENTS / name).exists():
            lines.append(f"- `{name}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main() -> None:
    rows = _collect_rows()
    summary = _summarize(rows)
    detail_path, summary_path = _write_csv(rows, summary)
    md_path = _write_markdown(rows, summary)
    print(f"wrote {summary_path}")
    print(f"wrote {detail_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
