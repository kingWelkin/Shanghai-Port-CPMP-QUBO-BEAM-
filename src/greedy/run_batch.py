"""
多贝位合并贪心：同一箱区 01+26（等）合成一场，允许跨贝位来回搬顶箱。

用法:
  python -m greedy
  python -m greedy --zone A1 --bays 01,26
  python -m greedy --refine
  python -m greedy --per-file   # 旧模式：每个 input 单独跑
"""

from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import runpy
    import sys
    from pathlib import Path

    _src_root = Path(__file__).resolve().parent.parent
    if str(_src_root) not in sys.path:
        sys.path.insert(0, str(_src_root))
    runpy.run_module("greedy", run_name="__main__", alter_sys=True)
    raise SystemExit(0)

import argparse
import time
import traceback
from pathlib import Path
from typing import List, Tuple

from .input_loader import format_layout_table, load_problem_from_input_txt
from .metrics import n0_from_problem
from .report import format_solution_placement_report

from .greedy_report import format_greedy_placement_report
from .multi_bay_loader import (
    discover_paired_cases,
    format_multi_bay_layout,
    load_combined_zone_bays,
)
from .solve import report_greedy_metrics, run_greedy_yard_solve

_PKG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PKG_DIR.parent.parent


def _parse_bays(s: str) -> List[str]:
    out: List[str] = []
    for x in s.split(","):
        t = x.strip()
        if not t:
            continue
        out.append(t.zfill(2) if t.isdigit() else t)
    return out


def _run_one_case(
    problem,
    case_name: str,
    input_desc: str,
    out_dir: Path,
    args,
    mode: str,
) -> Tuple[bool, str, float]:
    """返回 (成功, 摘要行, 耗时)。"""
    out_path = out_dir / f"result_{case_name}.txt"
    err_path = out_dir / f"result_{case_name}_ERROR.txt"
    t0 = time.perf_counter()
    try:
        n0 = n0_from_problem(problem)
        max_g = int(args.max_greedy_moves) if args.max_greedy_moves > 0 else None
        zone = getattr(problem, "zone", None) or "A1"

        best, best_e, meta, n_var, var_index, init_viol, planned = run_greedy_yard_solve(
            problem,
            refine=args.refine,
            refine_if_unsorted=not args.refine_always,
            sub_size=args.sub_size,
            iters=args.iters if args.refine else 0,
            seed=args.seed,
            penalty=args.penalty,
            reloc_scale=args.reloc_scale,
            stack_order_scale=args.stack_order_scale,
            terminal_stack_scale=args.terminal_stack_scale,
            sa_reads=args.sa_reads,
            verbose_iter=True,
            verbose_build=args.timing,
            greedy_max_moves=max_g,
        )

        if args.skip_if_moves_above > 0 and len(planned) > args.skip_if_moves_above:
            elapsed = time.perf_counter() - t0
            return False, f"SKIP {case_name}: moves={len(planned)}", elapsed

        elapsed = time.perf_counter() - t0
        meta["layout"] = "multi_bay" if getattr(problem, "combined_bays", None) else "single"
        metric_lines = report_greedy_metrics(problem, best, meta, var_index, init_viol)

        if meta.get("refined") and best is not None:
            report = format_solution_placement_report(
                problem,
                best,
                var_index,
                meta,
                zone=zone,
                bay=getattr(problem, "combined_bays", [1])[0],
                include_retrieval=False,
            )
        else:
            report = format_greedy_placement_report(
                problem, planned, meta, zone=zone
            )

        header = [
            f"greedy result mode={mode}",
            "=" * 40,
            f"case: {case_name}",
            f"inputs: {input_desc}",
            f"W={problem.W} H={problem.H} N={problem.N}",
            f"greedy_moves: {meta.get('greedy_moves')}",
            f"greedy_sec: {meta.get('greedy_sec')}",
                f"terminal_sorted: {meta.get('terminal_sorted')}",
                f"refined: {meta.get('refined', False)}",
                f"N(0)={n0}",
                f"G_end={meta.get('g_end')} G_cap={meta.get('g_cap')} g_ok={meta.get('g_ok')}",
            ]
        if getattr(problem, "combined_bays", None):
            header.append(f"combined_bays: {problem.combined_bays}")
        if meta.get("refined"):
            header.extend(
                [
                    f"n_var: {n_var}",
                    f"T_qubo: {meta.get('T_qubo')}",
                    f"best_energy: {best_e:.6f}",
                ]
            )
        header.extend([f"elapsed: {elapsed:.1f}s", ""])
        if getattr(problem, "combined_bays", None):
            header.append(format_multi_bay_layout(problem))
        else:
            header.append(format_layout_table(problem))
        header.append("")

        body = header + metric_lines + [""] + report
        out_path.write_text("\n".join(body), encoding="utf-8")
        summary = (
            f"OK {case_name}: {elapsed:.1f}s moves={meta.get('greedy_moves')} "
            f"sorted={meta.get('terminal_sorted')}"
        )
        print(f"  完成 → {out_path.name} ({elapsed:.1f}s)")
        return True, summary, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - t0
        err_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"  失败: {e} ({elapsed:.1f}s)")
        return False, f"ERR {case_name}: {e}", elapsed


def main() -> None:
    ap = argparse.ArgumentParser(
        description="多贝位合并贪心（01+26 等可跨贝位搬箱）"
    )
    ap.add_argument("--input-dir", type=str, default=str(_PROJECT_ROOT / "data" / "input"))
    ap.add_argument("--out-dir", type=str, default=str(_PROJECT_ROOT / "data" / "results"))
    ap.add_argument(
        "--bays",
        type=str,
        default="01,26",
        help="合并的贝位后缀，逗号分隔（默认 01,26）",
    )
    ap.add_argument(
        "--zone",
        type=str,
        default="",
        help="只跑指定箱区（如 A1）；默认扫描全部可配对箱区",
    )
    ap.add_argument(
        "--per-file",
        action="store_true",
        help="旧模式：每个 *_input.txt 单独跑（不合并贝位）",
    )
    ap.add_argument("--glob", type=str, default="*_input.txt", help="仅 --per-file 时用")
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--refine-always", action="store_true")
    ap.add_argument("--sub-size", type=int, default=40)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--penalty", type=float, default=30.0)
    ap.add_argument("--reloc-scale", type=float, default=0.3)
    ap.add_argument("--stack-order-scale", type=float, default=5.0)
    ap.add_argument("--terminal-stack-scale", type=float, default=50.0)
    ap.add_argument("--sa-reads", type=int, default=80)
    ap.add_argument("--max-greedy-moves", type=int, default=0)
    ap.add_argument("--skip-if-moves-above", type=int, default=0)
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bay_suffixes = _parse_bays(args.bays)
    mode = "multi_bay" + ("+qubo" if args.refine else "")

    summary_lines = [
        f"greedy 批量 mode={mode}",
        f"合并贝位: {','.join(bay_suffixes)}",
        "",
    ]
    ok_n = err_n = 0
    t_all = time.perf_counter()

    if args.per_file:
        files = sorted(in_dir.glob(args.glob))
        if not files:
            raise FileNotFoundError(f"未找到: {in_dir / args.glob}")
        for fp in files:
            print(f"\n=== {fp.name} ({mode}) ===")
            problem = load_problem_from_input_txt(fp)
            ok, line, _ = _run_one_case(
                problem,
                fp.stem,
                str(fp.resolve()),
                out_dir,
                args,
                mode,
            )
            summary_lines.append(line)
            if ok:
                ok_n += 1
            else:
                err_n += 1 if line.startswith("ERR") else 0
    else:
        cases = discover_paired_cases(in_dir, bay_suffixes)
        if args.zone:
            cases = [c for c in cases if c[0].upper() == args.zone.strip().upper()]
        if not cases:
            raise FileNotFoundError(
                f"未找到可合并案例（需同时存在 {{zone}}_<贝位>_input.txt，"
                f"贝位 in {bay_suffixes}）: {in_dir}"
            )
        for zone, suffixes, paths in cases:
            case_name = f"{zone}_{'_'.join(suffixes)}_paired"
            print(f"\n=== {case_name} ({mode}) ===")
            print(f"  合并: {', '.join(p.name for p in paths)}")
            problem = load_combined_zone_bays(in_dir, zone, suffixes)
            n0_all = n0_from_problem(problem)
            n0_bay = getattr(problem, "bay_n0_initial", None)
            print(
                f"  W={problem.W} ({len(suffixes)}×10堆) N={problem.N} "
                f"N(0)={n0_all}"
            )
            if n0_bay:
                drain = getattr(problem, "drain_bay", None)
                fill = getattr(problem, "fill_bay", None)
                print(
                    f"  各贝N(0)={n0_bay}  策略: 优先从贝{drain:02d}搬出 → 贝{fill:02d}"
                )
            ok, line, _ = _run_one_case(
                problem,
                case_name,
                "; ".join(str(p.resolve()) for p in paths),
                out_dir,
                args,
                mode,
            )
            summary_lines.append(line)
            if ok:
                ok_n += 1
            else:
                err_n += 1 if line.startswith("ERR") else 0

    total = time.perf_counter() - t_all
    summary_lines.extend(["", f"成功 {ok_n}  失败 {err_n}  合计 {total:.1f}s"])
    (out_dir / "SUMMARY.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\n汇总 → {out_dir / 'SUMMARY.txt'}")


if __name__ == "__main__":
    main()
