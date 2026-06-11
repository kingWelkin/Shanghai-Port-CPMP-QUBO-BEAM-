"""Run the current compact QUBO beam on open CPMP benchmark instances.

The first supported source is the DLTS CPMP repository:
https://github.com/ahottung/DLTS
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_compact_beam_qubo_n0 import (  # noqa: E402
    COMPACT_RESULT_FIELDS,
    _row_dict_from_instance_result,
    _validate_instance_result,
    make_compact_args,
    run_compact_beam_instance,
)
from container_reshuffle.core import Instance, ProblemVariant, validate_instance  # noqa: E402

DLTS_URL = "https://github.com/ahottung/DLTS.git"
OPEN_EXTRA_FIELDS = [
    "dataset",
    "family",
    "variant",
    "source_file",
    "reference_file",
    "reference_moves",
    "move_gap",
    "reference_tiers",
    "business_alpha",
    "business_ok",
    "goal_validation_ok",
    "validation_msg",
]
OPEN_RESULT_FIELDS = [field for field in OPEN_EXTRA_FIELDS if field not in COMPACT_RESULT_FIELDS] + COMPACT_RESULT_FIELDS


@dataclass(frozen=True)
class DltsReference:
    optimal_moves: int | None = None
    max_height: int | None = None


@dataclass(frozen=True)
class OpenCpmpCase:
    family: str
    case_id: str
    instance_path: Path
    reference_path: Path | None
    reference: DltsReference


def _case_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.stem
    except ValueError:
        return 10**9, path.stem


def load_dlts_dat(path: Path, max_height: int | None = None) -> Instance:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty DLTS instance file: {path}")
    header = [int(part) for part in lines[0].split()]
    if len(header) != 2:
        raise ValueError(f"bad DLTS header in {path}: {lines[0]!r}")
    stack_count, container_count = header
    stacks: list[list[int]] = []
    max_seen_height = 0
    for line in lines[1:]:
        values = [int(part) for part in line.split()]
        if not values:
            continue
        height, boxes = values[0], values[1:]
        if height != len(boxes):
            raise ValueError(f"declared height {height} does not match row in {path}: {line!r}")
        stacks.append(boxes)
        max_seen_height = max(max_seen_height, height)
    if len(stacks) != stack_count:
        raise ValueError(f"expected {stack_count} stacks in {path}, got {len(stacks)}")
    inferred_height = max_seen_height + 2
    instance = Instance(
        instance_id=f"{path.parent.name}_{path.stem}",
        variant=ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=stack_count,
        max_height=max_height if max_height is not None else inferred_height,
        container_count=container_count,
        stacks=stacks,
    )
    validate_instance(instance)
    return instance


def read_dlts_reference(path: Path | None) -> DltsReference:
    if path is None or not path.exists():
        return DltsReference()
    optimal_moves: int | None = None
    max_height: int | None = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("AA tiers-stacks-time"):
            parts = line.split()
            if len(parts) >= 3:
                max_height = int(parts[2])
        elif line.startswith("opt="):
            optimal_moves = int(line.split("=", 1)[1].strip())
    return DltsReference(optimal_moves=optimal_moves, max_height=max_height)


def ensure_dlts_source(source: Path, auto_clone: bool = True, refresh: bool = False) -> Path:
    if refresh and source.exists():
        subprocess.run(["git", "-C", str(source), "pull", "--ff-only"], check=True)
    if (source / "instances").is_dir():
        return source
    if not auto_clone:
        raise FileNotFoundError(f"DLTS source not found at {source}; pass --auto-clone or --source")
    source.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", DLTS_URL, str(source)], check=True)
    return source


def discover_dlts_cases(
    source: Path,
    families: list[str] | None = None,
    limit_per_family: int | None = None,
    max_total: int | None = None,
) -> list[OpenCpmpCase]:
    instances_dir = source / "instances"
    refs_dir = source / "reference_solutions"
    selected_families = set(families or [])
    cases: list[OpenCpmpCase] = []
    for family_dir in sorted(path for path in instances_dir.iterdir() if path.is_dir()):
        if selected_families and family_dir.name not in selected_families:
            continue
        dat_files = sorted(family_dir.glob("*.dat"), key=_case_sort_key)
        if limit_per_family is not None:
            dat_files = dat_files[:limit_per_family]
        for dat_path in dat_files:
            ref_path = refs_dir / family_dir.name / f"{dat_path.stem}.out"
            ref = read_dlts_reference(ref_path if ref_path.exists() else None)
            cases.append(
                OpenCpmpCase(
                    family=family_dir.name,
                    case_id=f"{family_dir.name}_{dat_path.stem}",
                    instance_path=dat_path,
                    reference_path=ref_path if ref_path.exists() else None,
                    reference=ref,
                )
            )
            if max_total is not None and len(cases) >= max_total:
                return cases
    return cases


def _numeric(row: dict[str, object], field: str, default: float = 0.0) -> float:
    value = row.get(field)
    if value in (None, ""):
        return default
    return float(value)


def _write_open_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OPEN_RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _family_size_sort_key(family: str) -> tuple[int, int, str]:
    match = re.match(r"(\d+)x(\d+)_", family)
    if not match:
        return 10**9, 10**9, family
    return int(match.group(1)), int(match.group(2)), family


def _write_summary(rows: list[dict[str, object]], output: Path, summary: Path, source: Path) -> None:
    lines = [
        "# Open CPMP Benchmark Summary",
        "",
        f"source: `{_display_path(source)}`",
        f"result_csv: `{_display_path(output)}`",
        "",
        "| variant | family | cases | N=0 solved | moves total | avg runtime sec | avg ref gap | qcalls | qtime sec |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["variant"]), str(row["family"])), []).append(row)
    for (variant, family), selected in sorted(groups.items(), key=lambda item: (item[0][0], _family_size_sort_key(item[0][1]))):
        solved = [row for row in selected if int(_numeric(row, "final_n", 10**9)) == 0]
        gaps = [
            _numeric(row, "move_gap")
            for row in solved
            if row.get("move_gap") not in (None, "")
        ]
        lines.append(
            f"| {variant} | {family} | {len(selected)} | {len(solved)} | "
            f"{int(sum(_numeric(row, 'moves') for row in solved))} | "
            f"{sum(_numeric(row, 'runtime_sec') for row in selected) / len(selected):.3f} | "
            f"{sum(gaps) / len(gaps) if gaps else 0.0:.3f} | "
            f"{int(sum(_numeric(row, 'qubo_calls') for row in selected))} | "
            f"{sum(_numeric(row, 'qubo_time_sec') for row in selected):.3f} |"
        )
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _variant_names(name: str) -> list[str]:
    return ["qubo", "classic_noqubo"] if name == "both" else [name]


def _args_for_variant(args: argparse.Namespace, variant: str) -> argparse.Namespace:
    overrides: dict[str, object] = {
        "alpha": args.alpha,
        "max_moves": args.max_moves,
        "solver_profile": args.solver_profile,
        "target_final_n": 0,
        "time_limit": args.time_limit,
    }
    if variant == "classic_noqubo":
        overrides.update(
            {
                "beam_width": args.no_qubo_beam_width,
                "core_width": args.no_qubo_beam_width,
                "max_qubo_calls": 0,
                "post_polish_qubo": False,
                "selection_policy": "classic",
                "use_qubo": False,
            }
        )
    return make_compact_args(**overrides)


def run_open_benchmark(args: argparse.Namespace) -> list[dict[str, object]]:
    source = ensure_dlts_source(args.source, auto_clone=args.auto_clone, refresh=args.refresh_source)
    families = None if args.families == ["all"] else args.families
    cases = discover_dlts_cases(
        source,
        families=families,
        limit_per_family=args.limit_per_family,
        max_total=args.max_total,
    )
    rows: list[dict[str, object]] = []
    total_runs = len(cases) * len(_variant_names(args.variant))
    run_idx = 0
    for case in cases:
        try:
            instance = load_dlts_dat(case.instance_path, max_height=case.reference.max_height)
        except ValueError as exc:
            if args.skip_invalid:
                print(f"CASE_SKIP {case.case_id} reason={exc}", flush=True)
                continue
            raise
        for variant in _variant_names(args.variant):
            run_idx += 1
            compact_args = _args_for_variant(args, variant)
            print(f"CASE_START {run_idx}/{total_runs} {variant} {case.case_id}", flush=True)
            result = run_compact_beam_instance(case.case_id, instance, compact_args)
            row = _row_dict_from_instance_result(result, instance, compact_args)
            business_ok, goal_ok, msg = _validate_instance_result(result, instance, business_alpha=args.alpha)
            reference_moves = case.reference.optimal_moves
            moves = row.get("moves")
            row.update(
                {
                    "dataset": "DLTS",
                    "family": case.family,
                    "variant": variant,
                    "source_file": str(case.instance_path),
                    "reference_file": str(case.reference_path) if case.reference_path else "",
                    "reference_moves": reference_moves if reference_moves is not None else "",
                    "move_gap": (int(moves) - reference_moves) if moves not in (None, "") and reference_moves is not None else "",
                    "reference_tiers": case.reference.max_height if case.reference.max_height is not None else "",
                    "business_alpha": args.alpha,
                    "business_ok": business_ok,
                    "goal_validation_ok": goal_ok,
                    "validation_msg": msg,
                }
            )
            rows.append(row)
            print(
                f"CASE_DONE {variant} {case.case_id} final_n={row['final_n']} moves={row['moves']} "
                f"ref={row['reference_moves']} runtime={float(row['runtime_sec']):.3f}s "
                f"goal_ok={goal_ok} business_ok={business_ok}",
                flush=True,
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "external" / "DLTS")
    parser.add_argument("--auto-clone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refresh-source", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--families", nargs="+", default=["5x5_cv1_test"])
    parser.add_argument("--limit-per-family", type=int, default=20)
    parser.add_argument("--max-total", type=int)
    parser.add_argument("--variant", choices=["qubo", "classic_noqubo", "both"], default="qubo")
    parser.add_argument("--skip-invalid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--solver-profile", choices=["speed_v2"], default="speed_v2")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--no-qubo-beam-width", type=int, default=1200)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "experiments" / "open_cpmp_dlts_speed_v2.csv",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.source = args.source if args.source.is_absolute() else ROOT / args.source
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    rows = run_open_benchmark(args)
    _write_open_csv(rows, args.output)
    summary = args.output.with_name(args.output.stem + "_summary.md")
    _write_summary(rows, args.output, summary, args.source)
    print(f"WROTE {args.output}", flush=True)
    print(f"WROTE {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
