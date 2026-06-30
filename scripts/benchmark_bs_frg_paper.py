"""Run paper-style BS-FRG on the BF and CV benchmark instances."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_classical_cpmp_baselines import FIELDS, row_from_solution  # noqa: E402
from container_reshuffle.core import Instance, ProblemVariant, validate_instance  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from container_reshuffle.solvers.bs_frg import BsFrgSolver  # noqa: E402


@dataclass(frozen=True)
class PaperCase:
    dataset: str
    family: str
    case_id: str
    instance_path: Path


_CV_NAME_RE = re.compile(r"^data(?P<height>\d+)-(?P<stacks>\d+)-(?P<idx>\d+)$")
_NATURAL_RE = re.compile(r"(\d+)")


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in _NATURAL_RE.split(value))


def _cv_name_parts(path: Path) -> tuple[int, int, int] | None:
    match = _CV_NAME_RE.match(path.stem)
    if not match:
        return None
    return (
        int(match.group("height")),
        int(match.group("stacks")),
        int(match.group("idx")),
    )


def load_bf_bay(path: Path) -> Instance:
    """Load a Bortfeldt-Forster ``.bay`` CPMP instance.

    The BF files store stacks bottom-to-top and may use repeated priority-group
    values, so the returned instance is intentionally not strict-ID validated.
    """

    label = path.stem
    width: int | None = None
    height: int | None = None
    containers: int | None = None
    stack_rows: dict[int, list[int]] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        key_lower = key.lower()
        if key_lower == "label":
            label = Path(value).stem if value else path.stem
        elif key_lower == "width":
            width = int(value)
        elif key_lower == "height":
            height = int(value)
        elif key_lower == "containers":
            containers = int(value)
        elif key_lower.startswith("stack"):
            stack_number = int(key.split()[1])
            stack_rows[stack_number - 1] = [int(part) for part in value.split()] if value else []

    if width is None or height is None:
        raise ValueError(f"missing Width/Height in {path}")
    stacks = [stack_rows.get(idx, []) for idx in range(width)]
    observed = sum(len(stack) for stack in stacks)
    container_count = containers if containers is not None else observed
    if observed != container_count:
        raise ValueError(f"{path} declares {container_count} containers but contains {observed}")
    min_priority = min((box for stack in stacks for box in stack), default=1)
    if min_priority <= 0:
        offset = 1 - min_priority
        stacks = [[box + offset for box in stack] for stack in stacks]
    return Instance(
        label,
        ProblemVariant.PREMARSHALLING_CPMP,
        width,
        height,
        container_count,
        stacks,
    )


def load_cv_dat(path: Path) -> Instance:
    """Load a Caserta-Voß ``dataH-S-i.dat`` CPMP instance.

    The original CV format fixes two empty relocation tiers, so the effective
    max height is ``H + 2`` when the filename exposes ``H``.
    """

    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty CV instance: {path}")
    header = [int(part) for part in rows[0].split()]
    if len(header) != 2:
        raise ValueError(f"expected CV header '<stacks> <containers>' in {path}; got {rows[0]!r}")
    stack_count, container_count = header
    name_parts = _cv_name_parts(path)
    if name_parts is not None and name_parts[1] != stack_count:
        raise ValueError(f"{path.name} encodes {name_parts[1]} stacks but header declares {stack_count}")
    stacks: list[list[int]] = []
    for raw_stack in rows[1:]:
        values = [int(part) for part in raw_stack.split()]
        if not values:
            continue
        height = values[0]
        boxes = values[1:]
        if len(boxes) != height:
            raise ValueError(f"stack row in {path} declares height {height} but has {len(boxes)} boxes")
        stacks.append(boxes)
    if len(stacks) != stack_count:
        raise ValueError(f"{path} declares {stack_count} stacks but contains {len(stacks)} rows")
    observed = sum(len(stack) for stack in stacks)
    if observed != container_count:
        raise ValueError(f"{path} declares {container_count} containers but contains {observed}")
    max_height = (name_parts[0] + 2) if name_parts is not None else (max((len(stack) for stack in stacks), default=0) + 2)
    return Instance(
        path.stem,
        ProblemVariant.PREMARSHALLING_CPMP,
        stack_count,
        max_height,
        container_count,
        stacks,
    )


def _bf_data_root(source: Path) -> Path:
    for candidate in (source / "BF" / "BF", source / "BF", source):
        if candidate.exists() and any(candidate.glob("BF*/*.bay")):
            return candidate
    raise FileNotFoundError(f"could not find BF*/.bay files under {source}")


def _cv_data_root(source: Path) -> Path:
    for candidate in (source / "CV" / "CRPTestcases_Caserta", source / "CRPTestcases_Caserta", source / "CV", source):
        if candidate.exists() and any(candidate.glob("data*-*-*.dat")):
            return candidate
    raise FileNotFoundError(f"could not find dataH-S-i.dat files under {source}")


def discover_paper_cases(
    source: Path,
    *,
    dataset: str,
    families: list[str] | None = None,
    limit_per_family: int | None = None,
) -> list[PaperCase]:
    dataset_key = "CV" if dataset.upper() in {"CV", "CVS"} else dataset.upper()
    family_filter = set(families or [])
    cases: list[PaperCase] = []

    if dataset_key == "BF":
        data_root = _bf_data_root(source)
        for family_dir in sorted((path for path in data_root.iterdir() if path.is_dir()), key=lambda path: _natural_key(path.name)):
            family = family_dir.name
            if family_filter and family not in family_filter:
                continue
            files = sorted(family_dir.glob("*.bay"), key=lambda path: _natural_key(path.name))
            for path in files[:limit_per_family]:
                cases.append(PaperCase("BF", family, path.stem, path))
        return cases

    if dataset_key == "CV":
        data_root = _cv_data_root(source)
        grouped: dict[str, list[Path]] = {}
        for path in data_root.glob("data*-*-*.dat"):
            parts = _cv_name_parts(path)
            if parts is None:
                continue
            family = f"{parts[0]}-{parts[1]}"
            if family_filter and family not in family_filter:
                continue
            grouped.setdefault(family, []).append(path)
        for family in sorted(grouped, key=_natural_key):
            for path in sorted(grouped[family], key=lambda item: _natural_key(item.name))[:limit_per_family]:
                cases.append(PaperCase("CV", family, path.stem, path))
        return cases

    raise ValueError(f"unknown dataset: {dataset}")


def _load_case(case: PaperCase) -> Instance:
    if case.dataset == "BF":
        return load_bf_bay(case.instance_path)
    return load_cv_dat(case.instance_path)


def _write_summary(output: Path, rows: list[dict[str, object]], solver_name: str) -> None:
    solved = [
        row
        for row in rows
        if row.get("goal_validation_ok") in (True, "True") and row.get("final_n") == 0 and row.get("moves") != ""
    ]
    moves = sum(int(row["moves"]) for row in solved)
    avg_runtime = sum(float(row["runtime_sec"]) for row in rows) / max(1, len(rows))
    invalid = [row for row in rows if row not in solved]
    lines = [
        "# Paper BS-FRG Benchmark Summary",
        "",
        f"Result CSV: `{output}`",
        "",
        "| solver | solved | moves_sum | avg_runtime_sec |",
        "|---|---:|---:|---:|",
        f"| {solver_name} | {len(solved)}/{len(rows)} | {moves if solved else ''} | {avg_runtime:.3f} |",
        "",
        "## Unsolved",
        "",
    ]
    if invalid:
        lines.extend(f"- `{row['family']}/{row['case']}`: {row['status']}" for row in invalid)
    else:
        lines.append("- none")
    output.with_name(output.stem + "_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "external" / "paper_cpmp")
    parser.add_argument("--dataset", choices=["BF", "CV", "CVS"], default="BF")
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--limit-per-family", type=int)
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--max-depth", type=int, default=1000)
    parser.add_argument("--beam-width", type=int, default=1000)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--engineered", action="store_true", help="run the cached/deduplicated engineering variant")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "experiments" / "bs_frg_paper.csv")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    cases = discover_paper_cases(
        source,
        dataset=args.dataset,
        families=args.families,
        limit_per_family=args.limit_per_family,
    )
    solver = BsFrgSolver(paper_pure=not args.engineered)
    config = SolverConfig(
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        candidate_limit_per_state=args.k,
        time_limit_sec=args.time_limit,
        entropy_alpha=10.0,
    )
    rows: list[dict[str, object]] = []

    for case in cases:
        instance = _load_case(case)
        try:
            validate_instance(instance)
            allow_priority_groups = False
        except ValueError:
            allow_priority_groups = True
        print(f"CASE_START {case.dataset} {case.family}/{case.case_id}", flush=True)
        solution = solver.solve(instance, config)
        row = row_from_solution(
            instance,
            case.family,
            case.case_id,
            solution,
            reference_moves=None,
            allow_priority_groups=allow_priority_groups,
        )
        rows.append(row)
        print(
            f"CASE_DONE {case.family}/{case.case_id} final_n={row['final_n']} moves={row['moves']} "
            f"runtime={float(row['runtime_sec']):.3f}s status={row['status']}",
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(output, rows, solver.name)
    print(f"WROTE {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
