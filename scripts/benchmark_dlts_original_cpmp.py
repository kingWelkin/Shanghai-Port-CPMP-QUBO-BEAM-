"""Wrapper for the original DLTS CPMP implementation.

This script intentionally calls ``external/DLTS/deep_learn.py`` instead of
re-implementing the neural tree search.  The goal is a strict same-problem
paper baseline for the container pre-marshalling problem.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


SOLVER_NAME = "dlts_original"
DLTS_FIELDS = [
    "family",
    "case",
    "solver",
    "moves",
    "runtime_sec",
    "nodes",
    "reference_moves",
    "move_gap",
    "status",
]


@dataclass(frozen=True)
class DltsRunConfig:
    dlts_root: Path
    family: str
    case_ids: list[str]
    output_dir: Path
    python_executable: str = sys.executable
    time_limit: float = 60.0
    run_id: int = 170909972
    upper_bound: int = 45
    tree_search_width: float = 0.1
    branch_strategy: str = "log"
    search_strategy: str = "dfs"
    use_value_model: bool = True
    n: int = 3
    param_d: float = 0.95
    param_d2: float = 1.0
    param_e: float = 0.95


def _case_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.stem
    except ValueError:
        return 10**9, path.stem


def _family_size(family: str) -> str:
    return family.split("_", 1)[0]


def select_dlts_models(dlts_root: Path, family: str) -> tuple[str, str]:
    size = _family_size(family)
    if size != "5x5":
        raise ValueError(
            f"original DLTS pre-trained model is not available for {family}; "
            "this checkout only contains pre-trained DLTS model files for 5x5 families"
        )
    policy = dlts_root / "pre-trained_models" / "pm_dnn_model_5x5.h5"
    value = dlts_root / "pre-trained_models" / "pm_dnn_value_model_5x5.h5"
    missing = [str(path) for path in (policy, value) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing original DLTS pre-trained model file(s): {', '.join(missing)}")
    return ("pre-trained_models/pm_dnn_model_5x5.h5", "pre-trained_models/pm_dnn_value_model_5x5.h5")


def discover_case_ids(dlts_root: Path, family: str, limit_per_family: int | None = None) -> list[str]:
    family_dir = dlts_root / "reference_solutions" / family
    if not family_dir.is_dir():
        raise FileNotFoundError(f"DLTS reference family not found: {family_dir}")
    paths = sorted(family_dir.glob("*.out"), key=_case_sort_key)
    if limit_per_family is not None:
        paths = paths[: max(0, limit_per_family)]
    return [path.stem for path in paths]


def read_reference_moves(reference_path: Path) -> int | None:
    if not reference_path.exists():
        return None
    for line in reference_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("opt="):
            return int(line.split("=", 1)[1].strip())
    return None


def reference_moves_by_case(dlts_root: Path, family: str, case_ids: list[str]) -> dict[str, int]:
    refs: dict[str, int] = {}
    for case_id in case_ids:
        moves = read_reference_moves(dlts_root / "reference_solutions" / family / f"{case_id}.out")
        if moves is not None:
            refs[case_id] = moves
    return refs


def build_dlts_command(config: DltsRunConfig) -> list[str]:
    policy_model, value_model = select_dlts_models(config.dlts_root, config.family)
    command = [
        config.python_executable,
        str(config.dlts_root / "deep_learn.py"),
        "-t",
        f"reference_solutions/{config.family}",
        "-m",
        policy_model,
        "-o",
        str(config.output_dir),
        "--ub",
        str(config.upper_bound),
        "-p",
        str(config.tree_search_width),
        "-n",
        str(config.n),
        "--param_d",
        str(config.param_d),
        "--param_d2",
        str(config.param_d2),
        "--param_e",
        str(config.param_e),
        "--branch_strategy",
        config.branch_strategy,
        "--search_strategy",
        config.search_strategy,
        "--val_timeout",
        str(config.time_limit + 0.1),
        "--run_id",
        str(config.run_id),
        "--val_instances",
        ",".join(config.case_ids),
        "--verbose",
        "1",
    ]
    if config.use_value_model:
        command.extend(["-s", "-v", value_model])
    return command


def parse_dlts_result_csv(
    result_path: Path,
    family: str,
    reference_by_case: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    reference_by_case = reference_by_case or {}
    rows: list[dict[str, object]] = []
    with result_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            case_id = str(row["id"])
            moves = int(float(row["moves"]))
            runtime = float(row["time"])
            nodes = int(float(row["nodes"]))
            reference = reference_by_case.get(case_id)
            rows.append(
                {
                    "family": family,
                    "case": f"{family}_{case_id}",
                    "solver": SOLVER_NAME,
                    "moves": moves if moves > 0 else "",
                    "runtime_sec": runtime,
                    "nodes": nodes,
                    "reference_moves": reference if reference is not None else "",
                    "move_gap": (moves - reference) if moves > 0 and reference is not None else "",
                    "status": "SOLVED" if moves > 0 else "UNSOLVED",
                }
            )
    return rows


def run_dlts_family(config: DltsRunConfig) -> tuple[list[dict[str, object]], subprocess.CompletedProcess[str]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "models").mkdir(parents=True, exist_ok=True)
    (config.output_dir / "solutions").mkdir(parents=True, exist_ok=True)
    command = build_dlts_command(config)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(config.dlts_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        return (
            [
                {
                    "family": config.family,
                    "case": f"{config.family}_{case_id}",
                    "solver": SOLVER_NAME,
                    "moves": "",
                    "runtime_sec": time.perf_counter() - started,
                    "nodes": "",
                    "reference_moves": "",
                    "move_gap": "",
                    "status": f"DLTS_RUN_FAILED; returncode={completed.returncode}",
                }
                for case_id in config.case_ids
            ],
            completed,
        )
    result_path = config.output_dir / f"results_{config.run_id}_{config.family}.csv"
    refs = reference_moves_by_case(config.dlts_root, config.family, config.case_ids)
    return parse_dlts_result_csv(result_path, config.family, refs), completed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "external" / "DLTS")
    parser.add_argument("--families", nargs="+", default=["5x5_cv1_test"])
    parser.add_argument("--limit-per-family", type=int, default=5)
    parser.add_argument("--case-ids", type=str, default="")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--time-limit", type=float, default=3.0)
    parser.add_argument("--run-id", type=int, default=170909972)
    parser.add_argument("--ub", type=int, default=45)
    parser.add_argument("--tree-search-width", type=float, default=0.1)
    parser.add_argument("--branch-strategy", default="log")
    parser.add_argument("--search-strategy", choices=["dfs", "lds", "wbs"], default="dfs")
    parser.add_argument("--without-value-model", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "experiments" / "dlts_original_run")
    parser.add_argument("--csv-output", type=Path, default=ROOT / "docs" / "experiments" / "dlts_original_cpmp.csv")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    dlts_root = args.source if args.source.is_absolute() else ROOT / args.source
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    csv_output = args.csv_output if args.csv_output.is_absolute() else ROOT / args.csv_output

    all_rows: list[dict[str, object]] = []
    failures: list[subprocess.CompletedProcess[str]] = []
    explicit_case_ids = [case.strip() for case in args.case_ids.split(",") if case.strip()]
    for family_index, family in enumerate(args.families):
        case_ids = explicit_case_ids or discover_case_ids(dlts_root, family, args.limit_per_family)
        run_id = args.run_id + family_index
        config = DltsRunConfig(
            dlts_root=dlts_root,
            family=family,
            case_ids=case_ids,
            output_dir=output_dir / family,
            python_executable=args.python_executable,
            time_limit=args.time_limit,
            run_id=run_id,
            upper_bound=args.ub,
            tree_search_width=args.tree_search_width,
            branch_strategy=args.branch_strategy,
            search_strategy=args.search_strategy,
            use_value_model=not args.without_value_model,
        )
        print(f"DLTS_START family={family} cases={','.join(case_ids)}", flush=True)
        rows, completed = run_dlts_family(config)
        all_rows.extend(rows)
        if completed.returncode != 0:
            failures.append(completed)
            print(f"DLTS_FAILED family={family} returncode={completed.returncode}", flush=True)
            if completed.stderr:
                print(completed.stderr.strip(), flush=True)
        else:
            solved = sum(1 for row in rows if row["status"] == "SOLVED")
            print(f"DLTS_DONE family={family} solved={solved}/{len(rows)}", flush=True)

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DLTS_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"WROTE {csv_output}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
