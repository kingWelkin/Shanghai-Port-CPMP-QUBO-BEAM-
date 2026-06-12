"""Run the current compact QUBO beam on open CPMP benchmark instances.

The first supported source is the DLTS CPMP repository:
https://github.com/ahottung/DLTS
"""

from __future__ import annotations

import argparse
import csv
import math
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
    SOLVER_NAME,
    _row_dict_from_instance_result,
    _validate_instance_result,
    make_compact_args,
    run_compact_beam_instance,
)
from container_reshuffle.core import Instance, Move, ProblemVariant, validate_instance  # noqa: E402
from container_reshuffle.features import future_disorder_count_stacks  # noqa: E402

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


def _priority_values(instance: Instance) -> list[int]:
    return [int(box) for stack in instance.stacks for box in stack]


def requires_priority_group_semantics(instance: Instance) -> bool:
    """Return True when an instance stores repeated priority labels, not unique box IDs."""

    values = _priority_values(instance)
    return sorted(values) != list(range(1, instance.container_count + 1))


def _validate_open_shape(instance: Instance, path: Path) -> None:
    if len(instance.stacks) != instance.stack_count:
        raise ValueError(f"expected {instance.stack_count} stacks in {path}, got {len(instance.stacks)}")
    box_count = 0
    for idx, stack in enumerate(instance.stacks):
        if len(stack) > instance.max_height:
            raise ValueError(f"stack {idx} exceeds max_height in {path}")
        for box in stack:
            if box <= 0:
                raise ValueError(f"priority values must be positive in {path}")
            if box > 255:
                raise ValueError(f"priority value {box} exceeds compact byte limit in {path}")
        box_count += len(stack)
    if box_count != instance.container_count:
        raise ValueError(f"expected {instance.container_count} boxes in {path}, got {box_count}")


def load_dlts_dat(path: Path, max_height: int | None = None, allow_priority_groups: bool = True) -> Instance:
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
    _validate_open_shape(instance, path)
    try:
        validate_instance(instance)
    except ValueError:
        if not allow_priority_groups:
            raise
    return instance


def _priority_group_sorted_stack(stack: list[int]) -> bool:
    return all(stack[idx] >= stack[idx + 1] for idx in range(len(stack) - 1))


def _simulate_priority_group_moves(
    instance: Instance,
    moves: list[Move],
) -> tuple[bool, str, list[list[int]], list[int], int]:
    stacks = [list(stack) for stack in instance.stacks]
    n_trace = [future_disorder_count_stacks(stacks)]
    max_g = n_trace[0]
    last_step = -1

    for depth, move in enumerate(moves, start=1):
        if move.step < last_step:
            return False, "move steps are not nondecreasing", stacks, n_trace, max_g
        last_step = move.step
        if move.from_stack == move.to_stack:
            return False, "source and destination must differ", stacks, n_trace, max_g
        if not (0 <= move.from_stack < instance.stack_count):
            return False, "source stack out of range", stacks, n_trace, max_g
        if not (0 <= move.to_stack < instance.stack_count):
            return False, "destination stack out of range", stacks, n_trace, max_g
        src = stacks[move.from_stack]
        dst = stacks[move.to_stack]
        if not src:
            return False, "source stack is empty", stacks, n_trace, max_g
        if len(dst) >= instance.max_height:
            return False, "destination stack is full", stacks, n_trace, max_g
        if src[-1] != move.box:
            return False, f"move box {move.box} is not source top {src[-1]}", stacks, n_trace, max_g
        if move.from_level is not None and move.from_level != len(src):
            return False, "from_level does not match state", stacks, n_trace, max_g
        if move.to_level is not None and move.to_level != len(dst) + 1:
            return False, "to_level does not match state", stacks, n_trace, max_g

        dst.append(src.pop())
        n_value = future_disorder_count_stacks(stacks)
        n_trace.append(n_value)
        max_g = max(max_g, depth + n_value)

    if not all(_priority_group_sorted_stack(stack) for stack in stacks):
        return False, "final state is not a priority-group goal", stacks, n_trace, max_g
    return True, "ok", stacks, n_trace, max_g


def _priority_group_row_dict_from_instance_result(
    result,
    instance: Instance,
    args: argparse.Namespace,
) -> dict[str, object]:
    n0 = future_disorder_count_stacks(instance.stacks)
    cap13 = int(math.floor(min(1.3, 2.0) * n0 + 1e-9))
    cap15 = int(math.floor(min(1.5, 2.0) * n0 + 1e-9))
    if result.moves:
        _ok, _msg, _stacks, n_trace, max_g = _simulate_priority_group_moves(instance, result.moves)
        final_n = n_trace[-1]
        moves_count = len(result.moves)
    else:
        final_n = n0
        moves_count = None
        max_g = n0
    row = {
        "case": result.case,
        "solver": SOLVER_NAME,
        "n0": n0,
        "cap_13": cap13,
        "cap_15": cap15,
        "moves": moves_count,
        "final_n": final_n,
        "reduction": n0 - final_n,
        "g_end": (moves_count + final_n) if moves_count is not None else None,
        "max_g": max_g,
        "g_ok_13": max_g <= cap13,
        "g_ok_15": max_g <= cap15,
        "runtime_sec": result.runtime_sec,
        "expanded_states": result.expanded,
        "generated_states": result.generated,
        "visited_states": result.visited,
        "qubo_calls": result.qubo_calls,
        "qubo_total_vars": result.qubo_vars,
        "qubo_time_sec": result.qubo_time_sec,
        "status": (
            f"{result.status}; solver={SOLVER_NAME}; representation=compact_bytes; "
            f"open_cpmp_semantics=priority_groups; target_stop=N0; proof=not_optimality_proof"
        ),
        "profile": getattr(args, "solver_profile", "speed_v2"),
        "solver_profile": getattr(args, "solver_profile", "speed_v2"),
        "qubo_influence_profile": getattr(args, "qubo_influence_profile", "stable"),
        "selection_policy": getattr(args, "selection_policy", "enhanced"),
        "priority": args.priority,
        "beam_width": args.beam_width,
        "core_width": args.core_width,
        "parent_core_cap": getattr(args, "parent_core_cap", 0),
        "peak_frontier": result.peak_frontier,
        "peak_candidates": result.peak_candidates,
        "depth_reached": result.depth_reached,
        "open_count": result.open_count,
        "pruned_bound": result.pruned_bound,
        "pruned_business": result.pruned_business,
        "dominated": result.dominated,
        "use_qubo": args.use_qubo,
        "qubo_mode": args.qubo_mode,
        "qubo_focus_n": args.qubo_focus_n,
        "qubo_first": args.qubo_first,
        "max_qubo_vars": args.max_qubo_vars,
        "qubo_core_width": args.qubo_core_width,
        "qubo_supplement_width": args.qubo_supplement_width,
        "qubo_cohorts_per_layer": args.qubo_cohorts_per_layer,
        "qubo_selected_total": result.qubo_selected_total,
        "qubo_selected_on_search_path": result.qubo_selected_on_search_path,
        "qubo_selected_on_path": result.qubo_selected_on_path,
        "qubo_ancestor_on_search_path": result.qubo_ancestor_on_search_path,
        "qubo_ancestor_on_path": result.qubo_ancestor_on_path,
        "group_qubo_supplement": getattr(args, "group_qubo_supplement", False),
        "group_qubo_calls": result.group_qubo_calls,
        "group_qubo_vars": result.group_qubo_vars,
        "group_qubo_time_sec": result.group_qubo_time_sec,
        "group_qubo_selected_total": result.group_qubo_selected_total,
        "qubo_group_max_vars": getattr(args, "qubo_group_max_vars", 0),
        "qubo_group_select": getattr(args, "qubo_group_select", 0),
        "qubo_group_take": getattr(args, "qubo_group_take", 0),
        "post_polish_attempts": result.post_polish_attempts,
        "post_polish_improvements": result.post_polish_improvements,
        "post_polish_moves_saved": result.post_polish_moves_saved,
        "global_reservoir": getattr(args, "global_reservoir", False),
        "reservoir_size": result.reservoir_size,
        "peak_reservoir": result.peak_reservoir,
        "reservoir_added_total": result.reservoir_added_total,
        "reservoir_pulled_total": result.reservoir_pulled_total,
        "stream_candidate_limit": getattr(args, "stream_candidate_limit", 0),
        "stream_candidate_trim_factor": getattr(args, "stream_candidate_trim_factor", 2.0),
        "stream_pruned_candidates": result.stream_pruned_candidates,
        "rank_pool_size": getattr(args, "rank_pool_size", 0),
        "rank_pool_min_candidates": getattr(args, "rank_pool_min_candidates", 0),
        "rank_pool_truncated_candidates": result.rank_pool_truncated_candidates,
        "adaptive_high_n": getattr(args, "adaptive_high_n", 0),
        "adaptive_low_n": getattr(args, "adaptive_low_n", 0),
        "adaptive_high_frontier": getattr(args, "adaptive_high_frontier", 0),
        "adaptive_mid_frontier": getattr(args, "adaptive_mid_frontier", 0),
        "adaptive_low_frontier": getattr(args, "adaptive_low_frontier", 0),
        "adaptive_core_ratio": getattr(args, "adaptive_core_ratio", 0.0),
        "adaptive_candidate_pool": getattr(args, "adaptive_candidate_pool", 0),
    }
    return row


def _open_row_dict_from_instance_result(result, instance: Instance, args: argparse.Namespace) -> dict[str, object]:
    if requires_priority_group_semantics(instance):
        return _priority_group_row_dict_from_instance_result(result, instance, args)
    return _row_dict_from_instance_result(result, instance, args)


def _validate_priority_group_result(
    result,
    instance: Instance,
    business_alpha: float,
) -> tuple[bool, bool, str]:
    if not result.moves:
        return False, False, result.status
    goal_ok, goal_msg, _stacks, _n_trace, max_g = _simulate_priority_group_moves(instance, result.moves)
    cap = int(math.floor(min(float(business_alpha), 2.0) * future_disorder_count_stacks(instance.stacks) + 1e-9))
    business_ok = max_g <= cap
    return business_ok, goal_ok, goal_msg


def _open_validate_instance_result(result, instance: Instance, business_alpha: float) -> tuple[bool, bool, str]:
    if requires_priority_group_semantics(instance):
        return _validate_priority_group_result(result, instance, business_alpha)
    return _validate_instance_result(result, instance, business_alpha=business_alpha)


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
    if name == "both":
        return ["qubo", "classic_noqubo"]
    if name == "all":
        return ["qubo", "qubo_adaptive", "classic_noqubo"]
    return [name]


def _args_for_variant(args: argparse.Namespace, variant: str, instance: Instance | None = None) -> argparse.Namespace:
    overrides: dict[str, object] = {
        "alpha": args.alpha,
        "max_moves": args.max_moves,
        "solver_profile": args.solver_profile,
        "target_final_n": 0,
        "time_limit": args.time_limit,
    }
    if variant == "qubo_adaptive":
        adaptive_high_frontier = args.adaptive_high_frontier
        adaptive_mid_frontier = args.adaptive_mid_frontier
        adaptive_low_frontier = args.adaptive_low_frontier
        if instance is not None and instance.stack_count <= 5:
            adaptive_high_frontier = adaptive_mid_frontier = adaptive_low_frontier = max(
                adaptive_high_frontier,
                adaptive_mid_frontier,
                adaptive_low_frontier,
            )
        overrides.update(
            {
                "adaptive_candidate_pool": args.adaptive_candidate_pool,
                "adaptive_high_frontier": adaptive_high_frontier,
                "adaptive_high_n": args.adaptive_high_n,
                "adaptive_low_frontier": adaptive_low_frontier,
                "adaptive_low_n": args.adaptive_low_n,
                "adaptive_mid_frontier": adaptive_mid_frontier,
                "selection_policy": "adaptive_qubo",
            }
        )
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
            compact_args = _args_for_variant(args, variant, instance)
            print(f"CASE_START {run_idx}/{total_runs} {variant} {case.case_id}", flush=True)
            result = run_compact_beam_instance(case.case_id, instance, compact_args)
            row = _open_row_dict_from_instance_result(result, instance, compact_args)
            business_ok, goal_ok, msg = _open_validate_instance_result(result, instance, business_alpha=args.alpha)
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
    parser.add_argument("--variant", choices=["qubo", "qubo_adaptive", "classic_noqubo", "both", "all"], default="qubo")
    parser.add_argument("--skip-invalid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--solver-profile", choices=["speed_v2"], default="speed_v2")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--no-qubo-beam-width", type=int, default=1200)
    parser.add_argument("--adaptive-high-n", type=int, default=16)
    parser.add_argument("--adaptive-low-n", type=int, default=8)
    parser.add_argument("--adaptive-high-frontier", type=int, default=800)
    parser.add_argument("--adaptive-mid-frontier", type=int, default=1000)
    parser.add_argument("--adaptive-low-frontier", type=int, default=1200)
    parser.add_argument("--adaptive-candidate-pool", type=int, default=0)
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
