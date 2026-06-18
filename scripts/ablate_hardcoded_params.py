from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPEN_BENCHMARK = ROOT / "scripts" / "open_cpmp_benchmarks.py"
DEFAULT_BASE_VARIANT = "qubo_scheduler_state_v5_wang_rescue_hardness_routed_polish"
DEFAULT_FAMILIES = (
    "5x5_cv1_test",
    "5x5_cv2_test",
)


@dataclass(frozen=True)
class AblationVariant:
    name: str
    variant: str = DEFAULT_BASE_VARIANT
    extra_args: tuple[str, ...] = ()


def default_ablation_variants(base_variant: str = DEFAULT_BASE_VARIANT) -> tuple[AblationVariant, ...]:
    return (
        AblationVariant("base", base_variant),
        AblationVariant("v5_rescue_polish_baseline", "qubo_scheduler_state_v5_wang_rescue_polish"),
        AblationVariant("risk_only_endpoint", "qubo_scheduler_state_v5_wang_rescue_risk_polish"),
        AblationVariant("light_only_endpoint", "qubo_scheduler_state_v5_wang_rescue_light_polish"),
        AblationVariant("route_threshold_055", base_variant, ("--hardness-route-light-threshold", "0.55")),
        AblationVariant("route_threshold_075", base_variant, ("--hardness-route-light-threshold", "0.75")),
        AblationVariant("route_threshold_099", base_variant, ("--hardness-route-light-threshold", "0.99")),
        AblationVariant("qubo_start_layer_4", base_variant, ("--qubo-start-layer", "4")),
        AblationVariant("qubo_start_layer_8", base_variant, ("--qubo-start-layer", "8")),
        AblationVariant("qubo_start_layer_12", base_variant, ("--qubo-start-layer", "12")),
        AblationVariant("wang_pair_start_layer_4", base_variant, ("--wang-pair-qubo-start-layer", "4")),
        AblationVariant("wang_pair_start_layer_8", base_variant, ("--wang-pair-qubo-start-layer", "8")),
        AblationVariant("wang_pair_start_layer_12", base_variant, ("--wang-pair-qubo-start-layer", "12")),
        AblationVariant(
            "all_qubo_start_layer_8",
            base_variant,
            ("--qubo-start-layer", "8", "--wang-pair-qubo-start-layer", "8"),
        ),
        AblationVariant("wang_pair_focus_8", base_variant, ("--wang-pair-qubo-focus-n", "8")),
        AblationVariant("wang_pair_off", base_variant, ("--no-wang-pair-qubo-enabled",)),
        AblationVariant("post_polish_off", base_variant, ("--no-post-polish-qubo",)),
    )


def _parse_csv_number(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_csv(path: Path, variant_name: str) -> dict[str, object]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    valid = [
        row
        for row in rows
        if str(row.get("business_ok", "")).lower() == "true"
        and str(row.get("goal_validation_ok", "")).lower() == "true"
    ]
    return {
        "ablation": variant_name,
        "cases": len(rows),
        "valid": len(valid),
        "invalid_or_unsolved": len(rows) - len(valid),
        "moves_total": int(sum(_parse_csv_number(row.get("moves")) for row in valid)),
        "full_quality_penalty_total": int(sum(_parse_csv_number(row.get("full_quality_penalty")) for row in rows)),
        "runtime_total": sum(_parse_csv_number(row.get("runtime_sec")) for row in rows),
        "runtime_avg": (sum(_parse_csv_number(row.get("runtime_sec")) for row in rows) / len(rows)) if rows else 0.0,
        "qubo_calls": int(sum(_parse_csv_number(row.get("qubo_calls")) for row in rows)),
        "wang_pair_qubo_calls": int(sum(_parse_csv_number(row.get("wang_pair_qubo_calls")) for row in rows)),
        "group_qubo_calls": int(sum(_parse_csv_number(row.get("group_qubo_calls")) for row in rows)),
        "post_polish_saved": int(sum(_parse_csv_number(row.get("post_polish_moves_saved")) for row in rows)),
        "output": str(path),
    }


def write_summary(rows: list[dict[str, object]], output: Path) -> None:
    fields = [
        "ablation",
        "cases",
        "valid",
        "invalid_or_unsolved",
        "moves_total",
        "full_quality_penalty_total",
        "runtime_total",
        "runtime_avg",
        "qubo_calls",
        "wang_pair_qubo_calls",
        "group_qubo_calls",
        "post_polish_saved",
        "output",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _variant_command(args: argparse.Namespace, ablation: AblationVariant, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(OPEN_BENCHMARK),
        "--source",
        str(args.source),
        "--no-auto-clone",
        "--families",
        *args.families,
        "--limit-per-family",
        str(args.limit_per_family),
        "--variant",
        ablation.variant,
        "--time-limit",
        str(args.time_limit),
        "--alpha",
        str(args.alpha),
        "--jobs",
        str(args.jobs),
        "--output",
        str(output),
    ]
    if args.case_ids:
        command.extend(["--case-ids", args.case_ids])
    command.extend(ablation.extra_args)
    return command


def run_ablation(args: argparse.Namespace) -> list[dict[str, object]]:
    variants = default_ablation_variants(args.base_variant)
    selected = {name.strip() for name in args.only.split(",") if name.strip()}
    if selected:
        variants = tuple(variant for variant in variants if variant.name in selected)
    rows: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ablation in variants:
        output = args.output_dir / f"{args.label}_{ablation.name}.csv"
        command = _variant_command(args, ablation, output)
        print("RUN", ablation.name, " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
            rows.append(summarize_csv(output, ablation.name))
    if not args.dry_run:
        write_summary(rows, args.output_dir / f"{args.label}_summary.csv")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "external" / "DLTS")
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--limit-per-family", type=int, default=20)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--jobs", default="8")
    parser.add_argument("--base-variant", default=DEFAULT_BASE_VARIANT)
    parser.add_argument("--only", default="")
    parser.add_argument("--label", default="hardcoded_ablation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "experiments" / "hardcoded_ablation")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.source = args.source if args.source.is_absolute() else ROOT / args.source
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    run_ablation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
