from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUITE_COLORS = {
    "Open120": "#2563eb",
    "BF": "#16a34a",
    "CV": "#f97316",
}

SOLVED_TEXT_COLOR = "#0f172a"
MISSING_COLOR = "#e5e7eb"


def _numbers(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", text)]


def parse_instance(path_value: Any) -> dict[str, float]:
    path = Path(str(path_value))
    lines = path.read_text(errors="ignore").splitlines()
    if not lines:
        return {
            "stack_count": np.nan,
            "max_height": np.nan,
            "container_count": np.nan,
            "density": np.nan,
        }

    width = None
    height = None
    declared_containers = None
    stacks: list[list[int]] = []

    if any(line.startswith("Width") for line in lines[:8]):
        for line in lines:
            if line.startswith("Width"):
                width = _numbers(line)[0]
            elif line.startswith("Height"):
                height = _numbers(line)[0]
            elif line.startswith("Containers"):
                declared_containers = _numbers(line)[0]
            elif line.startswith("Stack"):
                payload = line.split(":", 1)[1] if ":" in line else ""
                stacks.append(_numbers(payload))
    else:
        header = _numbers(lines[0])
        if header:
            width = header[0]
        if len(header) >= 2:
            declared_containers = header[1]
        for line in lines[1:]:
            nums = _numbers(line)
            if not nums:
                continue
            count = max(0, nums[0])
            stacks.append(nums[1 : 1 + count])
        if height is None and stacks:
            height = max((len(stack) for stack in stacks), default=0)

    stack_count = width or len(stacks)
    if height is None:
        height = max((len(stack) for stack in stacks), default=0)
    container_count = declared_containers or sum(len(stack) for stack in stacks)
    capacity = stack_count * height if stack_count and height else np.nan
    density = container_count / capacity if capacity and not math.isnan(capacity) else np.nan
    non_empty_ratio = (
        sum(1 for stack in stacks if stack) / stack_count if stack_count else np.nan
    )
    max_height_ratio = (
        max((len(stack) for stack in stacks), default=0) / height if height else np.nan
    )
    avg_height = container_count / stack_count if stack_count else np.nan

    return {
        "stack_count": float(stack_count),
        "max_height": float(height),
        "container_count": float(container_count),
        "capacity": float(capacity),
        "density": float(density),
        "non_empty_ratio": float(non_empty_ratio),
        "max_height_ratio": float(max_height_ratio),
        "avg_height": float(avg_height),
    }


def suite_label(row: pd.Series) -> str:
    dataset = str(row.get("dataset", ""))
    if dataset == "DLTS":
        return "Open120"
    if dataset in {"BF", "CV"}:
        return dataset
    return dataset or str(row.get("family", "unknown"))


def read_results(paths: list[Path], mode: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        frame["result_mode"] = mode
        frame["input_csv"] = str(path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True).copy()
    df["suite"] = df.apply(suite_label, axis=1)
    df["case_key"] = (
        df["dataset"].astype(str)
        + "|"
        + df["family"].astype(str)
        + "|"
        + df["case"].astype(str)
    )
    return df


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "ok"}


def _valid_solution_row(row: pd.Series) -> bool:
    try:
        final_n_ok = float(row.get("final_n", 1)) == 0.0
    except (TypeError, ValueError):
        final_n_ok = False
    goal_value = row.get("goal_validation_ok", True)
    goal_ok = True if pd.isna(goal_value) else _truthy(goal_value)
    try:
        moves_ok = not pd.isna(row.get("moves")) and str(row.get("moves")) != ""
    except TypeError:
        moves_ok = False
    return final_n_ok and goal_ok and moves_ok


def synthesize_stable_all_solved(
    repaired: pd.DataFrame,
    beam: pd.DataFrame,
    patches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    beam_by_key = {row["case_key"]: row for _, row in beam.iterrows()}
    patch_by_key = {}
    if patches is not None and not patches.empty:
        patch_by_key = {row["case_key"]: row for _, row in patches.iterrows()}

    rows: list[pd.Series] = []
    for _, repaired_row in repaired.iterrows():
        selected = repaired_row.copy()
        selected["stable_selected_source"] = "repaired"
        if not _valid_solution_row(repaired_row):
            key = repaired_row["case_key"]
            beam_row = beam_by_key.get(key)
            patch_row = patch_by_key.get(key)
            if beam_row is not None and _valid_solution_row(beam_row):
                selected = beam_row.copy()
                selected["stable_selected_source"] = "beam_existing"
            elif patch_row is not None and _valid_solution_row(patch_row):
                selected = patch_row.copy()
                selected["stable_selected_source"] = "single_case_guard_patch"
            else:
                selected["stable_selected_source"] = "unresolved"
        rows.append(selected)
    stable = pd.DataFrame(rows).reset_index(drop=True)
    stable["result_mode"] = "v4_stable_all_solved"
    invalid = stable[~stable.apply(_valid_solution_row, axis=1)]
    if not invalid.empty:
        names = ", ".join(invalid["case_key"].astype(str).head(8))
        raise RuntimeError(f"stable all-solved synthesis still has {len(invalid)} invalid rows: {names}")
    return stable


def add_hardness_columns(df: pd.DataFrame) -> pd.DataFrame:
    parsed = [parse_instance(path) for path in df["source_file"]]
    layout = pd.DataFrame(parsed)
    enriched = pd.concat([df.reset_index(drop=True), layout], axis=1)

    enriched["container_count"] = enriched["container_count"].replace(0, np.nan)
    enriched["bad_ratio"] = enriched["n0"].astype(float) / enriched["container_count"]
    enriched["scarcity_risk"] = enriched["initial_scarcity_risk"].astype(float)
    enriched["size_norm"] = np.log1p(enriched["container_count"]) / np.log1p(
        enriched["container_count"].max()
    )
    enriched["hardness_index"] = 100.0 * (
        0.30 * enriched["size_norm"].clip(0, 1)
        + 0.25 * enriched["density"].clip(0, 1)
        + 0.30 * enriched["bad_ratio"].clip(0, 1)
        + 0.15 * enriched["scarcity_risk"].clip(0, 1)
    )
    enriched["solved"] = enriched["final_n"].fillna(1).astype(float).eq(0)
    enriched["moves_per_container"] = (
        enriched["moves"].astype(float) / enriched["container_count"]
    )
    enriched["solution_quality_moves_per_container"] = enriched["moves_per_container"]
    enriched["quality_penalty_per_container"] = (
        enriched["full_quality_penalty"].astype(float) / enriched["container_count"]
    )
    return enriched


def plot_scatter(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.2, 7.4), dpi=180)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    for suite, color in SUITE_COLORS.items():
        subset = df[df["suite"].eq(suite)]
        ax.scatter(
            subset["hardness_index"],
            subset["solution_quality_moves_per_container"],
            s=34,
            alpha=0.68,
            color=color,
            label=f"{suite} ({len(subset)})",
            linewidth=0.35,
            edgecolor="#0f172a",
        )

    failures = df[~df["solved"]].copy()
    if not failures.empty:
        ax.scatter(
            failures["hardness_index"],
            failures["solution_quality_moves_per_container"],
            s=96,
            marker="x",
            color="#dc2626",
            linewidth=2.0,
            label=f"unsolved ({len(failures)})",
            zorder=5,
        )
        for _, row in failures.head(8).iterrows():
            label = f"{row['suite']} {row['family']}/{row['case']}"
            ax.annotate(
                label,
                (row["hardness_index"], row["solution_quality_moves_per_container"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=7.5,
                color="#7f1d1d",
            )

    ax.set_title(
        "Hardness vs. relocation quality across all 1,681 solved cases",
        fontsize=15,
        weight="bold",
        pad=14,
        color="#0f172a",
    )
    ax.set_xlabel("Hardness index (0-100, layout-only)", fontsize=11, color="#334155")
    ax.set_ylabel(
        "Relocations per container (lower is better)",
        fontsize=11,
        color="#334155",
    )
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.tick_params(colors="#475569")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1")

    total = len(df)
    solved = int(df["solved"].sum())
    note = (
        f"All cases: {solved}/{total} solved\n"
        "Hardness = 0.30 size + 0.25 density +\n"
        "           0.30 bad ratio + 0.15 scarcity"
    )
    ax.text(
        0.985,
        0.965,
        note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


HARDNESS_COMPONENTS = [
    (
        "size_norm",
        "Size score",
        "size_norm = log1p(container_count) / corpus max",
    ),
    ("density", "Density", "container_count / capacity"),
    ("bad_ratio", "Initial bad ratio", "n0 / container_count"),
    ("scarcity_risk", "Initial scarcity risk", "solver scarcity signal"),
]


def _draw_component_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    x_col: str,
    label: str,
    subtitle: str,
    show_legend: bool,
) -> None:
    ax.set_facecolor("#ffffff")
    for suite, color in SUITE_COLORS.items():
        subset = df[df["suite"].eq(suite)]
        ax.scatter(
            subset[x_col],
            subset["solution_quality_moves_per_container"],
            s=22,
            alpha=0.62,
            color=color,
            label=f"{suite} ({len(subset)})",
            linewidth=0.25,
            edgecolor="#0f172a",
        )
    ax.set_title(label, fontsize=12.5, weight="bold", color="#0f172a", pad=8)
    ax.set_xlabel(subtitle, fontsize=9.2, color="#334155")
    ax.set_ylabel("Relocations per container", fontsize=9.2, color="#334155")
    ax.grid(True, color="#e2e8f0", linewidth=0.7)
    ax.tick_params(colors="#475569", labelsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    if show_legend:
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8)


def plot_hardness_component_charts(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for x_col, label, subtitle in HARDNESS_COMPONENTS:
        fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=180)
        fig.patch.set_facecolor("#f8fafc")
        _draw_component_scatter(
            ax,
            df,
            x_col=x_col,
            label=f"{label} vs. relocation quality",
            subtitle=subtitle,
            show_legend=True,
        )
        fig.tight_layout()
        output_path = output_dir / f"hardness_component_{x_col}.png"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        paths[x_col] = str(output_path)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8), dpi=180)
    fig.patch.set_facecolor("#f8fafc")
    for ax, (x_col, label, subtitle) in zip(axes.ravel(), HARDNESS_COMPONENTS):
        _draw_component_scatter(
            ax,
            df,
            x_col=x_col,
            label=label,
            subtitle=subtitle,
            show_legend=(x_col == "size_norm"),
        )
    fig.suptitle(
        "Hardness components explain different parts of path-quality pressure",
        fontsize=15,
        weight="bold",
        color="#0f172a",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    grid_path = output_dir / "hardness_component_scatter_grid.png"
    fig.savefig(grid_path, bbox_inches="tight")
    plt.close(fig)
    paths["grid"] = str(grid_path)

    all_metrics = [
        (
            "hardness_index",
            "Total hardness index",
            "0-100 weighted score",
        ),
        *HARDNESS_COMPONENTS,
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), dpi=180)
    fig.patch.set_facecolor("#f8fafc")
    for ax, (x_col, label, subtitle) in zip(axes.ravel(), all_metrics):
        _draw_component_scatter(
            ax,
            df,
            x_col=x_col,
            label=label,
            subtitle=subtitle,
            show_legend=(x_col == "hardness_index"),
        )
        if x_col == "hardness_index":
            ax.set_xlim(0, 100)
    weight_ax = axes.ravel()[-1]
    weight_ax.set_facecolor("#ffffff")
    weight_labels = ["size", "density", "bad ratio", "scarcity"]
    weight_values = [0.30, 0.25, 0.30, 0.15]
    weight_colors = ["#64748b", "#94a3b8", "#475569", "#cbd5e1"]
    bars = weight_ax.barh(weight_labels, weight_values, color=weight_colors)
    weight_ax.invert_yaxis()
    weight_ax.set_xlim(0, 0.35)
    weight_ax.set_title("Index weights", fontsize=12.5, weight="bold", color="#0f172a", pad=8)
    weight_ax.set_xlabel("weight in total hardness", fontsize=9.2, color="#334155")
    weight_ax.tick_params(colors="#475569", labelsize=8.5)
    weight_ax.grid(True, axis="x", color="#e2e8f0", linewidth=0.7)
    weight_ax.spines[["top", "right", "left"]].set_visible(False)
    weight_ax.spines["bottom"].set_color("#cbd5e1")
    for bar, value in zip(bars, weight_values):
        weight_ax.text(
            value + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#334155",
        )
    fig.suptitle(
        "Hardness view: total score plus size, density, bad-ratio, and scarcity components",
        fontsize=15,
        weight="bold",
        color="#0f172a",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    all_metrics_path = output_dir / "hardness_all_metrics_grid.png"
    fig.savefig(all_metrics_path, bbox_inches="tight")
    plt.close(fig)
    paths["all_metrics_grid"] = str(all_metrics_path)
    return paths


def _infer_paper_dataset(family: object) -> str:
    value = str(family)
    return "CV" if re.match(r"^\d+-\d+$", value) else "BF"


def _coverage_entry(
    solved: int | None,
    total: int | None,
    *,
    moves_sum: int | None = None,
    moves_label: str | None = None,
    missing_label: str = "not run full",
) -> dict[str, object]:
    if solved is None or total is None or total == 0:
        return {"rate": np.nan, "label": missing_label, "moves": np.nan, "moves_label": ""}
    if moves_label is None:
        moves_label = f"moves {moves_sum}" if moves_sum is not None else ""
    return {"rate": solved / total, "label": f"{solved}/{total}", "moves": moves_sum, "moves_label": moves_label}


def _reference_solved_mask(frame: pd.DataFrame) -> pd.Series:
    final_n = pd.to_numeric(frame.get("final_n"), errors="coerce")
    if "goal_validation_ok" in frame:
        goal_ok = frame["goal_validation_ok"].astype(str).str.lower().isin(
            ["true", "1", "yes", "ok"]
        )
    else:
        goal_ok = pd.Series(True, index=frame.index)
    if "moves" in frame:
        moves = frame["moves"]
        moves_ok = moves.notna() & moves.astype(str).str.len().gt(0)
    else:
        moves_ok = pd.Series(False, index=frame.index)
    return final_n.eq(0) & goal_ok & moves_ok


def _coverage_from_result_csv(
    path: Path | None,
    *,
    invalid_label: str,
    missing_label: str = "not run full",
) -> dict[str, object]:
    if not path or not path.exists():
        return _coverage_entry(None, None, missing_label=missing_label)
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        return _coverage_entry(None, None, missing_label="empty run")
    if "status" in frame and frame["status"].astype(str).eq("ERROR").all():
        return _coverage_entry(None, None, missing_label=invalid_label)
    solved = _reference_solved_mask(frame)
    moves_sum = int(pd.to_numeric(frame.loc[solved, "moves"], errors="coerce").sum())
    return _coverage_entry(int(solved.sum()), len(frame), moves_sum=moves_sum)


def _real_entry_from_wide_csv(
    frame: pd.DataFrame,
    *,
    final_n_column: str,
    moves_column: str,
) -> dict[str, object]:
    if final_n_column not in frame or moves_column not in frame:
        return _coverage_entry(None, None, missing_label="not included")
    moves = pd.to_numeric(frame[moves_column], errors="coerce")
    solved = pd.to_numeric(frame[final_n_column], errors="coerce").eq(0) & moves.notna()
    return _coverage_entry(int(solved.sum()), len(frame), moves_sum=int(moves.loc[solved].sum()))


def build_reference_coverage(
    stable: pd.DataFrame,
    *,
    polish_quality: pd.DataFrame | None = None,
    bs_frg_open_csv: Path | None = None,
    paper_pure_csv: Path | None = None,
    bs_frg_real_csv: Path | None = None,
    wang_bs_b_open_csv: Path | None = None,
    wang_bs_b_paper_csv: Path | None = None,
    wang_bs_b_real_csv: Path | None = None,
    open_classic_csv: Path | None = None,
    paper_classic_csv: Path | None = None,
    astar_reference_csv: Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_row(algorithm: str, open_entry: dict[str, object], paper_entry: dict[str, object], real_entry: dict[str, object], note: str) -> None:
        rows.append(
            {
                "algorithm": algorithm,
                "Open120_rate": open_entry["rate"],
                "Open120_label": open_entry["label"],
                "Open120_moves": open_entry["moves"],
                "Open120_moves_label": open_entry["moves_label"],
                "Paper BF/CV_rate": paper_entry["rate"],
                "Paper BF/CV_label": paper_entry["label"],
                "Paper BF/CV_moves": paper_entry["moves"],
                "Paper BF/CV_moves_label": paper_entry["moves_label"],
                "Real30_rate": real_entry["rate"],
                "Real30_label": real_entry["label"],
                "Real30_moves": real_entry["moves"],
                "Real30_moves_label": real_entry["moves_label"],
                "note": note,
            }
        )

    open_stable = stable[stable["suite"].eq("Open120")]
    paper_stable = stable[stable["suite"].isin(["BF", "CV"])]
    open_quality = None
    paper_quality = None
    if polish_quality is not None:
        open_quality = polish_quality[polish_quality["suite"].eq("Open120")]
        paper_quality = polish_quality[polish_quality["suite"].isin(["BF", "CV"])]

    def stable_entry_with_quality_moves(
        coverage_subset: pd.DataFrame,
        quality_subset: pd.DataFrame | None,
    ) -> dict[str, object]:
        coverage_solved = int(coverage_subset["solved"].sum())
        total = len(coverage_subset)
        if quality_subset is None or quality_subset.empty:
            return _coverage_entry(
                coverage_solved,
                total,
                moves_sum=int(coverage_subset.loc[coverage_subset["solved"], "moves"].sum()),
            )
        quality_solved = quality_subset[quality_subset["solved"]]
        moves_sum = int(quality_solved["moves"].sum())
        label = f"polish moves {moves_sum}"
        return _coverage_entry(
            coverage_solved,
            total,
            moves_sum=moves_sum,
            moves_label=label,
        )

    stable_real_entry = _coverage_entry(None, None, missing_label="not included")
    if astar_reference_csv and astar_reference_csv.exists():
        real = pd.read_csv(astar_reference_csv, low_memory=False)
        stable_real_entry = _real_entry_from_wide_csv(
            real,
            final_n_column="qubo_enhanced_final_n",
            moves_column="qubo_enhanced_moves",
        )
    add_row(
        "V4.0 stable",
        stable_entry_with_quality_moves(open_stable, open_quality),
        stable_entry_with_quality_moves(paper_stable, paper_quality),
        stable_real_entry,
        "coverage uses stable all-solved; Open/Paper move totals use v4.0 polish/repaired mainline rows",
    )

    if (bs_frg_open_csv and bs_frg_open_csv.exists()) or (paper_pure_csv and paper_pure_csv.exists()):
        bs_frg_open_entry = _coverage_from_result_csv(
            bs_frg_open_csv,
            invalid_label="run invalid",
        )
        bs_frg_paper_entry = _coverage_from_result_csv(
            paper_pure_csv,
            invalid_label="run invalid",
        )
        bs_frg_real_entry = _coverage_from_result_csv(
            bs_frg_real_csv,
            invalid_label="run invalid",
            missing_label="not ported",
        )
        add_row(
            "BS-FRG pure",
            bs_frg_open_entry,
            bs_frg_paper_entry,
            bs_frg_real_entry,
            "paper-style pure run, no QUBO mainline; Open/Paper/Real30 t20",
        )

    if (
        (wang_bs_b_open_csv and wang_bs_b_open_csv.exists())
        or (wang_bs_b_paper_csv and wang_bs_b_paper_csv.exists())
        or (wang_bs_b_real_csv and wang_bs_b_real_csv.exists())
    ):
        add_row(
            "Wang BS-B paper",
            _coverage_from_result_csv(
                wang_bs_b_open_csv,
                invalid_label="run invalid",
            ),
            _coverage_from_result_csv(
                wang_bs_b_paper_csv,
                invalid_label="run invalid",
            ),
            _coverage_from_result_csv(
                wang_bs_b_real_csv,
                invalid_label="run invalid",
                missing_label="not run",
            ),
            "adapted Wang-Jin-Lim BS-B baby-move beam baseline",
        )

    if open_classic_csv and open_classic_csv.exists():
        open_entry = _coverage_from_result_csv(
            open_classic_csv,
            invalid_label="run invalid",
        )
        paper_entry = _coverage_from_result_csv(
            paper_classic_csv,
            invalid_label="full too slow",
            missing_label="full too slow",
        )
        real_entry = _coverage_entry(None, None)
        if astar_reference_csv and astar_reference_csv.exists():
            real = pd.read_csv(astar_reference_csv, low_memory=False)
            real_entry = _real_entry_from_wide_csv(
                real,
                final_n_column="baseline_beam_budget_best_final_n",
                moves_column="baseline_beam_budget_best_moves",
            )
        add_row(
            "Classic beam",
            open_entry,
            paper_entry,
            real_entry,
            "no-QUBO beam; Open/Paper t120; Real30 uses budget-best historical control",
        )

    if astar_reference_csv and astar_reference_csv.exists():
        real = pd.read_csv(astar_reference_csv, low_memory=False)
        add_row(
            "A* no-proof",
            _coverage_entry(None, None, missing_label="not scalable"),
            _coverage_entry(None, None, missing_label="not scalable"),
            _real_entry_from_wide_csv(
                real,
                final_n_column="astar_noproof_final_n",
                moves_column="astar_noproof_moves",
            ),
            "exact-search reference on 30 real single-bay cases only",
        )

    return pd.DataFrame(rows)


def plot_reference_coverage_matrix(coverage: pd.DataFrame, output_path: Path) -> None:
    scopes = ["Open120", "Paper BF/CV", "Real30"]
    matrix = coverage[[f"{scope}_rate" for scope in scopes]].to_numpy(dtype=float)
    cell_font_size = 8.2 if len(coverage) >= 5 else 9.0

    fig, ax = plt.subplots(figsize=(9.4, 5.8), dpi=180)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    masked = np.ma.masked_invalid(matrix)
    cmap = plt.colormaps["YlGn"].copy()
    cmap.set_bad(MISSING_COLOR)
    im = ax.imshow(masked, cmap=cmap, vmin=0.70, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(scopes)))
    ax.set_xticklabels(scopes, fontsize=10, color="#334155")
    ax.set_yticks(range(len(coverage)))
    ax.set_yticklabels(coverage["algorithm"], fontsize=10, color="#334155")
    ax.tick_params(length=0)
    ax.set_title(
        "Coverage and path length show stability and solution quality",
        fontsize=13.5,
        weight="bold",
        color="#0f172a",
        pad=12,
    )

    for i, row in coverage.iterrows():
        for j, scope in enumerate(scopes):
            label = row[f"{scope}_label"]
            value = row[f"{scope}_rate"]
            if pd.isna(value):
                text = "\n".join(str(label).split())
                color = "#64748b"
            else:
                moves_label = row.get(f"{scope}_moves_label", "")
                text = f"{label}\n{value:.1%}"
                if moves_label:
                    text = f"{text}\n{moves_label}"
                color = SOLVED_TEXT_COLOR
            ax.text(j, i, text, ha="center", va="center", fontsize=cell_font_size, color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(scopes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(coverage), 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("solve rate; cell text also shows solved moves", color="#334155", fontsize=9)
    cbar.ax.tick_params(labelsize=8, colors="#475569")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def paired_advantage(beam: pd.DataFrame, repaired: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "case_key",
        "suite",
        "family",
        "case",
        "moves",
        "solved",
        "hardness_index",
        "solution_quality_moves_per_container",
    ]
    merged = beam[cols].merge(
        repaired[cols],
        on="case_key",
        suffixes=("_beam", "_repaired"),
        how="inner",
    )
    merged["suite"] = merged["suite_repaired"]
    merged["both_solved"] = merged["solved_beam"] & merged["solved_repaired"]
    merged["moves_saved"] = merged["moves_beam"].astype(float) - merged[
        "moves_repaired"
    ].astype(float)
    return merged


def plot_advantage(paired: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows = []
    for suite in ["Open120", "BF", "CV"]:
        subset = paired[paired["suite"].eq(suite) & paired["both_solved"]]
        if subset.empty:
            continue
        rows.append(
            {
                "suite": suite,
                "both_solved_cases": int(len(subset)),
                "total_moves_saved": float(subset["moves_saved"].sum()),
                "median_moves_saved": float(subset["moves_saved"].median()),
                "improved_share": float((subset["moves_saved"] > 0).mean()),
                "worsened_share": float((subset["moves_saved"] < 0).mean()),
            }
        )
    summary = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9.4, 5.8), dpi=180)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")
    colors = [SUITE_COLORS.get(s, "#475569") for s in summary["suite"]]
    bars = ax.barh(summary["suite"], summary["total_moves_saved"], color=colors, alpha=0.88)
    ax.axvline(0, color="#0f172a", linewidth=1)

    for bar, (_, row) in zip(bars, summary.iterrows()):
        width = bar.get_width()
        label = (
            f"{width:,.0f} moves saved\n"
            f"{row['improved_share']:.0%} improved, n={int(row['both_solved_cases'])}"
        )
        x = width + (70 if width >= 0 else -70)
        ha = "left" if width >= 0 else "right"
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha=ha,
            fontsize=9,
            color="#334155",
        )

    ax.set_title(
        "Repair/polish layer cuts moves where both modes solve",
        fontsize=14,
        weight="bold",
        pad=12,
        color="#0f172a",
    )
    ax.set_xlabel("Total moves saved vs. beam-only", fontsize=10.5, color="#334155")
    ax.grid(True, axis="x", color="#e2e8f0", linewidth=0.8)
    ax.tick_params(colors="#475569")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return summary


def write_summary(
    repaired: pd.DataFrame,
    advantage: pd.DataFrame,
    coverage: pd.DataFrame,
    component_charts: dict[str, str],
    polish_quality: pd.DataFrame | None,
    output_path: Path,
    summary_json_path: Path,
) -> None:
    suite_summary = (
        repaired.groupby("suite")
        .agg(
            cases=("case_key", "count"),
            solved=("solved", "sum"),
            moves=("moves", "sum"),
            full_quality_penalty=("full_quality_penalty", "sum"),
            median_hardness=("hardness_index", "median"),
            p90_hardness=("hardness_index", lambda s: float(s.quantile(0.90))),
            median_relocations_per_container=(
                "solution_quality_moves_per_container",
                "median",
            ),
        )
        .reset_index()
    )
    totals = {
        "cases": int(len(repaired)),
        "solved": int(repaired["solved"].sum()),
        "moves": float(repaired["moves"].sum()),
        "full_quality_penalty": float(repaired["full_quality_penalty"].sum()),
        "median_hardness": float(repaired["hardness_index"].median()),
    }
    polish_quality_totals = None
    if polish_quality is not None:
        polish_solved = polish_quality[polish_quality["solved"]]
        polish_quality_totals = {
            "cases": int(len(polish_quality)),
            "solved": int(polish_quality["solved"].sum()),
            "moves": float(polish_solved["moves"].sum()),
            "full_quality_penalty": float(polish_quality["full_quality_penalty"].sum()),
            "median_hardness": float(polish_quality["hardness_index"].median()),
            "scope": "v4.0 polish/repaired mainline rows before stable guard patch synthesis",
        }
    payload = {
        "hardness_formula": {
            "size_norm": 0.30,
            "density": 0.25,
            "bad_ratio": 0.30,
            "initial_scarcity_risk": 0.15,
            "scale": "0-100",
            "solution_quality_axis": "moves / container_count",
            "raw_solution_quality": "moves",
        },
        "totals": totals,
        "polish_quality_totals": polish_quality_totals,
        "suite_summary": suite_summary.to_dict(orient="records"),
        "advantage_summary": advantage.to_dict(orient="records"),
        "coverage_summary": coverage.to_dict(orient="records"),
        "component_charts": component_charts,
    }
    def json_clean(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): json_clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_clean(item) for item in value]
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    summary_json_path.write_text(json.dumps(json_clean(payload), indent=2), encoding="utf-8")
    coverage_display = coverage.copy()
    for column in [
        "Open120_rate",
        "Open120_moves",
        "Paper BF/CV_rate",
        "Paper BF/CV_moves",
        "Real30_rate",
        "Real30_moves",
    ]:
        coverage_display[column] = coverage_display[column].map(
            lambda value: "" if pd.isna(value) else value
        )

    lines = [
        "# Hardness + Solution Quality Summary",
        "",
        "## Definitions",
        "",
        "- `hardness_index = 100 * (0.30 * size_norm + 0.25 * density + 0.30 * bad_ratio + 0.15 * initial_scarcity_risk)`.",
        "- `size_norm = log1p(container_count) / max(log1p(container_count))` across the analyzed corpus.",
        "- `solution_quality_moves_per_container = moves / container_count`; lower is better for cross-size scatter plots.",
        "- Raw `moves` remains the primary per-instance solution-quality metric when comparing same-case before/after runs.",
        "- `full_quality_penalty` is retained only as a feasibility/unfinished-solution guard, not the quality axis for the all-solved snapshot.",
        "- The V4.0 row separates coverage from path quality: coverage uses the stable all-solved snapshot, while Open/Paper move totals use the polish/repaired mainline rows.",
        "- The reference matrix reports both solved coverage and solved-case move totals; move totals are comparable only within the same benchmark scope.",
        "- Hardness intentionally excludes runtime, moves, and final solution status.",
        "",
        "## Repaired v4.0 Snapshot",
        "",
        suite_summary.to_markdown(index=False),
        "",
        "## V4.0 Polish Quality Totals",
        "",
        json.dumps(json_clean(polish_quality_totals), ensure_ascii=False) if polish_quality_totals else "not available",
        "",
        "## Beam-only vs. Repaired Advantage",
        "",
        advantage.to_markdown(index=False),
        "",
        "## Reference Coverage Matrix",
        "",
        coverage_display.to_markdown(index=False),
        "",
        "## Reading",
        "",
        "The scatter plot separates layout difficulty from solver outcome. Points drifting upward at high hardness are cases where quality pressure remains high.",
        "",
        "The component charts split the hardness index into size, density, initial bad ratio, and scarcity risk. They are diagnostic plots, not replacement scorecards.",
        "",
        "The advantage chart isolates the repair/polish layer on cases solved by both beam-only and repaired modes, so move savings are not inflated by feasibility differences.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-repaired", required=True, type=Path)
    parser.add_argument("--paper-repaired", required=True, type=Path)
    parser.add_argument("--open-beam", required=True, type=Path)
    parser.add_argument("--paper-beam", required=True, type=Path)
    parser.add_argument("--stable-patch", action="append", type=Path, default=[])
    parser.add_argument("--bs-frg-open-reference", type=Path)
    parser.add_argument("--paper-pure-reference", type=Path)
    parser.add_argument("--bs-frg-real-reference", type=Path)
    parser.add_argument("--wang-bs-b-open-reference", type=Path)
    parser.add_argument("--wang-bs-b-paper-reference", type=Path)
    parser.add_argument("--wang-bs-b-real-reference", type=Path)
    parser.add_argument("--open-classic-reference", type=Path)
    parser.add_argument("--paper-classic-reference", type=Path)
    parser.add_argument("--astar-reference", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    repaired_raw = read_results([args.open_repaired, args.paper_repaired], "v4_repaired")
    beam_raw = read_results([args.open_beam, args.paper_beam], "beam_only")
    patch_raw = (
        read_results(args.stable_patch, "single_case_guard_patch")
        if args.stable_patch
        else pd.DataFrame()
    )
    stable_raw = synthesize_stable_all_solved(
        repaired_raw,
        beam_raw,
        patch_raw if not patch_raw.empty else None,
    )
    repaired = add_hardness_columns(stable_raw)
    polish_quality = add_hardness_columns(repaired_raw)
    beam = add_hardness_columns(beam_raw)
    paired = paired_advantage(beam, repaired)

    repaired_csv = args.output_dir / "hardness_quality_all_cases.csv"
    paired_csv = args.output_dir / "beam_vs_repaired_paired_cases.csv"
    repaired.to_csv(repaired_csv, index=False, encoding="utf-8-sig")
    paired.to_csv(paired_csv, index=False, encoding="utf-8-sig")

    scatter_path = args.output_dir / "hardness_quality_scatter.png"
    advantage_path = args.output_dir / "beam_repair_move_savings.png"
    coverage_path = args.output_dir / "reference_coverage_matrix.png"
    plot_scatter(repaired, scatter_path)
    component_charts = plot_hardness_component_charts(repaired, args.output_dir)
    advantage = plot_advantage(paired, advantage_path)
    coverage = build_reference_coverage(
        repaired,
        polish_quality=polish_quality,
        bs_frg_open_csv=args.bs_frg_open_reference,
        paper_pure_csv=args.paper_pure_reference,
        bs_frg_real_csv=args.bs_frg_real_reference,
        wang_bs_b_open_csv=args.wang_bs_b_open_reference,
        wang_bs_b_paper_csv=args.wang_bs_b_paper_reference,
        wang_bs_b_real_csv=args.wang_bs_b_real_reference,
        open_classic_csv=args.open_classic_reference,
        paper_classic_csv=args.paper_classic_reference,
        astar_reference_csv=args.astar_reference,
    )
    coverage.to_csv(args.output_dir / "reference_coverage_summary.csv", index=False, encoding="utf-8-sig")
    plot_reference_coverage_matrix(coverage, coverage_path)

    write_summary(
        repaired,
        advantage,
        coverage,
        component_charts,
        polish_quality,
        args.output_dir / "hardness_quality_summary.md",
        args.output_dir / "hardness_quality_summary.json",
    )

    print(f"wrote {repaired_csv}")
    print(f"wrote {paired_csv}")
    print(f"wrote {scatter_path}")
    for component_path in component_charts.values():
        print(f"wrote {component_path}")
    print(f"wrote {advantage_path}")
    print(f"wrote {coverage_path}")


if __name__ == "__main__":
    main()
