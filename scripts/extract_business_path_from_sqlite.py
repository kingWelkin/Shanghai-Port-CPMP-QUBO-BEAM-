"""Extract and validate a physical move path from the exact SQLite search DB."""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from container_reshuffle.features import entropy_limit  # noqa: E402
from greedy.input_loader import load_problem_from_input_txt  # noqa: E402
from run_business_exact_target_search_sqlite import _encode, _split, _stack_n, _successors_with_n  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


def _read_found(path: Path) -> tuple[int, bytes, int | None, bytes | None]:
    text = path.read_text(encoding="utf-8").strip()
    depth = int(re.search(r"depth=(\d+)", text).group(1))  # type: ignore[union-attr]
    code = bytes.fromhex(re.search(r"code=([0-9a-f]+)", text).group(1))  # type: ignore[union-attr]
    parent_depth_match = re.search(r"parent_depth=(\d+)", text)
    parent_match = re.search(r"parent=([0-9a-f]+)", text)
    parent_depth = int(parent_depth_match.group(1)) if parent_depth_match else None
    parent = bytes.fromhex(parent_match.group(1)) if parent_match else None
    return depth, code, parent_depth, parent


def _n_value(code: bytes, stack_count: int, max_height: int) -> int:
    return sum(_stack_n(chunk, max_height) for chunk in _split(code, stack_count, max_height))


def _has_visited(conn: sqlite3.Connection, code: bytes, depth: int) -> bool:
    row = conn.execute("SELECT 1 FROM visited WHERE code = ? AND depth = ?", (sqlite3.Binary(code), depth)).fetchone()
    return row is not None


def _backtrack_codes(
    conn: sqlite3.Connection,
    target_code: bytes,
    target_depth: int,
    parent_code: bytes | None,
    parent_depth: int | None,
    stack_count: int,
    max_height: int,
) -> list[bytes]:
    codes = [target_code]
    current = parent_code if parent_code is not None else target_code
    depth = parent_depth if parent_depth is not None else target_depth
    if parent_code is not None:
        codes.append(parent_code)
    while depth > 0:
        current_n = _n_value(current, stack_count, max_height)
        previous = None
        for pred, _pred_n in _successors_with_n(current, current_n, stack_count, max_height):
            if _has_visited(conn, pred, depth - 1):
                previous = pred
                break
        if previous is None:
            raise RuntimeError(f"no predecessor found for depth {depth}")
        codes.append(previous)
        current = previous
        depth -= 1
    codes.reverse()
    return codes


def _rank_to_original(zone: str, bay: str) -> dict[int, int]:
    path = ROOT / "data" / "input" / f"{zone.upper()}_{bay}_input.txt"
    problem = load_problem_from_input_txt(path)
    placements: list[tuple[int, int, int]] = []
    for stack_idx, row in enumerate(problem.bay.seq):
        for level, seq in enumerate(row):
            if seq is not None:
                placements.append((int(seq), stack_idx, level))
    return {rank: seq for rank, (seq, _stack_idx, _level) in enumerate(sorted(placements), start=1)}


def _stack_label(index: int) -> str:
    return str(index + 1) if index < 9 else "A"


def _future_n(stacks: list[list[int]], stack_count: int, max_height: int) -> int:
    _code, n_value = _encode(stacks, stack_count, max_height)
    return n_value


def _lift_to_physical(
    initial_stacks: list[list[int]],
    codes: list[bytes],
    stack_count: int,
    max_height: int,
) -> list[dict[str, int | str]]:
    stacks = [list(stack) for stack in initial_stacks]
    moves: list[dict[str, int | str]] = []
    current_code, _ = _encode(stacks, stack_count, max_height)
    if current_code != codes[0]:
        raise RuntimeError("initial physical state does not match canonical path start")

    for step, next_code in enumerate(codes[1:], start=1):
        chosen = None
        for i, src in enumerate(stacks):
            if not src:
                continue
            for j, dst in enumerate(stacks):
                if i == j or len(dst) >= max_height:
                    continue
                trial = [list(stack) for stack in stacks]
                box = trial[i].pop()
                from_level = len(src)
                to_level = len(dst) + 1
                trial[j].append(box)
                code, n_after = _encode(trial, stack_count, max_height)
                if code == next_code:
                    chosen = {
                        "step": step,
                        "box_rank": box,
                        "from_stack": i,
                        "to_stack": j,
                        "from_level": from_level,
                        "to_level": to_level,
                        "n_after": n_after,
                        "g_after": step + n_after,
                    }
                    stacks = trial
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError(f"could not lift canonical transition at step {step}")
        moves.append(chosen)
    return moves


def run(zone: str, bay: str, db_path: Path, found_code_path: Path, output_csv: Path) -> int:
    instance = _instance_from_single_bay(zone, bay)
    target_depth, target_code, parent_depth, parent_code = _read_found(found_code_path)
    conn = sqlite3.connect(str(db_path))
    codes = _backtrack_codes(
        conn,
        target_code,
        target_depth,
        parent_code,
        parent_depth,
        instance.stack_count,
        instance.max_height,
    )
    moves = _lift_to_physical(instance.stacks, codes, instance.stack_count, instance.max_height)
    rank_to_original = _rank_to_original(zone, bay)
    cap = entropy_limit(instance, 1.3)
    n0 = _future_n(instance.stacks, instance.stack_count, instance.max_height)
    max_g = n0
    for move in moves:
        move["box_id"] = rank_to_original[int(move["box_rank"])]
        move["from_stack_label"] = _stack_label(int(move["from_stack"]))
        move["to_stack_label"] = _stack_label(int(move["to_stack"]))
        max_g = max(max_g, int(move["g_after"]))
    final_n = int(moves[-1]["n_after"]) if moves else n0

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "step",
            "box_rank",
            "box_id",
            "from_stack_label",
            "to_stack_label",
            "from_level",
            "to_level",
            "n_after",
            "g_after",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for move in moves:
            writer.writerow({name: move[name] for name in fieldnames})

    print(
        f"FINAL moves={len(moves)} initial_N={n0} final_N={final_n} max_G={max_g} cap={cap} "
        f"ok={final_n == 0 and max_g <= cap} output={output_csv}",
        flush=True,
    )
    return 0 if final_n == 0 and max_g <= cap else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default="C1")
    parser.add_argument("--bay", default="26")
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--found-code-path", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    return run(args.zone, args.bay, args.db_path, args.found_code_path, args.output_csv)


if __name__ == "__main__":
    raise SystemExit(main())
