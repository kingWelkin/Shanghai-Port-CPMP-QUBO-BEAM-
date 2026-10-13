"""Frozen acceptance rules for the fresh 30-case QUBO+Beam benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import exp, isfinite, log
from statistics import median
from typing import Mapping, Sequence


ZONES = tuple(f"{letter}{number}" for letter in "ABC" for number in range(1, 6))
OFFICIAL_CASES = tuple(f"{zone}_{bay}_single" for zone in ZONES for bay in ("01", "26"))
MOVE_CAPS = {"01": 95, "26": 412, "all": 507}
MEASURED_ROUNDS = 5
MIP_TIMEOUT_STATUSES = frozenset({"TIME_LIMIT_WITH_INCUMBENT", "TIME_LIMIT_NO_INCUMBENT"})


@dataclass(frozen=True)
class QuboRoundRow:
    case: str
    round_index: int
    started_at: str
    final_n: int
    moves: int
    legal: bool
    business_ok: bool
    goal_ok: bool
    solver_core_sec: float
    end_to_end_sec: float
    path_sha256: str


@dataclass(frozen=True)
class QuboAcceptance:
    moves_01: int
    moves_26: int
    total_moves: int
    median_core_by_case: dict[str, float]
    median_e2e_by_case: dict[str, float]
    official_row_by_case: dict[str, QuboRoundRow]


@dataclass(frozen=True)
class PureBeamRow:
    case: str
    started_at: str
    beam_width: int
    time_cap_sec: float
    solver_core_sec: float
    end_to_end_sec: float
    status: str
    solved: bool
    final_n: int
    moves: int | None
    legal: bool
    business_ok: bool
    goal_ok: bool
    qubo_calls: int
    path_sha256: str


@dataclass(frozen=True)
class PureBeamAcceptance:
    solved_total: int
    solved_01: int
    solved_26: int
    row_by_case: dict[str, PureBeamRow]


@dataclass(frozen=True)
class SpeedAcceptance:
    per_case_lower: dict[str, float]
    geomean_lower: float
    accepted: bool


def _parse_timestamp(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include timezone: {value!r}")
    return parsed


def _case_bay(case: str) -> str:
    if case.endswith("_01_single"):
        return "01"
    if case.endswith("_26_single"):
        return "26"
    raise ValueError(f"case outside official corpus: {case}")


def _validate_qubo_timing(
    value: object,
    *,
    field: str,
    case: str,
    round_index: int,
) -> None:
    error = f"invalid QUBO timing {field}: {case} round {round_index}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(error)
    try:
        finite = isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value <= 0.0:
        raise ValueError(error)


def validate_qubo_rounds(
    rows: Sequence[QuboRoundRow],
    *,
    frozen_at: str,
) -> QuboAcceptance:
    """Validate freshness, determinism, legality, and the three quality caps."""

    freeze_time = _parse_timestamp(frozen_at)
    grouped: dict[str, list[QuboRoundRow]] = {case: [] for case in OFFICIAL_CASES}
    for row in rows:
        if row.case not in grouped:
            raise ValueError(f"case outside official corpus: {row.case}")
        if _parse_timestamp(row.started_at) < freeze_time:
            raise ValueError(f"row predates freeze: {row.case} round {row.round_index}")
        grouped[row.case].append(row)

    expected_rounds = set(range(1, MEASURED_ROUNDS + 1))
    official: dict[str, QuboRoundRow] = {}
    median_core: dict[str, float] = {}
    median_e2e: dict[str, float] = {}
    for case in OFFICIAL_CASES:
        case_rows = sorted(grouped[case], key=lambda row: row.round_index)
        if {row.round_index for row in case_rows} != expected_rounds or len(case_rows) != MEASURED_ROUNDS:
            raise ValueError(f"expected five measured rounds for {case}")
        for row in case_rows:
            if not row.legal or not row.business_ok or not row.goal_ok or row.final_n != 0:
                raise ValueError(f"invalid QUBO result: {case} round {row.round_index}")
            if row.moves < 0:
                raise ValueError(f"invalid QUBO metrics: {case} round {row.round_index}")
            _validate_qubo_timing(
                row.solver_core_sec,
                field="solver_core_sec",
                case=case,
                round_index=row.round_index,
            )
            _validate_qubo_timing(
                row.end_to_end_sec,
                field="end_to_end_sec",
                case=case,
                round_index=row.round_index,
            )
        if len({row.moves for row in case_rows}) != 1:
            raise ValueError(f"non-deterministic move count: {case}")
        if len({row.path_sha256 for row in case_rows}) != 1:
            raise ValueError(f"non-deterministic path: {case}")
        official[case] = case_rows[0]
        median_core[case] = float(median(row.solver_core_sec for row in case_rows))
        median_e2e[case] = float(median(row.end_to_end_sec for row in case_rows))

    moves_01 = sum(row.moves for case, row in official.items() if _case_bay(case) == "01")
    moves_26 = sum(row.moves for case, row in official.items() if _case_bay(case) == "26")
    total_moves = moves_01 + moves_26
    if moves_01 > MOVE_CAPS["01"]:
        raise ValueError(f"bay 01 move cap exceeded: {moves_01} > {MOVE_CAPS['01']}")
    if moves_26 > MOVE_CAPS["26"]:
        raise ValueError(f"bay 26 move cap exceeded: {moves_26} > {MOVE_CAPS['26']}")
    if total_moves > MOVE_CAPS["all"]:
        raise ValueError(f"combined move cap exceeded: {total_moves} > {MOVE_CAPS['all']}")

    return QuboAcceptance(
        moves_01=moves_01,
        moves_26=moves_26,
        total_moves=total_moves,
        median_core_by_case=median_core,
        median_e2e_by_case=median_e2e,
        official_row_by_case=official,
    )


def validate_pure_beam_rows(
    rows: Sequence[PureBeamRow],
    *,
    expected_widths: Mapping[str, int],
    expected_caps: Mapping[str, float],
) -> PureBeamAcceptance:
    """Validate the one-run, same-width, time-matched QUBO-off ablation."""

    official_cases = set(OFFICIAL_CASES)
    if set(expected_widths) != official_cases or set(expected_caps) != official_cases:
        raise ValueError("pure Beam expectations must match the 30-case corpus")

    by_case: dict[str, PureBeamRow] = {}
    for row in rows:
        if row.case not in official_cases or row.case in by_case:
            raise ValueError(f"duplicate or foreign pure Beam row: {row.case}")
        if row.beam_width != int(expected_widths[row.case]):
            raise ValueError(f"pure Beam beam width drift: {row.case}")
        if abs(row.time_cap_sec - float(expected_caps[row.case])) > 1e-9:
            raise ValueError(f"pure Beam time cap drift: {row.case}")
        if row.qubo_calls != 0:
            raise ValueError(f"pure Beam QUBO-off violation: {row.case}")
        if row.solver_core_sec < 0.0 or row.end_to_end_sec <= 0.0:
            raise ValueError(f"invalid pure Beam timing: {row.case}")
        if row.solved:
            if (
                row.final_n != 0
                or row.moves is None
                or row.moves < 0
                or not row.legal
                or not row.business_ok
                or not row.goal_ok
                or len(row.path_sha256) != 64
            ):
                raise ValueError(f"invalid solved pure Beam row: {row.case}")
        elif row.moves is not None or row.path_sha256:
            raise ValueError(f"unsolved pure Beam row has a claimed path: {row.case}")
        by_case[row.case] = row

    if set(by_case) != official_cases:
        raise ValueError("pure Beam rows do not match the 30-case corpus")

    solved = [row for row in by_case.values() if row.solved]
    return PureBeamAcceptance(
        solved_total=len(solved),
        solved_01=sum(_case_bay(row.case) == "01" for row in solved),
        solved_26=sum(_case_bay(row.case) == "26" for row in solved),
        row_by_case=by_case,
    )


def conservative_speed_gate(
    *,
    q_times: Mapping[str, float],
    mip_rows: Sequence[Mapping[str, object]],
) -> SpeedAcceptance:
    """Compute the conservative geometric-mean exact-MIP speed lower bound."""

    if not q_times:
        raise ValueError("q_times must not be empty")
    if any(float(value) <= 0.0 for value in q_times.values()):
        raise ValueError("QUBO solver-core times must be positive")

    by_case: dict[str, Mapping[str, object]] = {}
    for row in mip_rows:
        case = str(row.get("case", ""))
        if case in by_case:
            raise ValueError(f"duplicate MIP row: {case}")
        by_case[case] = row
    if set(by_case) != set(q_times):
        raise ValueError("MIP rows must match QUBO timing cases exactly")

    per_case: dict[str, float] = {}
    for case, q_time_value in q_times.items():
        q_time = float(q_time_value)
        row = by_case[case]
        status = str(row.get("status", ""))
        if status == "OPTIMAL":
            numerator = float(row.get("solver_core_sec", 0.0))
        elif status in MIP_TIMEOUT_STATUSES:
            numerator = float(row.get("cap_sec", 0.0))
        else:
            raise ValueError(f"invalid MIP status for speed gate: {case} {status}")
        if numerator <= 0.0:
            raise ValueError(f"invalid MIP time for speed gate: {case}")
        per_case[case] = numerator / q_time

    geomean = exp(sum(log(value) for value in per_case.values()) / len(per_case))
    return SpeedAcceptance(
        per_case_lower=per_case,
        geomean_lower=geomean,
        accepted=geomean > 100.0,
    )
