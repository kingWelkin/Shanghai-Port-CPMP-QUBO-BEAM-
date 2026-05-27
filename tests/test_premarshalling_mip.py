from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

pytest.importorskip("gurobipy")

from greedy import (  # noqa: E402
    Move,
    PremarshallingInstance,
    Solve3DIntegerPositionMip,
    Solve4DSlotMip,
    validate_move_sequence,
)


def _sorted_instance() -> PremarshallingInstance:
    return PremarshallingInstance(
        StackCount=2,
        MaxHeight=2,
        ContainerCount=2,
        Stacks=[[2, 1], []],
    )


def _one_move_instance() -> PremarshallingInstance:
    return PremarshallingInstance(
        StackCount=2,
        MaxHeight=2,
        ContainerCount=2,
        Stacks=[[1, 2], []],
    )


@pytest.mark.parametrize("solver", [Solve3DIntegerPositionMip, Solve4DSlotMip])
def test_already_sorted_returns_zero_moves(solver):
    sol = solver(_sorted_instance(), maxSteps=2)

    assert sol.Feasible
    assert sol.Optimal
    assert sol.Objective == pytest.approx(0.0)
    assert sol.Moves == []


@pytest.mark.parametrize("solver", [Solve3DIntegerPositionMip, Solve4DSlotMip])
def test_one_move_instance_returns_valid_single_move(solver):
    sol = solver(_one_move_instance(), maxSteps=3)

    assert sol.Feasible
    assert sol.Optimal
    assert sol.Objective == pytest.approx(1.0)
    assert len(sol.Moves) == 1
    assert sol.Moves[0].Step == 1
    assert sol.Moves[0].Box == 2
    assert sol.Moves[0].FromStack == 1
    assert sol.Moves[0].ToStack == 2
    validate_move_sequence(_one_move_instance(), sol.Moves, expected_objective=sol.Objective)


@pytest.mark.parametrize("solver", [Solve3DIntegerPositionMip, Solve4DSlotMip])
def test_infeasible_with_zero_steps_writes_diagnostics(solver, tmp_path: Path):
    sol = solver(_one_move_instance(), maxSteps=0, diagnosticsDir=str(tmp_path))

    assert not sol.Feasible
    assert not sol.Optimal
    assert sol.Moves == []
    assert sol.Status == "INFEASIBLE"
    assert list(tmp_path.glob("*.lp"))
    assert list(tmp_path.glob("*.ilp"))


def test_3d_and_4d_return_same_optimum_on_small_case():
    inst = PremarshallingInstance(
        StackCount=3,
        MaxHeight=2,
        ContainerCount=3,
        Stacks=[[1, 3], [2], []],
    )

    sol3 = Solve3DIntegerPositionMip(inst, maxSteps=3)
    sol4 = Solve4DSlotMip(inst, maxSteps=3)

    assert sol3.Feasible and sol4.Feasible
    assert sol3.Optimal and sol4.Optimal
    assert sol3.Objective == pytest.approx(sol4.Objective)
    validate_move_sequence(inst, sol3.Moves, expected_objective=sol3.Objective)
    validate_move_sequence(inst, sol4.Moves, expected_objective=sol4.Objective)


@pytest.mark.parametrize(
    "instance",
    [
        PremarshallingInstance(2, 2, 2, [[1, 1], []]),
        PremarshallingInstance(2, 2, 3, [[1, 2], []]),
        PremarshallingInstance(2, 1, 2, [[1, 2], []]),
        PremarshallingInstance(2, 2, 2, [[1, 2]]),
    ],
)
def test_input_validation_rejects_bad_instances(instance):
    with pytest.raises(ValueError):
        Solve4DSlotMip(instance, maxSteps=1)


def test_validator_rejects_non_top_move():
    inst = _sorted_instance()
    bad = [Move(Step=1, Box=2, FromStack=1, ToStack=2, FromLevel=1, ToLevel=1)]

    with pytest.raises(ValueError, match="from level mismatch|not source top"):
        validate_move_sequence(inst, bad)


def test_validator_rejects_unsorted_final_layout():
    with pytest.raises(ValueError, match="final stacks"):
        validate_move_sequence(_one_move_instance(), [])
