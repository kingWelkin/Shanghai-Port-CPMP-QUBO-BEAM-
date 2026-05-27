"""Core state engine and validators for CRP and CPMP variants."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Dict, List, Optional, Tuple


class ProblemVariant(Enum):
    """Supported problem variants."""

    RETRIEVAL_CRP = "retrieval_crp"
    PREMARSHALLING_CPMP = "premarshalling_cpmp"


@dataclass
class Instance:
    """Container reshuffling instance with stacks represented bottom -> top."""

    instance_id: str
    variant: ProblemVariant
    stack_count: int
    max_height: int
    container_count: int
    stacks: List[List[int]]


@dataclass
class Move:
    """One relocation move.

    Stack indices are zero-based. Levels are one-based from bottom to top.
    """

    step: int
    box: int
    from_stack: int
    to_stack: int
    from_level: Optional[int] = None
    to_level: Optional[int] = None


@dataclass
class State:
    """Search state, including retrieval progress for CRP."""

    stacks: List[List[int]]
    next_retrieval_id: int = 1
    move_count: int = 0
    moves: List[Move] = field(default_factory=list)
    disorder_current: int = 0
    disorder_total: int = 0
    entropy_max: int = 0


@dataclass
class Solution:
    """Structured solver result and diagnostics."""

    solver_name: str
    feasible: bool
    optimal: bool
    objective: float
    runtime_sec: float
    moves: List[Move]
    status: str

    expanded_states: int = 0
    generated_states: int = 0
    visited_states: int = 0

    qubo_calls: int = 0
    qubo_total_vars: int = 0
    qubo_total_time_sec: float = 0.0
    qubo_repair_count: int = 0
    selected_avg_similarity: Optional[float] = None

    mip_best_bound: Optional[float] = None
    mip_gap: Optional[float] = None
    mip_node_count: Optional[float] = None
    mip_var_count: Optional[int] = None
    mip_constr_count: Optional[int] = None

    entropy_max: Optional[float] = None
    entropy_limit: Optional[float] = None
    entropy_alpha: Optional[float] = None
    disorder_total: Optional[float] = None
    disorder_initial: Optional[float] = None
    disorder_final: Optional[float] = None


def deep_copy_stacks(stacks: List[List[int]]) -> List[List[int]]:
    """Return a deep copy of a bottom-to-top stack layout."""

    return [list(stack) for stack in stacks]


def _coerce_variant(value: ProblemVariant | str) -> ProblemVariant:
    if isinstance(value, ProblemVariant):
        return value
    return ProblemVariant(value)


def validate_instance(instance: Instance) -> None:
    """Validate shape, capacity, and container IDs of an instance."""

    instance.variant = _coerce_variant(instance.variant)
    if instance.stack_count <= 0:
        raise ValueError("stack_count must be positive")
    if instance.max_height <= 0:
        raise ValueError("max_height must be positive")
    if instance.container_count < 0:
        raise ValueError("container_count must be non-negative")
    if len(instance.stacks) != instance.stack_count:
        raise ValueError("len(stacks) must equal stack_count")

    seen: List[int] = []
    for idx, stack in enumerate(instance.stacks):
        if len(stack) > instance.max_height:
            raise ValueError(f"stack {idx} exceeds max_height")
        seen.extend(int(box) for box in stack)

    expected = list(range(1, instance.container_count + 1))
    actual = sorted(seen)
    if actual != expected:
        raise ValueError(f"containers must be exactly 1..{instance.container_count}; got {actual}")


def create_initial_state(instance: Instance) -> State:
    """Create and normalize the initial state."""

    validate_instance(instance)
    state = State(stacks=deep_copy_stacks(instance.stacks))
    if instance.variant is ProblemVariant.RETRIEVAL_CRP:
        return normalize_retrieval_state(instance, state)
    current = future_disorder_count_stacks(state.stacks)
    state.disorder_current = current
    state.disorder_total = current
    state.entropy_max = current
    return state


def is_decreasing_stack(stack: List[int]) -> bool:
    """Return True if a stack is strictly decreasing bottom -> top."""

    return all(stack[i] > stack[i + 1] for i in range(len(stack) - 1))


def future_disorder_count_stacks(stacks: List[List[int]]) -> int:
    """Return business ``N(Future)`` for a bottom-to-top stack layout."""

    total = 0
    for stack in stacks:
        for idx in range(1, len(stack)):
            if stack[idx] > stack[idx - 1]:
                total += len(stack) - idx
                break
    return total


def is_goal(instance: Instance, state: State) -> bool:
    """Return whether a state satisfies the variant-specific terminal condition."""

    if instance.variant is ProblemVariant.RETRIEVAL_CRP:
        return state.next_retrieval_id > instance.container_count
    return all(is_decreasing_stack(stack) for stack in state.stacks)


def generate_legal_moves(instance: Instance, state: State) -> List[Move]:
    """Enumerate all legal top-container relocations from a state."""

    moves: List[Move] = []
    for i, src in enumerate(state.stacks):
        if not src:
            continue
        box = int(src[-1])
        if instance.variant is ProblemVariant.RETRIEVAL_CRP and box == state.next_retrieval_id:
            continue
        for j, dst in enumerate(state.stacks):
            if i == j or len(dst) >= instance.max_height:
                continue
            moves.append(
                Move(
                    step=state.move_count,
                    box=box,
                    from_stack=i,
                    to_stack=j,
                    from_level=len(src),
                    to_level=len(dst) + 1,
                )
            )
    return moves


def apply_move(instance: Instance, state: State, move: Move) -> State:
    """Apply one relocation after checking local feasibility."""

    if move.from_stack == move.to_stack:
        raise ValueError("source and destination must differ")
    if not (0 <= move.from_stack < instance.stack_count):
        raise ValueError("source stack out of range")
    if not (0 <= move.to_stack < instance.stack_count):
        raise ValueError("destination stack out of range")

    stacks = deep_copy_stacks(state.stacks)
    src = stacks[move.from_stack]
    dst = stacks[move.to_stack]
    if not src:
        raise ValueError("source stack is empty")
    if len(dst) >= instance.max_height:
        raise ValueError("destination stack is full")
    if src[-1] != move.box:
        raise ValueError(f"move box {move.box} is not source top {src[-1]}")

    actual_from = len(src)
    actual_to = len(dst) + 1
    if move.from_level is not None and move.from_level != actual_from:
        raise ValueError("from_level does not match state")
    if move.to_level is not None and move.to_level != actual_to:
        raise ValueError("to_level does not match state")

    box = src.pop()
    dst.append(box)
    fixed_move = Move(
        step=state.move_count,
        box=box,
        from_stack=move.from_stack,
        to_stack=move.to_stack,
        from_level=actual_from,
        to_level=actual_to,
    )
    new_state = State(
        stacks=stacks,
        next_retrieval_id=state.next_retrieval_id,
        move_count=state.move_count + 1,
        moves=state.moves + [fixed_move],
    )
    if instance.variant is ProblemVariant.RETRIEVAL_CRP:
        return normalize_retrieval_state(instance, new_state)
    previous_total = state.disorder_total
    previous_entropy = state.entropy_max
    if previous_total == 0 and state.move_count == 0:
        previous_total = future_disorder_count_stacks(state.stacks)
        previous_entropy = previous_total
    current_disorder = future_disorder_count_stacks(stacks)
    new_state.disorder_current = current_disorder
    new_state.disorder_total = previous_total + current_disorder
    new_state.entropy_max = max(previous_entropy, new_state.move_count + current_disorder)
    return new_state


def normalize_retrieval_state(instance: Instance, state: State) -> State:
    """Free-retrieve every currently available target container in CRP order."""

    if instance.variant is not ProblemVariant.RETRIEVAL_CRP:
        return state
    changed = True
    while changed and state.next_retrieval_id <= instance.container_count:
        changed = False
        target = state.next_retrieval_id
        for stack in state.stacks:
            if stack and stack[-1] == target:
                stack.pop()
                state.next_retrieval_id += 1
                changed = True
                break
    return state


def validate_solution(instance: Instance, solution: Solution) -> Tuple[bool, str, List[List[int]]]:
    """Validate a move sequence by independent simulation."""

    try:
        validate_instance(instance)
        state = create_initial_state(instance)
        last_step = -1
        for move in solution.moves:
            if move.step < last_step:
                return False, "move steps are not nondecreasing", state.stacks
            last_step = move.step
            state = apply_move(instance, state, move)
        if not is_goal(instance, state):
            return False, "final state is not a goal", state.stacks
        if isfinite(solution.objective) and len(solution.moves) != int(round(solution.objective)):
            return False, "move count does not equal objective", state.stacks
        return True, "ok", state.stacks
    except Exception as exc:  # noqa: BLE001 - validators should report, not raise.
        return False, str(exc), deep_copy_stacks(instance.stacks)


def hash_state(instance: Instance, state: State, canonical: bool = False) -> Tuple[Any, ...]:
    """Hash a state exactly or with stack-order symmetry canonicalization."""

    stacks_key = tuple(tuple(stack) for stack in state.stacks)
    if canonical:
        stacks_key = tuple(sorted(stacks_key))
    if instance.variant is ProblemVariant.RETRIEVAL_CRP:
        return stacks_key, state.next_retrieval_id
    return (stacks_key,)


def move_to_dict(move: Move) -> Dict[str, Any]:
    """Serialize a move to a JSON-compatible dict."""

    return {
        "step": move.step,
        "box": move.box,
        "from_stack": move.from_stack,
        "to_stack": move.to_stack,
        "from_level": move.from_level,
        "to_level": move.to_level,
    }


def move_from_dict(data: Dict[str, Any]) -> Move:
    """Deserialize a move from a JSON-compatible dict."""

    return Move(
        step=int(data["step"]),
        box=int(data["box"]),
        from_stack=int(data["from_stack"]),
        to_stack=int(data["to_stack"]),
        from_level=data.get("from_level"),
        to_level=data.get("to_level"),
    )
