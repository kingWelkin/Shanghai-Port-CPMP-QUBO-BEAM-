"""Instance generation and JSON IO."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

from ..core import Instance, ProblemVariant, validate_instance


def generate_random(instance_id: str, variant: ProblemVariant, stack_count: int, max_height: int, container_count: int, seed: int = 1) -> Instance:
    """Generate a random feasible layout."""

    rng = random.Random(seed)
    boxes = list(range(1, container_count + 1))
    rng.shuffle(boxes)
    stacks: List[List[int]] = [[] for _ in range(stack_count)]
    for box in boxes:
        choices = [i for i, st in enumerate(stacks) if len(st) < max_height]
        stacks[rng.choice(choices)].append(box)
    inst = Instance(instance_id, variant, stack_count, max_height, container_count, stacks)
    validate_instance(inst)
    return inst


def generate_nearly_sorted(instance_id: str, stack_count: int, max_height: int, container_count: int, seed: int = 1) -> Instance:
    """Generate a mostly sorted CPMP instance."""

    stacks = [[] for _ in range(stack_count)]
    for idx, box in enumerate(range(container_count, 0, -1)):
        stacks[idx % stack_count].append(box)
    rng = random.Random(seed)
    non_empty = [s for s in stacks if len(s) >= 2]
    if non_empty:
        st = rng.choice(non_empty)
        st[-1], st[-2] = st[-2], st[-1]
    return Instance(instance_id, ProblemVariant.PREMARSHALLING_CPMP, stack_count, max_height, container_count, stacks)


def generate_hard_blocking(instance_id: str, stack_count: int, max_height: int, container_count: int) -> Instance:
    """Generate a CRP instance with early targets buried."""

    stacks = [[] for _ in range(stack_count)]
    boxes = list(range(1, container_count + 1))
    for box in boxes:
        target_stack = 0 if len(stacks[0]) < max_height else min(range(stack_count), key=lambda i: len(stacks[i]))
        stacks[target_stack].append(box)
    return Instance(instance_id, ProblemVariant.RETRIEVAL_CRP, stack_count, max_height, container_count, stacks)


def generate_near_full(instance_id: str, variant: ProblemVariant, stack_count: int, max_height: int, empty_slots: int = 1, seed: int = 1) -> Instance:
    """Generate a near-full layout."""

    n = stack_count * max_height - empty_slots
    return generate_random(instance_id, variant, stack_count, max_height, n, seed)


def generate_adversarial(instance_id: str, variant: ProblemVariant, stack_count: int, max_height: int) -> Instance:
    """Generate a simple adversarial layout with increasing stacks."""

    n = stack_count * (max_height - 1)
    stacks = [[] for _ in range(stack_count)]
    box = 1
    for i in range(stack_count):
        for _ in range(max_height - 1):
            stacks[i].append(box)
            box += 1
    return Instance(instance_id, variant, stack_count, max_height, n, stacks)


def save_instance_json(instance: Instance, path: str | Path) -> None:
    """Save an instance to JSON."""

    data = {
        "instance_id": instance.instance_id,
        "variant": instance.variant.value,
        "stack_count": instance.stack_count,
        "max_height": instance.max_height,
        "container_count": instance.container_count,
        "stacks": instance.stacks,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_instance_json(path: str | Path) -> Instance:
    """Load an instance from JSON."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    inst = Instance(
        instance_id=data["instance_id"],
        variant=ProblemVariant(data["variant"]),
        stack_count=int(data["stack_count"]),
        max_height=int(data["max_height"]),
        container_count=int(data["container_count"]),
        stacks=[list(map(int, st)) for st in data["stacks"]],
    )
    validate_instance(inst)
    return inst
