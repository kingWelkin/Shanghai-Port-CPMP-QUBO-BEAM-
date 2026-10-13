from pathlib import Path
import itertools
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))

from container_reshuffle.core import Move
from trajectory_subsequence_qubo import SubsequenceModel


def sample(seed, steps=6):
    rng = random.Random(seed)
    stacks = [[5,4], [3,2], [1]]
    actions = []
    for _ in range(steps):
        s, d = rng.choice([(s, d) for s in range(3) if stacks[s]
                          for d in range(3) if s != d and len(stacks[d]) < 3])
        box = stacks[s].pop(); stacks[d].append(box)
        actions.append((box, s, d))
    initial = [list(s) for s in stacks]
    moves = []
    for box, s, d in reversed(actions):
        moves.append(Move(len(moves), box, d, s, len(stacks[d]), len(stacks[s])+1))
        stacks[s].append(stacks[d].pop())
    return initial, moves


def test_interval_constraints_match_exhaustive_identity_replay():
    for seed in range(20):
        stacks, moves = sample(seed)
        model = SubsequenceModel(stacks, 3, moves, max_options=32)
        q = model.compile_qubo()
        best = len(moves)
        for chosen in itertools.product(*model.groups):
            physical = True
            try:
                replayed = model.replay(chosen)
            except ValueError:
                physical = False
            assert model.feasible(chosen) == physical
            assert abs(q.evaluate(model.bits(chosen))-model.energy(chosen)) < 1e-6
            if physical:
                best = min(best, len(replayed))
        selected, meta = model.exact()
        assert meta['optimal']
        assert len(model.replay(selected)) == best


def test_repeated_priority_groups_preserve_box_identity_and_move_levels():
    stacks, moves = sample(91, 10)
    grouped = [[(v+1)//2 for v in stack] for stack in stacks]
    path = [Move(m.step, (m.box+1)//2, m.from_stack, m.to_stack, m.from_level, m.to_level) for m in moves]
    model = SubsequenceModel(grouped, 3, path)
    for blocks in (1, 2):
        selected, meta = model.descend(blocks)
        assert len(model.replay(selected)) <= len(path)
