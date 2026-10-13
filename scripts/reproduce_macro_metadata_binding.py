"""Small legal-state witness for a native candidate/macro metadata mismatch."""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'scripts', ROOT / 'src'):
    sys.path.insert(0, str(directory))
import benchmark_compact_beam_qubo_n0 as beam
from research_hardset_qubo_utility import write_json


def encode(stacks):
    return beam._encode(stacks, 3, 3)[0]


def instance(stacks):
    return SimpleNamespace(stacks=stacks, stack_count=3, max_height=3)


def reproduce():
    start = [[1, 3, 2], [4], []]
    target = [[1, 2], [4, 3], []]
    old_route = [start, [[1, 3], [4], [2]], [[1], [4, 3], [2]], target]
    new_parent_route = [start, [[1, 3], [4, 2], []], [[1], [4, 2], [3]],
                        [[1, 2], [4], [3]], [[1, 2, 3], [4], []]]
    old_codes = list(map(encode, old_route))
    new_codes = list(map(encode, new_parent_route))
    target_code, old_parent, new_parent = old_codes[-1], old_codes[0], new_codes[-1]
    old_moves = beam._codes_to_moves_instance(instance(start), old_codes)
    new_parent_moves = beam._codes_to_moves_instance(instance(start), new_codes)
    new_edge = beam._codes_to_moves_instance(instance(new_parent_route[-1]), [new_parent, target_code])
    goal = [[1], [4, 3, 2], []]
    old_tail = beam._codes_to_moves_instance(instance(target),
        [target_code, encode([[1], [4, 3], [2]]), encode(goal)])
    new_tail = beam._codes_to_moves_instance(instance(target), [target_code, encode(goal)])
    parent = {old_parent: None, target_code: old_parent}
    parent.update({b: a for a, b in zip(new_codes, new_codes[1:])})
    n = beam._code_n(target_code, 3, 3)
    candidates = {target_code: (n, len(old_moves), old_parent)}
    args = SimpleNamespace(
        _frg_macro_edge_path_by_code={target_code: tuple(old_codes[1:])},
        _frg_macro_edge_parent_by_code={target_code: old_parent},
        _frg_macro_edge_moves_by_code={target_code: tuple(old_moves)},
        _union_rollout_probe_cache={target_code: beam.UnionRolloutProbeResult(
            feasible=True, final_n=0, moves=len(old_tail), status='old_valid_tail',
            runtime_sec=0., move_path=tuple(old_tail))})
    before = dict(candidates)
    returned = beam._record_macro_chain_witness_candidate(candidates, args,
        parent_code=new_parent, parent_n=beam._code_n(new_parent, 3, 3), parent_depth=len(new_parent_moves),
        layout_path=(SimpleNamespace(stacks=target),), prefix_moves=tuple(new_edge),
        witness_moves=tuple(new_tail), rollout_final_n=0, rollout_feasible=True,
        rollout_status='shorter_valid_tail', rollout_runtime_sec=0., depth_cap=100,
        best_depth={target_code: len(old_moves)}, parent_chain=parent, stack_count=3, max_height=3)
    raw = beam._backtrack_with_macro_edges(parent, target_code, args._frg_macro_edge_path_by_code,
        macro_parent_by_code=args._frg_macro_edge_parent_by_code)
    try:
        beam._codes_to_moves_instance(instance(start), raw)
        lift_error = None
    except RuntimeError as error:
        lift_error = str(error)
    result = dict(returned_existing_candidate=returned == target_code,
        old_prefix_moves=len(old_moves), new_parent_depth=len(new_parent_moves),
        proposed_new_prefix_moves=len(new_parent_moves) + len(new_edge),
        old_tail_moves=len(old_tail), new_tail_moves=len(new_tail),
        candidate_preserved=candidates == before,
        candidate_parent_equals_macro_parent=candidates[target_code][2] == args._frg_macro_edge_parent_by_code[target_code],
        old_parent_equals_macro_parent=old_parent == args._frg_macro_edge_parent_by_code[target_code],
        resulting_raw_path_length=len(raw) - 1, lift_error=lift_error,
        legal_old_prefix=[asdict(m) for m in old_moves], legal_new_parent_prefix=[asdict(m) for m in new_parent_moves],
        legal_new_macro_edge=[asdict(m) for m in new_edge],
        legal_old_tail=[asdict(m) for m in old_tail], legal_new_tail=[asdict(m) for m in new_tail])
    assert returned == target_code and candidates == before
    assert len(old_moves) == 3 and len(new_parent_moves) == 4
    assert len(old_tail) == 2 and len(new_tail) == 1
    assert not result['candidate_parent_equals_macro_parent'] and lift_error
    # Local witness only: retain the existing prefix binding and share the better
    # tail from the identical endpoint. Production code is not patched here.
    args._frg_macro_edge_path_by_code[target_code] = tuple(old_codes[1:])
    args._frg_macro_edge_parent_by_code[target_code] = old_parent
    args._frg_macro_edge_moves_by_code[target_code] = tuple(old_moves)
    restored = beam._backtrack_with_macro_edges(parent, target_code, args._frg_macro_edge_path_by_code,
        macro_parent_by_code=args._frg_macro_edge_parent_by_code)
    restored_moves = beam._codes_to_moves_instance(instance(start), restored)
    complete = [replace(move, step=i) for i, move in enumerate(
        restored_moves + list(args._union_rollout_probe_cache[target_code].move_path))]
    stacks = [list(s) for s in start]
    for move in complete:
        assert move.from_stack != move.to_stack and stacks[move.from_stack][-1] == move.box
        assert len(stacks[move.to_stack]) < 3
        assert len(stacks[move.from_stack]) == move.from_level
        assert len(stacks[move.to_stack]) + 1 == move.to_level
        stacks[move.to_stack].append(stacks[move.from_stack].pop())
    final_n = beam._code_n(encode(stacks), 3, 3)
    assert len(complete) == 4 and final_n == 0
    result.update(local_prefix_preservation_verified=True, restored_prefix_moves=len(restored_moves),
        composed_complete_moves=len(complete), final_n=final_n,
        legal_composed_complete_path=[asdict(move) for move in complete],
        scope='Local synthetic witness; no production or full-corpus repair claim')
    return result


if __name__ == '__main__':
    result = reproduce()
    write_json(ROOT / 'docs/experiments/beam_bound_gate_20261011/macro_metadata_reproducer.json', result)
    print({k: v for k, v in result.items() if not k.startswith('legal_')})
