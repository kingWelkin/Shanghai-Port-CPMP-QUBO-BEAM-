"""Real legal-prefix regression tests for macro ownership and history recovery."""
from dataclasses import replace
from types import SimpleNamespace
import sys
from pathlib import Path

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark_compact_beam_qubo_n0 as beam
from macro_prefix_binding import MacroPrefixBinding, MacroEdge
from reproduce_macro_metadata_binding import encode, instance


def fixture(same_parent=False):
    start = beam._code_to_stacks(encode([[1, 3, 2], [4], []]), 3, 3)
    goal_state = beam._code_to_stacks(encode([[1, 2], [4, 3], []]), 3, 3)
    old_route = [start, [[1, 3], [4], [2]], [[1], [4, 3], [2]], goal_state]
    incoming_route = [start, [[1, 3], [4, 2], []], [[1], [4, 2], [3]],
                      [[1, 2], [4], [3]], [[1, 2, 3], [4], []]]
    old_codes, incoming_codes = list(map(encode, old_route)), list(map(encode, incoming_route))
    child, parent = old_codes[-1], old_codes[0]
    old_moves = beam._codes_to_moves_instance(instance(start), old_codes)
    target = encode([[1], [4, 3, 2], []])
    old_tail = beam._codes_to_moves_instance(instance(goal_state), [child, encode([[1], [4, 3], [2]]), target])
    new_tail = beam._codes_to_moves_instance(instance(goal_state), [child, target])
    parents = {parent: None, child: parent}
    parents.update({b: a for a, b in zip(incoming_codes, incoming_codes[1:])})
    args = SimpleNamespace(_frg_macro_edge_path_by_code={child: tuple(old_codes[1:])},
        _frg_macro_edge_parent_by_code={child: parent}, _frg_macro_edge_moves_by_code={child: tuple(old_moves)},
        _union_rollout_probe_cache={child: beam.UnionRolloutProbeResult(True, 0, len(old_tail), 'old', 0., tuple(old_tail))})
    candidates = {child: (beam._code_n(child, 3, 3), 3, parent)}
    new_parent, depth = incoming_codes[-1], 4
    layouts = [goal_state]
    prefix = beam._codes_to_moves_instance(instance(beam._code_to_stacks(new_parent, 3, 3)), [new_parent, child])
    if same_parent:
        new_parent, depth = parent, 0
        layouts = [old_route[1], start, *old_route[1:]]
        prefix = beam._codes_to_moves_instance(instance(start), [parent, *map(encode, layouts)])
    kwargs = dict(parent_code=new_parent, parent_n=beam._code_n(new_parent, 3, 3), parent_depth=depth,
        layout_path=tuple(SimpleNamespace(stacks=s) for s in layouts), prefix_moves=tuple(prefix),
        witness_moves=tuple(new_tail), rollout_final_n=0, rollout_feasible=True, rollout_status='better',
        rollout_runtime_sec=0., depth_cap=100, best_depth={child: 3}, parent_chain=parents, stack_count=3, max_height=3)
    return start, child, parent, parents, candidates, args, kwargs, old_codes


@pytest.mark.parametrize('same_parent', [False, True])
def test_better_tail_keeps_the_existing_shorter_prefix(same_parent):
    start, child, parent, parents, candidates, args, kwargs, old_codes = fixture(same_parent)
    fix = MacroPrefixBinding('guard')
    fix.install()
    try:
        result = beam._record_macro_chain_witness_candidate(candidates, args, **kwargs)
        assert result == child and candidates[child][1:] == (3, parent)
        assert args._frg_macro_edge_parent_by_code[child] == parent
        assert args._frg_macro_edge_path_by_code[child] == tuple(old_codes[1:])
        assert args._union_rollout_probe_cache[child].moves == 1
        assert fix.stats['prefix_bindings_preserved'] == 1
        prefix = beam._codes_to_moves_instance(instance(start),
            beam._backtrack_with_macro_edges(parents, child, args._frg_macro_edge_path_by_code,
                macro_parent_by_code=args._frg_macro_edge_parent_by_code))
        moves = [replace(m, step=i) for i, m in enumerate(prefix + list(args._union_rollout_probe_cache[child].move_path))]
        stacks = [list(s) for s in start]
        for m in moves:
            assert stacks[m.from_stack][-1] == m.box and len(stacks[m.to_stack]) < 3
            stacks[m.to_stack].append(stacks[m.from_stack].pop())
        assert len(moves) == 4 and beam._code_n(encode(stacks), 3, 3) == 0
    finally:
        fix.restore()


def test_record_only_then_recover_a_real_overwritten_ancestor_without_search():
    start, child, parent, parents, candidates, args, kwargs, old_codes = fixture()
    history = MacroPrefixBinding('observe')
    history.install()
    try:
        beam._record_macro_chain_witness_candidate(candidates, args, **kwargs)
        assert args._frg_macro_edge_parent_by_code[child] != parent
        with pytest.raises(RuntimeError):
            beam._codes_to_moves_instance(instance(start), beam._backtrack_with_macro_edges(parents, child,
                args._frg_macro_edge_path_by_code, macro_parent_by_code=args._frg_macro_edge_parent_by_code))
        paths, tags, _, restored = history.resolve_maps(beam._backtrack(parents, child),
            args._frg_macro_edge_path_by_code, args._frg_macro_edge_parent_by_code)
        recovered = beam._backtrack_with_macro_edges(parents, child, paths, macro_parent_by_code=tags)
        assert recovered == old_codes and len(restored) == 1
        assert len(beam._codes_to_moves_instance(instance(start), recovered)) == 3
        assert history.resolve_maps([b'unknown', child], {}, {})[3] == []
    finally:
        history.restore()


def test_repair_preserves_explicit_ownerless_macro_input_and_restores_hooks():
    start, child, _, parents, _, args, _, old_codes = fixture()
    original = beam._backtrack_with_macro_edges
    repair = MacroPrefixBinding('repair')
    repair.install()
    try:
        assert beam._backtrack_with_macro_edges(parents, child, args._frg_macro_edge_path_by_code) == old_codes
    finally:
        repair.restore()
    assert beam._backtrack_with_macro_edges is original


def test_ordinary_edge_is_never_replaced_by_a_longer_historical_macro():
    start, _, _, _, _, _, _, old_codes = fixture()
    parent, child = old_codes[:2]
    history = MacroPrefixBinding('repair')
    history.shape = (3, 3)
    history.edges[parent, child] = MacroEdge(parent, child, (child, parent, child), ())
    assert history.resolve_maps([parent, child], {}, {})[3] == []
    history.install()
    try:
        assert beam._backtrack_with_macro_edges({parent: None, child: parent}, child, {}) == [parent, child]
    finally:
        history.restore()


def test_tail_only_improvement_with_identical_prefix_keeps_new_tail_metadata():
    start, child, parent, _, candidates, args, kwargs, old_codes = fixture()
    kwargs.update(parent_code=parent, parent_n=beam._code_n(parent, 3, 3), parent_depth=0,
        layout_path=tuple(SimpleNamespace(stacks=beam._code_to_stacks(c, 3, 3)) for c in old_codes[1:]),
        prefix_moves=args._frg_macro_edge_moves_by_code[child])
    repair = MacroPrefixBinding('guard')
    repair.install()
    try:
        beam._record_macro_chain_witness_candidate(candidates, args, **kwargs)
        assert args._union_rollout_probe_cache[child].moves == 1
        assert child in args._macro_alt_signature_by_code
        assert repair.stats['prefix_bindings_preserved'] == 0
    finally:
        repair.restore()


def test_historical_move_witnesses_keep_canonical_parent_coordinates():
    start, child, _, parents, candidates, args, kwargs, old_codes = fixture()
    history = MacroPrefixBinding('observe')
    history.install()
    try:
        beam._record_macro_chain_witness_candidate(candidates, args, **kwargs)
    finally:
        history.restore()
    history.mode = 'repair'
    history.install()
    try:
        moves, codes = beam._codes_to_moves_instance_with_macro_witnesses(instance(start),
            beam._backtrack(parents, child), args._frg_macro_edge_moves_by_code, args._frg_macro_edge_parent_by_code)
        assert len(moves) == 3 and codes == old_codes
        assert history.stats['macro_lift_edges_restored'] == 1
    finally:
        history.restore()
