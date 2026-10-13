from dataclasses import asdict
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import deferred_pair_refinement as deferred
import deferred_pair_shared_pipeline as integration
import paired_shared_pipeline as shared
from audit_deferred_pair import audit_answer
from container_reshuffle.core import Move
from production_selector_replay import fingerprint
from research_receiver_allocation import physical_replay
from test_bounded_conditioned_refinement import example


def test_primary_finishes_before_independent_pair_without_clock_reset(monkeypatch):
    instance, path, _ = example()
    start = time.perf_counter()
    saved = deferred.pass_run
    calls = []
    def observed(instance, current, mode, original_start, budget, publish):
        calls.append((mode, tuple(current), original_start, budget))
        return saved(instance, current, mode, original_start, budget, publish)
    monkeypatch.setattr(deferred, 'pass_run', observed)
    events = []
    result = deferred.run(instance, path, 'qubo_deferred_pair', start, 2., events.append)
    assert result['moves'] == 2
    assert [(c[0], len(c[1])) for c in calls] == [
        ('single_qubo', 3), ('single_qubo', 2), ('pair', 3), ('pair', 2)]
    assert all(c[2] == start and c[3] == 2. for c in calls)
    assert calls[0][1] == calls[2][1] == tuple(path)
    assert [e['moves'] for e in events if e['type'] == 'INCUMBENT'] == [2]
    for event in events:
        if event['type'] == 'TRAJECTORY_END':
            physical_replay(instance.stacks, instance.max_height,
                tuple(Move(**value) for value in event['path']), goal=True)


def test_original_deadline_prevents_a_free_deferred_pair(monkeypatch):
    instance, path, _ = example()
    clock = [100.]
    def exhausted(instance, current, mode, start, budget, publish):
        assert mode == 'single_qubo'
        clock[0] = start + budget
        return dict(path=[asdict(move) for move in current])
    monkeypatch.setattr(deferred, 'REAL_TIME', lambda: clock[0])
    monkeypatch.setattr(deferred, 'pass_run', exhausted)
    result = deferred.run(instance, path, 'qubo_deferred_pair', 100., 20., lambda event: None)
    assert result['branch_passes'] == [1, 0]
    assert result['stop_reason'] == 'deadline'
    assert result['moves'] == len(path)


def test_independent_audit_rejects_premature_pair_switch():
    instance, path, _ = example()
    start = time.perf_counter()
    events = [dict(type='START_ACK', absolute_start=start, budget_sec=2.)]
    result = deferred.run(instance, path, 'qubo_deferred_pair', start, 2., events.append)
    events = [dict(event, received_sec=index*.0001) for index, event in enumerate(events)]
    initial = [asdict(move) for move in path]
    answer = dict(messages=events, moves=result['moves'])
    assert audit_answer(answer, initial, 'qubo_deferred_pair', 2.)['errors'] == []
    pair_begin = next(event for event in events if event['type']=='TRAJECTORY_BEGIN' and event['branch']==1)
    early = events.index(next(event for event in events if event['type']=='TRAJECTORY_END'))+1
    changed = events[:early]+[pair_begin]+events[early:]
    errors = audit_answer(dict(messages=changed, moves=result['moves']), initial, 'qubo_deferred_pair', 2.)['errors']
    assert any('before primary stagnation' in error for error in errors)


def test_shared_ledger_restores_modes_and_rejects_clock_rebase():
    instance, path, _ = example()
    raw = [asdict(move) for move in path]
    start = time.perf_counter()
    handoff = dict(path=raw, moves=len(path), path_signature=fingerprint(raw),
                   absolute_start=start, budget_sec=2.)
    modes, refine = shared.MODES, shared.refine
    events = []
    result = integration.run_continuation(instance, 'single_deferred_pair', handoff,
                                          start, 2., events.append)
    assert result['moves'] == 2
    assert shared.MODES == modes and shared.refine is refine
    try:
        integration.run_continuation(instance, 'qubo_deferred_pair', handoff,
                                     start+1., 2., events.append)
        raise AssertionError('Clock rebase accepted')
    except ValueError as error:
        assert 'rebased' in str(error)
    assert shared.MODES == modes and shared.refine is refine
