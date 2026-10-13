from dataclasses import asdict
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import serial_classical_guards as serial
from audit_serial_guards import audit_answer
from container_reshuffle.core import Move
from research_receiver_allocation import physical_replay
from test_bounded_conditioned_refinement import example


@pytest.mark.parametrize('mode', ['qubo_qsp', 'qubo_qps'])
def test_three_paid_independent_trajectories_share_one_clock(mode, monkeypatch):
    instance, path, _ = example()
    start = time.perf_counter()
    saved = serial.pass_run
    calls = []
    def observed(instance, current, policy, original_start, budget, publish):
        calls.append((policy, tuple(current), original_start, budget))
        return saved(instance, current, policy, original_start, budget, publish)
    monkeypatch.setattr(serial, 'pass_run', observed)
    events = [dict(type='START_ACK', absolute_start=start, budget_sec=3.)]
    result = serial.run(instance, path, mode, start, 3., events.append)
    assert result['moves'] == 2 and result['branch_passes'] == [2, 2, 2]
    assert [call[0] for call in calls] == [policy for policy in serial.POLICIES[mode] for _ in range(2)]
    assert all(calls[index][1] == tuple(path) for index in (0, 2, 4))
    assert all(call[2] == start and call[3] == 3. for call in calls)
    assert [event['moves'] for event in events if event['type'] == 'INCUMBENT'] == [2]
    for event in events:
        if event['type'] == 'TRAJECTORY_END':
            physical_replay(instance.stacks, instance.max_height,
                tuple(Move(**value) for value in event['path']), goal=True)
    events = [dict(event, received_sec=index*.0001) for index, event in enumerate(events)]
    assert audit_answer(dict(messages=events, moves=2), [asdict(move) for move in path], mode, 3.)['errors'] == []


def test_exhausted_primary_cannot_run_either_guard(monkeypatch):
    instance, path, _ = example()
    clock = [100.]
    def expire(instance, current, policy, start, budget, publish):
        assert policy == 'single_qubo'
        clock[0] = start + budget
        return dict(path=[asdict(move) for move in current])
    monkeypatch.setattr(serial, 'REAL_TIME', lambda: clock[0])
    monkeypatch.setattr(serial, 'pass_run', expire)
    result = serial.run(instance, path, 'qubo_qsp', 100., 20., lambda event: None)
    assert result['branch_passes'] == [1, 0, 0]
    assert result['moves'] == len(path) and result['stop_reason'] == 'deadline'
