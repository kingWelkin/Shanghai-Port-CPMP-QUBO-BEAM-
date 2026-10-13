from dataclasses import asdict
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from paired_shared_pipeline import run_continuation
from production_selector_replay import fingerprint
from container_reshuffle.core import Move
from research_receiver_allocation import physical_replay
from test_bounded_conditioned_refinement import example


@pytest.mark.parametrize('mode', ('single_cycle', 'pair_cycle', 'qubo_cycle'))
def test_solo_continuation_uses_original_clock_and_real_single_trajectory(mode):
    instance, path, _ = example()
    start = time.perf_counter()
    values = [asdict(move) for move in path]
    handoff = dict(path=values, moves=len(path), path_signature=fingerprint(values),
                   absolute_start=start, budget_sec=2.)
    events = []
    result = run_continuation(instance, mode, handoff, start, 2., events.append)
    assert result['moves'] == 2
    cycles = [event for event in events if event['type'] == 'CYCLE_BEGIN']
    assert cycles and not any(event['type'] == 'TRAJECTORY_BEGIN' for event in events)
    assert all(event['absolute_start'] == start and event['budget_sec'] == 2. for event in cycles)
    raw = [event for event in events if event['type'] == 'REFINEMENT_INCUMBENT']
    reported = [event for event in events if event['type'] == 'INCUMBENT']
    assert raw and reported[-1]['moves'] == 2
    assert reported[-1]['refinement_provenance']['trajectory_branch'] is None
    for event in raw + reported:
        physical_replay(instance.stacks, instance.max_height,
                        tuple(Move(**value) for value in event['path']), goal=True)
    with pytest.raises(ValueError, match='rebased'):
        run_continuation(instance, mode, handoff, start + 1., 2., events.append)
