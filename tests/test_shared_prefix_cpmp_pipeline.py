from dataclasses import asdict
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import shared_prefix_cpmp_pipeline as pipeline
from production_selector_replay import fingerprint
from research_receiver_allocation import physical_replay
from test_bounded_conditioned_refinement import example


def test_prefix_hook_publishes_valid_path_with_original_clock_and_restores(monkeypatch):
    instance, path, _ = example()
    saved = pipeline.main.capped
    start = time.perf_counter()
    events = []
    def native(entry, mode, received_start, budget, publish, **kwargs):
        assert mode == 'pair' and received_start == start
        return pipeline.main.capped(instance, path, 'pair_frg', received_start, budget, publish)
    monkeypatch.setattr(pipeline.main, 'run', native)
    result = pipeline.run_prefix({}, start, 2., events.append)
    assert result['path'] == [asdict(move) for move in path]
    assert len(events) == 1 and events[0]['type'] == 'HANDOFF'
    assert events[0]['absolute_start'] == start and events[0]['budget_sec'] == 2.
    assert pipeline.main.capped is saved


def test_continuations_preserve_input_and_physical_legality_and_reject_rebased_time():
    instance, path, _ = example()
    for mode in pipeline.MODES:
        start = time.perf_counter()
        values = [asdict(move) for move in path]
        handoff = dict(path=values, moves=len(path), path_signature=fingerprint(values),
                       absolute_start=start, budget_sec=2.)
        events = []
        result = pipeline.run_continuation(instance, mode, handoff, start, 2., events.append)
        assert result['moves'] <= len(path)
        received = next(event for event in events if event['type'] == 'BRANCH_INPUT')
        assert received['path_signature'] == handoff['path_signature']
        for event in events:
            if event['type'] == 'INCUMBENT':
                from container_reshuffle.core import Move
                physical_replay(instance.stacks, instance.max_height,
                                tuple(Move(**value) for value in event['path']), goal=True)
        with pytest.raises(ValueError, match='rebased'):
            pipeline.run_continuation(instance, mode, handoff, start + 1., 2., events.append)
