from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_cpmp_v50 as release
import research_paper_v43_panel as broad
import shared_wall_deadline_protocol_full as protocol
from test_bounded_conditioned_refinement import example


def response(prefix, refinement, issues=()):
    return dict(actors={
        'prefix': dict(messages=prefix, status='COMPLETE'),
        'refinement': dict(messages=refinement, status='TIME_LIMIT'),
    }, issues=list(issues), termination_lag_sec=0., cleanup_sec=0.01)


def test_release_keeps_timely_physical_solution_when_shorter_answer_arrives_late(monkeypatch):
    instance, path, _ = example()
    legal = dict(type='INCUMBENT', moves=len(path), path=[asdict(m) for m in path],
                 source='native', ready_sec=0.1, received_sec=0.2)
    # This shorter entry is also physically invalid. A late message must never
    # replace the already accepted valid plan, even if it advertises fewer moves.
    late = dict(type='INCUMBENT', moves=0, path=[], source='late',
                ready_sec=0.8, received_sec=1.1)
    monkeypatch.setattr(protocol, 'run_shared', lambda *a, **k: response([legal], [late]))
    monkeypatch.setattr(broad, 'load_case', lambda entry: instance)
    result = release.solve(dict(case='example'), 'qubo_pair', 1.)
    assert result['status'] == 'FEASIBLE'
    assert result['moves'] == 3 and result['path'] == legal['path']
    assert result['physical_replay_passed'] and result['global_cpmp_optimal'] is False


def test_release_reports_worker_failure_without_inventing_a_solution(monkeypatch):
    monkeypatch.setattr(protocol, 'run_shared', lambda *a, **k:
                        response([], [], [dict(role='prefix', error='failed')]))
    result = release.solve(dict(case='example'), 'qubo_pair', 1.)
    assert result['status'] == 'WORKER_ERROR' and result['feasible'] is False
    assert result['moves'] is None and result['path'] == []
