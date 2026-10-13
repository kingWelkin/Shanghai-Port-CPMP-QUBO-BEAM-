from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import sequential_conditioned_refiner as classical
import bounded_conditioned_refinement as bounded
from single_seed_conditioned_refinement import run
from fast_conditioned_refinement import run as reference
from test_bounded_conditioned_refinement import example


def test_cheap_seed_never_runs_pair_and_returns_verified_path(monkeypatch):
    instance, path, releases = example()
    def forbidden(*args, **kwargs):
        raise AssertionError('Pair work is forbidden in the cheap-seed arms')
    monkeypatch.setattr(classical, 'exchange', forbidden)
    saved = bounded.classical_seed
    for mode in ('single', 'single_qubo'):
        messages = []
        result = run(instance,path,mode,time.perf_counter(),2.,messages.append,
                     releases=releases,continue_frg=False)
        assert result['moves'] == 2 and bounded.classical_seed is saved
        models = [e['record'] for e in messages if e['type']=='MODEL']
        assert models and all(m['seed_metadata']['pair_trials']==[] for m in models)
        assert all(m['seed_kind']=='three_order_single' and m['nonoracle_milp_calls']==0 for m in models)
        assert any(e.get('source')=='single_seed' for e in messages)
        assert not any(e.get('source')=='pair_seed' for e in messages)


def test_strong_pair_control_is_unchanged():
    instance, path, releases = example()
    for mode, old_mode in (('pair','pair_frg'),('pair_qubo','capped_qubo_frg')):
        new = run(instance,path,mode,time.perf_counter(),2.,lambda e:None,
                  releases=releases,continue_frg=False)
        old = reference(instance,path,old_mode,time.perf_counter(),2.,lambda e:None,
                        releases=releases,continue_frg=False)
        assert new['path']==old['path']
        assert [r['selected'] for r in new['models']]==[r['selected'] for r in old['models']]
