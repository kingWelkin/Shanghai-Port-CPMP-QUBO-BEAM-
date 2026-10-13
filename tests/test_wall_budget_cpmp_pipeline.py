from pathlib import Path
from types import SimpleNamespace
import sys
import time
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import wall_budget_cpmp_pipeline as pipeline
from container_reshuffle.core import Instance,Move,ProblemVariant


def test_common_ledger_publishes_raw_before_graph_and_never_credits_late_paths():
    instance=Instance('ledger',ProblemVariant.PREMARSHALLING_CPMP,3,2,3,[[3,1],[2],[]])
    moves=(Move(0,1,0,2,2,1),Move(1,1,2,0,1,2))
    events=[];ledger=pipeline.CompletePathLedger(instance,time.perf_counter(),1.,events.append)
    assert ledger.offer(moves,'native_test')
    assert [e['moves'] for e in events]==[2,0]
    assert len(ledger.best)==0 and ledger.stats['graph_candidate_savings_sum']==2
    expired=pipeline.CompletePathLedger(instance,time.perf_counter()-2.,1.,events.append)
    assert not expired.offer(moves,'late') and expired.best is None


def test_reserve_requires_valid_incumbent_and_verified_frontier(monkeypatch):
    monkeypatch.setattr(pipeline,'REAL_TIME',lambda:19.)
    ledger=SimpleNamespace(best=None);observer=SimpleNamespace(pending_frontier=None)
    guard=pipeline.NativeReserveGuard(ledger,observer,18.,True)
    for _ in range(32):guard.request()
    ledger.best=();observer.pending_frontier=['pending']
    for _ in range(32):guard.request()
    observer.pending_frontier=None
    with pytest.raises(pipeline.NativeReserveReached):
        for _ in range(32):guard.request()
    disabled=pipeline.NativeReserveGuard(ledger,observer,18.,False)
    for _ in range(32):disabled.request()


def test_native_guard_restores_all_expansion_functions():
    before=(pipeline.beam._successors_with_n_uncached,pipeline.beam._successors_with_n_cached,
            pipeline.frg._bg_choices,pipeline.frg._reduction_choices,pipeline.frg._single_move_children)
    guard=pipeline.NativeReserveGuard(SimpleNamespace(best=None),None,0.,True)
    guard.install();guard.restore()
    after=(pipeline.beam._successors_with_n_uncached,pipeline.beam._successors_with_n_cached,
           pipeline.frg._bg_choices,pipeline.frg._reduction_choices,pipeline.frg._single_move_children)
    assert before==after
