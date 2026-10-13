from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import benchmark_compact_beam_qubo_n0 as beam
from frg_rollout_coordinates import replay_reference, graft_child_tail
from paper_priority_group_contract import PriorityGroupContract
from run_rollout_contract import cache_audit


@contextmanager
def group_contract():
    c=PriorityGroupContract()
    c.install()
    try:
        yield c
    finally:
        c.restore()


def test_group_cache_goal_with_repeated_labels_is_physically_valid():
    with group_contract():
        stacks=[[3,2,2],[3,1],[]]
        code,n=beam._encode(stacks,3,4)
        assert n==0
        rollout=beam.UnionRolloutProbeResult(True,0,0,'TEST',0.,())
        args=type('Args',(),dict(_stack_count=3,_max_height=4,_union_rollout_probe_cache={code:rollout}))()
        assert cache_audit(args)['contract_invalid']==0
        actual, endpoint, final_n=replay_reference(stacks,4,[])
        assert endpoint==code and final_n==0 and actual==stacks


def test_group_replay_preserves_box_and_level_checks():
    with group_contract():
        stacks=[[2,3],[2],[]]
        good=beam.Move(0,3,0,2,2,1)
        assert replay_reference(stacks,3,[good])[2]==0
        with pytest.raises(ValueError,match='source top'):
            replay_reference(stacks,3,[beam.Move(0,2,0,2,2,1)])
        with pytest.raises(ValueError,match='level mismatch'):
            replay_reference(stacks,3,[beam.Move(0,3,0,2,1,1)])


def test_group_graft_expands_and_checks_physical_prefix():
    inst=beam.Instance('group',beam.ProblemVariant.PREMARSHALLING_CPMP,3,3,3,[[2,3],[2],[]])
    initial,_=beam._encode(inst.stacks,3,3)
    child,n=beam._encode([[2],[2],[3]],3,3)
    assert n==0
    rollout=beam.UnionRolloutProbeResult(True,0,0,'TEST',0.,())
    with group_contract():
        moves=graft_child_tail(child,(0,1,initial),{initial:None},inst,rollout,10**9,respect_cap=False)
        assert moves is not None and len(moves)==1
        sol=beam.Solution('test',True,False,1.,0.,moves,'TEST')
        assert beam.validate_solution(inst,sol)[0]


def test_unique_id_helpers_are_restored_and_remain_strict():
    original=beam.create_initial_state
    inst=beam.Instance('unique',beam.ProblemVariant.PREMARSHALLING_CPMP,2,3,3,[[3,2],[1]])
    with group_contract():
        assert beam.create_initial_state(inst).stacks==inst.stacks
    assert beam.create_initial_state is original
