from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from research_paired_fresh import raw_ledger_view
from shared_wall_deadline_protocol_full import credited_path
from production_selector_replay import fingerprint
from test_bounded_conditioned_refinement import example
from container_reshuffle.core import Move


def test_raw_diagnostic_projection_cannot_credit_an_unreported_path():
    instance,path,_=example();values=[asdict(m) for m in path]
    shorter=[asdict(Move(0,4,0,2,4,2)),asdict(Move(1,3,0,3,3,2))]
    messages=[dict(type='START_ACK',absolute_start=100.,budget_sec=3.,received_sec=.01),
        dict(type='INCUMBENT',source='shared_handoff',path=values,moves=3,ready_sec=.1,received_sec=.11),
        dict(type='BRANCH_INPUT',model_input_signature=fingerprint(values),model_input_moves=3,received_sec=.12),
        dict(type='REFINEMENT_INCUMBENT',stage='refinement',source='branch0:qubo_joint',path=shorter,moves=2,
             ready_sec=2.99,received_sec=2.995),
        dict(type='INCUMBENT',source='refine:branch0:qubo_joint',path=shorter,moves=2,ready_sec=3.01,received_sec=3.02)]
    view=raw_ledger_view(dict(messages=messages),3.)
    assert view['moves']==2
    assert messages[3]['type']=='REFINEMENT_INCUMBENT'
    assert credited_path([],messages,3.)['moves']==3
