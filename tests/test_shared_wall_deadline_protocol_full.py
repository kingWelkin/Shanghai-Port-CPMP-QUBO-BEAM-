from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from shared_wall_deadline_protocol_full import run_shared, credited_path
from test_shared_wall_deadline_protocol import WORKER

STANDALONE = r'''
import json,sys,time
def emit(**message): print(json.dumps(message),flush=True)
emit(type='READY',role='native')
start=json.loads(sys.stdin.readline())
emit(type='ACK',absolute_start=start['start'],budget=start['budget'])
emit(type='INCUMBENT',moves=40,path=[40],source='private_native',ready_sec=time.perf_counter()-start['start'])
time.sleep(.3)
emit(type='DONE')
'''


def test_standalone_gets_full_clock_but_no_shared_handoff_or_score(tmp_path):
    commands={role:[sys.executable,'-u','-c',WORKER,role] for role in ('prefix','pair')}
    commands['native']=[sys.executable,'-u','-c',STANDALONE]
    result=run_shared(commands,cwd=tmp_path,env=os.environ.copy(),budget=1.,handoff_roles=('pair',))
    assert not result['issues']
    assert set(result['handoff_sent_sec'])=={'pair'}
    native=result['actors']['native']
    assert native['status']=='COMPLETE'
    ack=next(event for event in native['messages'] if event['type']=='ACK')
    assert ack['absolute_start']==result['absolute_start'] and ack['budget']==1.
    assert credited_path([],native['messages'],1.)['moves']==40
    assert credited_path(result['actors']['prefix']['messages'],[],1.)['moves']==10
