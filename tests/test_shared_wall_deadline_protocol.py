from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from shared_wall_deadline_protocol import run_shared, credited_path

WORKER = r'''
import json,sys,time
role=sys.argv[1]
def emit(**message): print(json.dumps(message),flush=True)
emit(type='READY',role=role)
start=json.loads(sys.stdin.readline())
emit(type='ACK',absolute_start=start['start'],budget=start['budget'])
def offer(n,ready=None):
 emit(type='INCUMBENT',moves=n,path=[n],source=role,
      ready_sec=time.perf_counter()-start['start'] if ready is None else ready)
if role.startswith('prefix'):
 time.sleep(.12)
 offer(10)
 if role=='prefix': emit(type='HANDOFF',path=[10],moves=10,ready_sec=time.perf_counter()-start['start'])
 emit(type='DONE')
else:
 message=json.loads(sys.stdin.readline())
 if message['type']=='STOP':
  emit(type='DONE',stopped=True)
 else:
  emit(type='INPUT',absolute_start=message['start'],elapsed=time.perf_counter()-start['start'])
  if role=='pair':
   offer(8)
   offer(1,ready=start['budget']+.1)
   emit(type='DONE')
  else:
   offer(9)
   time.sleep(start['budget'])
   offer(2)
   emit(type='DONE')
'''


def test_one_absolute_deadline_charges_prefix_and_rejects_late_claims(tmp_path):
    commands={role:[sys.executable,'-u','-c',WORKER,role] for role in ('prefix','pair','qubo')}
    result=run_shared(commands,cwd=tmp_path,env=os.environ.copy(),budget=.5)
    assert not result['issues']
    actors=result['actors']
    assert actors['pair']['status']=='COMPLETE'
    assert actors['qubo']['status']=='TIME_LIMIT'
    assert result['stop_requested_sec']<.65
    for actor in actors.values():
        ack=next(e for e in actor['messages'] if e['type']=='ACK')
        assert ack['absolute_start']==result['absolute_start'] and ack['budget']==.5
    for role in ('pair','qubo'):
        event=next(e for e in actors[role]['messages'] if e['type']=='INPUT')
        assert event['elapsed']>=.12 and event['absolute_start']==result['absolute_start']
    assert credited_path(actors['prefix']['messages'],actors['pair']['messages'],.5)['moves']==8
    assert credited_path(actors['prefix']['messages'],actors['qubo']['messages'],.5)['moves']==9


def test_no_handoff_preserves_prefix_path_and_stops_waiting_workers(tmp_path):
    commands={'prefix':[sys.executable,'-u','-c',WORKER,'prefix_nohandoff'],
              'pair':[sys.executable,'-u','-c',WORKER,'pair']}
    result=run_shared(commands,cwd=tmp_path,env=os.environ.copy(),budget=1.)
    assert not result['issues'] and result['handoff'] is None
    assert result['actors']['pair']['status']=='COMPLETE'
    assert credited_path(result['actors']['prefix']['messages'],result['actors']['pair']['messages'],1.)['moves']==10
