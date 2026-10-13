from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from research_paired_fresh_stream import captured_publisher,stream_errors


def test_native_stdout_redirect_cannot_capture_protocol_messages(monkeypatch):
    protocol=io.StringIO();diagnostic=io.StringIO()
    monkeypatch.setattr(sys,'stdout',protocol)
    publish=captured_publisher()
    with redirect_stdout(diagnostic):
        print('native diagnostic')
        publish(dict(type='BEAM',actual_frontier_verified=True))
        publish(dict(type='INCUMBENT',moves=2,ready_sec=.1))
    assert [json.loads(line)['type'] for line in protocol.getvalue().splitlines()]==['BEAM','INCUMBENT']
    assert diagnostic.getvalue()=='native diagnostic\n'


def test_stream_audit_rejects_missing_native_receipts_and_stderr_answers():
    actor=dict(stderr='{"type":"INCUMBENT","moves":2,"ready_sec":0.1}\n',messages=[
        dict(type='NATIVE_END',elapsed_sec=.2,received_sec=.21,beam_checks=3)])
    errors=stream_errors(dict(status='COMPLETE',actors={'native':actor}),1.)
    assert len(errors)==2
