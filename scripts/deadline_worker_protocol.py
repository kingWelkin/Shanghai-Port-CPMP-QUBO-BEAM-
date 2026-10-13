"""Receipt-timestamped worker protocol with a parent-enforced wall deadline."""
from __future__ import annotations

import json
import queue
import subprocess
import tempfile
import threading
import time


def run_worker(command,*,cwd,env,budget,startup_timeout=45.):
    incoming=queue.Queue();messages=[]
    with tempfile.TemporaryFile(mode='w+',encoding='utf-8') as errors:
        process=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=errors,text=True,encoding='utf-8',errors='replace',bufsize=1)
        def read():
            try:
                for line in process.stdout:
                    received=time.perf_counter()
                    try:incoming.put((received,json.loads(line)))
                    except json.JSONDecodeError:incoming.put((received,dict(type='PROTOCOL_ERROR',line=line)))
            finally:incoming.put((time.perf_counter(),dict(type='EOF')))
        reader=threading.Thread(target=read,daemon=True);reader.start()
        start=None;reason=None;stop_requested=None
        try:
            received,ready=incoming.get(timeout=startup_timeout)
            if ready.get('type')!='READY':raise RuntimeError('Worker did not send READY: '+str(ready))
            start=time.perf_counter()
            process.stdin.write(json.dumps(dict(start=start,budget=budget))+'\n');process.stdin.flush()
            while True:
                remaining=start+budget-time.perf_counter()
                if remaining<=0:reason='TIME_LIMIT';break
                try:received,message=incoming.get(timeout=remaining)
                except queue.Empty:reason='TIME_LIMIT';break
                messages.append(dict(message,received_sec=received-start))
                if message['type']=='DONE':reason='COMPLETE';break
                if message['type'] in ('ERROR','PROTOCOL_ERROR','EOF'):
                    raise RuntimeError('Worker failed during budget: '+str(message))
            stop_requested=time.perf_counter()-start
            if process.poll() is None and reason=='TIME_LIMIT':process.kill()
            try:process.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                process.kill();process.wait();raise RuntimeError('Worker failed to exit after termination')
            reader.join(timeout=2.)
            while True:
                try:received,message=incoming.get_nowait()
                except queue.Empty:break
                messages.append(dict(message,received_sec=received-start))
            if reason=='COMPLETE' and process.returncode:raise RuntimeError('Completed worker exited unsuccessfully')
            errors.seek(0);stderr=errors.read()
            return dict(status=reason,ready=ready,messages=messages,budget_sec=budget,
                stop_requested_sec=stop_requested,cleanup_sec=time.perf_counter()-start-stop_requested,
                termination_lag_sec=max(0.,stop_requested-budget),stderr=stderr)
        except Exception as error:
            if process.poll() is None:process.kill()
            process.wait(timeout=5.);reader.join(timeout=2.);errors.seek(0)
            raise RuntimeError(str(error)+'\n'+errors.read()[-3000:]) from error
        finally:
            process.stdin.close();process.stdout.close()
