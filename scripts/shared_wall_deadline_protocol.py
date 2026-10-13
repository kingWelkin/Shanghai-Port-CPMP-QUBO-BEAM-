"""Run a fresh shared prefix and isolated continuations under one absolute clock."""
from __future__ import annotations

import queue
import subprocess
import tempfile
import threading
import time
import json


def credited_path(prefix, continuation, budget):
    events = [event for event in prefix + continuation if event['type'] == 'INCUMBENT'
              and event['ready_sec'] <= budget and event['received_sec'] <= budget]
    return min(events, key=lambda event: event['moves']) if events else None


def run_shared(commands, *, cwd, env, budget, startup_timeout=60.):
    """commands contains prefix plus named continuation workers.

    Every worker first emits READY, then reads the SAME START. Continuations
    wait for HANDOFF or STOP. The producer's measured prefix is never replayed
    or granted extra time. Timestamping occurs in readers, before queue handling.
    """
    if 'prefix' not in commands or len(commands) < 2:
        raise ValueError('A producer and at least one continuation are required')
    incoming = queue.Queue()
    actors = {}
    readers = []
    start = None
    handoff = None
    delivered = {}
    issues = []

    def send(name, payload):
        process = actors[name]['process']
        process.stdin.write(json.dumps(payload, separators=(',', ':')) + '\n')
        process.stdin.flush()

    def cancel_waiters(reason):
        for name, actor in actors.items():
            if name != 'prefix' and actor['status'] == 'RUNNING' and name not in delivered:
                send(name, dict(type='STOP', reason=reason))

    try:
        for name, command in commands.items():
            errors = tempfile.TemporaryFile(mode='w+', encoding='utf-8')
            process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=errors, text=True, encoding='utf-8',
                errors='replace', bufsize=1)
            actors[name] = dict(process=process, errors=errors, status='STARTING', messages=[], ready=None)
            def read(role=name, pipe=process.stdout):
                try:
                    for line in pipe:
                        received = time.perf_counter()
                        try:
                            message = json.loads(line)
                        except json.JSONDecodeError:
                            message = dict(type='PROTOCOL_ERROR', line=line)
                        incoming.put((role, received, message))
                finally:
                    incoming.put((role, time.perf_counter(), dict(type='EOF')))
            reader = threading.Thread(target=read, daemon=True)
            reader.start()
            readers.append(reader)
        ready_deadline = time.perf_counter() + startup_timeout
        while any(actor['ready'] is None for actor in actors.values()):
            remaining = ready_deadline - time.perf_counter()
            if remaining <= 0:
                raise RuntimeError('Shared-prefix worker startup timed out')
            name, _, message = incoming.get(timeout=remaining)
            if message['type'] != 'READY' or actors[name]['ready'] is not None:
                raise RuntimeError('Unexpected startup message: ' + repr((name, message)))
            actors[name]['ready'] = message
        start = time.perf_counter()
        deadline = start + budget
        # Continuations can load their raw instance after START and then wait.
        # This work and producer-to-worker communication count in the same clock.
        for name in [key for key in actors if key != 'prefix'] + ['prefix']:
            actors[name]['status'] = 'RUNNING'
            send(name, dict(type='START', start=start, budget=budget))
        while any(actor['status'] == 'RUNNING' for actor in actors.values()):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                name, received, message = incoming.get(timeout=remaining)
            except queue.Empty:
                break
            event = dict(message, received_sec=received - start)
            actors[name]['messages'].append(event)
            if message['type'] == 'HANDOFF':
                if name != 'prefix' or handoff is not None:
                    raise ValueError('Unexpected or repeated shared handoff')
                if event['received_sec'] <= budget and message['ready_sec'] <= budget:
                    handoff = event
                    for role in actors:
                        if role == 'prefix' or actors[role]['status'] != 'RUNNING':
                            continue
                        if time.perf_counter() >= deadline:
                            break
                        send(role, dict(type='HANDOFF', handoff=message, start=start, budget=budget))
                        delivered[role] = time.perf_counter() - start
            elif message['type'] == 'DONE':
                actors[name]['status'] = 'COMPLETE'
                if name == 'prefix' and handoff is None:
                    cancel_waiters('PREFIX_FINISHED_WITHOUT_HANDOFF')
            elif message['type'] in ('ERROR', 'PROTOCOL_ERROR', 'EOF') and actors[name]['status'] == 'RUNNING':
                actors[name]['status'] = 'WORKER_ERROR'
                issues.append(dict(role=name, event=event))
                if name == 'prefix' and handoff is None:
                    cancel_waiters('PREFIX_ERROR')
        stop_requested = time.perf_counter() - start
        for actor in actors.values():
            if actor['status'] == 'RUNNING':
                actor['status'] = 'TIME_LIMIT'
                if actor['process'].poll() is None:
                    actor['process'].kill()
        for name, actor in actors.items():
            try:
                actor['process'].wait(timeout=5.)
            except subprocess.TimeoutExpired:
                actor['process'].kill()
                actor['process'].wait()
                issues.append(dict(role=name, error='Worker did not exit during cleanup'))
            if actor['status'] == 'COMPLETE' and actor['process'].returncode:
                issues.append(dict(role=name, error='Completed worker exited unsuccessfully'))
        for reader in readers:
            reader.join(timeout=2.)
        while True:
            try:
                name, received, message = incoming.get_nowait()
            except queue.Empty:
                break
            actors[name]['messages'].append(dict(message, received_sec=received - start))
        result = {}
        for name, actor in actors.items():
            actor['errors'].seek(0)
            result[name] = dict(status=actor['status'], messages=actor['messages'], ready=actor['ready'],
                                stderr=actor['errors'].read(), returncode=actor['process'].returncode)
        return dict(actors=result, absolute_start=start, budget_sec=budget,
            handoff=handoff, handoff_sent_sec=delivered, issues=issues,
            stop_requested_sec=stop_requested, termination_lag_sec=max(0., stop_requested - budget),
            cleanup_sec=time.perf_counter() - start - stop_requested)
    finally:
        for actor in actors.values():
            process = actor['process']
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5.)
        for reader in readers:
            reader.join(timeout=2.)
        for actor in actors.values():
            actor['process'].stdin.close()
            actor['process'].stdout.close()
            actor['errors'].close()
