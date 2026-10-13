"""Frozen development diagnostic for exact receiver-inversion QUBO quality."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import concurrent.futures
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'scripts', ROOT / 'src'):
    sys.path.insert(0, str(p))
from container_reshuffle.core import future_disorder_count_stacks
from container_reshuffle.solvers import bs_frg as frg
from receiver_allocation_qubo import Allocation, inversion_count
from research_hardset_qubo_utility import members, write_json
import research_paper_v43_panel as broad
from run_beam_selector_research import ENVIRONMENT, digest, write_csv

BASE = ROOT / 'docs/experiments/receiver_allocation_qubo_20261011'
OUT = BASE / 'diagnostic'
ARMS = ('native_sequence', 'linear_exact', 'marginal', 'pair_local', 'pair_exact')
SOURCES = ['scripts/receiver_allocation_qubo.py', 'scripts/research_receiver_allocation.py',
           'scripts/research_hardset_qubo_utility.py', 'scripts/research_paper_v43_panel.py',
           'container_reshuffle/solvers/bs_frg.py', 'container_reshuffle/core.py',
           'container_reshuffle/qubo/model.py']


def physical_replay(initial, height, moves, goal=False):
    stacks = [list(s) for s in initial]
    for t, move in enumerate(moves):
        if move.step != t or move.from_stack == move.to_stack:
            raise ValueError('Move step or identity mismatch')
        src, dst = stacks[move.from_stack], stacks[move.to_stack]
        if not src or src[-1] != move.box or len(dst) >= height:
            raise ValueError('Illegal physical move')
        if move.from_level != len(src) or move.to_level != len(dst) + 1:
            raise ValueError('Physical tier mismatch')
        dst.append(src.pop())
    if goal and future_disorder_count_stacks(stacks) != 0:
        raise ValueError('Final layout is not ordered')
    return tuple(tuple(s) for s in stacks)


def capture(instance):
    current = frg._initial_layout(instance)
    path = []
    seen = {current}
    packets, identities = [], set()
    original = frg._unblocking_assignment
    def observe(stacks, source, height, **kwargs):
        problem = Allocation.build(stacks, source, height)
        key = (problem.stacks, source)
        if 3 <= len(problem.boxes) <= 10 and 2 <= len(problem.receivers) < len(problem.boxes) and key not in identities:
            if problem.stacks != current.stacks:
                raise ValueError('Capture is not the live native state')
            if physical_replay(instance.stacks, height, path) != current.stacks:
                raise ValueError('Capture prefix does not replay')
            identities.add(key)
            packets.append(dict(stacks=problem.stacks, source=source, height=height,
                                prefix=[asdict(m) for m in path], counts=current.selected_counts,
                                active=current.active_reduce_stack))
        return original(stacks, source, height, **kwargs)
    generated = 0
    try:
        frg._unblocking_assignment = observe
        while len(path) < 600 and generated < 50000 and len(packets) < 3 and not frg._is_goal(current.stacks):
            children = frg._frg_step_children(current, instance.max_height, len(path), limit=None, paper_pure=True)
            generated += children[0].generated if children else 0
            child = next((c for c in children if c.layout not in seen), None)
            if child is None:
                break
            current = child.layout
            path.extend(child.moves)
            seen.add(current)
    finally:
        frg._unblocking_assignment = original
    return packets, dict(native_prefix_steps=len(path), native_generated=generated, windows=len(packets))


def execute(entry, cpu):
    broad.pin_cpu(cpu)
    instance = broad.load_case(entry)
    packets, capture_stats = capture(instance)
    outputs = []
    for window, packet in enumerate(packets):
        p = Allocation.build(packet['stacks'], packet['source'], packet['height'])
        assignments = {}
        selection = {}
        for arm in ARMS:
            start = time.perf_counter()
            if arm == 'native_sequence':
                a, meta = p.native(), dict(algorithm='native_sequence')
            elif arm == 'linear_exact':
                a, meta = p.exact(False)
            elif arm == 'marginal':
                a, meta = p.marginal(), dict(algorithm='marginal')
            elif arm == 'pair_local':
                a, meta = p.local(assignments['marginal'])
            else:
                a, meta = p.exact(True)
            selection[arm] = dict(**meta, selection_sec=time.perf_counter() - start)
            assignments[arm] = a
        model, x, slack, penalty = p.qubo()
        prefix = tuple(frg.Move(**move) for move in packet['prefix'])
        for arm in ARMS + ('native_repeat',):
            source_arm = 'native_sequence' if arm == 'native_repeat' else arm
            a = assignments[source_arm]
            if model.evaluate(p.encoded(a, x, slack)) != p.value(a):
                raise ValueError('QUBO energy and physical objective differ')
            after, macro = p.replay(a)
            counts = list(packet['counts'])
            counts[p.source] += 1
            layout = frg._FrgLayout(after, None, tuple(counts))
            started = time.perf_counter()
            tail = frg._frg_completion(layout, p.height, start_step=len(prefix) + len(macro),
                max_steps=800 - len(macro), deadline=float('inf'), max_generated=40000, paper_pure=True)
            suffix = tuple(replace(m, step=len(prefix) + i) for i, m in enumerate(macro)) + tail.moves
            full = prefix + suffix
            final = physical_replay(instance.stacks, p.height, full, goal=tail.feasible)
            if final != tail.layout.stacks:
                raise ValueError('Completion state and replay disagree')
            row = dict(case=entry['case'], dataset=entry['dataset'], family=entry['family'], window=window,
                arm=arm, pop=len(p.boxes), receivers=len(p.receivers), quadratic_clashes=len(p.clashes),
                prefix_steps=len(prefix), valid=tail.feasible, moves=len(full) if tail.feasible else None,
                partial_steps=len(full), final_n=tail.final_n, generated=tail.generated, status=tail.status,
                local_n=future_disorder_count_stacks(after), pair_energy=p.value(a),
                unary_energy=p.value(a, False), qubo_vars=len(model.var_index),
                continuation_sec=time.perf_counter() - started, **selection[source_arm])
            outputs.append(dict(row=row, assignment=a, path=[asdict(m) for m in full]))
    return dict(entry=entry, capture=capture_stats, packets=packets, outcomes=outputs)


def freeze():
    if (OUT / 'manifest.json').exists():
        raise FileExistsError('Diagnostic already frozen')
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [e for e in members() if e['split'] == 'development']
    protocol = (
        'All 35 existing hardset development originals, deterministic native paper-pure FRG prefix. '
        'Capture first three distinct dense unblocking states per original with3..10 popped boxes, '
        'at least two receivers and fewer receivers than popped boxes. Native capture stops at '
        '600 moves or50000 generated choices; no outcome-conditioned replacement. '
        'At each state every arm moves the same top min(source height,total free slots,10) boxes '
        'in the same pop order, then restarts native paper-pure FRG with incremented source count. '
        'Macro+tail800 move limit and40000 generated-choice limit. No wall deadline;120s block watchdog. '
        'Quality is the exact number of new priority-inversion pairs, no fitted coefficients. '
        'Native sequence, exact capacity-linear optimum, dynamic marginal greedy, one-move/swap '
        'descent, exact subset-DP quadratic optimum; repeat native continuation. Same destination '
        'domain, capacity and prefix. DP is a structural oracle, not claimed QUBO backend speed. '
        'Full physical replay, every failed continuation, all original-level denominators retained. '
        'This is an intervention diagnostic, not end-to-end v4.3 performance. Promotion requires '
        'independent originals and same-budget complete-start evidence, not lower inversion energy.')
    (OUT / 'protocol.md').write_text(protocol + '\n', encoding='utf-8')
    write_json(OUT / 'manifest.json', dict(cases=cases, arms=ARMS, cpus=[0, 2, 4, 6], watchdog_sec=120,
        frozen_at=datetime.now().isoformat(), source_sha256={p: digest(ROOT / p) for p in SOURCES},
        protocol_sha256=digest(OUT / 'protocol.md')))
    print(json.dumps(dict(status='FROZEN', originals=len(cases))))


def manifest():
    m = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    for path, expected in m['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise RuntimeError('Frozen source changed: ' + path)
    if digest(OUT / 'protocol.md') != m['protocol_sha256']:
        raise RuntimeError('Protocol changed')
    return m


def run():
    m = manifest()
    def stream(cpu_index):
        for index in range(cpu_index, len(m['cases']), len(m['cpus'])):
            entry = m['cases'][index]
            output = OUT / 'runs' / (entry['case'] + '.json')
            if output.exists():
                continue
            cmd = [sys.executable, str(Path(__file__).resolve()), 'worker', '--case', entry['case'],
                   '--cpu', str(m['cpus'][cpu_index])]
            try:
                done = subprocess.run(cmd, env=ENVIRONMENT, cwd=ROOT, capture_output=True, text=True,
                                      encoding='utf-8', timeout=m['watchdog_sec'])
                if done.returncode:
                    write_json(OUT / 'errors' / output.name, dict(stderr=done.stderr, command=cmd))
                    raise RuntimeError(done.stderr[-2000:])
                value = json.loads(output.read_text(encoding='utf-8'))
                print(json.dumps(dict(case=entry['case'], **value['capture'])), flush=True)
            except subprocess.TimeoutExpired:
                write_json(output, dict(entry=entry, capture=dict(windows=0, watchdog=True), outcomes=[], packets=[]))
                print(json.dumps(dict(case=entry['case'], watchdog=True)), flush=True)
    started = datetime.now().isoformat()
    write_json(OUT / 'status.json', dict(status='RUNNING', started=started, originals=len(m['cases'])))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(m['cpus'])) as pool:
        for f in concurrent.futures.as_completed([pool.submit(stream, i) for i in range(len(m['cpus']))]):
            f.result()
    analyze()
    write_json(OUT / 'status.json', dict(status='COMPLETE', started=started, finished=datetime.now().isoformat()))


def analyze():
    files = sorted((OUT / 'runs').glob('*.json'))
    rows, all_outputs = [], []
    repeats = []
    for path in files:
        data = json.loads(path.read_text(encoding='utf-8'))
        for window in range(len(data['packets'])):
            group = {o['row']['arm']: o for o in data['outcomes'] if o['row']['window'] == window}
            repeats.append(group['native_sequence']['path'] == group['native_repeat']['path'] and
                           group['native_sequence']['row']['generated'] == group['native_repeat']['row']['generated'])
        all_outputs.extend(data['outcomes'])
    rows = [o['row'] for o in all_outputs]
    by_key = {(r['case'], r['window'], r['arm']): r for r in rows}
    stats = {}
    for arm in ARMS:
        subset = [r for r in rows if r['arm'] == arm]
        stats[arm] = dict(rows=len(subset), originals=len({r['case'] for r in subset}),
            legal=sum(r['valid'] for r in subset), moves_sum=sum(r['moves'] or 0 for r in subset),
            local_n_sum=sum(r['local_n'] for r in subset), energy_sum=sum(r['pair_energy'] for r in subset),
            selection_sec=sum(r['selection_sec'] for r in subset))
    contrasts = {}
    for reference in ('native_sequence', 'linear_exact', 'marginal', 'pair_local'):
        pairs = [(r, by_key[(r['case'], r['window'], reference)]) for r in rows if r['arm'] == 'pair_exact']
        common = [(a, b) for a, b in pairs if a['valid'] and b['valid']]
        changes = [a['moves'] - b['moves'] for a, b in common]
        contrasts[reference] = dict(pairs=len(pairs), common=len(common),
            gain_success=sum(a['valid'] and not b['valid'] for a, b in pairs),
            lose_success=sum(b['valid'] and not a['valid'] for a, b in pairs),
            wins=sum(v < 0 for v in changes), losses=sum(v > 0 for v in changes), ties=sum(v == 0 for v in changes),
            net_moves=sum(changes), worst_loss=max(changes, default=None),
            objective_better=sum(a['pair_energy'] < b['pair_energy'] for a, b in pairs))
    summary = dict(originals_attempted=len(files), windows=len(repeats), repeats_ok=sum(repeats),
                   statistics=stats, pair_exact_contrasts=contrasts, decision='NO_PROMOTION')
    write_csv(OUT / 'results.csv', rows)
    write_json(OUT / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'run', 'worker', 'analyze'))
    parser.add_argument('--case')
    parser.add_argument('--cpu', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'worker':
        m = manifest()
        entry = next(e for e in m['cases'] if e['case'] == args.case)
        write_json(OUT / 'runs' / (args.case + '.json'), execute(entry, args.cpu))
    else:
        globals()[args.command]()
