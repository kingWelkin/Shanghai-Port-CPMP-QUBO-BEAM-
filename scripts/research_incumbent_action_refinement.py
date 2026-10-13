"""Only admit action-QUBO proposals that shorten a validated complete incumbent."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import concurrent.futures

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'scripts', ROOT / 'src'):
    sys.path.insert(0, str(p))
from container_reshuffle.core import Move, Instance, future_disorder_count_stacks
from container_reshuffle.solvers import bs_frg as frg
from container_reshuffle.solvers.base import SolverConfig
from source_option_qubo import SourceOptionQubo
from two_wave_action_qubo import ActionQubo, bounded_sources, greedy_two_wave, sequential_exact
from hardset_selection_value import MatchedClock, ExpansionBudget
from production_selector_replay import REAL_TIME, fingerprint
from research_receiver_allocation import physical_replay
from research_hardset_qubo_utility import write_json
import research_paper_v43_panel as broad
from run_beam_selector_research import ENVIRONMENT, digest, write_csv

BASE = ROOT / 'docs/experiments/action_coupling_qubo_20261011'
OUT = BASE / 'incumbent_refinement_pilot'
SOURCE = BASE / 'beam_bundle_native_work'
HORIZON = 120000
SOURCES = ['scripts/research_incumbent_action_refinement.py', 'scripts/source_option_qubo.py',
           'scripts/two_wave_action_qubo.py', 'scripts/hardset_selection_value.py',
           'scripts/research_receiver_allocation.py', 'scripts/research_paper_v43_panel.py',
           'container_reshuffle/solvers/bs_frg.py', 'container_reshuffle/core.py']


def suffix_states(instance, moves):
    state = [list(s) for s in instance.stacks]
    result = []
    for i, move in enumerate(moves):
        n = future_disorder_count_stacks(state)
        if n > 0 and len(moves) - i > n:
            result.append((i, tuple(tuple(s) for s in state), n))
        state[move.to_stack].append(state[move.from_stack].pop())
    return list(reversed(result))


def catalog(instance, incumbent, remaining, seed, points=64):
    args = broad.make_args(100., seed, instance, dict(dataset='BF'))
    args._overall_search_deadline = 1100.
    clock = MatchedClock([], 1000.)
    clock.continue_live()
    budget = ExpansionBudget(clock, args, remaining, [])
    best = {name: incumbent for name in ('frg', 'greedy', 'sequential', 'qubo')}
    records, calls = [], 0
    started = REAL_TIME()
    try:
        budget.install()
        time.perf_counter = clock
        for prefix, stacks, n in suffix_states(instance, incumbent)[:points]:
            if remaining - budget.stats['expansions'] < 4 * 502:
                break
            allowed = bounded_sources(stacks, instance.max_height)
            greedy, ga = greedy_two_wave(stacks, instance.max_height, allowed)
            plans = dict(frg=())
            q_meta = {}
            if greedy:
                action = ActionQubo(stacks, instance.max_height, allowed=allowed)
                if len(greedy) < len({b for b, d, w in action.x}):
                    q = SourceOptionQubo(action).solve(2)
                    calls += 1
                    if len(q['moves']) > len(greedy):
                        a, b = sequential_exact(stacks, instance.max_height, .2, allowed)
                        sequential = a.moves + tuple(replace(m, step=len(a.moves) + j) for j, m in enumerate(b.moves))
                        if a.optimal and b.optimal and len(q['moves']) > len(sequential):
                            plans.update(greedy=greedy, sequential=sequential, qubo=q['moves'])
                            q_meta = dict(q_count=len(q['moves']), sequential_count=len(sequential),
                                          energy=q['energy'], variables=q['variables'])
            common = None
            per_point = []
            for arm, macro in plans.items():
                limit = len(incumbent) - prefix - len(macro) - 1
                if limit < 0:
                    continue
                after = physical_replay(stacks, instance.max_height, macro)
                layout = frg._FrgLayout(after, None, (0,) * instance.stack_count)
                start_work = budget.stats['expansions']
                tail = frg._frg_completion(layout, instance.max_height,
                    start_step=prefix + len(macro), max_steps=limit, deadline=clock() + .05,
                    max_generated=None, paper_pure=True)
                candidate = None
                if tail.feasible:
                    full = incumbent[:prefix] + tuple(replace(m, step=prefix + j) for j, m in enumerate(macro)) + tail.moves
                    physical_replay(instance.stacks, instance.max_height, full, goal=True)
                    if len(full) >= len(incumbent):
                        raise ValueError('Refinement did not improve its original incumbent')
                    candidate = full
                    if len(full) < len(best[arm]):
                        best[arm] = full
                    if arm == 'frg':
                        common = full
                per_point.append(dict(arm=arm, feasible=tail.feasible, moves=len(candidate) if candidate else None,
                    work=budget.stats['expansions'] - start_work, status=tail.status))
            if common is not None:
                for arm in best:
                    if len(common) < len(best[arm]):
                        best[arm] = common
            records.append(dict(prefix=prefix, remaining_original=len(incumbent) - prefix, n=n,
                                q=q_meta, outcomes=per_point))
    finally:
        time.perf_counter = REAL_TIME
        budget.restore()
    return dict(best={a: [asdict(m) for m in p] for a, p in best.items()},
        work=budget.stats, q_calls=calls, records=records, real_sec=REAL_TIME() - started)


def bsfrg_reference(instance, incumbent, remaining, seed):
    if remaining <= 0:
        return dict(path=[asdict(m) for m in incumbent], work={}, real_sec=0.)
    args = broad.make_args(100., seed, instance, dict(dataset='BF'))
    args._overall_search_deadline = 1100.
    clock = MatchedClock([], 1000.)
    clock.continue_live()
    budget = ExpansionBudget(clock, args, remaining, [])
    started = REAL_TIME()
    try:
        budget.install()
        time.perf_counter = clock
        solution = frg.BsFrgSolver(paper_pure=True).solve(instance, SolverConfig(
            max_depth=max(1, len(incumbent) - 1), beam_width=1000, candidate_limit_per_state=3,
            enable_visited_pruning=True, time_limit_sec=100.,
            exact_incumbent_moves=list(incumbent), exact_stop_on_incumbent_improvement=False))
    finally:
        time.perf_counter = REAL_TIME
        budget.restore()
    best = incumbent
    if solution.feasible and len(solution.moves) < len(incumbent):
        candidate = tuple(replace(m, step=i) for i, m in enumerate(solution.moves))
        physical_replay(instance.stacks, instance.max_height, candidate, goal=True)
        best = candidate
    return dict(path=[asdict(m) for m in best], work=budget.stats, real_sec=REAL_TIME() - started)


def freeze():
    if (OUT / 'manifest.json').exists():
        raise FileExistsError('Incumbent refinement already frozen')
    m = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = (
        'New development test on the same12 hash-selected originals and seeds37/73. Reuse each '
        'validated frozen native fixed-work incumbent and its consumed request count. Additional '
        'refinement is limited to the unused portion of120000 compact/FRG requests; zero reserve '
        'means retain the incumbent. Never alter or relabel the native search trajectory. '
        'Scan at most64 suffix states,latest first,skipping tails already at their N lower bound. '
        'A common catalog evaluates native FRG and,only at a strict joint-count opportunity, '
        'greedy/sequential/QUBO macro plus native FRG completion. All branches share recorded '
        'information and execution budget; models see no future outcomes. Each continuation has '
        'at most500 virtual expansion requests(cooperative one-request overshoot allowed) and '
        'a strict original-suffix move bound. Reserve enough work for a complete four-branch point. '
        'Only replay-validated shorter full paths are accepted; each output retains the original '
        'incumbent and all common FRG improvements. Repeat the full catalog independently. '
        'Separately run strong BS-FRG continuation with the original incumbent and the entire same '
        'unused work budget; this control does not pay dummy QUBO calculations. '
        'Report all inputs,zero-reserve cases,actual total requests,full-path improvements and '
        'real overhead. Fixed-work scope excludes model arithmetic and physical bridge enumeration. '
        'No default or wall-time advantage claim; existing negative experiments remain unchanged.')
    (OUT / 'protocol.md').write_text(protocol + '\n', encoding='utf-8')
    inputs = {f"{b['case']}__s{b['seed']}": digest(SOURCE / 'runs' / f"{b['case']}__s{b['seed']}__native.json") for b in m['blocks']}
    write_json(OUT / 'manifest.json', dict(cases=m['cases'], blocks=m['blocks'], cpus=[0, 2, 4, 6], jobs=4,
        watchdog_sec=180, frozen_at=datetime.now().isoformat(), native_sha256=inputs,
        source_sha256={p: digest(ROOT / p) for p in SOURCES}, protocol_sha256=digest(OUT / 'protocol.md')))


def manifest():
    m = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    for p, sha in m['source_sha256'].items():
        if digest(ROOT / p) != sha:
            raise ValueError('Refinement source changed: ' + p)
    return m


def execute(entry, seed, cpu):
    broad.pin_cpu(cpu)
    p = SOURCE / 'runs' / f"{entry['case']}__s{seed}__native.json"
    key = f"{entry['case']}__s{seed}"
    if digest(p) != manifest()['native_sha256'][key]:
        raise ValueError('Native incumbent artifact changed')
    native = json.loads(p.read_text(encoding='utf-8'))
    instance = broad.load_case(entry)
    if not native['row']['valid']:
        return dict(case=entry['case'], seed=seed, skipped='NO_INCUMBENT')
    incumbent = tuple(Move(**m) for m in native['path'])
    physical_replay(instance.stacks, instance.max_height, incumbent, goal=True)
    remaining = max(0, HORIZON - native['row']['expansions'])
    if remaining < 2008:
        return dict(case=entry['case'], seed=seed, original=len(incumbent), remaining=remaining,
                    skipped='NO_REFINEMENT_RESERVE')
    first = catalog(instance, incumbent, remaining, seed)
    repeat = catalog(instance, incumbent, remaining, seed)
    reference = bsfrg_reference(instance, incumbent, remaining, seed)
    if first['best'] != repeat['best'] or first['work'] != repeat['work']:
        raise ValueError('Refinement catalog did not repeat')
    if first['work']['expansions'] > remaining or reference['work'].get('expansions', 0) > remaining:
        raise ValueError('Refinement exceeded the unused request budget')
    return dict(case=entry['case'], dataset=entry['dataset'], seed=seed, original=len(incumbent),
        remaining=remaining, repeat_ok=True, catalog=first, bsfrg=reference)


def analyze():
    records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT / 'runs').glob('*.json'))]
    rows = []
    for r in records:
        if r.get('skipped'):
            rows.append(dict(case=r['case'], seed=r['seed'], original=r.get('original'), skipped=r['skipped']))
            continue
        rows.append(dict(case=r['case'], dataset=r['dataset'], seed=r['seed'], original=r['original'],
            remaining=r['remaining'], work=r['catalog']['work']['expansions'], q_calls=r['catalog']['q_calls'],
            bsfrg=len(r['bsfrg']['path']), **{a: len(v) for a, v in r['catalog']['best'].items()}))
    active = [r for r in rows if 'qubo' in r]
    summary = dict(rows=len(rows), active=len(active), skipped=sum('skipped' in r for r in rows),
        total_q_calls=sum(r['q_calls'] for r in active),
        savings={a: sum(r['original'] - r[a] for r in active) for a in ('frg','greedy','sequential','qubo','bsfrg')},
        q_contrasts={a: dict(wins=sum(r['qubo'] < r[a] for r in active), losses=sum(r['qubo'] > r[a] for r in active),
                            net_moves=sum(r['qubo'] - r[a] for r in active)) for a in ('frg','greedy','sequential','bsfrg')},
        decision='NO_PROMOTION')
    write_csv(OUT / 'results.csv', rows)
    write_json(OUT / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


def run():
    m = manifest()
    write_json(OUT / 'status.json', dict(status='RUNNING', started=datetime.now().isoformat()))
    def stream(lane):
        for b in m['blocks']:
            if b['index'] % m['jobs'] != lane:
                continue
            output = OUT / 'runs' / f"{b['case']}__s{b['seed']}.json"
            if output.exists():
                continue
            command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--case', b['case'],
                       '--seed', str(b['seed']), '--cpu', str(m['cpus'][lane])]
            result = subprocess.run(command, cwd=ROOT, env=ENVIRONMENT, capture_output=True,
                text=True, encoding='utf-8', errors='replace', timeout=m['watchdog_sec'])
            if result.returncode:
                write_json(OUT / 'errors' / output.name, dict(stderr=result.stderr, command=command))
                raise RuntimeError(result.stderr[-4000:])
            print(json.dumps(dict(**b, complete=True)), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=m['jobs']) as pool:
        for f in concurrent.futures.as_completed([pool.submit(stream, i) for i in range(m['jobs'])]):
            f.result()
    analyze()
    write_json(OUT / 'status.json', dict(status='COMPLETE', finished=datetime.now().isoformat()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('freeze','run','worker','analyze'))
    p.add_argument('--case')
    p.add_argument('--seed', type=int, default=37)
    p.add_argument('--cpu', type=int, default=0)
    a = p.parse_args()
    if a.command == 'worker':
        e = next(e for e in manifest()['cases'] if e['case'] == a.case)
        write_json(OUT / 'runs' / f"{a.case}__s{a.seed}.json", execute(e, a.seed, a.cpu))
    else:
        globals()[a.command]()
