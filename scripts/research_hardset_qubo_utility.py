"""Frozen, paired development pilot for the hardset169 Union mechanism."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'scripts', ROOT / 'src'):
    sys.path.insert(0, str(directory))

import numpy as np
import research_paper_v43_panel as broad
from beam_component_ablation import scalar_settings
from hardset_qubo_utility import ARMS, RouteObserver
from frg_rollout_coordinates import CanonicalTailContract
from paper_priority_group_contract import PriorityGroupContract
from production_selector_replay import fingerprint
from run_beam_selector_research import ENVIRONMENT, digest, write_csv
from run_rollout_contract import cache_audit

OUT = ROOT / 'docs/experiments/hardset169_qubo_utility_20261010'
HARD = ROOT / 'docs/experiments/hardset169_innovation_20261010'
SOURCES = list(dict.fromkeys(broad.SOURCES + [
    'scripts/beam_component_ablation.py', 'scripts/research_hardset_qubo_utility.py',
    'scripts/hardset_qubo_utility.py', 'scripts/research_paper_selector_panel.py',
]))
CONTRASTS = [('generic_pair', 'linear'), ('recovery_pair', 'generic_pair'),
             ('recovery_core', 'recovery_pair'), ('recovery_core_sa', 'recovery_core')]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def members():
    hard = json.loads((HARD / 'manifest.json').read_text(encoding='utf-8'))
    parent = json.loads((broad.OUT / 'manifest.json').read_text(encoding='utf-8'))
    by_case = {entry['case']: entry for entry in parent['cases']}
    entries = []
    for member in hard['members']:
        entry = dict(by_case[member['case']])
        if entry['sha256'] != member['input_sha256'] or entry['split'] != member['split']:
            raise ValueError('Hardset input/split mismatch')
        entry['historical_case_key'] = member['historical_case_key']
        entries.append(entry)
    if len(entries) != 169 or sum(e['split'] == 'development' for e in entries) != 35:
        raise ValueError('Unexpected hardset partition')
    return sorted(entries, key=lambda entry: entry['case'])


def freeze():
    if (OUT / 'manifest.json').exists():
        raise FileExistsError('Pilot already frozen')
    cases = members()
    development = [e for e in cases if e['split'] == 'development']
    seeds = [37, 73]
    blocks = []
    for entry in development:
        for seed in seeds:
            index = len(blocks)
            shift = index % len(ARMS)
            order = list(ARMS[shift:] + ARMS[:shift])
            blocks.append(dict(index=index, case=entry['case'], seed=seed, cpu=(index % 8) * 2, order=order))
    source_map = {relative: digest(ROOT / relative) for relative in SOURCES}
    (OUT / 'frozen_protocol.md').write_text((OUT / 'task_plan.md').read_text(encoding='utf-8'), encoding='utf-8')
    manifest = dict(schema='hardset-qubo-utility/v1', frozen_at=datetime.now(timezone.utc).isoformat(),
        cases=cases, development_cases=[e['case'] for e in development], arms=list(ARMS), seeds=seeds,
        request_sec=20., watchdog_sec=100, jobs=8, affinity_cpus=list(range(0, 16, 2)), blocks=blocks,
        expected_blocks=len(blocks), expected_runs=len(blocks) * len(ARMS),
        source_sha256=source_map, hard_manifest_sha256=digest(HARD / 'manifest.json'),
        parent_manifest_sha256=digest(broad.OUT / 'manifest.json'),
        protocol='frozen_protocol.md', protocol_sha256=digest(OUT / 'frozen_protocol.md'),
        weights=dict(pair=.35, background=.25, generic_scale=1.23, background_representatives=32, cached_prefix_moves=6),
        scope='35 existing development originals x2 seeds x6 arms;134 evaluation originals not run. Native pays common feature cost. No new rollout, no default change, no promotion.')
    write_json(OUT / 'manifest.json', manifest)
    write_csv(OUT / 'corpus_inventory.csv', cases)
    write_json(OUT / 'status.json', dict(status='FROZEN', expected_blocks=len(blocks), expected_runs=manifest['expected_runs']))
    print(json.dumps(dict(status='FROZEN', blocks=len(blocks), runs=manifest['expected_runs'])), flush=True)


def manifest():
    result = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    for relative, expected in result['source_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Frozen source changed: ' + relative)
    if digest(HARD / 'manifest.json') != result['hard_manifest_sha256']:
        raise ValueError('Hardset membership manifest changed')
    if digest(broad.OUT / 'manifest.json') != result['parent_manifest_sha256']:
        raise ValueError('Parent input inventory changed')
    if digest(OUT / result['protocol']) != result['protocol_sha256']:
        raise ValueError('Frozen experimental protocol changed')
    return result


def execute(entry, arm, seed, cpu, cap):
    broad.pin_cpu(cpu)
    random.seed(seed)
    np.random.seed(seed)
    started = time.perf_counter()
    instance = broad.load_case(entry)
    args = broad.make_args(cap, seed, instance, entry)
    before = scalar_settings(args)
    group = PriorityGroupContract()
    tails = CanonicalTailContract()
    observer = RouteObserver(arm)
    args._union_selector_observer = observer
    configured = time.perf_counter()
    try:
        group.install()
        tails.install()
        observer.install()
        result = broad.beam.run_compact_beam_instance(entry['case'], instance, args)
    finally:
        observer.restore()
        tails.restore()
        group.restore()
    solved = time.perf_counter()
    business, goal, message = broad.opened._open_validate_instance_result(
        result, instance, 10., business_cap_enabled=False)
    group.install()
    try:
        audit = cache_audit(args)
    finally:
        group.restore()
    ended = time.perf_counter()
    if observer.pending_frontier is not None or any(not c['actual_frontier_verified'] for c in observer.commits):
        raise RuntimeError('Committed Beam not verified')
    if args.business_cap_enabled or args.inner_jobs != 1 or args.post_polish_inner_jobs != 1:
        raise RuntimeError('Common resource/input contract changed')
    changed = {key: [value, getattr(args, key, None)] for key, value in before.items()
               if not key.startswith('_') and getattr(args, key, None) != value}
    if changed:
        raise RuntimeError('Runtime patch changed native scalar configuration: ' + str(changed))
    path = [asdict(move) for move in result.moves]
    metrics = asdict(result)
    metrics.pop('moves', None)
    row = dict(case=entry['case'], dataset=entry['dataset'], family=entry['family'], split=entry['split'],
        arm=arm, seed=seed, cpu=cpu, request_sec=cap, execution_status='EXECUTED',
        actual_profile=args.solver_profile, union_enabled=bool(args.union_diversity_qubo_enabled),
        valid=business and goal, primary_ok=business and goal and ended - started <= cap and audit['contract_invalid'] == 0,
        within_budget=ended - started <= cap, moves=len(path) if goal else None,
        status=result.status, validation_message=message, cache_contract_invalid=audit['contract_invalid'],
        load_config_sec=configured - started, solver_wall_sec=solved - configured,
        validation_cache_sec=ended - solved, measured_end_to_end_sec=ended - started,
        full_path_signature=fingerprint(path), actual_beam_commits=len(observer.commits),
        actual_beam_verified=len(observer.commits), union_calls=len(observer.records),
        nontrivial_union_calls=sum(r['n'] > r['k'] for r in observer.records),
        semantic_sec=sum(r.get('semantic_sec', 0.) for r in observer.records),
        union_selector_sec=sum(r['selector_sec'] for r in observer.records),
        cached_pool_profiles=sum(r.get('cached_pool_profiles', 0) for r in observer.records),
        background_variable_calls=sum(r.get('background_shift_max', 0.) - r.get('background_shift_min', 0.) > 1e-10 for r in observer.records),
        configuration_preserved=True)
    audit_summary = {key: value for key, value in audit.items() if key != 'records'}
    audit_summary['violations'] = [r for r in audit['records'] if not r['contract_ok']]
    return dict(row=row, path=path, result_metrics=metrics, native_configuration=before,
        selector_records=observer.records, route_snapshots=observer.route_snapshots,
        commits=observer.commits, contract_stats=tails.stats, cache_audit=audit_summary)


def run_path(case, seed, arm):
    return OUT / 'runs' / f'{case}__s{seed}__{arm}.json'


def worker(case, arm, seed, cpu):
    m = manifest()
    entry = next(e for e in m['cases'] if e['case'] == case)
    data = execute(entry, arm, seed, cpu, m['request_sec'])
    write_json(run_path(case, seed, arm), data)
    print(json.dumps({key: data['row'][key] for key in ('case', 'arm', 'seed', 'valid', 'primary_ok', 'moves', 'union_calls')}), flush=True)


def run_block(block, m):
    case, seed = block['case'], block['seed']
    destination = OUT / 'blocks' / f'{case}__s{seed}.json'
    if destination.exists():
        return dict(case=case, seed=seed, existing=True)
    if any(run_path(case, seed, arm).exists() for arm in ARMS):
        raise RuntimeError('Partial block preserved; cannot splice or selectively rerun: ' + case)
    entry = next(e for e in m['cases'] if e['case'] == case)
    for arm in block['order']:
        command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--case', case,
                   '--arm', arm, '--seed', str(seed), '--cpu', str(block['cpu'])]
        began = time.perf_counter()
        try:
            process = subprocess.run(command, cwd=ROOT, env={**os.environ, **ENVIRONMENT},
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=m['watchdog_sec'])
            elapsed = time.perf_counter() - began
            if process.returncode:
                write_json(OUT / 'errors' / f'{case}__s{seed}__{arm}.json',
                    dict(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr))
                raise RuntimeError('Worker implementation error: ' + process.stderr[-1600:])
            data = json.loads(run_path(case, seed, arm).read_text(encoding='utf-8'))
            data['row']['process_wall_sec'] = elapsed
            write_json(run_path(case, seed, arm), data)
        except subprocess.TimeoutExpired as error:
            data = dict(row=dict(case=case, dataset=entry['dataset'], family=entry['family'], split='development',
                arm=arm, seed=seed, cpu=block['cpu'], request_sec=m['request_sec'], execution_status='WATCHDOG',
                valid=False, primary_ok=False, moves=None, status='EXTERNAL_WATCHDOG',
                process_wall_sec=time.perf_counter() - began), path=[])
            write_json(run_path(case, seed, arm), data)
            write_json(OUT / 'errors' / f'{case}__s{seed}__{arm}.json', dict(type='watchdog', stderr=str(error.stderr)))
    write_json(destination, dict(**block, complete=True,
        run_sha256={arm: digest(run_path(case, seed, arm)) for arm in ARMS}))
    return dict(case=case, seed=seed, existing=False)


def run():
    m = manifest()
    started = datetime.now(timezone.utc).isoformat()
    completed = len(list((OUT / 'blocks').glob('*.json')))
    write_json(OUT / 'status.json', dict(status='RUNNING', completed_blocks=completed,
        expected_blocks=m['expected_blocks'], expected_runs=m['expected_runs'], started_at=started))
    lock = threading.Lock()
    def stream(slot):
        nonlocal completed
        for block in m['blocks']:
            if block['cpu'] == m['affinity_cpus'][slot]:
                result = run_block(block, m)
                if not result['existing']:
                    with lock:
                        completed += 1
                        write_json(OUT / 'status.json', dict(status='RUNNING', completed_blocks=completed,
                            expected_blocks=m['expected_blocks'], completed_runs=completed * len(ARMS),
                            expected_runs=m['expected_runs'], started_at=started))
                print(json.dumps(dict(status='COMPLETE_BLOCK', **result)), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=m['jobs']) as executor:
        futures = [executor.submit(stream, slot) for slot in range(m['jobs'])]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    summarize()
    write_json(OUT / 'status.json', dict(status='COMPLETED', completed_blocks=m['expected_blocks'],
        expected_blocks=m['expected_blocks'], completed_runs=m['expected_runs'], started_at=started,
        completed_at=datetime.now(timezone.utc).isoformat()))


def pair_summary(rows, candidate, base, dataset='ALL', seed=None):
    chosen = [r for r in rows if (dataset == 'ALL' or r['dataset'] == dataset) and (seed is None or r['seed'] == seed)]
    index = {(r['case'], r['seed'], r['arm']): r for r in chosen}
    pairs = [(r, index[r['case'], r['seed'], base]) for r in chosen if r['arm'] == candidate]
    both_legal = [(a, b) for a, b in pairs if a['valid'] and b['valid']]
    both_strict = [(a, b) for a, b in pairs if a['primary_ok'] and b['primary_ok']]
    def outcome(a, b, key):
        if a[key] and b[key]:
            return 'win' if a['moves'] < b['moves'] else 'loss' if a['moves'] > b['moves'] else 'tie'
        return 'win' if a[key] else 'loss' if b[key] else 'both_failed'
    return dict(candidate=candidate, base=base, dataset=dataset, seed=seed, pairs=len(pairs),
        originals=len({a['case'] for a, b in pairs}),
        base_legal=sum(b['valid'] for a, b in pairs), candidate_legal=sum(a['valid'] for a, b in pairs),
        base_strict=sum(b['primary_ok'] for a, b in pairs), candidate_strict=sum(a['primary_ok'] for a, b in pairs),
        legal_outcomes=dict(collections.Counter(outcome(a, b, 'valid') for a, b in pairs)),
        strict_outcomes=dict(collections.Counter(outcome(a, b, 'primary_ok') for a, b in pairs)),
        both_legal=len(both_legal), legal_move_delta=sum(a['moves'] - b['moves'] for a, b in both_legal),
        both_strict=len(both_strict), strict_move_delta=sum(a['moves'] - b['moves'] for a, b in both_strict),
        worst_legal_regression=max((a['moves'] - b['moves'] for a, b in both_legal), default=None),
        median_sec_delta=statistics.median(a.get('measured_end_to_end_sec', 100.) - b.get('measured_end_to_end_sec', 100.) for a, b in pairs) if pairs else None)


def static_model_audit(data):
    rows = []
    for snapshot in data.get('route_snapshots', []):
        linear = snapshot['base_linear']
        size = len(linear)
        zeros = [[0.0] * size for _ in range(size)]
        from hardset_qubo_utility import greedy, repair, BACKGROUND_WEIGHT
        models = dict(linear=(linear, zeros), generic_pair=(linear, snapshot['generic']),
                      recovery_pair=(linear, snapshot['recovery']),
                      recovery_core=([a + BACKGROUND_WEIGHT * b for a, b in zip(linear, snapshot['background_shift'])], snapshot['recovery']))
        selected = {name: sorted(repair(greedy(a, p, snapshot['target']), a, p, snapshot['target'])[0])
                    for name, (a, p) in models.items()}
        rows.append(dict(case=data['row']['case'], seed=data['row']['seed'], layer=snapshot['layer'],
            n=size, k=snapshot['target'],
            generic_vs_linear=selected['generic_pair'] != selected['linear'],
            recovery_vs_generic=selected['recovery_pair'] != selected['generic_pair'],
            core_vs_recovery=selected['recovery_core'] != selected['recovery_pair'],
            selected=selected))
    return rows


def summarize():
    m = manifest()
    rows, static = [], []
    contexts = []
    for file in sorted((OUT / 'blocks').glob('*.json')):
        block = json.loads(file.read_text(encoding='utf-8'))
        for arm in ARMS:
            path = run_path(block['case'], block['seed'], arm)
            if digest(path) != block['run_sha256'][arm]:
                raise ValueError('Completed block changed')
            data = json.loads(path.read_text(encoding='utf-8'))
            rows.append(data['row'])
            contexts.extend(dict(record, case=data['row']['case'], run_seed=data['row']['seed'], run_arm=arm)
                            for record in data.get('selector_records', []))
            if arm == 'native':
                static.extend(static_model_audit(data))
    contrasts = CONTRASTS + [(arm, 'native') for arm in ARMS if arm != 'native']
    groups = [pair_summary(rows, candidate, base, dataset, seed)
              for dataset in ('ALL', 'CV', 'BF', 'Open120')
              for seed in (None, *m['seeds']) for candidate, base in contrasts
              if any(r['dataset'] == dataset for r in rows) or dataset == 'ALL']
    arm_summary = [dict(arm=arm, rows=len([r for r in rows if r['arm'] == arm]),
        legal=sum(r['valid'] for r in rows if r['arm'] == arm),
        strict=sum(r['primary_ok'] for r in rows if r['arm'] == arm),
        union_calls=sum(r.get('union_calls', 0) for r in rows if r['arm'] == arm),
        semantic_sec=sum(r.get('semantic_sec', 0.) for r in rows if r['arm'] == arm),
        selector_sec=sum(r.get('union_selector_sec', 0.) for r in rows if r['arm'] == arm)) for arm in ARMS]
    complete = len(rows) == m['expected_runs']
    audit = dict(complete=complete, rows=len(rows), expected_runs=m['expected_runs'],
        complete_blocks=len(rows) // len(ARMS), distinct_originals=len({r['case'] for r in rows}),
        cache_violations=sum(r.get('cache_contract_invalid', 0) for r in rows),
        verified_beam_commits=sum(r.get('actual_beam_verified', 0) for r in rows),
        unverified_beam_commits=sum(r.get('actual_beam_commits', 0) - r.get('actual_beam_verified', 0) for r in rows),
        watchdogs=sum(r['execution_status'] == 'WATCHDOG' for r in rows),
        mechanism_native_contexts=len(static),
        same_context_generic_changes=sum(r['generic_vs_linear'] for r in static),
        same_context_recovery_changes=sum(r['recovery_vs_generic'] for r in static),
        same_context_core_changes=sum(r['core_vs_recovery'] for r in static),
        inference='Case-seed pairs are repetitions of35 originals, not70 independent instances. Static changes are proxy evidence only.')
    write_csv(OUT / 'results.csv', rows)
    write_json(OUT / 'summary.json', dict(audit=audit, arms=arm_summary, contrasts=groups))
    write_json(OUT / 'native_context_audit.json', static)
    write_json(OUT / 'selector_contexts.json', contexts)
    write_csv(OUT / 'contrasts.csv', groups)
    lines = ['# hardset169：QUBO效用开发集试验', '',
        f"状态：{'完成' if complete else '部分结果'}；35个既有开发例、种子37/73、六分支、严格20秒。完整运行{len(rows)}/{m['expected_runs']}。",
        '', '原困难集169名单与35/134划分保留；134评价例未参与本轮。当前版本、原生供给和后处理不变。所有分支共同计算结构/缓存路线特征并计费；native是保持原生模型与修复的共同特征成本参考，不复用旧结果。',
        '', '研究分支共同使用原生归一化一元质量，删除宏配额，并统一初始化、计数修复、交换改进和最终排序的目标。背景项使用最多32个等间隔已选代表，属于有界近似；结构/缓存动作重复是未来搜索重复的代理，不是完成概率证书。',
        '', '| 分支 | 合法/70 | 预算内/70 | Union调用 | 特征秒合计 | 选集秒合计 |',
        '|---|---:|---:|---:|---:|---:|']
    for item in arm_summary:
        lines.append(f"| {item['arm']} | {item['legal']} | {item['strict']} | {item['union_calls']} | {item['semantic_sec']:.3f} | {item['selector_sec']:.3f} |")
    lines += ['', '差值为候选减基线；负数表示候选省移箱。严格差只统计双方预算内成功；合法差包含晚返回。各行共同成功集合不同，不能累加。',
        '', '| 候选/基线 | 基线→候选预算内成功 | 共同预算内 | 严格净移箱差 | 合法净差 | 最坏合法增加 |',
        '|---|---:|---:|---:|---:|---:|']
    for item in groups:
        if item['dataset'] == 'ALL' and item['seed'] is None:
            lines.append(f"| {item['candidate']}/{item['base']} | {item['base_strict']}→{item['candidate_strict']} | {item['both_strict']} | {item['strict_move_delta']} | {item['legal_move_delta']} | {item['worst_legal_regression']} |")
    lines += ['', f"路径缓存违规{audit['cache_violations']}；验证Beam提交{audit['verified_beam_commits']}；未验证提交{audit['unverified_beam_commits']}；外部watchdog{audit['watchdogs']}。",
        '', f"原生同快照代理检查{len(static)}个：通用二次项改变选择{audit['same_context_generic_changes']}，恢复画像相对通用项改变{audit['same_context_recovery_changes']}，背景项额外改变{audit['same_context_core_changes']}。这不是同前缀续跑或端到端因果归因。",
        '', '逐种子、数据集和胜负分布见summary.json/contrasts.csv。NO_PROMOTION：这是一轮开发集机制筛查，不能代替169例评价及全1681回归。',
        '', '复现：python scripts/research_hardset_qubo_utility.py run；汇总：python scripts/research_hardset_qubo_utility.py summarize。已完成原子块不会重跑；不完整块保留并拒绝拼接。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(audit), flush=True)


def smoke():
    entries = [e for e in members() if e['split'] == 'development' and e['dataset'] == 'CV']
    entry = max(entries, key=lambda e: (e['stacks'] * e['max_height'], e['initial_n'], e['case']))
    for arm in ARMS:
        data = execute(entry, arm, 37, 0, 3.)
        write_json(OUT / 'smoke' / f'{entry["case"]}__{arm}.json', data)
        print(json.dumps({key: data['row'][key] for key in ('case', 'arm', 'valid', 'primary_ok', 'moves', 'union_calls')}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'run', 'worker', 'summarize', 'smoke'))
    parser.add_argument('--case')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int, default=37)
    parser.add_argument('--cpu', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'worker':
        worker(args.case, args.arm, args.seed, args.cpu)
    else:
        globals()[args.command]()


if __name__ == '__main__':
    main()
