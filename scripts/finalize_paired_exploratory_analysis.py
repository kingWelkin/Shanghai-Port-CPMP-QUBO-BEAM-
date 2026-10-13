"""Recover post-run directory routing without changing frozen timed sources."""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime
import io
import json
import shutil

import research_paired_fresh_exploratory as exploratory


def analyze_existing(scope):
    """Keep the base, stream and exploratory analysis layers in one folder."""
    stream = exploratory.stream
    saved = stream.out
    try:
        stream.out = exploratory.out
        with redirect_stdout(io.StringIO()):
            exploratory.analyze(scope)
    finally:
        stream.out = saved


def main():
    scope = 'hardset169'
    study = exploratory.study
    folder = exploratory.out(scope)
    previous = study.read(folder / 'status.json')
    if previous['status'] == 'COMPLETE':
        raise FileExistsError('The study is already finalized')
    frozen = study.manifest(scope)
    blocks = 0
    for index, item in enumerate(frozen['cases']):
        for repeat in range(frozen['repeats']):
            data = study.read(folder / 'runs' / f'c{index}__r{repeat}.json')
            if data['status'] != 'COMPLETE' or data.get('issues') or data.get('physical_errors'):
                raise ValueError(f'Unfinished or invalid timed block: {index}/{repeat}')
            blocks += 1
    recovery = folder / 'analysis_recovery'
    recovery.mkdir(exist_ok=True)
    for name in ('status.json', 'summary.json', 'branch_audit.json', 'results.csv'):
        source, backup = folder / name, recovery / ('before_' + name)
        if source.exists() and not backup.exists():
            shutil.copyfile(source, backup)
    analyze_existing(scope)
    summary = study.read(folder / 'summary.json')
    if 'protocol_stream_errors' not in summary or 'measurement_intent' not in summary:
        raise ValueError('Both analysis wrapper layers must complete')
    record = dict(
        status='COMPLETE', completed=datetime.now().isoformat(), complete_blocks=blocks,
        previous_status=previous, timed_records_modified=False,
        frozen_sources_modified=False, timed_experiment_restarted=False,
        reason='Post-run stream analyzer used paired_fresh_stream_hardset169 instead of '
               'paired_fresh_exploratory_hardset169. Rebound only its analysis output locator '
               'in this separate process; reran the original scoring and audit rules.')
    study.write_json(recovery / 'recovery.json', record)
    study.write_json(folder / 'status.json', dict(
        status='COMPLETE', finished=record['completed'],
        analysis_recovery='analysis_recovery/recovery.json'))
    print(json.dumps(dict(blocks=blocks, budget_gate=summary['budget_gate'],
        audit_pass=summary['audit_pass'], forward_screen=summary['forward_screen'],
        protocol_stream_errors=summary['protocol_stream_errors'])))


if __name__ == '__main__':
    main()
