import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import finalize_paired_exploratory_analysis as recovery


def test_nested_analysis_reads_the_actual_exploratory_output(monkeypatch, tmp_path):
    """The original stream layer reads the wrong folder after base scoring."""
    exploratory = recovery.exploratory
    stream, study = exploratory.stream, exploratory.study
    intended, wrong = tmp_path / 'exploratory', tmp_path / 'old_stream'
    intended.mkdir()
    wrong.mkdir()
    original_locator = lambda scope: wrong
    monkeypatch.setattr(exploratory, 'out', lambda scope: intended)
    monkeypatch.setattr(study, 'out', lambda scope: intended)
    monkeypatch.setattr(stream, 'out', original_locator)
    monkeypatch.setattr(study, 'manifest', lambda scope: dict(
        cases=[], repeats=2, budget_sec=20., measurement_intent='EXPLORATORY',
        original_pilot_forward_screen=False))

    def base_score(scope):
        study.write_json(study.out(scope) / 'summary.json', dict(
            contrasts={}, audit_errors=[], audit_pass=True, forward_screen=False))

    monkeypatch.setattr(stream, 'original_analyze', base_score)
    recovery.analyze_existing('hardset169')
    result = json.loads((intended / 'summary.json').read_text())
    assert result['protocol_stream_errors'] == []
    assert result['measurement_intent'] == 'EXPLORATORY'
    assert result['confirmation_of_passed_pilot'] is False
    assert result['forward_screen'] is False
    assert not (wrong / 'summary.json').exists()
    assert stream.out is original_locator
