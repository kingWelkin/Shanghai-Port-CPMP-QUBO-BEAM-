from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_dlts_command_uses_original_script_and_pretrained_5x5_models(tmp_path):
    from benchmark_dlts_original_cpmp import DltsRunConfig, build_dlts_command

    config = DltsRunConfig(
        dlts_root=ROOT / "external" / "DLTS",
        family="5x5_cv1_test",
        case_ids=["0", "3"],
        output_dir=tmp_path / "out",
        python_executable="python",
        time_limit=3.0,
        run_id=123,
    )

    command = build_dlts_command(config)

    assert command[:2] == ["python", str(ROOT / "external" / "DLTS" / "deep_learn.py")]
    assert command[command.index("-t") + 1] == "reference_solutions/5x5_cv1_test"
    assert command[command.index("-m") + 1] == "pre-trained_models/pm_dnn_model_5x5.h5"
    assert command[command.index("-v") + 1] == "pre-trained_models/pm_dnn_value_model_5x5.h5"
    assert command[command.index("--val_instances") + 1] == "0,3"
    assert command[command.index("--run_id") + 1] == "123"
    assert "-s" in command


def test_dlts_runner_rejects_family_without_matching_pretrained_model():
    from benchmark_dlts_original_cpmp import DltsRunConfig, build_dlts_command

    config = DltsRunConfig(
        dlts_root=ROOT / "external" / "DLTS",
        family="7x5_cv1_test",
        case_ids=["0"],
        output_dir=ROOT / "docs" / "experiments",
    )

    with pytest.raises(ValueError, match="pre-trained DLTS model"):
        build_dlts_command(config)


def test_parse_dlts_result_csv_normalizes_rows(tmp_path):
    from benchmark_dlts_original_cpmp import parse_dlts_result_csv

    result_path = tmp_path / "results_123_5x5_cv1_test.csv"
    with result_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "id", "moves", "time", "opt", "nodes"])
        writer.writerow(["5x5_cv1_test", "0", "27", "0.52", "?", "1234"])

    rows = parse_dlts_result_csv(result_path, family="5x5_cv1_test", reference_by_case={"0": 26})

    assert rows == [
        {
            "family": "5x5_cv1_test",
            "case": "5x5_cv1_test_0",
            "solver": "dlts_original",
            "moves": 27,
            "runtime_sec": 0.52,
            "nodes": 1234,
            "reference_moves": 26,
            "move_gap": 1,
            "status": "SOLVED",
        }
    ]
