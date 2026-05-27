"""Run a container reshuffling experiment from JSON config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from container_reshuffle.experiments.runner import ExperimentRunner, load_experiment_config


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("example_config.json")))
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    rows, _ = ExperimentRunner(config).run()
    print(f"Wrote {config.output_csv}")
    print(f"Wrote {config.output_json}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
