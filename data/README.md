# Data

## `input/`

Raw yard input files. Current data contains 15 zones, each with paired bay files:

- `{ZONE}_01_input.txt`
- `{ZONE}_26_input.txt`

The input format is documented in [`../docs/data-format.md`](../docs/data-format.md).

## `results/`

Generated planner reports. Existing files are historical baseline outputs for the
current sample data. New runs write here by default.
