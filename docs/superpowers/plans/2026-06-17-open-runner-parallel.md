# Open Runner Parallel Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--jobs N` process-level parallel execution to `scripts/open_cpmp_benchmarks.py` while preserving default serial behavior and deterministic CSV row order.

**Architecture:** Keep the existing serial runner as the source of truth for `--jobs 1`. For `--jobs > 1`, build picklable top-level run tasks, execute each case/variant task in a `ProcessPoolExecutor`, and merge returned rows by task order before writing existing CSV and summary outputs. Portfolio child variants stay inside one worker task so no-regret selection remains local and unchanged.

**Tech Stack:** Python `argparse`, `dataclasses`, `concurrent.futures.ProcessPoolExecutor`, pytest, existing DLTS/open runner helpers.

---

### Task 1: Parser and Task Model

**Files:**
- Modify: `tests/test_qubo.py`
- Modify: `scripts/open_cpmp_benchmarks.py`

- [ ] **Step 1: Write the failing parser test**

Add near the open runner parser tests in `tests/test_qubo.py`:

```python
def test_open_runner_jobs_defaults_to_serial_and_accepts_parallel_count():
    from open_cpmp_benchmarks import build_parser

    assert build_parser().parse_args([]).jobs == 1
    assert build_parser().parse_args(["--jobs", "4"]).jobs == 4
```

- [ ] **Step 2: Run the parser test and verify RED**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py::test_open_runner_jobs_defaults_to_serial_and_accepts_parallel_count --basetemp .tmp_pytest_open_parallel -p no:cacheprovider -q
```

Expected: FAIL with `AttributeError` or argparse rejecting `--jobs`.

- [ ] **Step 3: Add `--jobs` to the parser**

In `scripts/open_cpmp_benchmarks.py`, add:

```python
parser.add_argument(
    "--jobs",
    type=int,
    default=1,
    help="Number of worker processes for open benchmark case/variant runs. Use 1 for serial execution.",
)
```

- [ ] **Step 4: Run the parser test and verify GREEN**

Run the same focused pytest command.

Expected: PASS.

### Task 2: Build Stable Parallel Task Units

**Files:**
- Modify: `tests/test_qubo.py`
- Modify: `scripts/open_cpmp_benchmarks.py`

- [ ] **Step 1: Write the failing task-order test**

Add this test near the open runner tests:

```python
def test_open_runner_builds_stable_tasks_by_case_then_variant(tmp_path):
    from open_cpmp_benchmarks import DltsReference, OpenCpmpCase, _build_open_run_tasks

    cases = [
        OpenCpmpCase("fam_a", "fam_a_0", tmp_path / "a0.dat", None, DltsReference()),
        OpenCpmpCase("fam_b", "fam_b_0", tmp_path / "b0.dat", None, DltsReference()),
    ]

    tasks = _build_open_run_tasks(cases, ["variant_a", "variant_b"])

    assert [(task.case.case_id, task.variant) for task in tasks] == [
        ("fam_a_0", "variant_a"),
        ("fam_a_0", "variant_b"),
        ("fam_b_0", "variant_a"),
        ("fam_b_0", "variant_b"),
    ]
    assert [task.task_index for task in tasks] == [0, 1, 2, 3]
```

- [ ] **Step 2: Run the task-order test and verify RED**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py::test_open_runner_builds_stable_tasks_by_case_then_variant --basetemp .tmp_pytest_open_parallel -p no:cacheprovider -q
```

Expected: FAIL because `_build_open_run_tasks` does not exist.

- [ ] **Step 3: Add task dataclasses and builder**

In `scripts/open_cpmp_benchmarks.py`, add near the existing dataclasses:

```python
@dataclass(frozen=True)
class OpenRunTask:
    task_index: int
    case: OpenCpmpCase
    variant: str


@dataclass(frozen=True)
class OpenRunTaskResult:
    task_index: int
    case_id: str
    variant: str
    rows: list[dict[str, object]]
    skip_reason: str = ""
```

Add:

```python
def _build_open_run_tasks(cases: list[OpenCpmpCase], variants: list[str]) -> list[OpenRunTask]:
    tasks: list[OpenRunTask] = []
    for case in cases:
        for variant in variants:
            tasks.append(OpenRunTask(task_index=len(tasks), case=case, variant=variant))
    return tasks
```

- [ ] **Step 4: Run the task-order test and verify GREEN**

Run the same focused pytest command.

Expected: PASS.

### Task 3: Extract Serial Task Execution Without Behavior Change

**Files:**
- Modify: `tests/test_qubo.py`
- Modify: `scripts/open_cpmp_benchmarks.py`

- [ ] **Step 1: Write the failing task execution test with monkeypatched row runner**

Add:

```python
def test_open_runner_task_execution_returns_selected_portfolio_row(monkeypatch, tmp_path):
    import argparse

    from open_cpmp_benchmarks import (
        DltsReference,
        OpenCpmpCase,
        OpenRunTask,
        _run_open_task,
    )

    case = OpenCpmpCase("fam", "fam_0", tmp_path / "0.dat", None, DltsReference(optimal_moves=10))
    args = argparse.Namespace(alpha=10.0, skip_invalid=True)
    calls = []

    def fake_load(path, max_height=None):
        return object()

    def fake_run(case_arg, instance, args_arg, variant):
        calls.append(variant)
        valid = variant.endswith("guarded_polish")
        return {
            "family": case_arg.family,
            "case": case_arg.case_id,
            "variant": variant,
            "solver_profile": variant,
            "moves": 9 if valid else 10,
            "final_n": 0,
            "runtime_sec": 1.0,
            "business_ok": True,
            "goal_validation_ok": True,
            "full_quality_penalty": 9 if valid else 10,
        }

    monkeypatch.setattr("open_cpmp_benchmarks.load_dlts_dat", fake_load)
    monkeypatch.setattr("open_cpmp_benchmarks._run_open_variant_row", fake_run)

    result = _run_open_task(
        OpenRunTask(0, case, "qubo_scheduler_state_v5_wang2017_task_guarded_portfolio"),
        vars(args),
    )

    assert calls == [
        "qubo_scheduler_state_v5_wang_rescue_polish",
        "qubo_scheduler_state_v5_wang2017_task_guarded_polish",
    ]
    assert result.task_index == 0
    assert result.rows[0]["variant"] == "qubo_scheduler_state_v5_wang2017_task_guarded_portfolio"
    assert result.rows[0]["portfolio_selected_profile"] == "qubo_scheduler_state_v5_wang2017_task_guarded_polish"
```

- [ ] **Step 2: Run the task execution test and verify RED**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py::test_open_runner_task_execution_returns_selected_portfolio_row --basetemp .tmp_pytest_open_parallel -p no:cacheprovider -q
```

Expected: FAIL because `_run_open_task` does not exist.

- [ ] **Step 3: Add `_run_open_task`**

In `scripts/open_cpmp_benchmarks.py`, add:

```python
def _run_open_task(task: OpenRunTask, args_payload: dict[str, object]) -> OpenRunTaskResult:
    args = argparse.Namespace(**args_payload)
    try:
        instance = load_dlts_dat(task.case.instance_path, max_height=task.case.reference.max_height)
    except ValueError as exc:
        if args.skip_invalid:
            return OpenRunTaskResult(task.task_index, task.case.case_id, task.variant, [], str(exc))
        raise

    if task.variant == "qubo_scheduler_state_v5_wang2017_task_guarded_portfolio":
        candidates: list[dict[str, object]] = []
        portfolio_runtime_sec = 0.0
        for child_variant in _WANG2017_TASK_GUARDED_PORTFOLIO_VARIANTS:
            child_row = _run_open_variant_row(task.case, instance, args, child_variant)
            portfolio_runtime_sec += _numeric(child_row, "runtime_sec", 0.0)
            candidates.append(child_row)
        selected = dict(_select_open_no_regret_row(candidates))
        selected["variant"] = task.variant
        selected["portfolio_profiles"] = ",".join(_WANG2017_TASK_GUARDED_PORTFOLIO_VARIANTS)
        selected["portfolio_selected_profile"] = selected.get("solver_profile", "")
        selected["portfolio_runtime_sec"] = f"{portfolio_runtime_sec:.6f}"
        selected["full_quality_penalty"] = _open_full_quality_penalty(selected)
        return OpenRunTaskResult(task.task_index, task.case.case_id, task.variant, [selected])

    return OpenRunTaskResult(
        task.task_index,
        task.case.case_id,
        task.variant,
        [_run_open_variant_row(task.case, instance, args, task.variant)],
    )
```

- [ ] **Step 4: Run the task execution test and verify GREEN**

Run the same focused pytest command.

Expected: PASS.

### Task 4: Add Serial/Parallel Row Runners

**Files:**
- Modify: `tests/test_qubo.py`
- Modify: `scripts/open_cpmp_benchmarks.py`

- [ ] **Step 1: Write the failing row-order test for parallel collection**

Add:

```python
def test_open_runner_collects_parallel_results_in_task_order(monkeypatch):
    import argparse

    from open_cpmp_benchmarks import OpenRunTaskResult, _collect_parallel_open_rows

    tasks = [object(), object(), object()]
    args = argparse.Namespace(jobs=2)

    def fake_run(task, payload):
        index = tasks.index(task)
        return OpenRunTaskResult(
            task_index=2 - index,
            case_id=f"case_{2 - index}",
            variant="v",
            rows=[{"case": f"case_{2 - index}", "variant": "v"}],
        )

    monkeypatch.setattr("open_cpmp_benchmarks._run_open_task", fake_run)

    rows = _collect_parallel_open_rows(tasks, args)

    assert [row["case"] for row in rows] == ["case_0", "case_1", "case_2"]
```

- [ ] **Step 2: Run the row-order test and verify RED**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py::test_open_runner_collects_parallel_results_in_task_order --basetemp .tmp_pytest_open_parallel -p no:cacheprovider -q
```

Expected: FAIL because `_collect_parallel_open_rows` does not exist.

- [ ] **Step 3: Implement serial and parallel collectors**

In `scripts/open_cpmp_benchmarks.py`, import:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
```

Add:

```python
def _args_payload(args: argparse.Namespace) -> dict[str, object]:
    return dict(vars(args))


def _flatten_task_results(results: list[OpenRunTaskResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in sorted(results, key=lambda item: item.task_index):
        rows.extend(result.rows)
    return rows


def _collect_serial_open_rows(tasks: list[OpenRunTask], args: argparse.Namespace) -> list[dict[str, object]]:
    payload = _args_payload(args)
    results = [_run_open_task(task, payload) for task in tasks]
    return _flatten_task_results(results)


def _collect_parallel_open_rows(tasks: list[OpenRunTask], args: argparse.Namespace) -> list[dict[str, object]]:
    payload = _args_payload(args)
    results: list[OpenRunTaskResult] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as executor:
        futures = [executor.submit(_run_open_task, task, payload) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    return _flatten_task_results(results)
```

- [ ] **Step 4: Run the row-order test and verify GREEN**

Run the same focused pytest command.

Expected: PASS.

### Task 5: Wire `run_open_benchmark`

**Files:**
- Modify: `tests/test_qubo.py`
- Modify: `scripts/open_cpmp_benchmarks.py`

- [ ] **Step 1: Write the failing integration test for serial fallback**

Add:

```python
def test_open_runner_serial_fallback_uses_serial_collector(monkeypatch):
    from open_cpmp_benchmarks import build_parser, run_open_benchmark

    args = build_parser().parse_args(["--jobs", "1"])
    monkeypatch.setattr("open_cpmp_benchmarks.ensure_dlts_source", lambda source, auto_clone=True, refresh=False: source)
    monkeypatch.setattr("open_cpmp_benchmarks.discover_dlts_cases", lambda *a, **k: [])
    called = {"serial": 0, "parallel": 0}

    monkeypatch.setattr("open_cpmp_benchmarks._collect_serial_open_rows", lambda tasks, args: called.__setitem__("serial", called["serial"] + 1) or [])
    monkeypatch.setattr("open_cpmp_benchmarks._collect_parallel_open_rows", lambda tasks, args: called.__setitem__("parallel", called["parallel"] + 1) or [])

    assert run_open_benchmark(args) == []
    assert called == {"serial": 1, "parallel": 0}
```

- [ ] **Step 2: Write the failing integration test for `--jobs > 1`**

Add:

```python
def test_open_runner_jobs_uses_parallel_collector(monkeypatch):
    from open_cpmp_benchmarks import build_parser, run_open_benchmark

    args = build_parser().parse_args(["--jobs", "2"])
    monkeypatch.setattr("open_cpmp_benchmarks.ensure_dlts_source", lambda source, auto_clone=True, refresh=False: source)
    monkeypatch.setattr("open_cpmp_benchmarks.discover_dlts_cases", lambda *a, **k: [])
    called = {"serial": 0, "parallel": 0}

    monkeypatch.setattr("open_cpmp_benchmarks._collect_serial_open_rows", lambda tasks, args: called.__setitem__("serial", called["serial"] + 1) or [])
    monkeypatch.setattr("open_cpmp_benchmarks._collect_parallel_open_rows", lambda tasks, args: called.__setitem__("parallel", called["parallel"] + 1) or [])

    assert run_open_benchmark(args) == []
    assert called == {"serial": 0, "parallel": 1}
```

- [ ] **Step 3: Run both integration tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py::test_open_runner_serial_fallback_uses_serial_collector tests\test_qubo.py::test_open_runner_jobs_uses_parallel_collector --basetemp .tmp_pytest_open_parallel -p no:cacheprovider -q
```

Expected: FAIL until `run_open_benchmark` is routed through collectors.

- [ ] **Step 4: Wire `run_open_benchmark` to collectors**

Replace the inner serial loop in `run_open_benchmark` with:

```python
variants = _variant_names(args.variant)
tasks = _build_open_run_tasks(cases, variants)
if int(getattr(args, "jobs", 1)) <= 1:
    return _collect_serial_open_rows(tasks, args)
return _collect_parallel_open_rows(tasks, args)
```

- [ ] **Step 5: Run both integration tests and verify GREEN**

Run the same focused pytest command.

Expected: PASS.

### Task 6: Verification and Commit

**Files:**
- Modify: `docs/superpowers/plans/2026-06-17-open-runner-parallel.md`
- Create benchmark artifacts only if the verification command produces useful final evidence.

- [ ] **Step 1: Run focused open runner tests**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py -k "open_runner or parallel" --basetemp .tmp_pytest_open_parallel -p no:cacheprovider -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run full QUBO test suite**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; pytest tests\test_qubo.py --basetemp .tmp_pytest_full -p no:cacheprovider -q
```

Expected: all tests pass.

- [ ] **Step 3: Run serial vs parallel smoke benchmark**

Run serial:

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test --limit-per-family 4 --variant qubo_scheduler_state_v5_wang_rescue_polish --time-limit 20 --alpha 10 --jobs 1 --output docs\experiments\scratch_open_parallel_serial_5x5_cv1_4_t20_20260617.csv
```

Run parallel:

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test --limit-per-family 4 --variant qubo_scheduler_state_v5_wang_rescue_polish --time-limit 20 --alpha 10 --jobs 4 --output docs\experiments\scratch_open_parallel_jobs4_5x5_cv1_4_t20_20260617.csv
```

Compare rows on case, variant, final_n, moves, goal_validation_ok, business_ok, and full_quality_penalty. Runtime can differ.

- [ ] **Step 4: Clean test temp directories**

Remove only known temp directories:

```powershell
Remove-Item -LiteralPath .tmp_pytest_open_parallel,.tmp_pytest_full -Recurse -Force -ErrorAction SilentlyContinue
```

- [ ] **Step 5: Commit**

Stage only implementation, tests, plan updates, and selected benchmark artifacts. Do not stage `AGENTS.md`.

```powershell
git add scripts/open_cpmp_benchmarks.py tests/test_qubo.py docs/superpowers/plans/2026-06-17-open-runner-parallel.md
git add -f docs/experiments/scratch_open_parallel_serial_5x5_cv1_4_t20_20260617.csv docs/experiments/scratch_open_parallel_serial_5x5_cv1_4_t20_20260617_summary.md docs/experiments/scratch_open_parallel_jobs4_5x5_cv1_4_t20_20260617.csv docs/experiments/scratch_open_parallel_jobs4_5x5_cv1_4_t20_20260617_summary.md
git commit -m "Add open runner parallel jobs"
```
