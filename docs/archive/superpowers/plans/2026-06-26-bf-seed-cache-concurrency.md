# BF Seed Cache Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in FRG seed materialization and consumption path so BF hard-family `jobs=1` and `jobs=8` runs can use identical seed evidence.

**Architecture:** Keep the compact solver's default live seed path unchanged. Add JSONL seed cache helpers in `scripts/benchmark_compact_beam_qubo_n0.py`, expose cache telemetry through existing result rows, and let `scripts/benchmark_compact_paper_set.py` optionally materialize a cache in the parent process before serial or parallel row collection.

**Tech Stack:** Python, JSONL, existing `Move`/`FrgSeedResult` dataclasses, compact paper runner, pytest.

---

### Task 1: Compact Runner Seed Cache Consumption

**Files:**
- Modify: `tests/test_qubo.py`
- Modify: `scripts/benchmark_compact_beam_qubo_n0.py`

- [ ] **Step 1: Write the failing artifact-consumption test**

Add this test near the existing FRG seed tests in `tests/test_qubo.py`:

```python
def test_run_compact_beam_uses_materialized_frg_seed_before_live(monkeypatch, tmp_path):
    import json

    import benchmark_compact_beam_qubo_n0 as bench

    instance = Instance(
        "seed_cached",
        ProblemVariant.PREMARSHALLING_CPMP,
        8,
        5,
        4,
        [[1, 2], [2, 1], [], [], [], [], [], []],
    )
    cached_moves = (Move(step=0, box=2, from_stack=0, to_stack=1, from_level=2, to_level=1),)
    args = bench.make_compact_args(
        solver_profile="speed_v2",
        frg_seed_enabled=True,
        frg_seed_allow_large_over_cap=True,
        frg_seed_cache_input=tmp_path / "frg_seed_cache.jsonl",
        time_limit=0.000001,
        alpha=10.0,
    )
    cache_key = bench._frg_seed_cache_key("seed_cached", instance, args)
    record = bench._frg_seed_record_from_result(
        "seed_cached",
        instance,
        args,
        bench.FrgSeedResult(
            moves=cached_moves,
            final_n=0,
            generated=11,
            runtime_sec=0.123,
            status="FRG_FEASIBLE",
            cap_ok=False,
        ),
    )
    args.frg_seed_cache_input.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        bench,
        "_frg_seed_incumbent",
        lambda instance, args, deadline: (_ for _ in ()).throw(AssertionError("live seed should not run")),
    )

    result = bench.run_compact_beam_instance("seed_cached", instance, args)

    assert result.moves == list(cached_moves)
    assert result.frg_seed_used is True
    assert result.frg_seed_moves == 1
    assert result.frg_seed_status == "FRG_FEASIBLE"
    assert result.frg_seed_generated == 11
    assert result.frg_seed_cache_hit is True
    assert result.frg_seed_cache_source == "artifact"
    assert result.frg_seed_cache_key == cache_key
```

- [ ] **Step 2: Write the failing stale-artifact miss test**

Add this test in the same FRG seed test block:

```python
def test_run_compact_beam_reports_materialized_frg_seed_cache_miss(monkeypatch, tmp_path):
    import json

    import benchmark_compact_beam_qubo_n0 as bench

    instance = Instance(
        "seed_miss",
        ProblemVariant.PREMARSHALLING_CPMP,
        8,
        5,
        4,
        [[1, 2], [2, 1], [], [], [], [], [], []],
    )
    cache_path = tmp_path / "frg_seed_cache.jsonl"
    cache_path.write_text(json.dumps({"cache_key": "different-case", "moves": []}) + "\n", encoding="utf-8")
    live_seed = bench.FrgSeedResult(
        moves=(Move(step=0, box=2, from_stack=0, to_stack=1, from_level=2, to_level=1),),
        final_n=0,
        generated=3,
        runtime_sec=0.004,
        status="FRG_FEASIBLE",
        cap_ok=False,
    )
    monkeypatch.setattr(bench, "_frg_seed_incumbent", lambda instance, args, deadline: live_seed)
    args = bench.make_compact_args(
        solver_profile="speed_v2",
        frg_seed_enabled=True,
        frg_seed_allow_large_over_cap=True,
        frg_seed_cache_input=cache_path,
        time_limit=0.000001,
        alpha=10.0,
    )

    result = bench.run_compact_beam_instance("seed_miss", instance, args)

    assert result.frg_seed_used is True
    assert result.frg_seed_cache_hit is False
    assert result.frg_seed_cache_source == "miss_live"
    assert result.frg_seed_generated == 3
```

- [ ] **Step 3: Run the focused tests to verify red**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; python -m pytest tests\test_qubo.py::test_run_compact_beam_uses_materialized_frg_seed_before_live tests\test_qubo.py::test_run_compact_beam_reports_materialized_frg_seed_cache_miss -q --basetemp .tmp_pytest_bf_seed_cache_red
```

Expected before implementation: FAIL with `AttributeError` for `_frg_seed_cache_key` or missing result telemetry fields.

- [ ] **Step 4: Add compact seed cache fields and helpers**

In `scripts/benchmark_compact_beam_qubo_n0.py`, add `import json`, add these fields to `COMPACT_EXTRA_FIELDS`, `CompactBeamResult`, and `make_compact_args` defaults:

```python
"frg_seed_cache_input",
"frg_seed_cache_key",
"frg_seed_cache_hit",
"frg_seed_cache_source",
```

Use these defaults:

```python
"frg_seed_cache_input": None,
"frg_seed_cache_key": "",
"frg_seed_cache_hit": False,
"frg_seed_cache_source": "",
```

Add helpers near `_frg_seed_incumbent(...)`:

```python
def _frg_seed_move_to_record(move: Move) -> dict[str, int | None]:
    return {
        "step": move.step,
        "box": move.box,
        "from_stack": move.from_stack,
        "to_stack": move.to_stack,
        "from_level": move.from_level,
        "to_level": move.to_level,
    }


def _frg_seed_move_from_record(record: dict[str, object]) -> Move:
    return Move(
        step=int(record["step"]),
        box=int(record["box"]),
        from_stack=int(record["from_stack"]),
        to_stack=int(record["to_stack"]),
        from_level=None if record.get("from_level") in (None, "") else int(record["from_level"]),
        to_level=None if record.get("to_level") in (None, "") else int(record["to_level"]),
    )


def _frg_seed_cache_key(case: str, instance: Instance, args: argparse.Namespace) -> str:
    profile = str(getattr(args, "solver_profile", ""))
    parts = [
        f"case={case}",
        f"profile={profile}",
        f"alpha={float(getattr(args, 'alpha', 0.0) or 0.0):.6g}",
        f"stacks={instance.stack_count}",
        f"height={instance.max_height}",
        f"containers={instance.container_count}",
        f"time={float(getattr(args, 'frg_seed_time_limit', 0.0) or 0.0):.6g}",
        f"min_steps={int(getattr(args, 'frg_seed_min_max_steps', 0) or 0)}",
        f"mult={int(getattr(args, 'frg_seed_max_steps_multiplier', 0) or 0)}",
        f"cap={int(getattr(args, 'frg_seed_max_steps_cap', 0) or 0)}",
        f"respect_cap={bool(getattr(args, 'frg_seed_respect_cap', True))}",
        f"paper_pure={bool(getattr(args, 'frg_seed_paper_pure', True))}",
    ]
    return "|".join(parts)


def _frg_seed_record_from_result(
    case: str,
    instance: Instance,
    args: argparse.Namespace,
    seed: FrgSeedResult,
) -> dict[str, object]:
    return {
        "cache_key": _frg_seed_cache_key(case, instance, args),
        "case": case,
        "profile": str(getattr(args, "solver_profile", "")),
        "alpha": float(getattr(args, "alpha", 0.0) or 0.0),
        "frg_seed_time_limit": float(getattr(args, "frg_seed_time_limit", 0.0) or 0.0),
        "frg_seed_min_max_steps": int(getattr(args, "frg_seed_min_max_steps", 0) or 0),
        "frg_seed_max_steps_multiplier": int(getattr(args, "frg_seed_max_steps_multiplier", 0) or 0),
        "frg_seed_max_steps_cap": int(getattr(args, "frg_seed_max_steps_cap", 0) or 0),
        "frg_seed_respect_cap": bool(getattr(args, "frg_seed_respect_cap", True)),
        "frg_seed_paper_pure": bool(getattr(args, "frg_seed_paper_pure", True)),
        "stack_count": instance.stack_count,
        "max_height": instance.max_height,
        "container_count": instance.container_count,
        "moves": [_frg_seed_move_to_record(move) for move in seed.moves],
        "final_n": seed.final_n,
        "generated": seed.generated,
        "runtime_sec": seed.runtime_sec,
        "status": seed.status,
        "cap_ok": seed.cap_ok,
    }
```

- [ ] **Step 5: Wire artifact consumption before live seed**

Add a path loader that reads JSONL by `cache_key` and returns `FrgSeedResult | None`. In `run_compact_beam_instance(...)`, replace the direct call to `_frg_seed_incumbent(...)` with:

```python
frg_seed_cache_key = _frg_seed_cache_key(case, instance, args)
frg_seed, frg_seed_cache_hit, frg_seed_cache_source = _frg_seed_from_cache(case, instance, args, frg_seed_cache_key)
if frg_seed is None:
    frg_seed = _frg_seed_incumbent(instance, args, deadline)
    if frg_seed is not None:
        frg_seed_cache_source = "miss_live" if getattr(args, "frg_seed_cache_input", None) else "live"
    elif getattr(args, "frg_seed_cache_input", None):
        frg_seed_cache_source = "miss"
```

Include `frg_seed_cache_key`, `frg_seed_cache_hit`, and `frg_seed_cache_source` in the returned `CompactBeamResult` and row dictionaries.

- [ ] **Step 6: Run the focused tests to verify green**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; python -m pytest tests\test_qubo.py::test_run_compact_beam_uses_materialized_frg_seed_before_live tests\test_qubo.py::test_run_compact_beam_reports_materialized_frg_seed_cache_miss -q --basetemp .tmp_pytest_bf_seed_cache_green
```

Expected after implementation: both tests pass.

- [ ] **Step 7: Commit Task 1**

Run:

```powershell
git add tests\test_qubo.py scripts\benchmark_compact_beam_qubo_n0.py
git commit -m "feat: consume materialized frg seed cache"
```

### Task 2: Paper Runner Seed Materialization

**Files:**
- Modify: `tests/test_bs_frg_paper_benchmarks.py`
- Modify: `scripts/benchmark_compact_paper_set.py`

- [ ] **Step 1: Write the failing paper runner materialization test**

Add this test to `tests/test_bs_frg_paper_benchmarks.py`:

```python
def test_paper_runner_materializes_frg_seed_cache_before_collecting_rows(monkeypatch, tmp_path: Path):
    from argparse import Namespace

    import benchmark_compact_paper_set as paper
    import benchmark_compact_beam_qubo_n0 as bench
    from benchmark_bs_frg_paper import PaperCase

    cache_path = tmp_path / "paper_seed_cache.jsonl"
    case_path = tmp_path / "case.bay"
    case_path.write_text("", encoding="utf-8")
    task = paper.PaperRunTask(
        task_index=0,
        case=PaperCase("BF", "BF14", "case_1", case_path),
        variant="qubo_scheduler_state_v5_qubo_diversity_union_polish",
    )
    instance = Instance(
        "case_1",
        ProblemVariant.PREMARSHALLING_CPMP,
        8,
        5,
        4,
        [[1, 2], [2, 1], [], [], [], [], [], []],
    )
    seed = bench.FrgSeedResult(
        moves=(Move(step=0, box=2, from_stack=0, to_stack=1, from_level=2, to_level=1),),
        final_n=0,
        generated=5,
        runtime_sec=0.01,
        status="FRG_FEASIBLE",
        cap_ok=False,
    )
    monkeypatch.setattr(paper, "_load_paper_case", lambda case: instance)
    monkeypatch.setattr(bench, "_frg_seed_incumbent", lambda instance, args, deadline: seed)
    args = Namespace(
        alpha=10.0,
        time_limit=20.0,
        frg_seed_cache_output=cache_path,
        frg_seed_cache_input=None,
    )

    payload = paper._materialize_frg_seed_cache([task], args)

    assert payload["frg_seed_cache_input"] == cache_path
    text = cache_path.read_text(encoding="utf-8")
    assert '"case": "case_1"' in text
    assert '"status": "FRG_FEASIBLE"' in text
```

- [ ] **Step 2: Run the focused test to verify red**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; python -m pytest tests\test_bs_frg_paper_benchmarks.py::test_paper_runner_materializes_frg_seed_cache_before_collecting_rows -q --basetemp .tmp_pytest_bf_seed_paper_red
```

Expected before implementation: FAIL with `AttributeError` for `_materialize_frg_seed_cache`.

- [ ] **Step 3: Add paper runner CLI arguments**

In `scripts/benchmark_compact_paper_set.py`, add parser arguments:

```python
parser.add_argument("--frg-seed-cache-input", type=Path)
parser.add_argument("--frg-seed-cache-output", type=Path)
```

- [ ] **Step 4: Implement parent-process materialization**

Import `json`, `time`, and these helpers from `benchmark_compact_beam_qubo_n0`:

```python
_apply_dynamic_profile,
_apply_qubo_influence_profile,
_apply_solver_profile,
_frg_seed_incumbent,
_frg_seed_record_from_result,
```

Add `_materialize_frg_seed_cache(tasks, args)`:

```python
def _materialize_frg_seed_cache(tasks: list[PaperRunTask], args: argparse.Namespace) -> dict[str, object]:
    payload = dict(vars(args))
    output = getattr(args, "frg_seed_cache_output", None)
    if output is None:
        return payload
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in tasks:
        instance = _load_paper_case(task.case)
        compact_args = _args_for_variant(args, task.variant, instance)
        _apply_solver_profile(compact_args)
        _apply_qubo_influence_profile(compact_args)
        _apply_dynamic_profile(compact_args)
        configured_limit = float(getattr(compact_args, "frg_seed_time_limit", 0.0) or 0.0)
        deadline = time.perf_counter() + max(0.001, configured_limit)
        seed = _frg_seed_incumbent(instance, compact_args, deadline)
        if seed is not None:
            rows.append(_frg_seed_record_from_result(task.case.case_id, instance, compact_args, seed))
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    payload["frg_seed_cache_input"] = output
    return payload
```

In `_collect_rows(...)`, call `_materialize_frg_seed_cache(tasks, args)` before serial or parallel dispatch and use the returned payload for every `_run_task(...)`.

- [ ] **Step 5: Pass cache input through paper task execution**

No extra worker object is needed. The existing `payload = dict(vars(args))` path is already picklable when the value is a `Path`. Add `frg_seed_cache_input` to the `_args_for_variant(...)` passthrough list in `scripts/open_cpmp_benchmarks.py` next to the existing `frg_seed_*` fields:

```python
"frg_seed_cache_input",
```

- [ ] **Step 6: Run the focused test to verify green**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; python -m pytest tests\test_bs_frg_paper_benchmarks.py::test_paper_runner_materializes_frg_seed_cache_before_collecting_rows -q --basetemp .tmp_pytest_bf_seed_paper_green
```

Expected after implementation: the test passes.

- [ ] **Step 7: Commit Task 2**

Run:

```powershell
git add tests\test_bs_frg_paper_benchmarks.py scripts\benchmark_compact_paper_set.py scripts\open_cpmp_benchmarks.py
git commit -m "bench: materialize frg seed cache for paper runs"
```

### Task 3: Verification and BF Smoke

**Files:**
- Create: `docs/experiments/bf_seed_cache_concurrency_smoke_20260626.md`

- [ ] **Step 1: Run focused regression tests**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; python -m pytest tests\test_qubo.py::test_run_compact_beam_uses_materialized_frg_seed_before_live tests\test_qubo.py::test_run_compact_beam_reports_materialized_frg_seed_cache_miss tests\test_bs_frg_paper_benchmarks.py::test_paper_runner_materializes_frg_seed_cache_before_collecting_rows -q --basetemp .tmp_pytest_bf_seed_cache_focus
```

Expected: all selected tests pass.

- [ ] **Step 2: Run FRG and paper regression slice**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; python -m pytest tests\test_qubo.py -k "frg_seed or open_runner_accepts_frg_seed or union_diversity" tests\test_bs_frg_paper_benchmarks.py -q --basetemp .tmp_pytest_bf_seed_cache_regression
```

Expected: selected tests pass.

- [ ] **Step 3: Compile touched scripts**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; python -m py_compile scripts\benchmark_compact_beam_qubo_n0.py scripts\benchmark_compact_paper_set.py scripts\open_cpmp_benchmarks.py
```

Expected: command exits 0 with no syntax errors.

- [ ] **Step 4: Run targeted BF smoke if paper data is present**

Run:

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF --families BF14 BF32 --sample-per-family 1 --variant qubo_scheduler_state_v5_qubo_diversity_union_polish --alpha 10 --time-limit 20 --jobs 1 --frg-seed-cache-output docs\experiments\bf_seed_cache_smoke_20260626.jsonl --output docs\experiments\bf_seed_cache_smoke_jobs1_20260626.csv
```

If `external\paper_cpmp` is unavailable, record that the smoke was skipped because the source data is absent. If it is present, rerun with `--jobs 2` and `--frg-seed-cache-input docs\experiments\bf_seed_cache_smoke_20260626.jsonl` to verify parallel consumption:

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF --families BF14 BF32 --sample-per-family 1 --variant qubo_scheduler_state_v5_qubo_diversity_union_polish --alpha 10 --time-limit 20 --jobs 2 --frg-seed-cache-input docs\experiments\bf_seed_cache_smoke_20260626.jsonl --output docs\experiments\bf_seed_cache_smoke_jobs2_20260626.csv
```

- [ ] **Step 5: Record smoke outcome**

If the BF smoke runs, create `docs/experiments/bf_seed_cache_concurrency_smoke_20260626.md` with:

```markdown
# BF Seed Cache Concurrency Smoke, 2026-06-26

Scope:
- Variant: `qubo_scheduler_state_v5_qubo_diversity_union_polish`
- Families: `BF14`, `BF32`
- Purpose: verify that parent-process FRG seed materialization can be consumed by a parallel paper runner.

Outcome:
- Focused tests: PASS
- Regression slice: PASS
- Compile check: PASS
- BF smoke: PASS

Interpretation:
- This artifact path is experimental and does not promote lazy portfolio or CV receiver-debt work.
```

If the local BF paper source data is absent, create the same file with this exact outcome block instead:

```markdown
Outcome:
- Focused tests: PASS
- Regression slice: PASS
- Compile check: PASS
- BF smoke: SKIPPED, `external\paper_cpmp` was not available in this checkout.
```

- [ ] **Step 6: Run diff and status checks**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; status shows only intended files plus pre-existing untracked temp artifacts.

- [ ] **Step 7: Commit Task 3**

Run:

```powershell
git add docs\experiments\bf_seed_cache_concurrency_smoke_20260626.md
git commit -m "docs: record bf seed cache smoke"
```

### Self-Review

- Task 1 implements opt-in artifact consumption and row telemetry.
- Task 2 implements parent-process seed materialization for paper runs.
- Task 3 verifies focused behavior, script syntax, and a BF smoke if local paper data exists.
- Default live FRG seed behavior stays unchanged unless `frg_seed_cache_input` or `--frg-seed-cache-*` is supplied.
- CV receiver-debt and lazy portfolio remain outside this implementation.
