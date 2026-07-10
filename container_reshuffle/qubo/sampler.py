"""QUBO sampler interfaces and fallback implementations."""

from __future__ import annotations

import json
import math
import csv
import os
import random
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .model import QuboModel


@dataclass
class QuboSamplerConfig:
    """QUBO sampler configuration."""

    num_reads: int = 50
    random_seed: int = 1
    time_limit_sec: float = 60.0
    respect_time_limit: bool = True
    external_command: Optional[str] = None
    working_dir: Optional[str] = None
    sa_iterations: int = 5000
    initial_temperature: float = 5.0
    final_temperature: float = 0.01
    sampler_name: str = "sa"
    api_base_url: Optional[str] = None
    api_endpoint: Optional[str] = None
    api_gear: Optional[int] = None
    api_problem_type: str = "binary"
    api_timeout_sec: float = 0.0
    api_working_dir: Optional[str] = None
    api_project_src: Optional[str] = None
    api_fallback: bool = True


@dataclass
class QuboSample:
    """One QUBO binary sample."""

    binary_values: Dict[int, int]
    energy: float


@dataclass
class QuboSampleSet:
    """Sampler result."""

    samples: List[QuboSample] = field(default_factory=list)
    sampler_name: str = ""
    runtime_sec: float = 0.0


@dataclass(frozen=True)
class _CompiledQuboTerms:
    offset: float
    linear: List[float]
    quadratic: List[tuple[int, int, float]]
    adjacency: List[List[tuple[int, float]]]

    def energy(self, values: List[int]) -> float:
        energy = self.offset
        for idx, coeff in enumerate(self.linear):
            if values[idx]:
                energy += coeff
        for i, j, coeff in self.quadratic:
            if values[i] and values[j]:
                energy += coeff
        return float(energy)

    def flip_delta(self, values: List[int], bit: int) -> float:
        direction = 1.0 if values[bit] == 0 else -1.0
        local = self.linear[bit]
        for other, coeff in self.adjacency[bit]:
            if values[other]:
                local += coeff
        return float(direction * local)


def _compile_qubo_terms(model: QuboModel) -> _CompiledQuboTerms:
    n = len(model.var_index)
    linear = [0.0] * n
    quadratic: List[tuple[int, int, float]] = []
    adjacency: List[List[tuple[int, float]]] = [[] for _ in range(n)]
    for (i, j), coeff in model.Q.items():
        i = int(i)
        j = int(j)
        coeff = float(coeff)
        if i == j:
            linear[i] += coeff
            continue
        quadratic.append((i, j, coeff))
        adjacency[i].append((j, coeff))
        adjacency[j].append((i, coeff))
    return _CompiledQuboTerms(float(model.offset), linear, quadratic, adjacency)


class IQuboSampler:
    """QUBO sampler protocol."""

    def sample(self, model: QuboModel, config: QuboSamplerConfig) -> QuboSampleSet:
        """Sample a QUBO model."""

        raise NotImplementedError


class SimulatedAnnealingQuboSampler(IQuboSampler):
    """Simple Metropolis simulated annealing fallback."""

    name = "simulated_annealing"

    def sample(self, model: QuboModel, config: QuboSamplerConfig) -> QuboSampleSet:
        """Return best samples from repeated random annealing runs."""

        start = time.perf_counter()
        rng = random.Random(config.random_seed)
        n = len(model.var_index)
        reads = max(1, config.num_reads)
        iterations = max(1, config.sa_iterations)
        samples: List[QuboSample] = []
        deadline = start + max(0.001, config.time_limit_sec) if config.respect_time_limit else math.inf
        compiled = _compile_qubo_terms(model)

        if n == 0:
            return QuboSampleSet(
                samples=[QuboSample({}, compiled.offset)],
                sampler_name=self.name,
                runtime_sec=time.perf_counter() - start,
            )

        for _ in range(reads):
            x = [rng.randint(0, 1) for _ in range(n)]
            energy = compiled.energy(x)
            best_x = list(x)
            best_energy = energy
            for it in range(iterations):
                if time.perf_counter() > deadline:
                    break
                frac = it / max(1, iterations - 1)
                temp = config.initial_temperature * ((config.final_temperature / config.initial_temperature) ** frac)
                bit = rng.randrange(n) if n else 0
                delta = compiled.flip_delta(x, bit)
                x[bit] = 1 - x[bit]
                new_energy = energy + delta
                accept = delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-12))
                if accept:
                    energy = new_energy
                    if energy < best_energy:
                        best_energy = energy
                        best_x = list(x)
                else:
                    x[bit] = 1 - x[bit]
            samples.append(QuboSample({i: value for i, value in enumerate(best_x)}, best_energy))
            if time.perf_counter() > deadline:
                break
        samples.sort(key=lambda s: s.energy)
        return QuboSampleSet(samples=samples, sampler_name=self.name, runtime_sec=time.perf_counter() - start)


def _samples_from_json(path: Path) -> List[QuboSample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for row in data.get("samples", []):
        values = {int(k): int(v) for k, v in row.get("binary_values", {}).items()}
        out.append(QuboSample(values, float(row.get("energy", 0.0))))
    out.sort(key=lambda s: s.energy)
    return out


class ExternalCommandQuboSampler(IQuboSampler):
    """JSON exchange sampler that falls back to internal SA on failure."""

    name = "external_command"

    def __init__(self, fallback: Optional[IQuboSampler] = None) -> None:
        self.fallback = fallback or SimulatedAnnealingQuboSampler()

    def sample(self, model: QuboModel, config: QuboSamplerConfig) -> QuboSampleSet:
        """Write ``qubo_input.json``, run an external command, read ``qubo_output.json``."""

        if not config.external_command:
            return self.fallback.sample(model, config)
        start = time.perf_counter()
        work = Path(config.working_dir or ".").resolve()
        work.mkdir(parents=True, exist_ok=True)
        input_path = work / "qubo_input.json"
        output_path = work / "qubo_output.json"
        model.to_json(input_path)
        try:
            subprocess.run(
                config.external_command,
                cwd=work,
                shell=True,
                check=True,
                timeout=config.time_limit_sec,
                capture_output=True,
                text=True,
            )
            samples = _samples_from_json(output_path)
            if not samples:
                raise RuntimeError("external sampler returned no samples")
            return QuboSampleSet(samples, self.name, time.perf_counter() - start)
        except Exception:
            return self.fallback.sample(model, config)


class ExternalPythonDimodSampler(IQuboSampler):
    """Optional dimod sampler using the active Python environment."""

    name = "python_dimod"

    def __init__(self, fallback: Optional[IQuboSampler] = None) -> None:
        self.fallback = fallback or SimulatedAnnealingQuboSampler()

    def sample(self, model: QuboModel, config: QuboSamplerConfig) -> QuboSampleSet:
        """Use dimod if installed, otherwise fall back to SA."""

        start = time.perf_counter()
        try:
            import dimod  # type: ignore

            bqm = dimod.BinaryQuadraticModel({}, {}, model.offset, dimod.BINARY)
            for (i, j), coeff in model.Q.items():
                if i == j:
                    bqm.add_variable(i, coeff)
                else:
                    bqm.add_interaction(i, j, coeff)
            sampler = dimod.SimulatedAnnealingSampler()
            ss = sampler.sample(bqm, num_reads=config.num_reads, seed=config.random_seed)
            samples = []
            for row in ss.data(["sample", "energy"]):
                values = {int(k): int(v) for k, v in row.sample.items()}
                samples.append(QuboSample(values, float(row.energy)))
            samples.sort(key=lambda s: s.energy)
            return QuboSampleSet(samples, self.name, time.perf_counter() - start)
        except Exception:
            return self.fallback.sample(model, config)


class _HttpCsvApiClient:
    """Small CSV API client used when the legacy package is unavailable."""

    def __init__(
        self,
        *,
        base_url: str,
        endpoint: str,
        gear: int,
        problem_type: str,
        timeout_sec: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint = "/" + endpoint.lstrip("/")
        self.gear = int(gear)
        self.problem_type = problem_type
        self.timeout_sec = max(0.001, float(timeout_sec))

    def submit_csv_file(self, csv_path: Path) -> Any:
        """Submit a matrix CSV with a requests client when available."""

        url = f"{self.base_url}{self.endpoint}"
        try:
            import requests  # type: ignore

            csv_text = csv_path.read_text(encoding="utf-8")
            response = requests.post(
                url,
                files={"data": csv_text},
                params={"gear": self.gear, "type": self.problem_type},
                timeout=self.timeout_sec,
            )
            try:
                response.raise_for_status()
            except Exception as exc:
                body = response.text[:500].replace("\n", " ")
                raise RuntimeError(f"API HTTP {response.status_code}: {body}") from exc
            return response.json()
        except ImportError:
            return self._submit_with_urllib(url, csv_path)

    def _submit_with_urllib(self, url: str, csv_path: Path) -> Any:
        boundary = f"----container-reshuffle-{int(time.time() * 1000)}"
        csv_bytes = csv_path.read_text(encoding="utf-8").encode("utf-8")
        parts = [
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="data"; filename="data"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode("utf-8"),
            csv_bytes,
            f"\r\n--{boundary}--\r\n".encode("utf-8"),
        ]
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}gear={self.gear}&type={self.problem_type}"
        request = urllib.request.Request(
            url,
            data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)


class IsingMatrixApiQuboSampler(IQuboSampler):
    """Sampler bridge to the Ising matrix API.

    The API sample is intentionally only a QUBO sample. Downstream builders
    still decode, repair, and validate through the classical engine.
    """

    name = "ising_matrix_api"

    def __init__(
        self,
        *,
        project_src: Optional[str] = None,
        base_url: Optional[str] = None,
        endpoint: Optional[str] = None,
        gear: Optional[int] = None,
        problem_type: str = "binary",
        fallback: Optional[IQuboSampler] = None,
    ) -> None:
        self.project_src = project_src or os.getenv("ISING_MATRIX_PROJECT_SRC")
        self.base_url = base_url or os.getenv("ISING_API_BASE_URL", "http://192.168.137.100:5001")
        self.endpoint = endpoint or os.getenv("ISING_API_ENDPOINT", "/api/v1/job/")
        self.gear = int(gear if gear is not None else os.getenv("ISING_API_GEAR", "2"))
        self.problem_type = problem_type
        self.fallback = fallback or SimulatedAnnealingQuboSampler()

    def sample(self, model: QuboModel, config: QuboSamplerConfig) -> QuboSampleSet:
        """Compile QUBO to CSV, submit to API, and normalize returned samples."""

        start = time.perf_counter()
        try:
            project_src = config.api_project_src or self.project_src
            modules = self._load_modules(project_src)
            Q_float = _qubo_model_to_dense_upper(model)
            terms = modules["QuboTerms"](Q=Q_float, offset=model.offset, name="container_reshuffle_qubo")
            compiler_cfg = modules["Int8CompilerConfig"](
                bits=8,
                beta_mode="probe_rank",
                random_seed=int(config.random_seed),
                probe_count=min(512, max(64, 8 * max(1, len(model.var_index)))),
            )
            compiled = modules["compile_qubo_int8"](terms, compiler_cfg)
            work = Path(config.api_working_dir or config.working_dir or "tmp/ising_api_qubo").resolve()
            work.mkdir(parents=True, exist_ok=True)
            csv_path = work / f"qubo_{int(time.time() * 1000)}.csv"
            _write_matrix_csv(csv_path, compiled.Q_int)
            timeout_sec = float(config.api_timeout_sec or config.time_limit_sec)
            client = modules["IsingApiClient"](
                base_url=config.api_base_url or self.base_url,
                endpoint=config.api_endpoint or self.endpoint,
                gear=int(config.api_gear if config.api_gear is not None else self.gear),
                problem_type=config.api_problem_type or self.problem_type,
                timeout_sec=timeout_sec,
            )
            response = client.submit_csv_file(csv_path)
            raw_results = modules["extract_api_results"](response)
            samples = _api_results_to_samples(raw_results, model)
            if not samples:
                raise RuntimeError("API returned no usable sample")
            sampler_name = self.name
            if bool(config.api_fallback):
                fallback_set = self.fallback.sample(model, config)
                samples.extend(fallback_set.samples)
                sampler_name = f"{self.name}_shadow_{fallback_set.sampler_name}"
            samples.sort(key=lambda s: s.energy)
            return QuboSampleSet(samples, sampler_name, time.perf_counter() - start)
        except Exception:
            if not bool(config.api_fallback):
                raise
            fallback_set = self.fallback.sample(model, config)
            fallback_set.sampler_name = f"{self.name}_fallback_{fallback_set.sampler_name}"
            return fallback_set

    def _load_modules(self, project_src: Optional[str] = None) -> Dict[str, object]:
        errors: List[Exception] = []
        for src in _candidate_api_project_sources(project_src):
            if src and src not in sys.path:
                sys.path.insert(0, src)
            try:
                from ising_matrix_synthetic import (  # type: ignore
                    Int8CompilerConfig,
                    IsingApiClient,
                    QuboTerms,
                    compile_qubo_int8,
                    extract_api_results,
                )

                return {
                    "Int8CompilerConfig": Int8CompilerConfig,
                    "IsingApiClient": IsingApiClient,
                    "QuboTerms": QuboTerms,
                    "compile_qubo_int8": compile_qubo_int8,
                    "extract_api_results": extract_api_results,
                }
            except Exception as exc:
                errors.append(exc)
                try:
                    from qubo_matrix_compiler._runtime.compiler import (  # type: ignore
                        Int8CompilerConfig,
                        compile_qubo_int8,
                    )
                    from qubo_matrix_compiler._runtime.qubo import QuboTerms  # type: ignore

                    return {
                        "Int8CompilerConfig": Int8CompilerConfig,
                        "IsingApiClient": _HttpCsvApiClient,
                        "QuboTerms": QuboTerms,
                        "compile_qubo_int8": compile_qubo_int8,
                        "extract_api_results": _extract_api_results,
                    }
                except Exception as split_exc:
                    errors.append(split_exc)
        if errors:
            raise errors[-1]
        raise ImportError("no Ising API compiler modules found")


def _candidate_api_project_sources(project_src: Optional[str]) -> List[str]:
    candidates = [
        project_src,
        os.getenv("ISING_MATRIX_PROJECT_SRC"),
        os.getenv("QUBO_MATRIX_COMPILER_SRC"),
        os.getenv("QUBO_PIPELINE_CONTRACTS_SRC"),
    ]
    root = Path(__file__).resolve().parents[2]
    desktop = Path.home() / "Desktop"
    candidates.extend(
        [
            str(root.parent / "ising_matrix_synthetic_project" / "src"),
            str(root.parent / "qubo-matrix-compiler" / "src"),
            str(root.parent.parent / "qubo-matrix-compiler" / "src"),
            str(desktop / "qubo-matrix-compiler" / "src"),
            str(root.parent / "qubo-pipeline-contracts" / "src"),
            str(root.parent.parent / "qubo-pipeline-contracts" / "src"),
            str(desktop / "qubo-pipeline-contracts" / "src"),
        ]
    )
    out: List[str] = []
    for item in candidates:
        if not item:
            continue
        text = str(Path(item).resolve())
        if text not in out:
            out.append(text)
    return out


def _qubo_model_to_dense_upper(model: QuboModel):
    import numpy as np

    n = len(model.var_index)
    Q = np.zeros((n, n), dtype=float)
    for (i, j), coeff in model.Q.items():
        a, b = sorted((int(i), int(j)))
        Q[a, b] += float(coeff)
    return Q


def _write_matrix_csv(path: Path, matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in matrix:
            writer.writerow([int(v) for v in row])


def _extract_api_results(response: Any) -> List[Dict[str, Any]]:
    if isinstance(response, str):
        response = json.loads(response)
    results: List[Dict[str, Any]] = []

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            sample = None
            for key in ("sample", "solution", "bits", "spins", "spin_config", "x", "state", "binary_values"):
                if key in obj:
                    sample = obj[key]
                    break
            if sample is not None:
                results.append(
                    {
                        "sample": _api_raw_sample_to_list(sample),
                        "energy": obj.get("energy", obj.get("objective", obj.get("cost"))),
                        "runtime_sec": obj.get("runtime_sec", obj.get("runtime")),
                    }
                )
            for key in ("samples", "results", "solutions", "data", "result", "outputs", "candidates"):
                if key in obj:
                    visit(obj[key])
        elif isinstance(obj, list):
            if obj and all(isinstance(v, (bool, int, float)) for v in obj):
                results.append({"sample": _api_raw_sample_to_list(obj)})
            else:
                for item in obj:
                    visit(item)

    visit(response)
    return [item for item in results if item.get("sample")]


def _api_raw_sample_to_list(sample: Any) -> List[int]:
    if isinstance(sample, dict):
        return [int(v) for _k, v in sorted(sample.items(), key=lambda kv: int(kv[0]))]
    return [int(v) for v in list(sample)]


def _api_results_to_samples(raw_results: Any, model: QuboModel) -> List[QuboSample]:
    samples: List[QuboSample] = []
    n = len(model.var_index)
    for item in raw_results:
        raw = _api_raw_sample_to_list(item.get("sample", []))
        if len(raw) < n:
            continue
        values = _api_sample_to_binary(raw[:n])
        samples.append(QuboSample(values, model.evaluate(values)))
    return samples


def _api_sample_to_binary(sample: List[int]) -> Dict[int, int]:
    if any(v < 0 for v in sample):
        return {i: 1 if int(v) > 0 else 0 for i, v in enumerate(sample)}
    return {i: 1 if int(v) else 0 for i, v in enumerate(sample)}
