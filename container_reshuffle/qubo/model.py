"""Pure Python QUBO data structure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


class QuboModel:
    """Sparse QUBO model using ``E=sum_i Qii xi + sum_i<j Qij xi xj + offset``."""

    def __init__(self) -> None:
        self.var_index: Dict[str, int] = {}
        self.index_var: Dict[int, str] = {}
        self.Q: Dict[Tuple[int, int], float] = {}
        self.offset: float = 0.0

    def add_binary_var(self, name: str) -> int:
        """Add or return a binary variable index."""

        if name in self.var_index:
            return self.var_index[name]
        idx = len(self.var_index)
        self.var_index[name] = idx
        self.index_var[idx] = name
        return idx

    def add_linear(self, i: int, coeff: float) -> None:
        """Add a linear coefficient to variable ``i``."""

        self.Q[(i, i)] = self.Q.get((i, i), 0.0) + float(coeff)

    def add_quadratic(self, i: int, j: int, coeff: float) -> None:
        """Add a quadratic coefficient. Same-index terms become linear."""

        if i == j:
            self.add_linear(i, coeff)
            return
        a, b = sorted((i, j))
        self.Q[(a, b)] = self.Q.get((a, b), 0.0) + float(coeff)

    def add_exactly_k_penalty(self, vars: List[int], k: int, penalty: float) -> None:
        """Add ``penalty * (sum(vars)-k)^2``."""

        a = float(penalty)
        self.offset += a * k * k
        for var in vars:
            self.add_linear(var, a * (1 - 2 * k))
        for idx, vi in enumerate(vars):
            for vj in vars[idx + 1 :]:
                self.add_quadratic(vi, vj, 2 * a)

    def add_one_hot_penalty(self, vars: List[int], penalty: float) -> None:
        """Add an exactly-one penalty."""

        self.add_exactly_k_penalty(vars, 1, penalty)

    def add_at_most_one_penalty(self, vars: List[int], penalty: float) -> None:
        """Add ``penalty * sum_i<j xi*xj``."""

        for idx, vi in enumerate(vars):
            for vj in vars[idx + 1 :]:
                self.add_quadratic(vi, vj, penalty)

    def evaluate(self, binary_values: Dict[int, int]) -> float:
        """Evaluate a binary assignment."""

        energy = self.offset
        for (i, j), coeff in self.Q.items():
            xi = int(binary_values.get(i, 0))
            xj = int(binary_values.get(j, 0))
            energy += coeff * xi if i == j else coeff * xi * xj
        return float(energy)

    def to_dict(self) -> Dict:
        """Serialize model to a JSON-compatible dict."""

        return {
            "var_index": self.var_index,
            "Q": [{"i": i, "j": j, "coeff": coeff} for (i, j), coeff in sorted(self.Q.items())],
            "offset": self.offset,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "QuboModel":
        """Deserialize a QUBO model from a dict."""

        model = cls()
        model.var_index = {str(k): int(v) for k, v in data.get("var_index", {}).items()}
        model.index_var = {v: k for k, v in model.var_index.items()}
        model.Q = {
            (int(row["i"]), int(row["j"])): float(row["coeff"])
            for row in data.get("Q", [])
        }
        model.offset = float(data.get("offset", 0.0))
        return model

    def to_json(self, path: str | Path) -> None:
        """Write model as JSON."""

        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "QuboModel":
        """Read model from JSON."""

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
