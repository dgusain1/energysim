# -*- coding: utf-8 -*-
"""Deterministic parameter samplers for :class:`energysim.ensemble.Ensemble`.

Every sampler exposes ``sample(n=None) -> list[dict]`` (one parameter dict per
run) and ``__len__`` (raises ``TypeError`` when the size is chosen by the
caller).  All random samplers are deterministic given their ``seed``.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _py(x):
    return x.item() if isinstance(x, np.generic) else x


def _bounds(spec: Mapping[str, Sequence[float]], what: str) -> Tuple[List[str], np.ndarray]:
    if not spec:
        raise ValueError(f"{what}: at least one parameter is required.")
    names = list(spec)
    arr = np.empty((len(names), 2), dtype=float)
    for i, k in enumerate(names):
        v = spec[k]
        if len(v) != 2:
            raise ValueError(f"{what}: '{k}' must be a pair, got {v!r}.")
        arr[i] = [float(v[0]), float(v[1])]
    return names, arr


class Sampler:
    """Base class; subclasses implement :meth:`sample`."""

    n: Optional[int] = None

    def sample(self, n: Optional[int] = None) -> List[Dict[str, Any]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def __len__(self):
        if self.n is None:
            raise TypeError(f"{type(self).__name__} has no fixed size.")
        return int(self.n)

    def _n(self, n):
        n = self.n if n is None else n
        if n is None:
            raise ValueError(f"{type(self).__name__}: the number of samples is required.")
        n = int(n)
        if n < 1:
            raise ValueError("The number of samples must be >= 1.")
        return n


class Grid(Sampler):
    """Full factorial design: ``Grid({'a': [1, 2], 'b': [10, 20, 30]})`` -> 6 runs (first key varies slowest)."""

    def __init__(self, spec: Mapping[str, Sequence[Any]]):
        if not spec:
            raise ValueError("Grid: at least one parameter is required.")
        self.spec = {k: [_py(x) for x in v] for k, v in spec.items()}
        size = 1
        for k, v in self.spec.items():
            if not v:
                raise ValueError(f"Grid: parameter '{k}' has no values.")
            size *= len(v)
        self.n = size

    def sample(self, n: Optional[int] = None):
        if n is not None and int(n) != self.n:
            raise ValueError(f"Grid has {self.n} points; n_runs={n} does not match (omit n_runs).")
        names = list(self.spec)
        return [dict(zip(names, combo)) for combo in itertools.product(*self.spec.values())]


class Scenarios(Sampler):
    """Explicit list of parameter dicts."""

    def __init__(self, scenarios: Sequence[Mapping[str, Any]]):
        self.scenarios = [{k: _py(v) for k, v in dict(s).items()} for s in scenarios]
        if not self.scenarios:
            raise ValueError("Scenarios: at least one scenario is required.")
        self.n = len(self.scenarios)

    def sample(self, n: Optional[int] = None):
        if n is not None and int(n) != self.n:
            raise ValueError(f"Scenarios has {self.n} entries; n_runs={n} does not match (omit n_runs).")
        return [dict(s) for s in self.scenarios]


class Uniform(Sampler):
    """Independent uniform draws: ``Uniform({'x': (lo, hi)}, n=100, seed=1)``."""

    def __init__(self, spec: Mapping[str, Sequence[float]], n: Optional[int] = None, seed: Optional[int] = 0):
        self.names, self.bounds = _bounds(spec, "Uniform")
        self.n, self.seed = n, seed

    def _unit(self, n):
        return np.random.default_rng(self.seed).random((n, len(self.names)))

    def sample(self, n: Optional[int] = None):
        n = self._n(n)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        x = lo + self._unit(n) * (hi - lo)
        return [dict(zip(self.names, map(float, row))) for row in x]


class Normal(Sampler):
    """Independent normal draws: ``Normal({'x': (mu, sigma)}, n=100, seed=1)``."""

    def __init__(self, spec: Mapping[str, Sequence[float]], n: Optional[int] = None, seed: Optional[int] = 0):
        self.names, self.params = _bounds(spec, "Normal")
        if np.any(self.params[:, 1] < 0):
            raise ValueError("Normal: sigma must be >= 0.")
        self.n, self.seed = n, seed

    def sample(self, n: Optional[int] = None):
        n = self._n(n)
        z = np.random.default_rng(self.seed).standard_normal((n, len(self.names)))
        x = self.params[:, 0] + z * self.params[:, 1]
        return [dict(zip(self.names, map(float, row))) for row in x]


def _numpy_lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube sample in [0, 1)^d: exactly one point per stratum in every dimension."""
    u = rng.random((n, d))
    out = np.empty((n, d))
    for j in range(d):
        out[:, j] = (rng.permutation(n) + u[:, j]) / n
    return out


class LatinHypercube(Sampler):
    """Latin hypercube sample over ``(lo, hi)`` ranges.

    Uses :mod:`scipy.stats.qmc` when SciPy is installed and a pure-NumPy
    implementation otherwise (``method='numpy'`` forces the latter; the two
    give different but equally valid designs).
    """

    def __init__(self, spec: Mapping[str, Sequence[float]], n: Optional[int] = None,
                 seed: Optional[int] = 0, method: str = 'auto'):
        self.names, self.bounds = _bounds(spec, "LatinHypercube")
        if method not in ('auto', 'scipy', 'numpy'):
            raise ValueError("method must be 'auto', 'scipy' or 'numpy'.")
        self.n, self.seed, self.method = n, seed, method

    def _unit(self, n):
        d = len(self.names)
        if self.method != 'numpy':
            try:
                from scipy.stats import qmc
                return qmc.LatinHypercube(d=d, seed=self.seed).random(n)
            except ImportError:
                if self.method == 'scipy':
                    raise ImportError("LatinHypercube(method='scipy') requires scipy.") from None
        return _numpy_lhs(n, d, np.random.default_rng(self.seed))

    def sample(self, n: Optional[int] = None):
        n = self._n(n)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        x = lo + self._unit(n) * (hi - lo)
        return [dict(zip(self.names, map(float, row))) for row in x]


class Sobol(LatinHypercube):
    """Scrambled Sobol' sequence over ``(lo, hi)`` ranges (requires SciPy; n should be a power of two)."""

    def __init__(self, spec: Mapping[str, Sequence[float]], n: Optional[int] = None, seed: Optional[int] = 0):
        super().__init__(spec, n=n, seed=seed)

    def _unit(self, n):
        try:
            from scipy.stats import qmc
        except ImportError:
            raise ImportError("Sobol sampling requires scipy (pip install scipy).") from None
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return qmc.Sobol(d=len(self.names), scramble=True, seed=self.seed).random(n)


__all__ = ["Sampler", "Grid", "Scenarios", "Uniform", "Normal", "LatinHypercube", "Sobol"]
