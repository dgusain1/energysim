# -*- coding: utf-8 -*-
"""Seeded input perturbation for uncertainty ensembles (weather / price / demand)."""
from __future__ import annotations

from typing import Callable, Union

import numpy as np


class _Const:
    def __init__(self, v):
        self.v = v

    def __call__(self, t):
        return self.v


class _SeriesFn:
    def __init__(self, t, y):
        self.t, self.y = t, y

    def __call__(self, t):
        return float(np.interp(t, self.t, self.y))


class NoisyInput:
    """Callable ``f(t)`` returning ``base(t)`` perturbed by seeded (optionally AR(1)) noise.

    The noise is piecewise constant on a grid of width ``dt`` seconds and is a
    *pure function of t* for a given seed (values are generated lazily and
    cached), so the result does not depend on how often or in which order the
    signal is evaluated.  ``n_k = ar1 * n_{k-1} + sqrt(1 - ar1^2) * sigma * eps_k``
    keeps the marginal standard deviation equal to ``sigma``.

    Parameters
    ----------
    base : callable ``f(t)``, number, or ``pandas.Series`` indexed by time (linear interpolation).
    sigma : noise standard deviation (fraction of ``base`` when ``rel=True``).
    seed : integer seed (``None`` for non-reproducible noise).
    rel : multiplicative noise ``base * (1 + n)`` if True, additive ``base + n`` otherwise.
    ar1 : lag-1 autocorrelation in ``[0, 1)``; ``0`` gives white noise.
    dt : width of the noise grid in seconds.
    """

    def __init__(self, base: Union[Callable[[float], float], float], sigma: float, seed=0,
                 rel: bool = True, ar1: float = 0.9, dt: float = 1.0):
        if not 0.0 <= ar1 < 1.0:
            raise ValueError("ar1 must be in [0, 1).")
        if sigma < 0:
            raise ValueError("sigma must be >= 0.")
        if dt <= 0:
            raise ValueError("dt must be > 0.")
        self.base = self._as_callable(base)
        self.sigma, self.seed, self.rel = float(sigma), seed, bool(rel)
        self.ar1, self.dt = float(ar1), float(dt)
        self.reset()

    @staticmethod
    def _as_callable(base):
        if callable(base):
            return base
        if isinstance(base, (int, float, np.number)):
            return _Const(float(base))
        import pandas as pd
        if isinstance(base, pd.Series):
            return _SeriesFn(base.index.to_numpy(dtype=float), base.to_numpy(dtype=float))
        raise TypeError(f"NoisyInput base must be callable, a number or a Series, got {type(base).__name__}.")

    def reset(self):
        self._rng = np.random.default_rng(self.seed)
        self._noise = []

    def noise(self, t: float) -> float:
        """Noise value (relative or absolute) in effect at time ``t``."""
        k = max(int(np.floor(float(t) / self.dt + 1e-12)), 0)
        a = self.ar1
        innov = np.sqrt(1.0 - a * a) * self.sigma
        while len(self._noise) <= k:
            eps = float(self._rng.standard_normal())
            if not self._noise:
                self._noise.append(self.sigma * eps)
            else:
                self._noise.append(a * self._noise[-1] + innov * eps)
        return self._noise[k]

    def __call__(self, t: float) -> float:
        b = float(self.base(t))
        n = self.noise(t)
        return b * (1.0 + n) if self.rel else b + n


def noisy_signal(base_fn, sigma: float, seed=0, rel: bool = True, ar1: float = 0.9, dt: float = 1.0) -> NoisyInput:
    """One-liner: ``w.add_signal('price', noisy_signal(base, 0.1, seed))`` (see :class:`NoisyInput`)."""
    return NoisyInput(base_fn, sigma, seed=seed, rel=rel, ar1=ar1, dt=dt)


__all__ = ["NoisyInput", "noisy_signal"]
