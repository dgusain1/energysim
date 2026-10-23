# -*- coding: utf-8 -*-
"""
Forecast providers for controllers.

Every provider is called as ``forecast(t, times, measurements=None)`` and
returns a float array with one value per entry of ``times`` (the start times
of the horizon intervals, in seconds of simulation time).  ``t`` is the
decision time; ``measurements`` are the controller's latest input values
(used by :class:`PersistenceForecast`).

Providers are reset by the controller at every ``init()`` so that repeated
runs are identical (:class:`NoisyForecast` re-seeds its generator).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np


class Forecast:
    """Base class of forecast providers."""

    def reset(self) -> None:
        """Called at every controller ``init()``; restore the initial state."""

    def __call__(self, t: float, times: Sequence[float],
                 measurements: Optional[Mapping[str, Any]] = None) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _shape(values, times) -> np.ndarray:
        times = np.atleast_1d(np.asarray(times, dtype=float))
        out = np.asarray(values, dtype=float)
        if out.ndim == 0:
            return np.full(times.shape, float(out))
        if out.shape != times.shape:
            raise ValueError(f"Forecast returned shape {out.shape}, expected {times.shape}.")
        return out


class PerfectForecast(Forecast):
    """Read ahead in a known time series (an oracle forecast).

    Parameters
    ----------
    data : DataFrame, Series, str (CSV path), dict or ``(times, values)`` tuple
        The time series.  For a DataFrame / CSV the time axis is the
        ``time_column`` (default ``'time'`` if present, else the index); for a
        Series its index.  Times are seconds.
    column : str, optional
        Value column (required for DataFrames with more than one candidate).
    periodic : float, optional
        Period in seconds; times are wrapped modulo the period (e.g. 86400 to
        repeat a daily profile).  Without it values beyond the series hold
        the first / last sample.
    scale : float, default 1.0
        Multiplier applied to the values (unit conversion, e.g. 1e-3 kW->MW).
    """

    def __init__(self, data, column: Optional[str] = None, time_column: Optional[str] = None,
                 periodic: Optional[float] = None, scale: float = 1.0):
        times, values = self._load(data, column, time_column)
        order = np.argsort(times, kind="stable")
        self.times = times[order]
        self.values = values[order] * float(scale)
        if self.times.size == 0:
            raise ValueError("PerfectForecast needs a non-empty time series.")
        self.periodic = None if periodic is None else float(periodic)
        self.column = column

    @staticmethod
    def _load(data, column, time_column):
        import pandas as pd

        if isinstance(data, tuple) and len(data) == 2:
            return np.asarray(data[0], dtype=float), np.asarray(data[1], dtype=float)
        if isinstance(data, str):
            data = pd.read_csv(data)
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        if isinstance(data, pd.Series):
            return np.asarray(data.index, dtype=float), np.asarray(data.values, dtype=float)
        if isinstance(data, pd.DataFrame):
            tcol = time_column if time_column is not None else ("time" if "time" in data.columns else None)
            if tcol is not None:
                if tcol not in data.columns:
                    raise KeyError(f"Time column '{tcol}' not in data columns {list(data.columns)}.")
                times = np.asarray(data[tcol], dtype=float)
            else:
                times = np.asarray(data.index, dtype=float)
            if column is None:
                cand = [c for c in data.columns if c != tcol and not str(c).startswith("Unnamed")]
                if len(cand) != 1:
                    raise ValueError(f"PerfectForecast: specify column=..., candidates are {cand}.")
                column = cand[0]
            if column not in data.columns:
                raise KeyError(f"Column '{column}' not in data columns {list(data.columns)}.")
            return times, np.asarray(data[column], dtype=float)
        raise TypeError(f"Unsupported forecast data type {type(data).__name__}.")

    def __call__(self, t, times, measurements=None):
        q = np.atleast_1d(np.asarray(times, dtype=float))
        if self.periodic:
            q = np.mod(q, self.periodic)
        return np.interp(q, self.times, self.values)


class PersistenceForecast(Forecast):
    """Repeat the latest measured value over the whole horizon.

    Parameters
    ----------
    input : str, optional
        Name of the controller input holding the measured value.
    default : float, default 0.0
        Value used when no measurement is available.
    """

    def __init__(self, input: Optional[str] = None, default: float = 0.0):
        self.input = input
        self.default = float(default)
        self._last = self.default

    def reset(self):
        self._last = self.default

    def __call__(self, t, times, measurements=None):
        if self.input is not None and measurements is not None and self.input in measurements:
            val = measurements[self.input]
            if val is not None and np.isfinite(val):
                self._last = float(val)
        return self._shape(self._last, times)


class CallableForecast(Forecast):
    """Wrap ``fn(t, horizon_times) -> array | scalar``."""

    def __init__(self, fn: Callable[[float, np.ndarray], Any]):
        if not callable(fn):
            raise TypeError("CallableForecast needs a callable fn(t, horizon_times).")
        self.fn = fn

    def __call__(self, t, times, measurements=None):
        times = np.atleast_1d(np.asarray(times, dtype=float))
        return self._shape(self.fn(t, times), times)


class NoisyForecast(Forecast):
    """Add Gaussian noise to another forecast (for robustness / ensemble studies).

    ``sigma`` is absolute (same unit as the base forecast) unless
    ``relative=True`` (fraction of the base value).  The noise standard
    deviation grows linearly with lead time: ``sigma_k = sigma * (1 + growth * k)``.
    The first horizon value is left noise-free when ``exact_first=True``.
    The generator is re-seeded at every ``reset()`` (i.e. every run), so equal
    seeds give identical runs.
    """

    def __init__(self, base: Forecast, sigma: float, seed: Optional[int] = None, relative: bool = False,
                 growth: float = 0.0, exact_first: bool = False):
        if not callable(base):
            raise TypeError("NoisyForecast base must be a Forecast or callable.")
        self.base = base
        self.sigma = float(sigma)
        self.seed = seed
        self.relative = bool(relative)
        self.growth = float(growth)
        self.exact_first = bool(exact_first)
        self.reset()

    def reset(self):
        if isinstance(self.base, Forecast):
            self.base.reset()
        self.rng = np.random.default_rng(self.seed)

    def _noise(self, base_vals: np.ndarray, size) -> np.ndarray:
        k = np.arange(base_vals.shape[-1])
        sig = self.sigma * (1.0 + self.growth * k)
        if self.relative:
            sig = sig * np.abs(base_vals)
        eps = self.rng.standard_normal(size) * sig
        if self.exact_first:
            eps[..., 0] = 0.0
        return eps

    def __call__(self, t, times, measurements=None):
        base_vals = np.asarray(self.base(t, times, measurements), dtype=float)
        return base_vals + self._noise(base_vals, base_vals.shape)

    def ensemble(self, t, times, n: int, measurements=None) -> np.ndarray:
        """Return ``n`` noisy samples, shape ``(n, len(times))``."""
        base_vals = np.asarray(self.base(t, times, measurements), dtype=float)
        return base_vals[None, :] + self._noise(base_vals, (int(n), base_vals.size))


def as_forecast(spec) -> Forecast:
    """Coerce numbers, arrays, callables ``f(t)`` or Forecasts into a :class:`Forecast`."""
    if isinstance(spec, Forecast):
        return spec
    if isinstance(spec, (int, float, np.floating, np.integer)):
        val = float(spec)
        return CallableForecast(lambda t, times: val)
    if callable(spec):
        def _vec(t, times, _f=spec):
            return np.array([float(_f(x)) for x in times])
        return CallableForecast(_vec)
    raise TypeError(
        f"Cannot use {type(spec).__name__} as a forecast; pass a Forecast, a number or a callable f(t)."
    )


__all__ = [
    "Forecast", "PerfectForecast", "PersistenceForecast", "CallableForecast", "NoisyForecast", "as_forecast",
]
