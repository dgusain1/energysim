# -*- coding: utf-8 -*-
"""Parameter sweeps (one-at-a-time and full-factorial) built on :class:`energysim.ensemble.Ensemble`.

Every run uses a *fresh* world, so there is no state leakage between runs and
the order of parameters never changes results.
"""
from __future__ import annotations

import copy
import itertools
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .ensemble.core import Ensemble, EnsembleError, apply_params
from .ensemble.samplers import Scenarios

METRICS = ('last', 'mean', 'max', 'min', 'sum', 'integral')


def _seconds(index: pd.Index) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex):
        return index.astype('int64').to_numpy(dtype=float) / 1e9
    return index.to_numpy(dtype=float)


def compute_metric(series: pd.Series, metric: str) -> float:
    """Reduce a time series to one number (``integral`` = zero-order-hold integral over time in seconds)."""
    s = series.astype(float)
    if metric == 'last':
        return float(s.iloc[-1])
    if metric == 'mean':
        return float(s.mean())
    if metric == 'max':
        return float(s.max())
    if metric == 'min':
        return float(s.min())
    if metric == 'sum':
        return float(s.sum())
    if metric == 'integral':
        t = _seconds(s.index)
        return float(np.sum(s.to_numpy()[:-1] * np.diff(t))) if len(s) > 1 else 0.0
    raise ValueError(f"Unknown metric '{metric}'. Choose one of {METRICS}.")


class _OutputMetric:
    """Picklable KPI callable: metric of one 'sim.var' output."""

    def __init__(self, output: str, metric: str):
        self.output, self.metric = output, metric

    def __call__(self, res) -> float:
        return compute_metric(res[self.output], self.metric)


class _WorldCopyFactory:
    """Factory(params, seed) -> deep copy of a pristine template world with the parameters applied."""

    def __init__(self, template):
        self.template = template

    def __call__(self, params, seed):
        w = copy.deepcopy(self.template)
        if params:
            apply_params(w, params)
        return w


def _is_world(obj) -> bool:
    return hasattr(obj, 'simulator_dict') and hasattr(obj, 'simulate') and hasattr(obj, 'options')


class SweepResult:
    """Outcome of :meth:`Sweep.sweep`.

    ``table`` is a DataFrame: for ``kind='single'`` the columns are
    ``parameter, value, result``; for ``'parametric'`` one column per parameter plus ``result``
    (``result`` is the chosen ``metric`` of the output parameter).  ``baseline`` holds the
    unmodified-world value (single sweeps over a world instance), ``ensemble`` the underlying
    :class:`~energysim.ensemble.EnsembleResults`.
    """

    def __init__(self, table: pd.DataFrame, kind: str, parameters: List[str], output_parameter: str,
                 metric: str, ensemble, baseline: Optional[Dict[str, Any]] = None):
        self.table, self.kind, self.parameters = table, kind, list(parameters)
        self.output_parameter, self.metric = output_parameter, metric
        self.ensemble, self.baseline = ensemble, baseline

    def __repr__(self):
        return f"SweepResult(kind={self.kind!r}, runs={len(self.table)}, output={self.output_parameter!r}, metric={self.metric!r})"

    def export_to_csv(self, path) -> str:
        """Write :attr:`table` to a CSV file and return the path."""
        self.table.to_csv(path, index=False)
        return str(path)

    def plot(self, show: bool = False, kind: str = 'metric'):
        """Return a matplotlib figure (``plt.show()`` only if ``show=True``).

        ``kind='metric'`` (single: one panel per parameter, result vs value; parametric with exactly two
        parameters: heatmap) or ``kind='curves'`` (single sweeps: output time series for each value;
        needs ``keep='all'``).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("SweepResult.plot() requires matplotlib "
                              "(pip install matplotlib or pip install energysim[plotting]).") from None
        t = self.table
        if self.kind == 'parametric':
            if len(self.parameters) != 2:
                raise ValueError("Heatmap plots need exactly two parameters; use .table for more.")
            a, b = self.parameters
            piv = t.pivot(index=b, columns=a, values='result')
            fig, ax = plt.subplots()
            im = ax.imshow(piv.to_numpy(), origin='lower', aspect='auto',
                           extent=(piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()))
            ax.set_xlabel(a)
            ax.set_ylabel(b)
            ax.set_title(f"{self.metric} of {self.output_parameter}")
            fig.colorbar(im, ax=ax, label=f"{self.metric}({self.output_parameter})")
        else:
            n = len(self.parameters)
            fig, axes = plt.subplots(n, 1, squeeze=False, figsize=(6, 3 * n))
            for ax, p in zip(axes[:, 0], self.parameters):
                sub = t[t['parameter'] == p]
                if kind == 'curves':
                    for idx, val in zip(sub.index, sub['value']):
                        key = int(idx)
                        ax.plot(self.ensemble.run(key)[self.output_parameter], label=f"{p}={val:g}")
                    ax.set_xlabel('time')
                    ax.set_ylabel(self.output_parameter)
                    ax.legend(fontsize='small')
                elif kind == 'metric':
                    ax.plot(sub['value'], sub['result'], marker='o')
                    ax.set_xlabel(p)
                    ax.set_ylabel(f"{self.metric}({self.output_parameter})")
                else:
                    raise ValueError("kind must be 'metric' or 'curves'.")
                ax.set_title(f"Sensitivity of {self.output_parameter} to {p}")
            fig.tight_layout()
        if show:
            plt.show()
        return fig


class Sweep:
    """Parameter sweeps on a co-simulation world.

    ``Sweep(world_or_factory, sensitivity_info, kind='single'|'parametric')`` where

    * *world_or_factory* is a configured ``world`` (a pristine deep copy is used for every run, the
      instance itself is never simulated or modified), or a factory ``f(params: dict, seed) -> world``
      (``params`` holds only the swept values; absent parameters must take their baseline in the factory);
    * ``sensitivity_info = {'parameters': ['sim.var', ...], 'bounds': [[lo, hi, N], ...],
      'output_parameter': 'sim.var', 'metric': 'last'}``; ``N`` defaults to 10; ``metric`` is one of
      ``last, mean, max, min, sum, integral`` (default ``last``); optional ``'seed'`` (default 0, shared by
      all runs);
    * ``kind='single'`` sweeps each parameter on its own around the baseline (the others untouched);
      ``kind='parametric'`` runs the full factorial grid.

    ``sweep()`` returns a :class:`SweepResult` (``table``, ``export_to_csv``, ``plot``).  Extra keyword
    arguments (``parallel``, ``max_workers``, ``keep``, ``on_error``) go to :class:`Ensemble`.
    """

    def __init__(self, _cosimulation_world, _sensitivity_data, kind: str = 'single', **ensemble_kwargs):
        if kind not in ('single', 'parametric'):
            raise ValueError(f"Unsupported sweep kind '{kind}'. Choose 'single' or 'parametric'.")
        self.info = dict(_sensitivity_data)
        self.kind = kind
        self.ensemble_kwargs = ensemble_kwargs
        self.world = _cosimulation_world if _is_world(_cosimulation_world) else None
        # attributes kept for backward compatibility
        self.simulators_dict = dict(getattr(self.world, 'simulator_dict', {}) or {})
        self.powerflow_exists = any(getattr(v, 'sim_type', None) == 'pf' for v in self.simulators_dict.values())
        if self.world is not None:
            try:
                self._template = copy.deepcopy(self.world)
            except Exception as exc:
                raise EnsembleError(
                    f"The world cannot be copied ({type(exc).__name__}: {exc}); it holds non-copyable "
                    f"resources. Pass a factory f(params, seed) -> world instead of a world instance."
                ) from exc
            self.factory: Callable = _WorldCopyFactory(self._template)
        elif callable(_cosimulation_world):
            self.factory = _cosimulation_world
        else:
            raise TypeError("Sweep needs a world or a factory f(params, seed) -> world.")
        for key in ('parameters', 'bounds', 'output_parameter'):
            if key not in self.info:
                raise KeyError(f"sensitivity_info is missing '{key}'.")
        self.parameters_list = list(self.info['parameters'])
        if len(self.info['bounds']) != len(self.parameters_list):
            raise ValueError("'bounds' must have one entry per parameter.")
        self.op_param = self.info['output_parameter']
        self.metric = self.info.get('metric', 'last')
        if self.metric not in METRICS:
            raise ValueError(f"Unknown metric '{self.metric}'. Choose one of {METRICS}.")
        if kind == 'parametric' and len(self.parameters_list) < 2:
            raise ValueError("kind='parametric' needs at least two parameters.")
        self.values = [self._grid(b) for b in self.info['bounds']]
        self.seed = int(self.info.get('seed', 0))

    @staticmethod
    def _grid(item) -> np.ndarray:
        if len(item) == 2:
            return np.linspace(item[0], item[1], 10)
        if len(item) == 3:
            return np.linspace(item[0], item[1], int(item[2]))
        raise ValueError('Bounds matrix should have elements with either two or three values '
                         'to define lower, upper bounds and number of values.')

    def sweep(self) -> SweepResult:
        if self.kind == 'single':
            return self.single_parameter_sweep()
        return self.parametric_sweep()

    def _baseline(self) -> Optional[Dict[str, Any]]:
        if self.world is None:
            return None
        base = {}
        for p in self.parameters_list:
            sim, var = p.split('.', 1)
            try:
                base[p] = copy.deepcopy(self._template).get_adapter(sim).get_value([var], 0)[0]
            except Exception:
                return None
        return base

    def _run(self, scenarios: List[dict]):
        kw = dict(self.ensemble_kwargs)
        kw.setdefault('on_error', 'raise')
        ens = Ensemble(self.factory, sampler=Scenarios(scenarios), seeds=[self.seed] * len(scenarios),
                       kpis={'result': _OutputMetric(self.op_param, self.metric)}, **kw)
        return ens.run()

    def single_parameter_sweep(self) -> SweepResult:
        scenarios, rows = [], []
        for p, vals in zip(self.parameters_list, self.values):
            for v in vals:
                scenarios.append({p: float(v)})
                rows.append((p, float(v)))
        res = self._run(scenarios)
        table = pd.DataFrame(rows, columns=['parameter', 'value'])
        table['result'] = res.kpis()['result'].to_numpy()
        baseline = self._baseline()
        base_info = None
        if baseline is not None:
            r0 = self._run([{}]) if self.world is not None else None
            base_info = {'values': baseline, 'result': float(r0.kpis()['result'].iloc[0])}
        return SweepResult(table, 'single', self.parameters_list, self.op_param, self.metric, res, base_info)

    def parametric_sweep(self) -> SweepResult:
        scenarios = [dict(zip(self.parameters_list, map(float, combo)))
                     for combo in itertools.product(*self.values)]
        res = self._run(scenarios)
        table = pd.DataFrame(scenarios)[self.parameters_list]
        table['result'] = res.kpis()['result'].to_numpy()
        return SweepResult(table, 'parametric', self.parameters_list, self.op_param, self.metric, res)

    def evaluation_function(self):  # kept for backward compatibility
        pass
