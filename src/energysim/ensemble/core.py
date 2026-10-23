# -*- coding: utf-8 -*-
"""Monte-Carlo / scenario ensembles of co-simulation runs."""
from __future__ import annotations

import json
import os
import pickle
import time
import traceback
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..results import Results
from .samplers import Sampler, Scenarios

_SEED_COL = 'seed'
_DEFAULT_KPI_NAMES = ('min', 'max', 'mean', 'time_weighted_mean')


class EnsembleError(RuntimeError):
    """Raised for invalid ensemble configuration or (``on_error='raise'``) a failed run."""


# ---------------------------------------------------------------------------
# copy / pickle support for worlds
# ---------------------------------------------------------------------------

def _ensure_copyable_connections():
    """``ParsedConnection`` has a custom ``__new__``; make worlds deep-copyable/picklable if it is not yet."""
    import copy
    import copyreg

    from ..base import ParsedConnection
    if ParsedConnection in copyreg.dispatch_table:
        return
    try:
        copy.deepcopy(ParsedConnection(("a", "b", "c"), ("d", "e")))
        return
    except Exception:
        pass

    def _reduce(pc):
        return ParsedConnection, (pc[0], pc[1], pc.scale, pc.offset, pc.from_unit, pc.to_unit,
                                  pc.user_scale, pc.user_offset)

    copyreg.pickle(ParsedConnection, _reduce)


_ensure_copyable_connections()


# ---------------------------------------------------------------------------
# parameter application
# ---------------------------------------------------------------------------

def _split_dotted(name: str):
    if '.' not in name:
        raise ValueError(f"Parameter '{name}' must be given as 'simulator.variable'.")
    sim, var = name.split('.', 1)
    return sim, var


def apply_params(world, params: Mapping[str, Any]):
    """Set ``{'sim.var': value}`` as initial values of *world* (merged into ``options({'init': ...})``).

    Values are applied at the start of every run, exactly like
    ``world.options({'init': {sim: ([vars], [values])}})``; an existing
    ``init`` entry for the same variable is overwritten.  Returns *world*.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for sim, p in (getattr(world, 'init_dict', None) or {}).items():
        if isinstance(p, dict):
            merged[sim] = dict(p)
        else:
            merged[sim] = dict(zip(list(p[0]), list(p[1])))
    for name, value in params.items():
        sim, var = _split_dotted(name)
        merged.setdefault(sim, {})[var] = value
    world.options({'init': {s: (list(d), list(d.values())) for s, d in merged.items()}})
    return world


class _MapFactory:
    """Picklable factory used by :meth:`Ensemble.from_world`."""

    def __init__(self, world_factory: Callable[[], Any], parameter_map: Optional[Mapping[str, str]] = None):
        self.world_factory = world_factory
        self.parameter_map = dict(parameter_map) if parameter_map else None

    def __call__(self, params, seed):
        w = self.world_factory()
        mapped = {}
        for k, v in params.items():
            target = self.parameter_map.get(k, k) if self.parameter_map else k
            mapped[target] = v
        if mapped:
            apply_params(w, mapped)
        try:
            w.seed = seed
        except Exception:  # pragma: no cover
            pass
        return w


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

def _compute_kpis(res: Results, kpis: Optional[Mapping[str, Callable]], variables, kpi_names) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if variables is not None or not kpis:
        names = list(kpi_names) if kpi_names else list(_DEFAULT_KPI_NAMES)
        table = res.kpis(names=names, variables=variables)
        for var, row in table.iterrows():
            for k, v in row.items():
                if pd.notna(v):
                    out[f"{var}:{k}"] = float(v)
    for name, fn in (kpis or {}).items():
        out[name] = float(fn(res))
    return out


def _execute(factory, params, seed, kpis, variables, kpi_names, keep):
    """Run one member; never raises (errors are returned as text)."""
    t0 = time.perf_counter()
    try:
        w = factory(dict(params), seed)
        res = w.simulate(pbar=False)
        k = _compute_kpis(res, kpis, variables, kpi_names)
        return {'status': 'completed', 'error': '', 'kpis': k,
                'results': res if keep == 'all' else None,
                'duration_s': time.perf_counter() - t0}
    except BaseException as exc:  # noqa: BLE001 - isolate every failure
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return {'status': 'failed', 'error': f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                'kpis': {}, 'results': None, 'duration_s': time.perf_counter() - t0}


def _check_picklable(obj, what: str):
    name = getattr(obj, '__qualname__', '')
    if '<lambda>' in name or '<locals>' in name:
        raise EnsembleError(
            f"{what} {name!r} is a lambda or closure and cannot be sent to worker processes. "
            f"Define it at module level (an importable function in a .py file), or use parallel='serial'."
        )
    try:
        pickle.dumps(obj)
    except Exception as exc:
        raise EnsembleError(
            f"{what} cannot be pickled for parallel='process' ({type(exc).__name__}: {exc}). "
            f"Define it at module level so it can be imported by the worker processes."
        ) from None


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

class EnsembleResults:
    """Outcome of :meth:`Ensemble.run`.

    Attributes / methods: ``params`` (one row per run, incl. the ``seed``
    column), ``status`` (``status``, ``error``, ``duration_s`` per run),
    ``runs`` / ``run(key)`` / ``ensemble`` (completed runs; needs
    ``keep='all'``), ``kpis()``, ``summary()``, ``to_frame(variable)``,
    ``fan(variable)``, ``save(path)`` / ``EnsembleResults.load(path)``.
    Run keys are the integers ``0..n-1``.
    """

    def __init__(self, params: pd.DataFrame, status: pd.DataFrame, kpis: pd.DataFrame,
                 results: Dict[int, Results], keep: str = 'all', ensemble: Optional['Ensemble'] = None):
        self._params = params
        self._status = status
        self._kpis = kpis
        self._results = dict(results)
        self.keep = keep
        self._ensemble = ensemble

    # -- basic ---------------------------------------------------------------
    @property
    def params(self) -> pd.DataFrame:
        return self._params.copy()

    @property
    def status(self) -> pd.DataFrame:
        return self._status.copy()

    @property
    def completed(self) -> List[int]:
        return [k for k in self._status.index if self._status.loc[k, 'status'] == 'completed']

    @property
    def failed(self) -> List[int]:
        return [k for k in self._status.index if self._status.loc[k, 'status'] != 'completed']

    @property
    def run_keys(self) -> List[int]:
        return list(self._params.index)

    def __len__(self):
        return len(self._params)

    def __repr__(self):
        return (f"EnsembleResults(runs={len(self)}, completed={len(self.completed)}, "
                f"failed={len(self.failed)}, keep={self.keep!r})")

    def _need_runs(self):
        if self.keep != 'all' or not self._results:
            raise EnsembleError("Run results were not kept (keep='kpis' or no run completed); "
                                "re-run the ensemble with keep='all'.")

    @property
    def runs(self) -> List[Results]:
        self._need_runs()
        return [self._results[k] for k in self.completed]

    def run(self, key) -> Results:
        self._need_runs()
        if key not in self._results:
            raise KeyError(f"No completed run {key!r}. Completed: {self.completed}; failed: {self.failed}")
        return self._results[key]

    @property
    def ensemble(self) -> Results:
        """Completed runs combined with :meth:`Results.concat` (run dimension)."""
        self._need_runs()
        keys = self.completed
        return Results.concat([self._results[k] for k in keys], keys=keys,
                              params=self._params.loc[keys].reset_index(drop=True))

    def rerun(self, run_key) -> Results:
        """Re-execute a single run (same params and seed) and return its :class:`Results`."""
        if self._ensemble is None:
            raise EnsembleError("This result is not bound to an Ensemble (loaded from disk); "
                                "call Ensemble.rerun(run_key) on the original ensemble.")
        return self._ensemble.rerun(run_key)

    # -- statistics ----------------------------------------------------------------
    def kpis(self) -> pd.DataFrame:
        """Per-run KPI table (index run; NaN rows for failed runs)."""
        return self._kpis.reindex(self._params.index).copy()

    def summary(self, quantiles: Sequence[float] = (0.1, 0.5, 0.9)) -> pd.DataFrame:
        """Distribution of every KPI over the completed runs: count, mean, std, min, quantiles, max."""
        k = self._kpis.loc[[i for i in self.completed if i in self._kpis.index]]
        rows = {}
        for col in k.columns:
            s = k[col].astype(float).dropna()
            row = {'count': int(s.count()), 'mean': s.mean(), 'std': s.std(), 'min': s.min()}
            for q in quantiles:
                row[f"q{q:g}"] = float(np.quantile(s.to_numpy(), q)) if len(s) else np.nan
            row['max'] = s.max()
            rows[col] = row
        cols = ['count', 'mean', 'std', 'min'] + [f"q{q:g}" for q in quantiles] + ['max']
        return pd.DataFrame.from_dict(rows, orient='index', columns=cols)

    def to_frame(self, variable: str) -> pd.DataFrame:
        """Run-by-time frame for one ``'sim.var'``: index time (union grid, zero-order hold), columns run keys."""
        self._need_runs()
        cols = {}
        for k in self.completed:
            f = self._results[k].to_frame(variables=variable)
            cols[k] = f.iloc[:, 0]
        wide = pd.concat(cols, axis=1, join='outer', sort=True).ffill()
        wide.columns.name = 'run'
        return wide

    def fan(self, variable: str, quantiles: Sequence[float] = (0.1, 0.5, 0.9)) -> pd.DataFrame:
        """Quantile bands over runs for *variable* (index time, one column per quantile)."""
        wide = self.to_frame(variable)
        q = np.nanquantile(wide.to_numpy(dtype=float), list(quantiles), axis=1)
        return pd.DataFrame(q.T, index=wide.index, columns=[f"q{x:g}" for x in quantiles])

    # -- persistence ----------------------------------------------------------------
    def save(self, path) -> str:
        """Write ``all_params.csv``, ``status.csv``, ``kpis.csv`` and, with ``keep='all'``, the runs
        in the :meth:`Results.save` ensemble layout (``run_NNNN/``, ``params.csv``, ``ensemble.json``)."""
        path = os.fspath(path)
        os.makedirs(path, exist_ok=True)
        if self.keep == 'all' and self._results:
            self.ensemble.save(path)
        self._params.to_csv(os.path.join(path, 'all_params.csv'))
        self._status.to_csv(os.path.join(path, 'status.csv'))
        self._kpis.to_csv(os.path.join(path, 'kpis.csv'))
        with open(os.path.join(path, 'ensemble_meta.json'), 'w', encoding='utf-8') as f:
            json.dump({'n_runs': len(self), 'keep': self.keep}, f)
        return os.path.abspath(path)

    @classmethod
    def load(cls, path) -> 'EnsembleResults':
        path = os.fspath(path)
        mp = os.path.join(path, 'ensemble_meta.json')
        if not os.path.exists(mp):
            raise FileNotFoundError(f"No saved EnsembleResults at {path!r}.")
        with open(mp, encoding='utf-8') as f:
            meta = json.load(f)
        params = pd.read_csv(os.path.join(path, 'all_params.csv'), index_col=0)
        status = pd.read_csv(os.path.join(path, 'status.csv'), index_col=0).fillna({'error': ''})
        kp = pd.read_csv(os.path.join(path, 'kpis.csv'), index_col=0)
        results: Dict[int, Results] = {}
        if meta.get('keep') == 'all' and os.path.exists(os.path.join(path, 'ensemble.json')):
            loaded = Results.load(path)
            done = [k for k in status.index if status.loc[k, 'status'] == 'completed']
            for k, r in zip(done, loaded.runs):
                results[int(k)] = r
        return cls(params, status, kp, results, keep=meta.get('keep', 'all'))


# ---------------------------------------------------------------------------
# ensemble
# ---------------------------------------------------------------------------

class Ensemble:
    """Run many variants of a co-simulation and collect the outcomes.

    Parameters
    ----------
    world_factory : ``f(params: dict, seed: int) -> world`` building a *fresh* world for every run.
        For ``parallel='process'`` it must be an importable module-level function
        (no lambdas / closures); worker processes use the ``spawn`` start method.
    n_runs : number of runs (required for random samplers without their own ``n``; with no
        sampler/scenarios it gives that many parameter-free, seed-only Monte-Carlo runs).
    sampler : a :mod:`energysim.ensemble.samplers` object; ``scenarios`` is a shortcut for a list of dicts.
    seeds : ``None`` (master seed 0), an ``int`` master seed, or an explicit list of per-run seeds.
        Run seeds are derived with ``numpy.random.SeedSequence(master).spawn`` so they do not depend
        on the execution mode.
    parallel : ``'serial'`` or ``'process'``; ``max_workers`` bounds the pool.
    out_dir : if set, :meth:`run` saves the results there (see :meth:`EnsembleResults.save`).
    keep : ``'all'`` keeps every run's :class:`Results`; ``'kpis'`` keeps only the KPI table.
    kpis : ``{'name': callable(Results) -> float}``; ``variables`` / ``kpi_names`` additionally request
        registered KPIs (default ``min,max,mean,time_weighted_mean``) for ``'sim.var'`` names, stored as
        ``'sim.var:kpi'``.  With neither given, built-in KPIs are computed for all variables.
    on_error : ``'record'`` (default; failed runs get status ``'failed'`` plus the traceback) or ``'raise'``.
    """

    def __init__(self, world_factory: Callable[[dict, int], Any], n_runs: Optional[int] = None,
                 sampler: Optional[Sampler] = None, scenarios: Optional[Sequence[Mapping]] = None,
                 seeds=None, parallel: str = 'serial', max_workers: Optional[int] = None,
                 out_dir=None, keep: str = 'all', kpis: Optional[Mapping[str, Callable]] = None,
                 on_error: str = 'record', variables=None, kpi_names: Optional[Sequence[str]] = None):
        if not callable(world_factory):
            raise TypeError("world_factory must be a callable f(params, seed) -> world.")
        if parallel not in ('serial', 'process'):
            raise ValueError("parallel must be 'serial' or 'process'.")
        if keep not in ('all', 'kpis'):
            raise ValueError("keep must be 'all' or 'kpis'.")
        if on_error not in ('record', 'raise'):
            raise ValueError("on_error must be 'record' or 'raise'.")
        if sampler is not None and scenarios is not None:
            raise ValueError("Pass either sampler or scenarios, not both.")
        if scenarios is not None:
            sampler = Scenarios(scenarios)
        self.world_factory = world_factory
        self.n_runs, self.sampler = n_runs, sampler
        self.seeds, self.parallel, self.max_workers = seeds, parallel, max_workers
        self.out_dir, self.keep, self.kpis = out_dir, keep, dict(kpis) if kpis else None
        self.on_error = on_error
        self.variables = [variables] if isinstance(variables, str) else (list(variables) if variables else None)
        self.kpi_names = list(kpi_names) if kpi_names else None
        self.results: Optional[EnsembleResults] = None

    @classmethod
    def from_world(cls, world_factory: Callable[[], Any], parameter_map: Optional[Mapping[str, str]] = None,
                   **kwargs) -> 'Ensemble':
        """Ensemble from a parameter-free ``world_factory() -> world``.

        Sampled parameters are applied through ``world.options({'init': ...})``;
        ``parameter_map`` maps sampled names to ``'sim.var'`` (identity when omitted,
        i.e. sample directly over ``'sim.var'`` names).  The run seed is stored on ``world.seed``.
        """
        return cls(_MapFactory(world_factory, parameter_map), **kwargs)

    # -- planning ------------------------------------------------------------------
    def _plan(self):
        if self.sampler is not None:
            plist = self.sampler.sample(self.n_runs)
        else:
            if self.n_runs is None:
                raise EnsembleError("Give n_runs, a sampler or scenarios.")
            plist = [{} for _ in range(int(self.n_runs))]
        n = len(plist)
        if isinstance(self.seeds, (list, tuple, np.ndarray)):
            seeds = [int(s) for s in self.seeds]
            if len(seeds) != n:
                raise EnsembleError(f"{len(seeds)} seeds given for {n} runs.")
        else:
            master = 0 if self.seeds is None else int(self.seeds)
            children = np.random.SeedSequence(master).spawn(n)
            seeds = [int(c.generate_state(1)[0]) for c in children]
        for p in plist:
            if _SEED_COL in p:
                raise EnsembleError("'seed' is reserved for the per-run seed; rename that parameter.")
        return plist, seeds

    def _args(self, params, seed):
        return (self.world_factory, params, seed, self.kpis, self.variables, self.kpi_names, self.keep)

    # -- execution -------------------------------------------------------------------
    def _run_outcomes(self, plist, seeds):
        n = len(plist)
        if self.parallel == 'serial' or n == 1:
            outs = []
            for p, s in zip(plist, seeds):
                o = _execute(*self._args(p, s))
                outs.append(o)
                if o['status'] == 'failed' and self.on_error == 'raise':
                    break
            return outs
        _check_picklable(self.world_factory, 'world_factory')
        for name, fn in (self.kpis or {}).items():
            _check_picklable(fn, f"KPI '{name}'")
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        try:
            ctx = mp.get_context('spawn')
            pool = ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx)
        except (ImportError, NotImplementedError, OSError) as exc:
            raise EnsembleError(f"multiprocessing is unavailable ({exc}); use parallel='serial'.") from exc
        with pool:
            futs = [pool.submit(_execute, *self._args(p, s)) for p, s in zip(plist, seeds)]
            return [f.result() for f in futs]

    def run(self) -> EnsembleResults:
        plist, seeds = self._plan()
        outs = self._run_outcomes(plist, seeds)
        if self.on_error == 'raise':
            for i, o in enumerate(outs):
                if o['status'] == 'failed':
                    raise EnsembleError(f"Run {i} failed (params={plist[i]}, seed={seeds[i]}):\n{o['error']}")
        n = len(plist)
        idx = pd.Index(range(n), name='run')
        params = pd.DataFrame(plist, index=idx)
        params[_SEED_COL] = seeds
        status = pd.DataFrame({'status': [o['status'] for o in outs], 'error': [o['error'] for o in outs],
                               'duration_s': [o['duration_s'] for o in outs]}, index=idx)
        kp = pd.DataFrame([o['kpis'] for o in outs], index=idx)
        results = {i: o['results'] for i, o in enumerate(outs) if o['results'] is not None}
        self.results = EnsembleResults(params, status, kp, results, keep=self.keep, ensemble=self)
        if self.out_dir is not None:
            self.results.save(self.out_dir)
        return self.results

    def rerun(self, run_key) -> Results:
        """Re-execute one run (same parameters and seed) in this process; raises if it fails."""
        plist, seeds = self._plan()
        if not isinstance(run_key, (int, np.integer)) or not 0 <= run_key < len(plist):
            raise KeyError(f"No run {run_key!r}; valid keys are 0..{len(plist) - 1}.")
        o = _execute(self.world_factory, plist[run_key], seeds[run_key], None, None, None, 'all')
        if o['status'] == 'failed':
            raise EnsembleError(f"Run {run_key} failed:\n{o['error']}")
        return o['results']


__all__ = ["Ensemble", "EnsembleResults", "EnsembleError", "apply_params"]
