# -*- coding: utf-8 -*-
"""
Simulation results for energysim.

:class:`Results` is what :meth:`energysim.world.simulate` returns.  It holds
one table per simulator (rows = recorded time points, columns = output
variables, typed float / int / bool / str), plus units and run metadata, and
offers selection, wide-frame alignment, summaries, KPIs, persistence
(Parquet / CSV / HDF5) and an interactive dashboard.

Recording during a run is done by the internal :class:`Recorder`, which
buffers rows per simulator and converts them chunk-wise into typed NumPy
columns (optionally streaming the chunks to Parquet part files).

KPIs are pluggable::

    from energysim.results import register_kpi
    register_kpi('peak_kw', lambda s, unit: float(s.max()) if unit == 'kW' else None)
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import sys
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .utils import safe_filename_component, parse_unit

__all__ = [
    'Results', 'Recorder', 'register_kpi', 'unregister_kpi', 'available_kpis',
    'scenario_hash',
]

_TIME = 'time'
_META_FILE = 'meta.json'
_ENSEMBLE_FILE = 'ensemble.json'
_PARAMS_FILE = 'params.csv'
_FORMATS = ('auto', 'parquet', 'csv', 'hdf5')


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

def _have_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def _require_pyarrow():
    if not _have_pyarrow():
        raise ImportError(
            "Parquet support requires 'pyarrow'. Install it with: pip install pyarrow "
            "(or pip install energysim[results]), or use format='csv'."
        )


def _require_tables():
    try:
        import tables  # noqa: F401
    except ImportError:
        raise ImportError(
            "HDF5 support requires 'tables' (PyTables). Install it with: pip install tables "
            "(or pip install energysim[hdf5]), or use format='parquet' / 'csv'."
        ) from None


def _versions() -> Dict[str, Optional[str]]:
    from . import __version__
    try:
        import pyarrow
        pa_v = pyarrow.__version__
    except ImportError:
        pa_v = None
    return {
        'energysim': __version__,
        'python': platform.python_version(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'pyarrow': pa_v,
        'platform': platform.platform(),
    }


def scenario_hash(scenario: Any) -> str:
    """Stable SHA-256 of a scenario dict (key order independent)."""
    blob = json.dumps(scenario, sort_keys=True, default=repr, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def make_run_dir(out_dir: Union[str, os.PathLike], digest: str,
                 now: Optional[_dt.datetime] = None) -> str:
    """Create and return ``<out_dir>/<YYYYmmdd-HHMMSS>_<hash8>/`` (unique)."""
    now = now or _dt.datetime.now()
    base = f"{now.strftime('%Y%m%d-%H%M%S')}_{digest[:8]}"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(os.fspath(out_dir), base)
    i = 1
    while os.path.exists(path):
        path = os.path.join(os.fspath(out_dir), f"{base}-{i}")
        i += 1
    os.makedirs(path)
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Typed column helpers
# ---------------------------------------------------------------------------

def _to_object_array(vals: Sequence[Any]) -> np.ndarray:
    out = np.empty(len(vals), dtype=object)
    for i, v in enumerate(vals):
        out[i] = v
    return out


def _column_from_values(vals: Sequence[Any]) -> np.ndarray:
    """Convert a list of recorded values into a typed 1-D column.

    bool -> ``bool``, integers -> ``int64``, numbers (incl. ``None``/NaN) ->
    ``float64``, anything else (strings, mixed, vectors) -> ``object``.
    """
    try:
        arr = np.asarray(vals)
    except (ValueError, TypeError):
        return _to_object_array(vals)
    if arr.ndim != 1:
        return _to_object_array(vals)
    k = arr.dtype.kind
    if k == 'b':
        return arr
    if k in 'iu':
        try:
            return arr.astype(np.int64)
        except (OverflowError, ValueError):
            return arr.astype(np.float64)
    if k == 'f':
        return arr.astype(np.float64, copy=False)
    if k == 'O':
        # numbers mixed with None -> float with NaN
        if all(v is None or (isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)))
               for v in vals):
            try:
                return np.array([np.nan if v is None else v for v in vals], dtype=np.float64)
            except (TypeError, ValueError):
                pass
    return _to_object_array(vals)


def _concat_columns(chunks: List[np.ndarray]) -> np.ndarray:
    """Concatenate typed column chunks, widening bool -> int -> float -> object."""
    if not chunks:
        return np.empty(0, dtype=np.float64)
    if len(chunks) == 1:
        return chunks[0]
    kinds = {c.dtype.kind for c in chunks}
    if len(kinds) == 1:
        return np.concatenate(chunks)
    if kinds & {'O', 'U', 'S', 'c'}:
        target = object
    elif 'f' in kinds:
        target = np.float64
    else:
        target = np.int64
    return np.concatenate([c.astype(target) for c in chunks])


def _frame_from_arrays(t: np.ndarray, arrays: List[np.ndarray], variables: List[str]) -> pd.DataFrame:
    idx = pd.Index(np.asarray(t, dtype=np.float64), name=_TIME)
    df = pd.DataFrame({i: a for i, a in enumerate(arrays)}, index=idx)
    df.columns = list(variables)
    return df


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Make object columns Parquet-friendly (all-str or None), converting mixed values to str."""
    out = None
    for i, _col in enumerate(df.columns):
        s = df.iloc[:, i]
        if s.dtype == object:
            vals = s.to_numpy()
            if not all(v is None or isinstance(v, str) for v in vals):
                if out is None:
                    out = df.copy()
                out.isetitem(i, pd.Series([None if v is None else str(v) for v in vals],
                                          index=df.index, dtype=object))
    return df if out is None else out


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class Recorder:
    """Buffers rows ``[time, v1, v2, ...]`` per simulator into typed columns.

    Rows are appended to a plain list (O(1) per step, no name lookups) and
    converted into NumPy column chunks every ``chunk_rows`` rows.  With
    ``stream_dir`` set (requires pyarrow) each chunk is written to
    ``<stream_dir>/_parts/<sim>/part-NNNNN.parquet`` and dropped from memory;
    :meth:`finalize` then writes ``<stream_dir>/<sim>.parquet``.
    """

    def __init__(self, columns: Dict[str, List[str]], chunk_rows: int = 65536,
                 stream_dir: Optional[str] = None):
        self.columns = {sim: list(v) for sim, v in columns.items()}
        self.chunk_rows = max(1, int(chunk_rows))
        self.stream_dir = stream_dir
        if stream_dir is not None:
            _require_pyarrow()
        self._pending: Dict[str, list] = {sim: [] for sim in self.columns}
        self._chunks: Dict[str, list] = {sim: [] for sim in self.columns}
        self._parts: Dict[str, List[str]] = {sim: [] for sim in self.columns}
        self._nrows: Dict[str, int] = {sim: 0 for sim in self.columns}
        self.files: Dict[str, str] = {}

    def add_rows(self, sim: str, rows: Iterable[Sequence[Any]]):
        """Append several rows for *sim*."""
        pending = self._pending[sim]
        pending.extend(rows)
        if len(pending) >= self.chunk_rows:
            self._flush(sim)

    def add_row(self, sim: str, row: Sequence[Any]):
        """Append a single row for *sim*."""
        pending = self._pending[sim]
        pending.append(row)
        if len(pending) >= self.chunk_rows:
            self._flush(sim)

    def n_rows(self, sim: str) -> int:
        return self._nrows[sim] + len(self._pending[sim])

    def _flush(self, sim: str):
        rows = self._pending[sim]
        if not rows:
            return
        self._pending[sim] = []
        variables = self.columns[sim]
        width = len(variables) + 1
        for r in rows:
            if len(r) != width:
                raise ValueError(
                    f"Simulator '{sim}' returned a row with {len(r) - 1} value(s) for "
                    f"{len(variables)} output(s) {variables}: {list(r)!r}"
                )
        cols = list(zip(*rows))
        t = np.asarray(cols[0], dtype=np.float64)
        arrays = [_column_from_values(c) for c in cols[1:]]
        self._nrows[sim] += len(rows)
        if self.stream_dir is None:
            self._chunks[sim].append((t, arrays))
        else:
            part_dir = os.path.join(self.stream_dir, '_parts', safe_filename_component(sim))
            os.makedirs(part_dir, exist_ok=True)
            path = os.path.join(part_dir, f"part-{len(self._parts[sim]):05d}.parquet")
            df = _frame_from_arrays(t, arrays, [f"c{i}" for i in range(len(arrays))])
            _parquet_safe(df).to_parquet(path)
            self._parts[sim].append(path)

    def _collect(self, sim: str):
        """Return (time, [columns]) for everything recorded for *sim*."""
        self._flush(sim)
        variables = self.columns[sim]
        if self.stream_dir is None:
            chunks = self._chunks[sim]
        else:
            chunks = []
            for p in self._parts[sim]:
                df = pd.read_parquet(p)
                chunks.append((df.index.to_numpy(dtype=np.float64),
                               [df.iloc[:, i].to_numpy() for i in range(df.shape[1])]))
        t = (np.concatenate([c[0] for c in chunks]) if chunks
             else np.empty(0, dtype=np.float64))
        arrays = [_concat_columns([c[1][i] for c in chunks]) for i in range(len(variables))]
        return t, arrays

    def finalize(self) -> Dict[str, Any]:
        """Return ``{sim: DataFrame}`` (memory) or ``{sim: loader}`` (streaming)."""
        frames: Dict[str, Any] = {}
        used = set()
        for sim, variables in self.columns.items():
            t, arrays = self._collect(sim)
            df = _frame_from_arrays(t, arrays, variables)
            if self.stream_dir is None:
                frames[sim] = df
                self._chunks[sim] = []
            else:
                fname = _unique_name(safe_filename_component(sim), used) + '.parquet'
                path = os.path.join(self.stream_dir, fname)
                _write_sim_parquet(df, path)
                self.files[sim] = fname
                frames[sim] = _ParquetLoader(path, variables)
                del df, arrays
        if self.stream_dir is not None:
            shutil.rmtree(os.path.join(self.stream_dir, '_parts'), ignore_errors=True)
        return frames


def _unique_name(base: str, used: set) -> str:
    name, i = base, 1
    while name.lower() in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name.lower())
    return name


def _write_sim_parquet(df: pd.DataFrame, path: str):
    out = _parquet_safe(df)
    cols = list(out.columns)
    if len(set(map(str, cols))) != len(cols) or _TIME in cols:
        raise ValueError(f"Cannot store duplicate/reserved column names {cols} in {path!r}.")
    out = out.reset_index()
    out.columns = [_TIME] + [str(c) for c in cols]
    out.to_parquet(path, index=False)


class _ParquetLoader:
    """Lazily load one simulator table from a Parquet file."""

    def __init__(self, path: str, variables: List[str]):
        self.path = path
        self.variables = list(variables)

    def __call__(self) -> pd.DataFrame:
        df = pd.read_parquet(self.path)
        return _restore_index(df)


def _restore_index(df: pd.DataFrame) -> pd.DataFrame:
    if _TIME in df.columns:
        df = df.set_index(_TIME)
    df.index = pd.Index(np.asarray(df.index, dtype=np.float64), name=_TIME)
    return df


# ---------------------------------------------------------------------------
# KPI registry
# ---------------------------------------------------------------------------

KpiFunc = Callable[[pd.Series, Optional[str]], Optional[float]]
_KPI_REGISTRY: Dict[str, KpiFunc] = {}


def register_kpi(name: str, func: KpiFunc, overwrite: bool = True):
    """Register a KPI ``func(series, unit) -> float | None``.

    *series* is a numeric ``pandas.Series`` indexed by time in seconds (sorted,
    NaNs dropped); *unit* is the variable's unit string or ``None``.  Return
    ``None`` when the KPI does not apply to the variable.
    """
    if not callable(func):
        raise TypeError("KPI function must be callable.")
    if name in _KPI_REGISTRY and not overwrite:
        raise ValueError(f"KPI '{name}' is already registered.")
    _KPI_REGISTRY[name] = func


def unregister_kpi(name: str):
    """Remove a registered KPI."""
    _KPI_REGISTRY.pop(name, None)


def available_kpis() -> List[str]:
    """Names of all registered KPIs."""
    return list(_KPI_REGISTRY)


def _zoh_integral(s: pd.Series) -> float:
    t = s.index.to_numpy(dtype=np.float64)
    v = s.to_numpy(dtype=np.float64)
    if len(t) < 2:
        return 0.0
    return float(np.sum(v[:-1] * np.diff(t)))


def _trapz_integral(s: pd.Series) -> float:
    t = s.index.to_numpy(dtype=np.float64)
    v = s.to_numpy(dtype=np.float64)
    if len(t) < 2:
        return 0.0
    return float(np.sum(0.5 * (v[:-1] + v[1:]) * np.diff(t)))


def _power_to_kw(unit: Optional[str]) -> Optional[float]:
    """Factor converting *unit* to kW, or None when *unit* is not an active-power unit."""
    if not unit:
        return None
    try:
        u = parse_unit(unit)
    except ValueError:
        return None
    if u.dimension != 'power':
        return None
    return u.factor / 1e3


def _kpi_energy(integrator):
    def kpi(s: pd.Series, unit: Optional[str]) -> Optional[float]:
        f = _power_to_kw(unit)
        if f is None:
            return None
        return integrator(s) * f / 3600.0
    return kpi


def _kpi_time_weighted_mean(s: pd.Series, unit: Optional[str]) -> Optional[float]:
    if len(s) == 0:
        return None
    span = float(s.index[-1] - s.index[0])
    if span <= 0:
        return float(s.iloc[-1])
    return _zoh_integral(s) / span


register_kpi('min', lambda s, unit: float(s.min()) if len(s) else None)
register_kpi('max', lambda s, unit: float(s.max()) if len(s) else None)
register_kpi('mean', lambda s, unit: float(s.mean()) if len(s) else None)
register_kpi('time_weighted_mean', _kpi_time_weighted_mean)
register_kpi('energy_kwh', _kpi_energy(_zoh_integral))
register_kpi('energy_kwh_trapz', _kpi_energy(_trapz_integral))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def _seconds_of(rule) -> float:
    if isinstance(rule, (int, float, np.integer, np.floating)) and not isinstance(rule, bool):
        sec = float(rule)
    else:
        sec = pd.Timedelta(rule).total_seconds()
    if sec <= 0:
        raise ValueError(f"resample step must be > 0, got {rule!r}.")
    return sec


class Results:
    """Results of one co-simulation run (or an ensemble of runs).

    Access
    ------
    ``res.sims``                    simulator names
    ``res.variables(sim=None)``     variable names (``'sim.var'`` when *sim* is None)
    ``res['sim.var']``              ``pandas.Series`` indexed by time
    ``res['sim']``                  ``pandas.DataFrame`` (index ``time``)
    ``res.to_frame(...)``           wide frame on the union time grid, columns ``'sim.var'``
    ``res.units`` / ``res.meta``    units per ``'sim.var'`` and run metadata
    ``res.summary()`` / ``res.kpis()``

    Time is in seconds (float) unless ``start_datetime`` is set, in which case
    a ``DatetimeIndex`` is returned.
    """

    def __init__(self, frames: Optional[Dict[str, Any]] = None, meta: Optional[dict] = None,
                 units: Optional[Dict[str, Optional[str]]] = None, start_datetime=None):
        self._frames: Dict[str, Any] = dict(frames or {})
        self._meta: dict = dict(meta or {})
        self._units: Dict[str, Optional[str]] = dict(units or {})
        if start_datetime is None:
            start_datetime = self._meta.get('start_datetime')
        self._start = pd.Timestamp(start_datetime) if start_datetime is not None else None
        # ensemble state
        self._runs: Optional[List['Results']] = None
        self._run_keys: List[Any] = []
        self._params: pd.DataFrame = pd.DataFrame()

    # -- basic access -------------------------------------------------------

    def _raw(self, sim: str) -> pd.DataFrame:
        """Stored frame for *sim* (float-seconds index). Do not mutate."""
        if sim not in self._frames:
            raise KeyError(f"No simulator '{sim}' in results. Available: {self.sims}")
        f = self._frames[sim]
        if callable(f) and not isinstance(f, pd.DataFrame):
            f = f()
            self._frames[sim] = f
        return f

    def _index(self, t: pd.Index, start_datetime=None) -> pd.Index:
        start = pd.Timestamp(start_datetime) if start_datetime is not None else self._start
        if start is None:
            return pd.Index(np.asarray(t, dtype=np.float64), name=_TIME)
        return pd.DatetimeIndex(start + pd.to_timedelta(np.asarray(t, dtype=np.float64), unit='s'),
                                name=_TIME)

    @property
    def is_ensemble(self) -> bool:
        return self._runs is not None

    @property
    def sims(self) -> List[str]:
        if self.is_ensemble:
            seen: Dict[str, None] = {}
            for r in self._runs:
                for s in r.sims:
                    seen.setdefault(s, None)
            return list(seen)
        return list(self._frames)

    def variables(self, sim: Optional[str] = None) -> List[str]:
        """Variable names of *sim*, or all ``'sim.var'`` names when *sim* is None."""
        if self.is_ensemble:
            seen: Dict[str, None] = {}
            for r in self._runs:
                for v in r.variables(sim):
                    seen.setdefault(v, None)
            return list(seen)
        if sim is not None:
            return self._variables_of(sim)
        return [f"{s}.{v}" for s in self.sims for v in self._variables_of(s)]

    def _variables_of(self, sim: str) -> List[str]:
        f = self._frames.get(sim)
        if f is None:
            raise KeyError(f"No simulator '{sim}' in results. Available: {self.sims}")
        if isinstance(f, _ParquetLoader):
            return list(f.variables)
        return [str(c) for c in self._raw(sim).columns]

    def _split(self, key: str):
        if not isinstance(key, str):
            raise KeyError(key)
        if key in self.sims:
            return key, None
        if '.' in key:
            sim, var = key.split('.', 1)
            if sim in self.sims:
                return sim, var
        raise KeyError(f"'{key}' is neither a simulator nor a 'sim.var' name. Simulators: {self.sims}")

    def __getitem__(self, key: str):
        if self.is_ensemble:
            sim, var = self._split(key)
            parts = {k: r[key] for k, r in zip(self._run_keys, self._runs) if sim in r.sims}
            if var is None:
                return pd.concat(parts, axis=1, names=['run', 'variable'])
            return pd.concat(parts, axis=1, names=['run'])
        sim, var = self._split(key)
        df = self._raw(sim)
        if var is None:
            out = df.copy()
            out.index = self._index(df.index)
            return out
        if var not in df.columns:
            raise KeyError(f"Simulator '{sim}' has no variable '{var}'. Variables: {list(df.columns)}")
        s = df[var].copy()
        s.index = self._index(df.index)
        s.name = key
        return s

    def __contains__(self, key) -> bool:
        try:
            sim, var = self._split(key)
        except KeyError:
            return False
        return var is None or var in self.variables(sim)

    def __iter__(self):
        return iter(self.sims)

    def __len__(self) -> int:
        return len(self.sims)

    def keys(self):
        return self.sims

    def items(self):
        return [(s, self[s]) for s in self.sims]

    def values(self):
        return [self[s] for s in self.sims]

    def __repr__(self) -> str:
        if self.is_ensemble:
            return f"<Results ensemble: {len(self._runs)} runs, sims={self.sims}>"
        rows = {s: (self._raw(s).shape[0] if isinstance(self._frames[s], pd.DataFrame) else '?')
                for s in self.sims}
        status = self._meta.get('status', '')
        return f"<Results {status} sims={len(rows)} rows={rows}>"

    @property
    def units(self) -> Dict[str, Optional[str]]:
        if self.is_ensemble:
            u: Dict[str, Optional[str]] = {}
            for r in self._runs:
                for k, v in r.units.items():
                    u.setdefault(k, v)
            return u
        return {v: self._units.get(v) for v in self.variables()}

    @property
    def meta(self) -> dict:
        return self._meta

    @property
    def status(self) -> Optional[str]:
        return self._meta.get('status')

    @property
    def run_dir(self) -> Optional[str]:
        return self._meta.get('run_dir')

    @property
    def start_datetime(self):
        return self._start

    # -- wide frames ----------------------------------------------------------

    def _select(self, sims=None, variables=None) -> List[tuple]:
        sims = self.sims if sims is None else ([sims] if isinstance(sims, str) else list(sims))
        for s in sims:
            if s not in self._frames:
                raise KeyError(f"No simulator '{s}' in results. Available: {self.sims}")
        wanted = None if variables is None else (
            [variables] if isinstance(variables, str) else list(variables))
        out = []
        for s in sims:
            for v in self._variables_of(s):
                full = f"{s}.{v}"
                if wanted is None or full in wanted or v in wanted:
                    out.append((s, v))
        if wanted is not None:
            found = {f"{s}.{v}" for s, v in out} | {v for _, v in out}
            missing = [w for w in wanted if w not in found]
            if missing:
                raise KeyError(f"Unknown variable(s) {missing}.")
        return out

    def to_frame(self, sims=None, variables=None, resample=None, ffill=True,
                 start_datetime=None) -> pd.DataFrame:
        """Wide ``DataFrame`` with columns ``'sim.var'`` on the union of all time points.

        Parameters
        ----------
        sims, variables : selection (``variables`` accepts ``'sim.var'`` or bare names).
        resample : number of seconds or a pandas timedelta string (``'15min'``):
            sample every column (zero-order hold) on a regular grid from the
            first to the last time point.
        ffill : forward-fill gaps created by aligning simulators with different
            time grids (zero-order hold).  Ignored when ``resample`` is given.
        start_datetime : override the run's ``start_datetime`` for the index.
        """
        if self.is_ensemble:
            parts = {k: r.to_frame(sims=sims, variables=variables, resample=resample, ffill=ffill,
                                   start_datetime=start_datetime)
                     for k, r in zip(self._run_keys, self._runs)}
            return pd.concat(parts, axis=1, names=['run', 'variable'])
        sel = self._select(sims, variables)
        series = []
        for s, v in sel:
            df = self._raw(s)
            col = df[v]
            if col.index.has_duplicates:
                col = col[~col.index.duplicated(keep='last')]
            col = col.rename(f"{s}.{v}")
            series.append(col)
        if not series:
            return pd.DataFrame(index=self._index(pd.Index([], dtype=np.float64), start_datetime))
        wide = pd.concat(series, axis=1, join='outer', sort=True)
        if resample is not None:
            step = _seconds_of(resample)
            t = wide.index.to_numpy(dtype=np.float64)
            grid = np.arange(t[0], t[-1] + step * 1e-9, step) if len(t) else np.empty(0)
            wide = wide.ffill().reindex(pd.Index(grid, name=_TIME), method='ffill')
        elif ffill:
            wide = wide.ffill()
        wide.index = self._index(wide.index, start_datetime)
        wide.columns.name = None
        return wide

    # -- statistics ----------------------------------------------------------------

    @staticmethod
    def _numeric(col: pd.Series) -> Optional[pd.Series]:
        if pd.api.types.is_bool_dtype(col):
            return col.astype(np.float64)
        if pd.api.types.is_numeric_dtype(col):
            return col.astype(np.float64)
        return None

    def summary(self) -> pd.DataFrame:
        """Per-variable ``min, max, mean, last, count`` as a clean **numeric** frame.

        Index ``variable`` (``'sim.var'``); every column is numeric (``count`` is integer,
        ``last`` is ``NaN`` for non-numeric variables), so ``res.summary().round(2)`` and
        ``res.summary().loc[row, ['min', 'max']].round()`` work.  Units and dtypes are in
        :meth:`describe_variables`.
        """
        if self.is_ensemble:
            return pd.concat({k: r.summary() for k, r in zip(self._run_keys, self._runs)},
                             names=['run', 'variable'])
        rows, idx = [], []
        for s in self.sims:
            df = self._raw(s)
            for v in df.columns:
                col = df[v]
                num = self._numeric(col)
                if num is not None:
                    last_num = num.dropna()
                    row = {'min': last_num.min() if len(last_num) else np.nan,
                           'max': last_num.max() if len(last_num) else np.nan,
                           'mean': last_num.mean() if len(last_num) else np.nan,
                           'last': float(num.iloc[-1]) if len(num) else np.nan}
                else:
                    row = {'min': np.nan, 'max': np.nan, 'mean': np.nan, 'last': np.nan}
                row['count'] = int(col.notna().sum())
                rows.append(row)
                idx.append(f"{s}.{v}")
        cols = ['min', 'max', 'mean', 'last', 'count']
        out = pd.DataFrame(rows, index=pd.Index(idx, name='variable'), columns=cols)
        return out.astype({'min': float, 'max': float, 'mean': float, 'last': float, 'count': int})

    def describe_variables(self) -> pd.DataFrame:
        """Per-variable ``unit`` and ``dtype`` (same index as :meth:`summary`).

        ``unit`` is ``None`` where the variable's unit is unspecified.
        """
        if self.is_ensemble:
            return self._runs[0].describe_variables()
        rows, idx = [], []
        for s in self.sims:
            df = self._raw(s)
            for v in df.columns:
                rows.append({'unit': self._units.get(f"{s}.{v}"), 'dtype': str(df[v].dtype)})
                idx.append(f"{s}.{v}")
        return pd.DataFrame(rows, index=pd.Index(idx, name='variable'), columns=['unit', 'dtype'])

    def kpis(self, names: Optional[Sequence[str]] = None, sims=None, variables=None) -> pd.DataFrame:
        """Evaluate registered KPIs for every numeric variable.

        Returns a ``DataFrame`` (index ``'sim.var'``, one column per KPI); NaN
        where a KPI does not apply (e.g. ``energy_kwh`` for non-power units).
        Built-ins: ``min``, ``max``, ``mean``, ``time_weighted_mean``,
        ``energy_kwh`` (zero-order hold integral of power, in kWh) and
        ``energy_kwh_trapz`` (trapezoid rule).
        """
        if self.is_ensemble:
            return pd.concat({k: r.kpis(names, sims, variables) for k, r in zip(self._run_keys, self._runs)},
                             names=['run', 'variable'])
        names = list(_KPI_REGISTRY) if names is None else ([names] if isinstance(names, str) else list(names))
        unknown = [n for n in names if n not in _KPI_REGISTRY]
        if unknown:
            raise KeyError(f"Unknown KPI(s) {unknown}. Registered: {available_kpis()}")
        rows, idx = [], []
        for s, v in self._select(sims, variables):
            col = self._numeric(self._raw(s)[v])
            if col is None:
                continue
            col = col.dropna()
            if not col.index.is_monotonic_increasing:
                col = col.sort_index(kind='stable')
            unit = self._units.get(f"{s}.{v}")
            row = {}
            for n in names:
                val = _KPI_REGISTRY[n](col, unit)
                row[n] = np.nan if val is None else val
            rows.append(row)
            idx.append(f"{s}.{v}")
        return pd.DataFrame(rows, index=pd.Index(idx, name='variable'), columns=names)

    # -- ensembles -------------------------------------------------------------------

    @classmethod
    def concat(cls, results: Sequence['Results'], keys: Optional[Sequence[Any]] = None,
               params=None) -> 'Results':
        """Combine several runs into an ensemble with a ``run`` dimension.

        ``keys`` labels the runs (default ``0..n-1``); if the keys are dicts
        they are taken as the run parameters.  ``params`` (DataFrame or list of
        dicts, one row per run) describes the parameters of each run.
        """
        results = list(results)
        if not results:
            raise ValueError("Results.concat() needs at least one Results object.")
        for r in results:
            if not isinstance(r, Results):
                raise TypeError(f"Expected Results, got {type(r).__name__}.")
        n = len(results)
        if keys is not None:
            keys = list(keys)
            if len(keys) != n:
                raise ValueError(f"{len(keys)} keys for {n} results.")
            if params is None and all(isinstance(k, dict) for k in keys):
                params, keys = list(keys), list(range(n))
        else:
            keys = list(range(n))
        if len(set(map(repr, keys))) != n:
            raise ValueError("Run keys must be unique.")
        if params is None:
            p = pd.DataFrame(index=pd.Index(keys, name='run'))
        else:
            p = params.copy() if isinstance(params, pd.DataFrame) else pd.DataFrame(list(params))
            if len(p) != n:
                raise ValueError(f"params has {len(p)} rows for {n} runs.")
            p.index = pd.Index(keys, name='run')
        ens = cls(meta={'ensemble': True, 'n_runs': n,
                        'runs': [r.meta.get('run_id') for r in results]})
        ens._runs = results
        ens._run_keys = keys
        ens._params = p
        return ens

    @property
    def runs(self) -> List['Results']:
        """Per-run :class:`Results` (a single run returns ``[self]``)."""
        return list(self._runs) if self.is_ensemble else [self]

    @property
    def run_keys(self) -> List[Any]:
        return list(self._run_keys) if self.is_ensemble else [0]

    @property
    def params(self) -> pd.DataFrame:
        """Run parameters (one row per run, index = run key)."""
        return self._params.copy()

    def run(self, key) -> 'Results':
        """One run of an ensemble, by key (or by position if *key* is not a key)."""
        if not self.is_ensemble:
            if key in (0, -1, None):
                return self
            raise KeyError(f"Single-run results have no run {key!r}.")
        for k, r in zip(self._run_keys, self._runs):
            if k == key:
                return r
        if isinstance(key, (int, np.integer)):
            return self._runs[key]
        raise KeyError(f"No run {key!r}. Runs: {self._run_keys}")

    # -- persistence ---------------------------------------------------------------------

    def _meta_for_storage(self, fmt: str, files: Dict[str, str]) -> dict:
        meta = dict(self._meta)
        meta['units'] = self.units
        meta['storage'] = {
            'format': fmt,
            'files': files,
            'variables': {s: self._variables_of(s) for s in self.sims},
            'dtypes': {s: {str(c): str(t) for c, t in self._raw(s).dtypes.items()} for s in self.sims},
        }
        if self._start is not None:
            meta['start_datetime'] = self._start.isoformat()
        return meta

    def _write_meta(self, directory: str, fmt: str, files: Dict[str, str]):
        meta = self._meta_for_storage(fmt, files)
        with open(os.path.join(directory, _META_FILE), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, default=str)

    @staticmethod
    def _resolve_format(path, format: str) -> str:
        if format not in _FORMATS:
            raise ValueError(f"format must be one of {_FORMATS}, got {format!r}.")
        if format != 'auto':
            return format
        if str(path).lower().endswith(('.h5', '.hdf5', '.hdf')):
            return 'hdf5'
        if _have_pyarrow():
            return 'parquet'
        warnings.warn("pyarrow is not installed; saving results as CSV instead of Parquet "
                      "(pip install pyarrow).", UserWarning, stacklevel=3)
        return 'csv'

    def save(self, path, format: str = 'auto') -> str:
        """Save to *path* and return its absolute path.

        ``'parquet'`` / ``'csv'``: *path* is a directory receiving one file per
        simulator plus ``meta.json``.  ``'hdf5'``: *path* is a single ``.h5`` file
        (needs ``tables``).  ``'auto'``: HDF5 for ``.h5``/``.hdf5`` paths, else
        Parquet if pyarrow is installed, else CSV (with a warning).
        """
        path = os.fspath(path)
        fmt = self._resolve_format(path, format)
        if fmt == 'hdf5':
            return self.to_hdf(path)
        if fmt == 'parquet':
            _require_pyarrow()
        os.makedirs(path, exist_ok=True)
        if self.is_ensemble:
            for i, r in enumerate(self._runs):
                r.save(os.path.join(path, f"run_{i:04d}"), format=fmt)
            self._params.to_csv(os.path.join(path, _PARAMS_FILE))
            with open(os.path.join(path, _ENSEMBLE_FILE), 'w', encoding='utf-8') as f:
                json.dump({'n_runs': len(self._runs), 'keys': [repr(k) for k in self._run_keys],
                           'format': fmt}, f, indent=2)
            return os.path.abspath(path)
        files: Dict[str, str] = {}
        used: set = set()
        for s in self.sims:
            df = self._raw(s)
            base = _unique_name(safe_filename_component(s), used)
            if fmt == 'parquet':
                files[s] = base + '.parquet'
                _write_sim_parquet(df, os.path.join(path, files[s]))
            else:
                files[s] = base + '.csv'
                df.to_csv(os.path.join(path, files[s]), index=True, index_label=_TIME)
        self._write_meta(path, fmt, files)
        return os.path.abspath(path)

    def to_parquet(self, directory) -> str:
        """Write one Parquet file per simulator (+ ``meta.json``) into *directory*."""
        return self.save(directory, format='parquet')

    def to_csv(self, directory) -> str:
        """Write one CSV file per simulator (+ ``meta.json``) into *directory*."""
        return self.save(directory, format='csv')

    def to_hdf(self, path) -> str:
        """Write all simulator tables into one HDF5 file (requires ``tables``)."""
        _require_tables()
        if self.is_ensemble:
            raise NotImplementedError("Save ensembles with format='parquet' or 'csv'.")
        path = os.fspath(path)
        files = {s: f"sim_{i}" for i, s in enumerate(self.sims)}
        meta = self._meta_for_storage('hdf5', files)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pd.HDFStore(path, mode='w') as store:
                for s, key in files.items():
                    store.put(key, self._raw(s))
                store.put('_meta', pd.Series([json.dumps(meta, default=str)]))
        return os.path.abspath(path)

    @classmethod
    def load(cls, path) -> 'Results':
        """Load results written by :meth:`save` / a run directory / an HDF5 file."""
        path = os.fspath(path)
        if os.path.isfile(path):
            return cls._load_hdf(path)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"No results at {path!r}.")
        if os.path.exists(os.path.join(path, _ENSEMBLE_FILE)):
            subdirs = sorted(d for d in os.listdir(path)
                             if d.startswith('run_') and os.path.isdir(os.path.join(path, d)))
            runs = [cls.load(os.path.join(path, d)) for d in subdirs]
            params = pd.read_csv(os.path.join(path, _PARAMS_FILE), index_col=0)
            keys = list(params.index)
            ens = cls.concat(runs, keys=keys, params=params.reset_index(drop=True))
            return ens
        meta: dict = {}
        mpath = os.path.join(path, _META_FILE)
        if os.path.exists(mpath):
            with open(mpath, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        storage = meta.pop('storage', {}) or {}
        files = storage.get('files') or {}
        dtypes = storage.get('dtypes') or {}
        if not files:
            for fn in sorted(os.listdir(path)):
                stem, ext = os.path.splitext(fn)
                if ext.lower() in ('.parquet', '.csv'):
                    files[stem] = fn
        frames: Dict[str, Any] = {}
        for sim, fn in files.items():
            fp = os.path.join(path, fn)
            if fn.lower().endswith('.parquet'):
                _require_pyarrow()
                frames[sim] = _ParquetLoader(fp, (storage.get('variables') or {}).get(sim)
                                             or list(pd.read_parquet(fp).columns.drop(_TIME, errors='ignore')))
            else:
                df = _restore_index(pd.read_csv(fp))
                frames[sim] = _restore_dtypes(df, dtypes.get(sim, {}))
        units = meta.pop('units', {}) or {}
        meta.setdefault('run_dir', meta.get('run_dir'))
        return cls(frames, meta=meta, units=units)

    @classmethod
    def _load_hdf(cls, path: str) -> 'Results':
        _require_tables()
        with pd.HDFStore(path, mode='r') as store:
            meta = json.loads(store['_meta'].iloc[0])
            storage = meta.pop('storage', {}) or {}
            frames = {sim: store[key] for sim, key in (storage.get('files') or {}).items()}
        units = meta.pop('units', {}) or {}
        return cls(frames, meta=meta, units=units)

    # -- dashboard -----------------------------------------------------------------------------

    def dashboard(self, path=None, open: bool = False, max_points: int = 5000,
                  offline: bool = False, title: str = "energysim Result Explorer") -> str:
        """Write the interactive HTML dashboard and return its path.

        Defaults to ``<run_dir>/dashboard.html`` for runs with a run directory,
        else a temporary file.  The browser is only opened when ``open=True``.
        """
        from .dashboard import generate_dashboard
        if path is None and self.run_dir and os.path.isdir(self.run_dir):
            path = os.path.join(self.run_dir, 'dashboard.html')
        return generate_dashboard(self, output_path=path, auto_open=open, title=title,
                                  max_points=max_points, offline=offline)


def _restore_dtypes(df: pd.DataFrame, dtypes: Dict[str, str]) -> pd.DataFrame:
    for c, t in dtypes.items():
        if c not in df.columns:
            continue
        try:
            if t == 'object' and df[c].dtype != object:
                df[c] = df[c].astype(object).where(df[c].notna(), None).map(
                    lambda v: v if v is None else str(v))
            elif t == 'bool' and df[c].dtype != bool:
                df[c] = df[c].astype(bool)
        except (TypeError, ValueError):
            pass
    return df
