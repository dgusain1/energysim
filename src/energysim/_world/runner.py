# -*- coding: utf-8 -*-
"""Time grid, run lifecycle, simulate() and results assembly."""

from __future__ import annotations
from typing import Any, Dict

import os
import math
import warnings
import time as _time
import datetime as _dt
from fractions import Fraction
from functools import reduce
import numpy as np
import yaml
from tqdm import tqdm

from ..results import Results, Recorder, make_run_dir, scenario_hash, _have_pyarrow, _versions
from ._common import (logger, ITERATIVE_REMOVED_MSG, _COUPLINGS, OPTION_KEYS, _to_fraction)


class RunnerMixin:
    """Time grid, run lifecycle, simulate() and results assembly."""

    # ------------------------------------------------------------------
    # Time grid
    # ------------------------------------------------------------------

    def get_lcm(self):
        """Least common multiple of all simulator step sizes (exact rational arithmetic)."""
        steps = [entry.step_size for entry in self.simulator_dict.values()]
        if not steps:
            raise ValueError("Cannot compute the step-size LCM: no simulators registered.")
        fracs = []
        for s in steps:
            f = _to_fraction(s)
            if f <= 0:
                raise ValueError(f"Step sizes must be > 0, got {s!r}.")
            fracs.append(f)

        def lcm2(a, b):
            return a * b // math.gcd(a, b)

        num = reduce(lcm2, (f.numerator for f in fracs))
        den = reduce(math.gcd, (f.denominator for f in fracs))
        return float(Fraction(num, den))

    def _macro_step(self):
        if self.exchange is None or self.exchange == 0:
            return self.get_lcm()
        if self.exchange < 0:
            raise ValueError(f"Macro step size t_macro ({self.exchange}) must be > 0.")
        return self.exchange

    def _compute_total_steps(self, macro):
        """Number of macro steps; raise ValueError unless the span is an integer multiple of *macro*.

        A relative tolerance of 1e-9 absorbs floating-point error
        (e.g. 0.3 / 0.1); anything else is a configuration error.
        """
        span = self.stop_time - self.start_time
        if macro <= 0:
            raise ValueError(f"Macro step ({macro}) must be > 0.")
        n_float = span / macro
        n = round(n_float)
        if abs(n_float - n) > 1e-9 * max(1.0, abs(n_float)):
            lo = self.start_time + math.floor(n_float) * macro
            hi = self.start_time + math.ceil(n_float) * macro
            raise ValueError(
                f"Simulation span stop_time - start_time = {span} is not an integer multiple of the "
                f"macro step {macro}. Use e.g. stop_time={lo:g} or stop_time={hi:g}."
            )
        if n < 1:
            raise ValueError(
                f"stop_time - start_time ({span}) must be at least one macro step ({macro})."
            )
        return int(n)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all state of the previous run (recorded-output map, exchange history, time grid).

        The configured simulators, connections and options are kept;
        ``last_results`` is cleared.
        """
        self.last_results = None
        self.sim_dict = {}
        self._exchange_history = {}
        self.macro_tstep = None
        self._total_steps = None
        self._current_dt = None

    @staticmethod
    def _normalize_params(params, sim_name):
        """Accept ``([names], [values])`` or ``{name: value}``; return a dict."""
        if isinstance(params, dict):
            return dict(params)
        if isinstance(params, (list, tuple)) and len(params) == 2:
            names, values = list(params[0]), list(params[1])
            if len(names) != len(values):
                raise ValueError(
                    f"Parameters for '{sim_name}': {len(names)} names but {len(values)} values."
                )
            return dict(zip(names, values))
        raise TypeError(
            f"Parameters for '{sim_name}' must be ([names], [values]) or {{name: value}}, got {params!r}."
        )

    def _collect_parameters(self):
        merged: Dict[str, Dict[str, Any]] = {}
        for source in (self.init_dict, self._parameter_set):
            for sim_name, params in (source or {}).items():
                if sim_name not in self.simulator_dict:
                    raise ValueError(f"Parameters given for unknown simulator '{sim_name}'.")
                merged.setdefault(sim_name, {}).update(self._normalize_params(params, sim_name))
        return merged

    def init(self):
        '''
        Prepares a fresh run: resets run state, calls ``configure(start, stop, macro)`` on every adapter, then
        ``init()`` on every adapter with initial parameters applied before or
        after ``init()`` according to ``adapter.SET_PARAMS_BEFORE_INIT``.
        '''
        logger.info("Simulation started..")
        self.reset()

        self.macro_tstep = self._macro_step()
        if self.t_macro_min is not None and self.t_macro_min <= 0:
            raise ValueError(f"t_macro_min ({self.t_macro_min}) must be > 0.")
        if self.variable_step:
            span = self.stop_time - self.start_time
            if span <= 0:
                raise ValueError(f"stop_time - start_time ({span}) must be > 0.")
            self._total_steps = max(1, math.ceil(span / self.macro_tstep - 1e-9))
        else:
            self._total_steps = self._compute_total_steps(self.macro_tstep)
        self._reset_events()
        params = self._collect_parameters()

        self.sim_dict = {name: list(entry.outputs) for name, entry in self.simulator_dict.items()}

        for sim_name, entry in self.simulator_dict.items():
            configure = getattr(entry.adapter, 'configure', None)
            if callable(configure):
                try:
                    configure(self.start_time, self.stop_time, self.macro_tstep)
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not configure the {entry.sim_type} simulator '{sim_name}': {exc}"
                    ) from exc

        for sim_name, entry in self.simulator_dict.items():
            adapter = entry.adapter
            p = params.get(sim_name)
            before = bool(getattr(adapter, 'SET_PARAMS_BEFORE_INIT', False))
            if p and before:
                self._apply_parameters(sim_name, p)
            try:
                adapter.init()
            except Exception as exc:
                raise RuntimeError(
                    f"Could not initialize the {entry.sim_type} simulator '{sim_name}': {exc}"
                ) from exc
            if p and not before:
                self._apply_parameters(sim_name, p)

    def _apply_parameters(self, sim_name, params):
        entry = self.simulator_dict[sim_name]
        setter = getattr(entry.adapter, 'set_parameters', None)
        if callable(setter):
            setter(dict(params))
        else:
            entry.adapter.set_value(list(params.keys()), list(params.values()))

    def _cleanup_all(self):
        """Call cleanup() on every adapter; errors are logged, never raised."""
        for sim_name, entry in self.simulator_dict.items():
            cleanup = getattr(entry.adapter, 'cleanup', None)
            if not callable(cleanup):
                continue
            try:
                cleanup()
            except Exception:
                logger.warning("Error during cleanup of simulator '%s'", sim_name, exc_info=True)

    def step_simulator(self, sim_name, entry, time, record_all, local_stop_time):
        """Step a single simulator from *time* to *local_stop_time*.

        Returns a list of [time, val1, val2, …] rows for recording.
        """
        adapter = entry.adapter
        outputs = entry.outputs
        tmp_var = []

        if record_all:
            # Delegate to the adapter's advance_with_recording(), which
            # respects internal solver logic (e.g. FMU ME adaptive stepping)
            tmp_var = adapter.advance_with_recording(time, local_stop_time, outputs)
        else:
            # Record snapshot at macro boundary BEFORE stepping
            if outputs:
                tmp = [time] + list(adapter.get_value(outputs, time))
            else:
                tmp = [time]
            tmp_var.append(tmp)
            # Hand the full interval to advance()
            adapter.advance(time, local_stop_time)

        return tmp_var

    def simulate(self, pbar=True, record_all=None, validate=True, out_dir=None):
        '''
        Simulates the world object and returns a :class:`~energysim.results.Results`.

        Supports two coupling modes (set via ``world(..., coupling=...)``):
        - ``'jacobi'``: Exchange all variables first, then step all simulators.
        - ``'gauss-seidel'``: Step simulators in topological order; after each
          simulator steps, immediately propagate its outputs to downstream
          simulators before they step (algebraic loops use a one-step delay).

        Every simulator gets one row per macro step (taken before stepping;
        with ``record_all`` one row per micro step) plus a final row at
        ``stop_time`` holding the state at the end of the run.  Rows may be
        irregular when events / hints shorten macro steps (``t_macro_min``,
        :meth:`schedule`, :meth:`add_trigger`); fired events are listed in
        ``Results.meta['events']``.

        ``record_all`` defaults to the value given via ``options({'record_all': ...})``
        or True.  ``out_dir`` (default: the world's ``out_dir``) persists the run
        to ``<out_dir>/<YYYYmmdd-HHMMSS>_<hash8>/``; otherwise nothing is written.
        The results are also stored as ``world.last_results``; if the run
        fails, the partial results (``meta['status'] == 'failed'``) are stored
        there and the error is re-raised.  ``cleanup()`` is always called on
        every adapter.  ``simulate()`` may be called repeatedly.
        '''
        if self.coupling == 'iterative':
            raise ValueError(ITERATIVE_REMOVED_MSG)
        if self.coupling not in _COUPLINGS:
            raise ValueError(f"coupling must be 'jacobi' or 'gauss-seidel', got '{self.coupling}'")
        if record_all is None:
            record_all = True if self.record_all is None else bool(self.record_all)
        self._log_unit_system()
        self._log_conversions('info')
        if validate:
            self.validate(raise_on_error=True)

        self.last_results = None
        out_dir = self.out_dir if out_dir is None else out_dir
        scenario = self._scenario_snapshot()
        digest = scenario_hash(scenario)
        now = _dt.datetime.now()
        run_dir = make_run_dir(out_dir, digest, now=now) if out_dir else None
        stream = run_dir is not None and _have_pyarrow()
        recorder = Recorder({name: list(entry.outputs) for name, entry in self.simulator_dict.items()},
                            chunk_rows=self._results_chunk_rows,
                            stream_dir=run_dir if stream else None)
        run_info = dict(scenario=scenario, digest=digest, created=now, run_dir=run_dir,
                        record_all=record_all, t0=_time.perf_counter())

        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                try:
                    self.init()
                    self._run_loop(recorder, record_all, pbar)
                    self._record_final_state(recorder)
                finally:
                    self._cleanup_all()
        except BaseException as exc:
            try:
                self.last_results = self._finish_results(recorder, run_info, 'failed', caught, exc)
            except Exception:
                logger.warning("Could not assemble partial results of the failed run", exc_info=True)
            raise
        finally:
            for w in caught:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source)

        self.last_results = self._finish_results(recorder, run_info, 'completed', caught)
        return self.last_results

    def _run_loop(self, recorder, record_all, pbar):
        """Macro-step loop: ``while t < stop: fire events, plan the step, couple, record``.

        Fixed-step (default ``t_macro_min == t_macro``): steps follow the grid
        ``start_time + k * t_macro`` exactly as before 3.0; scheduled events
        insert extra boundaries and the next step re-aligns with the grid.
        Variable step: see :meth:`_next_step` / :mod:`energysim._world.events`.
        """
        exec_order = self._execution_order or list(self.simulator_dict.keys())
        start, stop, macro = self.start_time, self.stop_time, self.macro_tstep
        eps = 1e-9 * max(1.0, abs(stop), abs(macro))
        t, k = start, 0
        first = True
        self._n_steps = 0
        with tqdm(total=self._total_steps, disable=not pbar) as bar:
            while t < stop - eps:
                self._evaluate_triggers(t, initial=first)
                first = False
                self._fire_due_events(t)
                t_end, regular = self._plan_step(t, k)
                self._current_dt = macro if regular else t_end - t

                if self.coupling == 'jacobi':
                    tmp_dict = self._step_jacobi(exec_order, t, t_end, record_all)
                else:
                    tmp_dict = self._step_gauss_seidel(exec_order, t, t_end, record_all)
                for sim_name, rows in tmp_dict.items():
                    recorder.add_rows(sim_name, rows)
                self._n_steps += 1

                if regular:
                    k += 1
                    t_next = start + k * macro      # same grid arithmetic as the fixed-step loop
                else:
                    t_next = t_end
                    idx = round((t_end - start) / macro)
                    if not self.variable_step and abs(start + idx * macro - t_end) <= eps:
                        k, t_next = idx, start + idx * macro
                    else:
                        k = None
                bar.update(min(max((t_next - t) / macro, 0.0), bar.total - bar.n) if bar.total else 0)
                t = t_next
        self._evaluate_triggers(stop, initial=first)
        self._finish_events()

    # ------------------------------------------------------------------
    # Results assembly
    # ------------------------------------------------------------------

    def _record_final_state(self, recorder):
        """Append one row per simulator at ``stop_time`` (state at the end of the run)."""
        t_end = self.stop_time
        for sim_name, entry in self.simulator_dict.items():
            outputs = entry.outputs
            try:
                row = [t_end] + (list(entry.adapter.get_value(outputs, t_end)) if outputs else [])
            except Exception as exc:
                warnings.warn(f"Could not record the final state of '{sim_name}' at t={t_end}: {exc}",
                              RuntimeWarning, stacklevel=2)
                continue
            recorder.add_row(sim_name, row)

    def _scenario_snapshot(self):
        """Serialisable scenario description used for the metadata and the run hash."""
        prev, logger.disabled = logger.disabled, True   # no "cannot serialise" noise per run
        try:
            scenario = self.to_dict()
        except Exception as exc:  # pragma: no cover - defensive
            scenario = {'error': f"scenario not serialisable: {exc}"}
        finally:
            logger.disabled = prev
        scenario.get('world', {}).pop('out_dir', None)
        return scenario

    def _results_units(self):
        """``{'sim.var': unit}`` with the same resolution as connections (see :meth:`resolve_variable_unit`)."""
        conn_units = {}
        for conn in self._parsed_connections:
            out = conn[0]
            spec = getattr(conn, '_spec_units', (None, None))[0]
            if spec and not (isinstance(out, tuple) and isinstance(out[0], tuple)):
                conn_units.setdefault((out[0], out[1]), spec)
        units = {}
        for sim_name, entry in self.simulator_dict.items():
            for var in entry.outputs:
                units[f"{sim_name}.{var}"] = self.resolve_variable_unit(
                    sim_name, var, conn_units.get((sim_name, var)))[0]
        return units

    def _finish_results(self, recorder, info, status, caught, error=None):
        frames = recorder.finalize()
        units = self._results_units()
        start_dt = None
        if self.start_datetime is not None:
            import pandas as pd
            start_dt = pd.Timestamp(self.start_datetime).isoformat()
        created = info['created']
        meta = {
            'run_id': f"{created.strftime('%Y%m%d-%H%M%S')}_{info['digest'][:8]}",
            'created': created.isoformat(timespec='seconds'),
            'status': status,
            'error': None if error is None else f"{type(error).__name__}: {error}",
            'wall_time_s': round(_time.perf_counter() - info['t0'], 6),
            'versions': _versions(),
            'scenario': info['scenario'],
            'scenario_sha256': info['digest'],
            'seed': self.seed,
            'start_datetime': start_dt,
            'start_time': self.start_time,
            'stop_time': self.stop_time,
            't_macro': self.macro_tstep,
            't_macro_min': self._t_macro_min_eff() if self.macro_tstep is not None else self.t_macro_min,
            'n_macro_steps': getattr(self, '_n_steps', None),
            'events': [dict(e) for e in getattr(self, '_event_log', [])],
            'coupling': self.coupling,
            'extrapolation': self.extrapolation,
            'record_all': info['record_all'],
            'units': units,
            'unit_system': self.unit_system.to_dict(),
            'unit_policy': self.unit_policy,
            'warnings': [{'category': w.category.__name__, 'message': str(w.message)}
                         for w in list(caught)[:1000]],
            'n_rows': {s: recorder.n_rows(s) for s in recorder.columns},
            'run_dir': info['run_dir'],
        }
        res = Results(frames, meta=meta, units=units, start_datetime=self.start_datetime)
        run_dir = info['run_dir']
        if run_dir:
            if recorder.stream_dir is not None:
                res._write_meta(run_dir, 'parquet', dict(recorder.files))
            else:
                warnings.warn("pyarrow is not installed; the run directory stores CSV files instead "
                              "of Parquet (pip install pyarrow).", UserWarning, stacklevel=3)
                res.save(run_dir, format='csv')
            try:
                scenario_yaml = yaml.safe_dump(info['scenario'], sort_keys=False, default_flow_style=False)
            except Exception:
                scenario_yaml = repr(info['scenario'])
            with open(os.path.join(run_dir, 'scenario.yaml'), 'w', encoding='utf-8') as f:
                f.write(scenario_yaml)
        return res

    def options(self, settings):
        """Set run options.

        Recognised keys (anything else raises ``ValueError``):

        - ``init``: ``{sim_name: ([names], [values])}`` or ``{sim_name: {name: value}}``
          initial values, applied at the start of every run.
        - ``parameter_set``: same format, merged with ``init`` (see :meth:`set_parameters`).
        - ``modify_signal``: ``{'sim.var': [gain]}`` or ``{'sim.var': [gain, offset]}``.
        - ``record_all``: bool, default for ``simulate(record_all=...)``.
        """
        if not isinstance(settings, dict):
            raise TypeError(f"options() expects a dict, got {type(settings).__name__}.")
        unknown = sorted(set(settings) - OPTION_KEYS)
        if unknown:
            raise ValueError(f"Unknown option(s) {unknown}. Valid options: {sorted(OPTION_KEYS)}.")
        if not self.simulator_dict:
            raise ValueError("Cannot add settings to world when no simulators are specified!")

        if 'init' in settings:
            init = settings['init'] or {}
            if not isinstance(init, dict):
                raise TypeError("options['init'] must be a dict {sim_name: ([names], [values])}.")
            for sim_name, params in init.items():
                if sim_name not in self.simulator_dict:
                    raise ValueError(f"options['init'] references unknown simulator '{sim_name}'.")
                self._normalize_params(params, sim_name)
            self.init_dict = init

        if 'parameter_set' in settings:
            self.set_parameters(settings['parameter_set'] or {})

        if 'modify_signal' in settings:
            mods = settings['modify_signal'] or {}
            if not isinstance(mods, dict):
                raise TypeError("options['modify_signal'] must be a dict {'sim.var': [gain, offset]}.")
            for key, mod in mods.items():
                if not isinstance(mod, (list, tuple)) or len(mod) not in (1, 2):
                    raise ValueError(
                        f"Invalid modify_signal entry for '{key}': {mod!r} (expected [gain] or [gain, offset])."
                    )
            self.modify_dict = mods

        if 'record_all' in settings:
            ra = settings['record_all']
            if not isinstance(ra, (bool, np.bool_)):
                raise TypeError(f"options['record_all'] must be a bool, got {ra!r}.")
            self.record_all = bool(ra)

    def set_parameters(self, parameter_dictionary):
        """Register parameter values ``{sim_name: ([names], [values])}`` (or ``{sim_name: {name: value}}``).

        Since 3.0 values are *not* applied immediately: they are applied at
        the start of every run, before or after ``adapter.init()`` as the
        adapter's ``SET_PARAMS_BEFORE_INIT`` flag requires.
        """
        if not isinstance(parameter_dictionary, dict):
            raise TypeError("set_parameters() expects a dict {sim_name: ([names], [values])}.")
        for sim_name, params in parameter_dictionary.items():
            if sim_name not in self.simulator_dict:
                raise ValueError(f"set_parameters(): unknown simulator '{sim_name}'.")
            normalized = self._normalize_params(params, sim_name)
            self._parameter_set.setdefault(sim_name, {}).update(normalized)

    def check_signal_output(self, fx):
        tmp = fx(1)
        if isinstance(tmp, list):
            return True
        logger.warning('Function must return list. Signal function will not be added.')
        return False

    def plot_results(self, xlab='Time(s)', ylab='', y=None, scientific=False):
        """Plot simulation results using matplotlib.

        Parameters
        ----------
        xlab : str
            X-axis label.
        ylab : str
            Y-axis label.
        y : list[str] or None
            Variables to plot, specified as 'sim_name.var_name' or 'var_name'.
            If empty/None, plots all recorded variables.
        scientific : bool
            Whether to use scientific notation for ticks.
        """
        import matplotlib.pyplot as plt
        if y is None:
            y = []
        res = self.last_results
        if res is None:
            raise RuntimeError("No simulation results available. Call simulate() first.")
        plt.figure(figsize=(10, 6))
        for sim_name in res.sims:
            df = res[sim_name]
            vars_to_plot = list(df.columns)
            if y:
                vars_to_plot = [c for c in vars_to_plot if f"{sim_name}.{c}" in y or c in y]
            for var in vars_to_plot:
                plt.plot(df.index, df[var], label=f"{sim_name}.{var}")
        plt.ticklabel_format(useOffset=scientific)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend()
        plt.tight_layout()
        plt.show()

