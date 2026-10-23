# -*- coding: utf-8 -*-
"""Event-driven / variable macro step: scheduled events, threshold triggers and step-size hints.

There is **no rollback**: events are handled *predictively*.

* The macro step ``t_macro`` is the maximum (and default) step.  With
  ``t_macro_min < t_macro`` the world asks every adapter for
  :meth:`~energysim.base.SimulatorAdapter.max_step` /
  :meth:`~energysim.base.SimulatorAdapter.next_event_time` and takes the
  minimum, clamped to ``[t_macro_min, t_macro]``.  With the default
  ``t_macro_min = t_macro`` the run is fixed-step and bit-identical to
  energysim < 3.0 (hints are not even queried).
* Scheduled events (:meth:`EventsMixin.schedule`) always shorten the step so
  that a step boundary lands exactly on the event time; they fire at the
  *start* of the step beginning there, before the exchange of that step.
  In fixed-step mode the following step re-aligns with the
  ``start_time + k * t_macro`` grid.
* Threshold triggers (:meth:`EventsMixin.add_trigger`) are evaluated at step
  boundaries.  A crossing detected at boundary ``t`` (it happened somewhere
  in the previous step) fires its action at the start of the step beginning
  at ``t``.  A declarative ``set`` on a connected input is held for the step
  in which it fires (the exchange skips that input once).  Localisation accuracy is therefore one macro step; event hints
  (``next_event_time``) and a smaller ``t_macro_min`` improve it.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Union

from .._safeexpr import compile_condition
from ._common import logger

_DIRECTIONS = ('rising', 'falling', 'both')
_EVENT_KEYS = frozenset({'time', 'name', 'set'})
_TRIGGER_KEYS = frozenset({'name', 'condition', 'direction', 'set', 'variables'})
TRIGGER_NOTE = ("crossing localised to the macro step that ended at detected_time "
                "(accuracy limited to the step; use t_macro_min / event hints to refine)")


def _check_action(action, what):
    if callable(action):
        return action
    if isinstance(action, dict):
        unknown = sorted(set(action) - {'set'})
        if unknown or 'set' not in action:
            raise ValueError(f"{what}: a declarative action must be {{'set': {{'sim.var': value}}}}, "
                             f"got keys {sorted(action)}.")
        targets = action['set']
        if not isinstance(targets, dict) or not targets:
            raise ValueError(f"{what}: 'set' must be a non-empty mapping {{'sim.var': value}}.")
        for k in targets:
            if not isinstance(k, str) or '.' not in k:
                raise ValueError(f"{what}: 'set' target {k!r} must be 'sim.var'.")
        return {'set': dict(targets)}
    raise TypeError(f"{what}: action must be a callable(world) or {{'set': {{'sim.var': value}}}}, "
                    f"got {type(action).__name__}.")


class EventsMixin:
    """Scheduled events, threshold triggers and the variable macro step."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(self, t: float, action: Union[Callable, Dict[str, Any]], name: Optional[str] = None):
        """Schedule *action* at simulation time *t*.

        *action* is a ``callable(world)`` or a declarative ``{'set': {'sim.var': value}}``
        (values are written with ``set_value`` as given - no unit conversion).
        A step boundary always lands exactly on *t* (the step is shortened);
        the action runs at the start of the step beginning at *t*, before that
        step's exchange.  A declarative ``set`` on a *connected* input is **held**
        for that step: the exchange skips that input for the step in which the
        event fires and resumes with the next step (a callable action that writes
        to an adapter directly is not held).
        Firing is recorded in ``Results.meta['events']``.
        """
        t = float(t)
        if not math.isfinite(t):
            raise ValueError(f"Event time must be finite, got {t!r}.")
        self._event_seq += 1
        seq = self._event_seq
        name = name or f"event_{seq}"
        ev = {'time': t, 'action': _check_action(action, f"Event '{name}'"), 'name': name, 'seq': seq}
        self._scheduled.append(ev)
        self._scheduled.sort(key=lambda e: (e['time'], e['seq']))
        return name

    def add_trigger(self, name: str, condition: Union[str, Callable[[Dict[str, Any]], bool]],
                    action: Union[Callable, Dict[str, Any]], direction: str = 'rising',
                    variables: Optional[List[str]] = None, bisect: bool = False):
        """Fire *action* when *condition* changes at a step boundary.

        *condition* is a safe expression string over ``'sim.var'`` names
        (``"grid.Bus1.vm_pu < 0.95"``, quote names that are not attribute-like)
        or a callable on ``{'sim.var': value}`` (pass ``variables=[...]``).
        ``direction``: ``'rising'`` (False -> True, default), ``'falling'``
        (True -> False) or ``'both'``.  The action (callable(world) or
        ``{'set': {...}}``) runs at the start of the step following the
        crossing; the event is recorded with the detection time and a note
        that localisation is limited to the step.  ``bisect=True`` (locating
        the crossing inside the step) needs rollback and is not supported.
        """
        if bisect:
            raise NotImplementedError(
                "bisect=True would need rollback of every simulator, which energysim does not support; "
                "reduce t_macro / t_macro_min or provide event hints (next_event_time) instead.")
        if direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {_DIRECTIONS}, got {direction!r}.")
        if not name or any(tr['name'] == name for tr in self._triggers):
            raise ValueError(f"Trigger name {name!r} must be non-empty and unique.")
        if isinstance(condition, str):
            cond = compile_condition(condition)
            vars_ = list(variables or cond.variables)
        elif callable(condition):
            cond = condition
            if not variables:
                raise ValueError(f"Trigger '{name}': a callable condition needs variables=['sim.var', ...].")
            vars_ = list(variables)
        else:
            raise TypeError("condition must be an expression string or a callable.")
        for v in vars_:
            if not isinstance(v, str) or '.' not in v:
                raise ValueError(f"Trigger '{name}': variable {v!r} must be 'sim.var'.")
        self._triggers.append({'name': name, 'condition': cond, 'source': condition if isinstance(condition, str)
                               else None, 'action': _check_action(action, f"Trigger '{name}'"),
                               'direction': direction, 'variables': vars_})
        return name

    @property
    def scheduled_events(self) -> List[Dict[str, Any]]:
        """``[{'time', 'name', 'action'}]`` of the scheduled events (sorted by time)."""
        return [{'time': e['time'], 'name': e['name'], 'action': e['action']} for e in self._scheduled]

    @property
    def triggers(self) -> List[Dict[str, Any]]:
        return [dict(tr) for tr in self._triggers]

    def clear_events(self) -> None:
        """Remove all scheduled events and triggers."""
        self._scheduled = []
        self._triggers = []

    # ------------------------------------------------------------------
    # Step-size selection
    # ------------------------------------------------------------------

    @property
    def variable_step(self) -> bool:
        """True when ``t_macro_min < t_macro`` (event hints may shrink the macro step)."""
        macro = self.macro_tstep if self.macro_tstep is not None else self._macro_step()
        return self._t_macro_min_eff() < macro - 1e-12

    def _t_macro_min_eff(self) -> float:
        macro = self.macro_tstep if self.macro_tstep is not None else self._macro_step()
        if self.t_macro_min is None:
            return macro
        return min(float(self.t_macro_min), macro)

    def _next_step(self, t: float) -> float:
        """Hint-based macro step at *t*: min over adapter hints, clamped to ``[t_macro_min, t_macro]``.

        Does not consider scheduled events / stop_time (see :meth:`_plan_step`).
        """
        macro = self.macro_tstep
        t_min = self._t_macro_min_eff()
        if t_min >= macro - 1e-12:
            return macro
        h = math.inf
        for sim_name, entry in self.simulator_dict.items():
            a = entry.adapter
            try:
                ms = float(a.max_step(t)) if callable(getattr(a, 'max_step', None)) else math.inf
                ne = float(a.next_event_time(t)) if callable(getattr(a, 'next_event_time', None)) else math.inf
            except Exception as exc:
                logger.warning("Step hint of '%s' failed at t=%s: %s", sim_name, t, exc)
                continue
            if ne - t > 1e-12:
                h = min(h, ne - t)
            if ms > 0:
                h = min(h, ms)
        if not math.isfinite(h):
            return macro
        return max(t_min, min(macro, h))

    def _plan_step(self, t: float, k: int):
        """``(t_end, regular)`` of the step starting at *t*; *k* is the grid index of *t* (or None)."""
        macro = self.macro_tstep
        eps = 1e-9 * max(1.0, abs(self.stop_time), abs(macro))
        stop = self.stop_time
        nxt = self._next_scheduled_time(t, eps)
        if not self.variable_step:
            # fixed step: grid start + k*macro; events insert extra boundaries
            if k is not None:
                t_end, regular = t + macro, True        # bit-identical to the fixed-step loop
                g = t_end
            else:
                g_idx = math.floor((t - self.start_time) / macro + 1e-9) + 1
                g = self.start_time + g_idx * macro
                t_end, regular = g, False
            if nxt is not None and nxt < g - eps:
                t_end, regular = nxt, False
            if t_end > stop + eps:
                t_end, regular = stop, False
            return t_end, regular
        dt = self._next_step(t)
        t_end = t + dt
        if nxt is not None and nxt < t_end + eps:
            t_end = nxt
        if t_end > stop - eps:
            t_end = stop
        return t_end, False

    def _next_scheduled_time(self, t, eps):
        for ev in self._pending_events:
            if ev['time'] > t + eps:
                return ev['time']
        return None

    # ------------------------------------------------------------------
    # Firing
    # ------------------------------------------------------------------

    def _run_action(self, action):
        if callable(action):
            action(self)
            return
        for key, value in action['set'].items():
            sim, var = self._split_var(key)
            if sim not in self.simulator_dict:
                raise KeyError(f"unknown simulator '{sim}' in set target '{key}'")
            self.simulator_dict[sim].adapter.set_value([var], [value])
            # hold: the exchange of this step must not overwrite the value just set
            self._held_inputs.add((sim, var))

    def _fire(self, record, action):
        try:
            self._run_action(action)
            record['success'] = True
            record['error'] = None
        except Exception as exc:
            record['success'] = False
            record['error'] = f"{type(exc).__name__}: {exc}"
            logger.warning("Event '%s' at t=%s failed: %s", record['name'], record['time'], record['error'])
        self._event_log.append(record)
        logger.info("Event '%s' fired at t=%s (%s)", record['name'], record['time'],
                    'ok' if record['success'] else 'failed')

    def _fire_due_events(self, t: float) -> None:
        self._held_inputs = set()       # holds last for the one step in which the ``set`` fired
        eps = 1e-9 * max(1.0, abs(self.stop_time))
        while self._pending_events and self._pending_events[0]['time'] <= t + eps:
            ev = self._pending_events.pop(0)
            self._fire({'time': t, 'scheduled_time': ev['time'], 'name': ev['name'], 'kind': 'scheduled'},
                       ev['action'])
        for tr, detected, t_prev in self._pending_triggers:
            self._fire({'time': t, 'name': tr['name'], 'kind': 'trigger', 'detected_time': detected,
                        'interval': [t_prev, detected], 'direction': tr['direction'], 'note': TRIGGER_NOTE},
                       tr['action'])
        self._pending_triggers = []

    def _trigger_values(self, tr, t):
        values = {}
        for key in tr['variables']:
            sim, var = self._split_var(key)
            values[key] = self.simulator_dict[sim].adapter.get_value([var], t)[0]
        return values

    def _evaluate_triggers(self, t: float, initial: bool = False) -> None:
        """Evaluate all trigger conditions at boundary *t*; queue actions of crossings."""
        for tr in self._triggers:
            try:
                values = self._trigger_values(tr, t)
                cond = tr['condition']
                state = bool(cond(values, t) if tr['source'] is not None else cond(values))
            except Exception as exc:
                logger.warning("Trigger '%s' could not be evaluated at t=%s: %s", tr['name'], t, exc)
                continue
            prev = self._trigger_state.get(tr['name'])
            self._trigger_state[tr['name']] = state
            prev_t = self._trigger_time.get(tr['name'], t)
            self._trigger_time[tr['name']] = t
            if initial or prev is None or prev == state:
                continue
            rising = state and not prev
            if tr['direction'] == 'both' or (tr['direction'] == 'rising') == rising:
                self._pending_triggers.append((tr, t, prev_t))

    def _reset_events(self) -> None:
        self._pending_events = list(self._scheduled)
        self._pending_triggers = []
        self._trigger_state = {}
        self._trigger_time = {}
        self._event_log = []

    def _finish_events(self) -> None:
        """Record crossings detected at stop_time and events that never fired."""
        for tr, detected, t_prev in self._pending_triggers:
            self._event_log.append({'time': None, 'name': tr['name'], 'kind': 'trigger', 'success': None,
                                    'error': None, 'detected_time': detected, 'interval': [t_prev, detected],
                                    'direction': tr['direction'],
                                    'note': 'detected at stop_time; no step left to run the action'})
        self._pending_triggers = []
        for ev in self._pending_events:
            self._event_log.append({'time': None, 'scheduled_time': ev['time'], 'name': ev['name'],
                                    'kind': 'scheduled', 'success': None, 'error': None,
                                    'note': 'not fired: at or after stop_time'})
        self._pending_events = []

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _events_to_list(self):
        out = []
        for ev in self._scheduled:
            if callable(ev['action']):
                logger.warning("Event '%s' has a Python callable action and cannot be serialised.", ev['name'])
                continue
            out.append({'time': ev['time'], 'name': ev['name'], 'set': dict(ev['action']['set'])})
        return out

    def _triggers_to_list(self):
        out = []
        for tr in self._triggers:
            if tr['source'] is None or callable(tr['action']):
                logger.warning("Trigger '%s' uses Python callables and cannot be serialised.", tr['name'])
                continue
            out.append({'name': tr['name'], 'condition': tr['source'], 'direction': tr['direction'],
                        'set': dict(tr['action']['set'])})
        return out

    def _load_events(self, events, triggers):
        for i, ev in enumerate(events or []):
            if not isinstance(ev, dict):
                raise ValueError(f"events[{i}] must be a mapping {{time: ..., set: {{...}}}}.")
            unknown = sorted(set(ev) - _EVENT_KEYS)
            if unknown:
                raise ValueError(f"events[{i}]: unknown key(s) {unknown}; only 'time', 'name' and the "
                                 f"declarative 'set' action are allowed in scenario files.")
            if 'time' not in ev or 'set' not in ev:
                raise ValueError(f"events[{i}] needs 'time' and 'set'.")
            self.schedule(ev['time'], {'set': ev['set']}, name=ev.get('name'))
        for i, tr in enumerate(triggers or []):
            if not isinstance(tr, dict):
                raise ValueError(f"triggers[{i}] must be a mapping.")
            unknown = sorted(set(tr) - _TRIGGER_KEYS)
            if unknown:
                raise ValueError(f"triggers[{i}]: unknown key(s) {unknown}; allowed: {sorted(_TRIGGER_KEYS)}.")
            if not isinstance(tr.get('condition'), str):
                raise ValueError(f"triggers[{i}] needs a 'condition' expression string.")
            self.add_trigger(tr.get('name') or f"trigger_{i}", tr['condition'], {'set': tr.get('set')},
                             direction=tr.get('direction', 'rising'), variables=tr.get('variables'))


def _init_event_state(w) -> None:
    """Initialise the event attributes on a new world."""
    w._scheduled = []
    w._triggers = []
    w._event_seq = 0
    w._held_inputs = set()
    w._reset_events()
