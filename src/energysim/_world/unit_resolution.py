# -*- coding: utf-8 -*-
"""Unit resolution, connection unit conversion and unit visibility of the world.

Resolution order for every connected (and recorded) variable:

1. ``declared`` - the simulator-level ``units={var: unit}`` mapping, or the
   simulator's ``power_unit`` for power-like names without a unit suffix
   (``power_unit`` is shorthand for the units of ``p``, ``P_cmd``, ``power``...);
2. ``component`` - the adapter's own ``units`` dict (models, fleets, networks);
3. ``connection`` - ``from_unit`` / ``to_unit`` / ``unit`` given on the connection;
4. ``suffix`` - a conventional name suffix (``p_mw``, ``t_in_degc``, ``vm_pu``);
5. ``default-by-quantity`` - the name denotes a quantity (``P_cmd``, ``T_amb``,
   ``soc``); the world's :class:`~energysim.units.UnitSystem` default is used;
6. ``unspecified`` - no unit; the value is never scaled.  This is not an error
   unless the world was created with ``unit_policy='strict'``.

Values are always converted into the **receiver's** resolved unit.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..units import (
    UnitSystem, get_unit_conversion, infer_unit_from_name, is_known_unit,
    resolve_unit as _resolve_by_name,
)
from ..utils import looks_like_power
from ._common import logger

#: unit sources that are explicit enough to raise on incompatible dimensions
EXPLICIT_SOURCES = frozenset({'declared', 'component', 'connection', 'suffix'})
UNIT_POLICIES = ('lenient', 'strict')
_PRESETS = {'default': UnitSystem.default, 'energysim': UnitSystem.default,
            'energysim-default': UnitSystem.default, 'si': UnitSystem.si}


def coerce_unit_system(spec: Any) -> UnitSystem:
    """``None`` / preset name / overrides dict / ``{'name':..., 'overrides':...}`` / UnitSystem."""
    if isinstance(spec, dict) and ('name' in spec or 'overrides' in spec or 'defaults' in spec):
        name = spec.get('name') or 'custom'
        base = _PRESETS.get(str(name).lower(), UnitSystem.default)()
        overrides = dict(spec.get('overrides') or {})
        overrides.update(spec.get('defaults') or {})
        return UnitSystem.from_overrides(overrides, base=base, name=name)
    return UnitSystem.coerce(spec)


def unit_system_to_dict(system: UnitSystem) -> Dict[str, Any]:
    """``{'name': ..., 'overrides': {...}}`` relative to the preset named by ``system.name``."""
    base = _PRESETS.get(str(system.name).lower(), UnitSystem.default)()
    overrides = {q: u for q, u in system.defaults.items() if base.defaults.get(q) != u}
    out: Dict[str, Any] = {'name': system.name}
    if overrides:
        out['overrides'] = overrides
    return out


def _fmt_num(x: float) -> str:
    return f"{x:.10g}"


def format_conversion(src: str, from_unit: Optional[str], dst: str, to_unit: Optional[str],
                      scale: float, offset: float, source: str) -> str:
    """``A.y [kW] -> B.u [MW]  x0.001  (source: declared -> suffix)``."""
    conv = f"x{_fmt_num(scale)}"
    if offset:
        conv += f" {'+' if offset > 0 else '-'}{_fmt_num(abs(offset))}"
    return (f"{src} [{from_unit or 'unspecified'}] -> {dst} [{to_unit or 'unspecified'}]  "
            f"{conv}  (source: {source})")


class UnitsMixin:
    """Unit resolution and visibility (``unit_table``, ``connection_units``, logs)."""

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve_variable_unit(self, sim_name: str, var: str,
                              connection_unit: Optional[str] = None) -> Tuple[Optional[str], str]:
        """Resolve the unit of ``sim_name.var`` -> ``(unit, source)`` (see module docstring)."""
        entry = self.simulator_dict.get(sim_name)
        if entry is not None:
            if entry.units and entry.units.get(var):
                return entry.units[var], 'declared'
            if entry.power_unit and looks_like_power(var) and infer_unit_from_name(var) is None:
                return entry.power_unit, 'declared'
            comp = getattr(entry.adapter, 'units', None)
            if isinstance(comp, dict) and comp.get(var):
                return comp[var], 'component'
        if connection_unit:
            return connection_unit, 'connection'
        return _resolve_by_name(var, None, self.unit_system)

    def _conversion_for(self, src: Tuple[str, str], dst: Tuple[str, str],
                        from_conn: Optional[str], to_conn: Optional[str]) -> Dict[str, Any]:
        """Resolve both ends and compute ``(scale, offset)`` into the receiver's unit."""
        fu, fs = self.resolve_variable_unit(src[0], src[1], from_conn)
        tu, ts = self.resolve_variable_unit(dst[0], dst[1], to_conn)
        for (sim, var), unit, source, conn_unit in ((src, fu, fs, from_conn), (dst, tu, ts, to_conn)):
            if conn_unit and source in ('declared', 'component') and unit != conn_unit:
                logger.warning("Connection unit '%s' for %s.%s ignored: the simulator declares '%s' (%s).",
                               conn_unit, sim, var, unit, source)
        note = None
        scale, offset = 1.0, 0.0
        if fu is None or tu is None:
            note = 'unspecified'
        elif fu == tu:
            pass
        elif not (is_known_unit(fu) and is_known_unit(tu)):
            note = 'unknown'
        else:
            try:
                scale, offset = get_unit_conversion(fu, tu)
            except ValueError:
                if fs in EXPLICIT_SOURCES and ts in EXPLICIT_SOURCES:
                    raise
                note = 'incompatible-guess'
        return {'from_unit': fu, 'from_source': fs, 'to_unit': tu, 'to_source': ts,
                'scale': scale, 'offset': offset, 'note': note}

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def _unit_system_label(self) -> str:
        return f"{self.unit_system.name} [{self.unit_system.hash}]"

    def _log_unit_system(self) -> None:
        logger.info("Default unit system: %s", self.unit_system.describe())

    def unit_table(self, show: bool = False) -> str:
        """The world's unit system as a text table (``UnitSystem.format_table()``); printed if *show*."""
        table = self.unit_system.format_table()
        if show:
            print(table)
        return table

    def _connection_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for conn in self._parsed_connections:
            output_, input_ = conn[0], conn[1]
            if isinstance(output_, tuple) and isinstance(output_[0], tuple):
                src = ' + '.join(k for _, _, k in output_)
            else:
                src = output_[2]
            targets = input_ if isinstance(input_, list) else [input_]
            convs = getattr(conn, 'target_conversions', None)
            infos = getattr(conn, '_unit_info', None) or []
            for i, (in_sim, in_var) in enumerate(targets):
                info = infos[i] if i < len(infos) else {}
                sc, off = convs[i] if convs else (conn.scale, conn.offset)
                rows.append({
                    'from': src, 'to': f"{in_sim}.{in_var}",
                    'from_unit': info.get('from_unit', conn.from_unit),
                    'to_unit': info.get('to_unit', conn.to_unit),
                    'scale': sc, 'offset': off,
                    'unit_scale': info.get('scale', conn.unit_scale),
                    'unit_offset': info.get('offset', conn.unit_offset),
                    'user_scale': conn.user_scale, 'user_offset': conn.user_offset,
                    'from_source': info.get('from_source', conn.from_source),
                    'to_source': info.get('to_source', conn.to_source),
                    'note': info.get('note'),
                })
        return rows

    def connection_units(self):
        """DataFrame with one row per (connection, receiver): units, sources and effective scale/offset."""
        import pandas as pd
        cols = ['from', 'to', 'from_unit', 'to_unit', 'scale', 'offset', 'unit_scale', 'unit_offset',
                'user_scale', 'user_offset', 'from_source', 'to_source', 'note']
        return pd.DataFrame(self._connection_rows(), columns=cols)

    def conversion_lines(self) -> List[str]:
        """One human-readable line per (connection, receiver), see :func:`format_conversion`."""
        lines = []
        for r in self._connection_rows():
            if r['from_source'] == 'manual':
                source = 'manual scale'
            else:
                fs, ts = r['from_source'] or 'unspecified', r['to_source'] or 'unspecified'
                source = fs if fs == ts else f"{fs} -> {ts}"
            lines.append(format_conversion(r['from'], r['from_unit'], r['to'], r['to_unit'],
                                           r['scale'], r['offset'], source))
        return lines

    def _unit_warnings(self) -> List[str]:
        """Messages for connections with an unspecified / unknown / mismatching unit."""
        out = []
        for r in self._connection_rows():
            note = r['note']
            if not note or r['from_source'] == 'manual':
                continue
            if note == 'unspecified':
                which = [f"{r['from']}" if r['from_unit'] is None else None,
                         f"{r['to']}" if r['to_unit'] is None else None]
                which = ' and '.join(w for w in which if w)
                out.append(f"Connection {r['from']} -> {r['to']}: unit of {which} is unspecified; "
                           f"the value is passed through unscaled (declare units={{...}} or "
                           f"from_unit/to_unit).")
            elif note == 'unknown':
                out.append(f"Connection {r['from']} -> {r['to']}: unit '{r['from_unit']}' or "
                           f"'{r['to_unit']}' is not a known unit; the value is passed through unscaled.")
            else:
                out.append(f"Connection {r['from']} -> {r['to']}: inferred units '{r['from_unit']}' "
                           f"({r['from_source']}) and '{r['to_unit']}' ({r['to_source']}) are incompatible; "
                           f"the value is passed through unscaled (declare the units explicitly).")
        return out

    def _log_conversions(self, level: str = 'info') -> None:
        log = getattr(logger, level)
        for line in self.conversion_lines():
            log("Unit conversion: %s", line)
