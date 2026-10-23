# -*- coding: utf-8 -*-
"""Scenario serialisation (to_dict / from_dict / YAML) and safe mode."""

from __future__ import annotations

import os
import yaml

from ..utils import (
    looks_like_dotted_import, PICKLE_EXTS, MODEL_FILE_EXTS,
)
from .unit_resolution import unit_system_to_dict
from ._common import (logger, ITERATIVE_REMOVED_MSG, UNSAFE_SIM_TYPES,
                      _MODEL_ALIAS_KEYS, _WORLD_KEYS,
                      RES_FILENAME_REMOVED_MSG, _REMOVED_WORLD_KEYS)


class ScenarioMixin:
    """Scenario serialisation (to_dict / from_dict / YAML) and safe mode."""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self):
        """Serialize the world configuration, simulators, and connections to a dictionary."""
        sims_list = []
        for name, entry in self.simulator_dict.items():
            sim_info = {
                'name': name,
                'type': entry.sim_type,
                'step_size': entry.step_size,
                'inputs': list(entry.inputs),
                'outputs': list(entry.outputs),
                'power_unit': entry.power_unit,
            }
            if entry.units:
                sim_info['units'] = dict(entry.units)
            if entry.sim_type == 'signal':
                spec = entry.model
                if isinstance(spec, (str, dict, int, float)) and not isinstance(spec, bool):
                    sim_info['signal'] = spec
                else:
                    logger.warning(
                        "Signal '%s' is a Python callable and cannot be serialised; use an expression "
                        "string or a structured spec to make it round-trip.", name)
            elif entry.model is not None and isinstance(entry.model, str):
                sim_info['model'] = entry.model
                sim_info['location'] = entry.model
            else:
                for attr in ('sim_loc', 'fmu_loc', 'net_loc', 'csv_location', 'dss_location', 'script_loc'):
                    if hasattr(entry.adapter, attr):
                        loc_val = getattr(entry.adapter, attr)
                        if isinstance(loc_val, str):
                            sim_info['model'] = loc_val
                            sim_info['location'] = loc_val
                            break

            if entry.mode:
                sim_info['mode'] = entry.mode
            opts = {}
            if getattr(entry.adapter, 'user_breakpoints', None):
                opts['breakpoints'] = list(entry.adapter.user_breakpoints)
            if getattr(entry.adapter, 'align_to_rows', False):
                opts['align_to_rows'] = True
            if opts:
                sim_info['options'] = opts
            sims_list.append(sim_info)

        conns_list = []
        for conn in self._parsed_connections:
            output_, input_ = conn[0], conn[1]
            scale = getattr(conn, 'user_scale', getattr(conn, 'scale', 1.0))
            offset = getattr(conn, 'user_offset', getattr(conn, 'offset', 0.0))
            from_unit = getattr(conn, 'from_unit', None)
            to_unit = getattr(conn, 'to_unit', None)

            if isinstance(output_, tuple) and isinstance(output_[0], tuple):
                from_val = [item[2] for item in output_]
            else:
                from_val = output_[2]

            if isinstance(input_, list):
                to_val = [f"{item[0]}.{item[1]}" for item in input_]
            else:
                to_val = f"{input_[0]}.{input_[1]}"

            conn_dict = {'from': from_val, 'to': to_val}
            if scale != 1.0:
                conn_dict['scale'] = scale
            if offset != 0.0:
                conn_dict['offset'] = offset
            if from_unit and to_unit:
                conn_dict['from_unit'] = from_unit
                conn_dict['to_unit'] = to_unit
            conns_list.append(conn_dict)

        out = {
            'world': {
                'start_time': self.start_time,
                'stop_time': self.stop_time,
                't_macro': self.exchange,
                **({'t_macro_min': self.t_macro_min} if self.t_macro_min is not None else {}),
                'coupling': self.coupling,
                'extrapolation': self.extrapolation,
                **({'power_unit': self.power_unit} if self.power_unit else {}),
                'logging': self.logging,
                'unit_system': unit_system_to_dict(self.unit_system),
                **({'unit_policy': self.unit_policy} if self.unit_policy != 'lenient' else {}),
                **({'start_datetime': str(self.start_datetime)} if self.start_datetime is not None else {}),
                **({'seed': self.seed} if self.seed is not None else {}),
                **({'out_dir': str(self.out_dir)} if self.out_dir is not None else {}),
            },
            'simulators': sims_list,
            'connections': conns_list,
        }
        events = self._events_to_list()
        if events:
            out['events'] = events
        triggers = self._triggers_to_list()
        if triggers:
            out['triggers'] = triggers
        return out

    def to_yaml(self, filepath=None):
        """Export world scenario configuration to YAML string or file."""
        data = self.to_dict()
        yaml_str = yaml.dump(data, sort_keys=False, default_flow_style=False)
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(yaml_str)
        return yaml_str

    @staticmethod
    def _check_safe_simulator(name, sim_type, model, options):
        """Reject code-executing simulator definitions in safe mode."""
        st = (sim_type or '').lower()
        if st in UNSAFE_SIM_TYPES:
            raise ValueError(
                f"Simulator '{name}': type '{sim_type}' runs arbitrary code and is not allowed in "
                f"safe mode."
            )
        refs = [('model', model)] + [(k, v) for k, v in (options or {}).items()]
        for key, val in refs:
            if not isinstance(val, (str, os.PathLike)):
                continue
            s = os.fspath(val)
            if s.lower().endswith(PICKLE_EXTS):
                raise ValueError(
                    f"Simulator '{name}': pickle file {s!r} ({key}) can execute arbitrary code when "
                    f"loaded and is not allowed in safe mode."
                )
            if key in ('model', 'location') + _MODEL_ALIAS_KEYS and looks_like_dotted_import(s):
                raise ValueError(
                    f"Simulator '{name}': dotted import reference {s!r} is not allowed in safe mode."
                )

    @classmethod
    def from_dict(cls, data, base_dir=None, safe=False, allow_import=False):
        """Construct a world instance from a dictionary configuration.

        Parameters
        ----------
        safe : bool
            Refuse simulator types that execute arbitrary code (external, script,
            matlab, powerfactory), dotted-import model references and pickle
            (``.p`` / ``.pkl``) model files.  Use for untrusted scenario files.
        allow_import : bool
            Allow dotted-import model references (``'pkg.module.factory'``) to
            be imported and called.  Ignored (False) in safe mode.
        """
        if not isinstance(data, dict):
            raise ValueError("Scenario must be a mapping with 'world', 'simulators' and 'connections'.")
        w_cfg = data.get('world', {}) or {}
        removed = sorted(set(w_cfg) & _REMOVED_WORLD_KEYS)
        if removed:
            raise ValueError(f"World keys {removed} are no longer supported: {ITERATIVE_REMOVED_MSG}.")
        if 'res_filename' in w_cfg:
            raise ValueError(RES_FILENAME_REMOVED_MSG)
        unknown = sorted(set(w_cfg) - _WORLD_KEYS)
        if unknown:
            raise ValueError(f"Unknown world key(s) {unknown}. Valid keys: {sorted(_WORLD_KEYS)}.")
        if safe:
            allow_import = False
        w = cls(
            start_time=w_cfg.get('start_time', 0),
            stop_time=w_cfg.get('stop_time', 1000),
            t_macro=w_cfg.get('t_macro', 60),
            t_macro_min=w_cfg.get('t_macro_min'),
            coupling=w_cfg.get('coupling', 'jacobi'),
            extrapolation=w_cfg.get('extrapolation', 'zero-order'),
            start_datetime=w_cfg.get('start_datetime'),
            seed=w_cfg.get('seed'),
            out_dir=w_cfg.get('out_dir'),
            power_unit=w_cfg.get('power_unit'),
            logging=w_cfg.get('logging', False),
            unit_system=w_cfg.get('unit_system'),
            unit_policy=w_cfg.get('unit_policy', 'lenient'),
        )

        for sim_cfg in data.get('simulators', []) or []:
            if not isinstance(sim_cfg, dict) or 'name' not in sim_cfg or 'type' not in sim_cfg:
                raise ValueError(f"Each simulator needs at least 'name' and 'type': {sim_cfg!r}")
            s_name = sim_cfg['name']
            s_type = sim_cfg['type']
            st = str(s_type).lower()
            s_model = sim_cfg.get('model') if sim_cfg.get('model') is not None else sim_cfg.get('location')
            kwargs = dict(sim_cfg.get('options', {}) or {})
            if 'signal' in sim_cfg:
                kwargs['signal'] = sim_cfg['signal']

            if safe:
                cls._check_safe_simulator(s_name, s_type, s_model, kwargs)

            if (st != 'signal' and base_dir and isinstance(s_model, str) and s_model
                    and not os.path.isabs(s_model)):
                candidate_path = os.path.normpath(os.path.join(base_dir, s_model))
                if os.path.exists(candidate_path) or s_model.lower().endswith(MODEL_FILE_EXTS):
                    s_model = candidate_path

            w.add_simulator(
                sim_type=s_type,
                sim_name=s_name,
                model=s_model,
                inputs=sim_cfg.get('inputs', []),
                outputs=sim_cfg.get('outputs', []),
                step_size=sim_cfg.get('step_size', 1.0),
                mode=sim_cfg.get('mode') or sim_cfg.get('pf_mode') or 'ac',
                power_unit=sim_cfg.get('power_unit'),
                units=sim_cfg.get('units'),
                allow_import=allow_import,
                **kwargs
            )

        conns = data.get('connections', [])
        if conns:
            w.add_connections(conns)

        # events / triggers: only the declarative 'set' action exists in scenario
        # files, so they are safe-mode compatible by construction
        w._load_events(data.get('events'), data.get('triggers'))
        return w

    @classmethod
    def from_yaml(cls, path_or_str, safe=False, allow_import=False):
        """Load a world from a YAML file path or YAML string (see :meth:`from_dict` for ``safe``)."""
        if os.path.isfile(path_or_str):
            base_dir = os.path.dirname(os.path.abspath(path_or_str))
            with open(path_or_str, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        else:
            base_dir = None
            data = yaml.safe_load(path_or_str)

        return cls.from_dict(data, base_dir=base_dir, safe=safe, allow_import=allow_import)

