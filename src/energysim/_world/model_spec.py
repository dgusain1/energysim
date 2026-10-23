# -*- coding: utf-8 -*-
"""Scenario-file (YAML) support for ``type: model`` simulators.

A ``type: model`` entry names a component instead of passing a Python object::

    - name: ely
      type: model
      model: PEMElectrolyzer          # a class exported by energysim.models / energysim.networks
      parameters: {p_nominal_mw: 2.0}   # constructor parameters (alias: options)
    - name: bess
      type: model
      model: BESS_4h_2MW              # a catalog archetype
    - name: mine
      type: model
      model: my_models.py:Plant       # a class in a file (NOT allowed in safe mode)

:func:`build_model` creates the instance; :func:`model_to_spec` is the inverse used by
``world.to_dict``: it succeeds only when the model is an instance of a known exported class and
its constructor parameters can be recovered and verified (see its docstring).
"""

from __future__ import annotations

import difflib
import inspect
import os
from typing import Any, Dict, Optional, Tuple

from ..utils import looks_like_dotted_import
from ._common import _filter_kwargs

#: ``EnergyNetwork`` topology keys carried in a scenario file under ``topology:``.
_SIM_LEVEL = ('step_size', 'inputs', 'outputs')
_TOPOLOGY_KEYS = ('producers', 'consumers', 'storages', 'junctions', 'pipes')


def model_registry() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(classes, archetypes)``: exported component classes and catalog archetype factories."""
    from .. import models as M
    from .. import networks as N
    from ..models.common import ModelComponent
    classes: Dict[str, Any] = {}
    for mod in (M, N):
        for name in getattr(mod, '__all__', ()):
            obj = getattr(mod, name, None)
            if inspect.isclass(obj) and issubclass(obj, ModelComponent) and obj is not ModelComponent:
                classes[name] = obj
    archetypes = {n: getattr(M.catalog, n) for n in M.catalog.__all__ if n != 'PROVENANCE'}
    return classes, archetypes


def _is_file_ref(ref: str) -> bool:
    path_part, _, attr = ref.rpartition(':')
    return bool(path_part) and path_part.lower().endswith('.py') and attr.isidentifier()


def is_file_model_ref(ref: Any) -> bool:
    """True for ``'path/to/file.py:ClassName'`` references."""
    return isinstance(ref, str) and _is_file_ref(ref)


def _unknown(sim_name: str, ref: str, classes, archetypes) -> ValueError:
    pool = sorted(classes) + sorted(archetypes)
    close = difflib.get_close_matches(ref, pool, n=4, cutoff=0.5)
    hint = f" Did you mean: {', '.join(close)}?" if close else ''
    return ValueError(
        f"Simulator '{sim_name}': unknown model {ref!r}.{hint} Use a class exported by energysim.models "
        f"or energysim.networks (e.g. 'PEMElectrolyzer'), a catalog archetype (e.g. 'BESS_4h_2MW'), or "
        f"'path/to/file.py:ClassName'.")


def build_model(sim_name: str, ref: Any, params: Optional[Dict[str, Any]] = None, *, load_module=None,
                base_dir: Optional[str] = None, safe: bool = False, allow_import: bool = False):
    """Instantiate the component named by *ref* with constructor *params*.

    ``topology`` in *params* (network classes only) is a network spec dict
    (``producers``, ``consumers``, ``storages``, ``junctions``, ``pipes``).
    Unknown constructor parameters raise ``ValueError`` (the model base class would
    otherwise ignore typos silently).
    """
    from ..networks import EnergyNetwork
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError(f"Simulator '{sim_name}': type 'model' needs a model name, e.g. "
                         f"model: PEMElectrolyzer.")
    ref = ref.strip()
    params = dict(params or {})
    classes, archetypes = model_registry()

    if _is_file_ref(ref):
        if safe:
            raise ValueError(f"Simulator '{sim_name}': model file reference {ref!r} runs arbitrary code "
                             f"and is not allowed in safe mode.")
        path_part, _, attr = ref.rpartition(':')
        if base_dir and not os.path.isabs(path_part):
            path_part = os.path.normpath(os.path.join(base_dir, path_part))
        if not os.path.isfile(path_part):
            raise FileNotFoundError(f"Simulator '{sim_name}': model file not found: {path_part!r}")
        module = load_module(path_part)
        try:
            factory = getattr(module, attr)
        except AttributeError:
            raise AttributeError(f"Simulator '{sim_name}': '{attr}' not found in {path_part!r}.") from None
    elif ref in classes:
        factory = classes[ref]
    elif ref in archetypes:
        factory = archetypes[ref]
    elif looks_like_dotted_import(ref):
        if safe or not allow_import:
            raise ValueError(f"Simulator '{sim_name}': dotted import reference {ref!r} requires "
                             f"allow_import=True (and is never allowed in safe mode).")
        import importlib
        mod_name, _, obj_name = ref.rpartition('.')
        try:
            factory = getattr(importlib.import_module(mod_name), obj_name)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"Simulator '{sim_name}': could not import {ref!r}: {exc}") from exc
    else:
        raise _unknown(sim_name, ref, classes, archetypes)

    topology = params.pop('topology', None)
    if topology is not None:
        if not (inspect.isclass(factory) and issubclass(factory, EnergyNetwork)):
            raise ValueError(f"Simulator '{sim_name}': 'topology' is only valid for network models "
                             f"(HeatNetwork, GasNetwork, EnergyNetwork).")
        if not isinstance(topology, dict):
            raise ValueError(f"Simulator '{sim_name}': 'topology' must be a mapping.")
        extra = sorted(set(topology) - set(_TOPOLOGY_KEYS))
        if extra:
            raise ValueError(f"Simulator '{sim_name}': unknown topology key(s) {extra}; "
                             f"valid: {list(_TOPOLOGY_KEYS)}.")
        spec = dict(topology)
        spec['name'] = params.pop('name', sim_name)
        spec['parameters'] = params
        instance = factory.from_dict(spec)
        return instance

    call = dict(_filter_kwargs(factory, {'name': sim_name}))
    call.update(params)
    try:
        instance = factory(**call)
    except TypeError as exc:
        raise ValueError(f"Simulator '{sim_name}': cannot construct {ref!r} with parameters "
                         f"{sorted(params)}: {exc}") from None
    _check_known_parameters(sim_name, ref, instance, params)
    return instance


def _check_known_parameters(sim_name, ref, instance, params) -> None:
    try:
        accepted = set(inspect.signature(type(instance).__init__).parameters)
    except (TypeError, ValueError):
        return
    unknown = sorted(set(params) - accepted - {'name'})
    if unknown:
        valid = sorted(a for a in accepted if a not in ('self', 'kwargs', 'args'))
        raise ValueError(f"Simulator '{sim_name}': {ref!r} has no parameter(s) {unknown}. "
                         f"Valid parameters: {valid}.")


# ---------------------------------------------------------------------------
# inverse: instance -> (model name, parameters)
# ---------------------------------------------------------------------------

def _plain(v: Any) -> Tuple[bool, Any]:
    """``(ok, converted)``: *v* as YAML-safe plain data."""
    if hasattr(v, 'item') and getattr(v, 'ndim', 1) == 0:
        v = v.item()
    if v is None or isinstance(v, (bool, int, float, str)):
        return True, v
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            ok, c = _plain(x)
            if not ok:
                return False, None
            out.append(c)
        return True, out
    if isinstance(v, dict) and all(isinstance(k, str) for k in v):
        out = {}
        for k, x in v.items():
            ok, c = _plain(x)
            if not ok:
                return False, None
            out[k] = c
        return True, out
    return False, None


def _same(a: Any, b: Any) -> bool:
    try:
        r = a == b
        return bool(r.all()) if hasattr(r, 'all') else bool(r)
    except Exception:
        return False


def model_to_spec(inst: Any) -> Tuple[Optional[str], Dict[str, Any], str]:
    """``(model name, parameters, reason)`` for an in-memory component, or ``(None, {}, reason)``.

    Serialisable means: ``type(inst)`` is a class exported by ``energysim.models`` /
    ``energysim.networks`` and its constructor parameters (``inst.describe()['parameters']``,
    the constructor defaults) can be read back from same-named attributes as plain
    values.  The instance is then rebuilt from the recovered parameters and every described
    parameter must compare equal on the copy; otherwise the model is *not* serialised.
    Network topologies (``EnergyNetwork.to_dict``) are carried under ``topology``.
    """
    from ..networks import EnergyNetwork
    classes, _ = model_registry()
    cls = type(inst)
    name = next((n for n, c in classes.items() if c is cls), None)
    if name is None:
        return None, {}, f"{cls.__name__} is not a class exported by energysim.models or energysim.networks"

    if isinstance(inst, EnergyNetwork):
        try:
            d = inst.to_dict()
            params = dict(d.get('parameters') or {})
            topo = {k: d[k] for k in _TOPOLOGY_KEYS if d.get(k)}
            probe = build_model(inst.name, name, {**params, **({'topology': topo} if topo else {})})
            if probe.to_dict() != d:
                return None, {}, "network could not be rebuilt identically from its topology"
        except Exception as exc:
            return None, {}, f"network is not serialisable: {exc}"
        if topo:
            params['topology'] = topo
        return name, params, ''

    try:
        defaults = {k: v for k, v in inst.describe().get('parameters', {}).items()
                    if k not in _SIM_LEVEL}   # step_size / inputs / outputs belong to the simulator entry
    except Exception as exc:
        return None, {}, f"describe() failed: {exc}"
    params: Dict[str, Any] = {}
    for key, default in defaults.items():
        if not hasattr(inst, key):
            continue
        value = getattr(inst, key)
        if _same(value, default):
            continue
        ok, plain = _plain(value)
        if not ok:
            return None, {}, f"parameter '{key}' ({type(value).__name__}) is not plain data"
        params[key] = plain
    try:
        probe = cls(name=inst.name, **params)
    except Exception as exc:
        return None, {}, f"rebuilding with {sorted(params)} failed: {exc}"
    for key in defaults:
        if hasattr(inst, key) and hasattr(probe, key) and not _same(getattr(inst, key), getattr(probe, key)):
            return None, {}, f"parameter '{key}' does not survive a rebuild"
    return name, params, ''
