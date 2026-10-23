# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:07:17 2019

@author: Digvijay

Python script adapter for energysim.

Usage
-----
``script_name`` is ``module_name.function_name`` (or just
``function_name`` when ``script_loc`` points at a ``.py`` file).
``script_loc`` is either the ``.py`` file itself or the directory that
contains ``module_name.py``.

The module is loaded from its file with
:func:`importlib.util.spec_from_file_location` under a unique,
per-instance module name, so two scripts with the same module name in
different directories never collide and ``sys.path`` is not permanently
modified (the script's directory is on ``sys.path`` only while the module
body executes, so sibling imports keep working).

The function is called each step with a dict of current input values
and must return a list of output values.
"""
from __future__ import annotations


import importlib
import importlib.util
import itertools
import os
import sys

from .base import SimulatorAdapter, SimulatorVariableError

_instance_counter = itertools.count()


def _load_module_from_file(path, module_name):
    """Execute *path* as a fresh module registered under *module_name*."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python script {path!r}.")
    module = importlib.util.module_from_spec(spec)
    script_dir = os.path.dirname(os.path.abspath(path))
    sys.modules[module_name] = module
    inserted = script_dir not in sys.path
    if inserted:
        sys.path.insert(0, script_dir)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        if inserted:
            try:
                sys.path.remove(script_dir)
            except ValueError:
                pass
    return module


class py_adapter(SimulatorAdapter):

    def __init__(self, script_name, script_loc, inputs=None, outputs=None):
        self.script_name = script_name
        self.script_loc = script_loc
        self.inputs = inputs or []
        self.outputs = outputs or []

        if '.' in script_name:
            self._module_name, self._func_name = script_name.rsplit('.', 1)
        else:
            self._module_name, self._func_name = None, script_name

        self._module = None
        self._func = None
        self._load()

        # Internal state
        self._input_dict = {}
        self._output_values = [0.0] * len(self.outputs)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _resolve_file(self):
        loc = os.fspath(self.script_loc) if self.script_loc is not None else ''
        if loc and os.path.isfile(loc):
            return loc
        if self._module_name is None:
            raise ImportError(
                f"Script '{self.script_name}': script_loc {loc!r} is not a file; "
                f"use 'module.function' with a directory or point script_loc at the .py file."
            )
        rel = self._module_name.replace('.', os.sep)
        for cand in (os.path.join(loc, rel + '.py'), os.path.join(loc, rel, '__init__.py')):
            if os.path.isfile(cand):
                return cand
        return None

    def _load(self):
        path = self._resolve_file()
        if path is not None:
            base = os.path.splitext(os.path.basename(path))[0]
            unique = f"_energysim_script_{next(_instance_counter)}_{base}"
            self._module = _load_module_from_file(path, unique)
        else:
            # Fallback: an installed / importable module (no file under script_loc)
            loc = os.fspath(self.script_loc) if self.script_loc else ''
            inserted = bool(loc) and loc not in sys.path
            if inserted:
                sys.path.insert(0, loc)
            try:
                self._module = importlib.import_module(self._module_name)
            finally:
                if inserted:
                    try:
                        sys.path.remove(loc)
                    except ValueError:
                        pass
        try:
            self._func = getattr(self._module, self._func_name)
        except AttributeError:
            raise AttributeError(
                f"Function '{self._func_name}' not found in script '{self.script_name}'."
            ) from None

    # ------------------------------------------------------------------
    # SimulatorAdapter interface
    # ------------------------------------------------------------------

    def init(self):
        # Re-load after cleanup() so a world can be simulated again.
        if self._module is None or self._func is None:
            self._load()
        self._input_dict = {}
        self._output_values = [0.0] * len(self.outputs)

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            self._input_dict[p] = v

    def get_value(self, parameters, time=None):
        vals = []
        for p in parameters:
            if p in self.outputs:
                idx = self.outputs.index(p)
                if idx < len(self._output_values):
                    vals.append(self._output_values[idx])
                else:
                    vals.append(0.0)
            else:
                raise SimulatorVariableError(
                    f"Variable '{p}' not found in script outputs {self.outputs}."
                )
        return vals

    def step(self, time):
        result = self._func(self._input_dict)
        if isinstance(result, (list, tuple)):
            self._output_values = list(result)
        else:
            self._output_values = [result]

    def cleanup(self):
        mod_name = getattr(self._module, '__name__', '') or ''
        if mod_name.startswith('_energysim_script_'):
            sys.modules.pop(mod_name, None)
        self._module = None
        self._func = None

    def get_available_variables(self):
        return {'inputs': list(self.inputs), 'outputs': list(self.outputs)}
