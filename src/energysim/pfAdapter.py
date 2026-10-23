# -*- coding: utf-8 -*-
"""
DIgSILENT PowerFactory adapter for energysim.

Provides a ``SimulatorAdapter`` that drives a PowerFactory project
via the official Python API (2020+).  The adapter connects to a
running PowerFactory instance, activates the specified project, and
runs load-flow / short-circuit / RMS calculations each time step.

Requirements
------------
- DIgSILENT PowerFactory 2020 or later with a valid licence.
- The ``powerfactory`` Python module must be importable.  When running
  outside the built-in PF Python interpreter, pass ``pf_path`` to the
  ``add_simulator`` call so the adapter can add it to ``sys.path``.

Variable naming
---------------
Inputs and outputs use ``"ElementName.attribute"`` dot-notation::

    'Load1.plini'       # set active power of Load1
    'Bus1.m:u'          # read voltage magnitude from Bus1

For *inputs* (``set_value``), the adapter writes to the element data
model using ``SetAttribute``.  For *outputs* (``get_value``), it reads
from the results using ``GetAttribute``.  Result attributes usually
start with ``m:`` (e.g. ``m:u``, ``m:P:bus1``).  Input attributes
usually start with ``e:`` but the prefix is added automatically if
omitted.
"""
from __future__ import annotations


import os
import sys
import warnings

from .base import (
    SimulatorAdapter, SimulatorVariableError, SimulatorElementNotFoundError, EnergysimError,
)


class pf_adapter(SimulatorAdapter):
    """Adapter for DIgSILENT PowerFactory (2020+ Python API).

    Elements are cached by ``loc_name``.  PowerFactory allows the same
    ``loc_name`` in different folders; when names collide the *first* object
    returned by ``GetCalcRelevantObjects`` wins, a warning lists the
    duplicates and they are available as ``adapter.duplicate_names``.

    ``cleanup()`` deactivates the project; ``init()`` may be called again.
    """

    SET_PARAMS_BEFORE_INIT = False

    def __init__(self, project_name=None, pf_loc=None, inputs=None,
                 outputs=None, mode='ac', pf=None, pf_path=None, model=None):
        """
        Parameters
        ----------
        project_name : str or None
            Name of the PowerFactory project to activate.
        pf_loc : str or None
            Backward-compatible location / project name argument.
        model : str or None
            PowerFactory project name identifier.
        inputs : list[str]
            Input variables in ``"ElementName.attribute"`` format.
        outputs : list[str]
            Output variables to record.
        mode : str
            Calculation mode: ``'ac'`` (AC load flow, default) or ``'dc'`` (DC load flow).
        pf_path : str or None
            Path to the PowerFactory Python directory, e.g.
            ``r"C:\\DIgSILENT\\PowerFactory 2024\\Python\\3.11"``.
            If *None*, ``powerfactory`` must already be importable.
        """
        self.project_name = project_name or model or pf_loc
        self.pf_loc = pf_loc or self.project_name
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.pf_path = pf_path

        # Standardized powerflow mode: strictly 'ac' or 'dc'
        raw_mode = (mode if (mode != 'ac' or pf is None) else pf) or 'ac'
        raw_mode = raw_mode.lower()
        if raw_mode in ('ac', 'ldf', 'pf'):
            self.mode = 'ac'
        elif raw_mode in ('dc', 'dcpf'):
            self.mode = 'dc'
        else:
            raise ValueError(
                f"Unsupported calculation mode '{raw_mode}'. "
                f"energysim only supports 'ac' and 'dc' power flow modes."
            )
        self.pf = self.mode  # backward compatibility alias

        # Populated by init()
        self._app = None
        self._project = None
        self._calc_cmd = None
        self._sim_cmd = None
        self._element_cache = {}   # name -> PF DataObject
        self.duplicate_names = {}  # name -> number of objects sharing it

    # ------------------------------------------------------------------
    # Life-cycle
    # ------------------------------------------------------------------

    def init(self):
        """Connect to PowerFactory, activate the project, cache elements."""
        if self.pf_path:
            if not os.path.isdir(self.pf_path):
                raise FileNotFoundError(
                    f"PowerFactory Python path does not exist: {self.pf_path}"
                )
            if self.pf_path not in sys.path:
                sys.path.insert(0, self.pf_path)

        try:
            import powerfactory  # noqa: delayed import
        except ImportError as exc:
            raise ImportError(
                "The 'powerfactory' module could not be imported. Pass pf_path=<PowerFactory "
                "Python directory> (e.g. r'C:\\Program Files\\DIgSILENT\\PowerFactory 2024\\Python\\3.11') "
                "and make sure a PowerFactory licence is available."
            ) from exc
        self._app = powerfactory.GetApplication()
        if self._app is None:
            raise RuntimeError(
                "Could not connect to PowerFactory. "
                "Make sure PowerFactory is running or the Python "
                "path is configured correctly."
            )

        # Activate project
        if not self.project_name:
            raise ValueError("No PowerFactory project name given.")
        err = self._app.ActivateProject(self.project_name)
        if err:
            self._app = None
            raise RuntimeError(
                f"Could not activate project '{self.project_name}' "
                f"(error code {err})."
            )
        self._project = self._app.GetActiveProject()
        if self._project is None:
            self._app = None
            raise RuntimeError(
                f"Project '{self.project_name}' was activated but GetActiveProject() returned None."
            )

        # Get the calculation command object (ComLdf for both AC and DC load flow)
        self._calc_cmd = self._app.GetFromStudyCase('ComLdf')

        if self._calc_cmd is None:
            raise RuntimeError(
                "Could not find calculation command 'ComLdf' in the active study case "
                "(is a study case active in the project?)."
            )

        # Set AC (0) vs DC (2) load flow option in PowerFactory ComLdf
        if self.mode == 'dc':
            self._calc_cmd.SetAttribute('iopt_net', 2)
        else:
            self._calc_cmd.SetAttribute('iopt_net', 0)

        # Build element cache from all calc-relevant objects
        self._build_element_cache()

        # Run initial calculation
        self._run_calculation()

    def cleanup(self):
        """Deactivate project and release the application handle."""
        if self._project is not None:
            try:
                self._project.Deactivate()
            except Exception:
                pass
        self._app = None
        self._project = None
        self._calc_cmd = None
        self._sim_cmd = None
        self._element_cache.clear()

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def step(self, time):
        """Run the configured calculation."""
        self._run_calculation()

    def set_value(self, parameters, values):
        """Set element attributes.

        Each parameter is ``"ElementName.attribute"``.  If the
        attribute does not start with a known PF prefix (``e:``,
        ``m:``, ``c:``, ``s:``), the ``e:`` prefix is added
        automatically (PowerFactory convention for input / editing
        attributes).
        """
        for param, value in zip(parameters, values):
            elem_name, attr = self._split(param)
            obj = self._get_element(elem_name)
            # Auto-prefix with 'e:' for input attributes
            if not attr.startswith(('e:', 'm:', 'c:', 's:')):
                attr = 'e:' + attr
            try:
                rc = obj.SetAttribute(attr, float(value))
            except Exception as exc:
                raise SimulatorVariableError(
                    f"Could not set '{attr}' on PowerFactory element '{elem_name}': {exc}"
                ) from exc
            if isinstance(rc, int) and not isinstance(rc, bool) and rc != 0:
                raise SimulatorVariableError(
                    f"PowerFactory rejected SetAttribute('{attr}') on '{elem_name}' (code {rc})."
                )

    def get_value(self, parameters, time=None):
        """Read element attributes (typically ``m:*`` results)."""
        result = []
        for param in parameters:
            elem_name, attr = self._split(param)
            obj = self._get_element(elem_name)
            try:
                val = obj.GetAttribute(attr)
            except Exception as exc:
                raise SimulatorVariableError(
                    f"Could not read '{attr}' from PowerFactory element '{elem_name}': {exc}"
                ) from exc
            if val is None:
                raise SimulatorVariableError(
                    f"Attribute '{attr}' of PowerFactory element '{elem_name}' returned no value "
                    f"(results available? attribute name correct?)."
                )
            try:
                result.append(float(val))
            except (TypeError, ValueError) as exc:
                raise SimulatorVariableError(
                    f"Attribute '{attr}' of '{elem_name}' is not numeric: {val!r}"
                ) from exc
        return result

    def get_available_variables(self):
        """Declared inputs / outputs (PowerFactory attributes are not enumerated)."""
        return {'inputs': list(self.inputs), 'outputs': list(self.outputs),
                'all': list(self.inputs) + list(self.outputs)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_element_cache(self):
        """Pre-cache element objects by name for fast lookup."""
        self._element_cache.clear()
        self.duplicate_names = {}
        all_objs = self._app.GetCalcRelevantObjects('*') or []
        for obj in all_objs:
            name = obj.GetAttribute('loc_name')
            if not name:
                continue
            if name in self._element_cache:
                self.duplicate_names[name] = self.duplicate_names.get(name, 1) + 1
            else:
                self._element_cache[name] = obj
        if self.duplicate_names:
            detail = ", ".join(f"'{n}' x{c}" for n, c in sorted(self.duplicate_names.items())[:20])
            warnings.warn(
                f"PowerFactory project '{self.project_name}' has duplicate element names ({detail}). "
                f"The first object with each name is used; rename the others to address them.",
                UserWarning, stacklevel=3)

    def _split(self, param):
        """Split ``'ElementName.attribute'`` at the first dot that follows a known element name."""
        if not isinstance(param, str) or '.' not in param:
            raise SimulatorVariableError(
                f"Malformed PowerFactory variable '{param}'. Expected 'ElementName.attribute'.")
        for i, c in enumerate(param):
            if c == '.' and param[:i] in self._element_cache:
                return param[:i], param[i + 1:]
        return tuple(param.split('.', 1))  # unknown element: _get_element raises a clear error

    def _get_element(self, name):
        """Retrieve a cached element, raising a clear error if missing."""
        if self._app is None:
            raise EnergysimError("PowerFactory simulator is not initialised (call init() first).")
        if name not in self._element_cache:
            raise SimulatorElementNotFoundError(
                f"Element '{name}' not found in PowerFactory "
                f"project '{self.project_name}'. Available (first 20): "
                f"{list(self._element_cache.keys())[:20]}"
            )
        return self._element_cache[name]

    def _run_calculation(self):
        """Execute the active calculation command (nonzero return code = failure)."""
        rc = self._calc_cmd.Execute()
        if isinstance(rc, int) and not isinstance(rc, bool) and rc != 0:
            raise EnergysimError(
                f"PowerFactory load flow ('ComLdf') failed with error code {rc} "
                f"(project '{self.project_name}', mode '{self.mode}')."
            )
