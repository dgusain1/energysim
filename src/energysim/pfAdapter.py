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

import sys

from .base import SimulatorAdapter


class pf_adapter(SimulatorAdapter):
    """Adapter for DIgSILENT PowerFactory (2020+ Python API)."""

    def __init__(self, project_name, pf_loc=None, inputs=None,
                 outputs=None, pf='ldf', pf_path=None):
        """
        Parameters
        ----------
        project_name : str
            Name of the PowerFactory project to activate.
        pf_loc : str or None
            Not used directly (project is opened by name inside PF).
            Kept for API symmetry with other adapters.
        inputs : list[str]
            Input variables in ``"ElementName.attribute"`` format.
        outputs : list[str]
            Output variables to record.
        pf : str
            Calculation type: ``'ldf'`` (load flow, default),
            ``'shc'`` (short circuit), or ``'rms'`` (RMS simulation).
        pf_path : str or None
            Path to the PowerFactory Python directory, e.g.
            ``r"C:\\DIgSILENT\\PowerFactory 2024\\Python\\3.11"``.
            If *None*, ``powerfactory`` must already be importable.
        """
        self.project_name = project_name
        self.pf_loc = pf_loc
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.pf = pf.lower()
        self.pf_path = pf_path

        assert self.pf in ('ldf', 'shc', 'rms'), (
            f"Unknown calculation type '{self.pf}'. "
            f"Choose from 'ldf', 'shc', 'rms'."
        )

        # Populated by init()
        self._app = None
        self._project = None
        self._calc_cmd = None
        self._sim_cmd = None
        self._element_cache = {}   # name -> PF DataObject

    # ------------------------------------------------------------------
    # Life-cycle
    # ------------------------------------------------------------------

    def init(self):
        """Connect to PowerFactory, activate the project, cache elements."""
        if self.pf_path and self.pf_path not in sys.path:
            sys.path.insert(0, self.pf_path)

        import powerfactory  # noqa: delayed import
        self._app = powerfactory.GetApplication()
        if self._app is None:
            raise RuntimeError(
                "Could not connect to PowerFactory. "
                "Make sure PowerFactory is running or the Python "
                "path is configured correctly."
            )

        # Activate project
        err = self._app.ActivateProject(self.project_name)
        if err:
            raise RuntimeError(
                f"Could not activate project '{self.project_name}' "
                f"(error code {err})."
            )
        self._project = self._app.GetActiveProject()

        # Get the calculation command object
        if self.pf == 'ldf':
            self._calc_cmd = self._app.GetFromStudyCase('ComLdf')
        elif self.pf == 'shc':
            self._calc_cmd = self._app.GetFromStudyCase('ComShc')
        elif self.pf == 'rms':
            self._calc_cmd = self._app.GetFromStudyCase('ComInc')
            self._sim_cmd = self._app.GetFromStudyCase('ComSim')

        if self._calc_cmd is None:
            raise RuntimeError(
                f"Could not find calculation command for mode "
                f"'{self.pf}' in the active study case."
            )

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
            elem_name, attr = param.split('.', 1)
            obj = self._get_element(elem_name)
            # Auto-prefix with 'e:' for input attributes
            if not attr.startswith(('e:', 'm:', 'c:', 's:')):
                attr = 'e:' + attr
            obj.SetAttribute(attr, float(value))

    def get_value(self, parameters, time=None):
        """Read element attributes (typically ``m:*`` results)."""
        result = []
        for param in parameters:
            elem_name, attr = param.split('.', 1)
            obj = self._get_element(elem_name)
            val = obj.GetAttribute(attr)
            result.append(float(val) if val is not None else 0.0)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_element_cache(self):
        """Pre-cache element objects by name for fast lookup."""
        self._element_cache.clear()
        all_objs = self._app.GetCalcRelevantObjects('*')
        for obj in all_objs:
            name = obj.GetAttribute('loc_name')
            if name:
                self._element_cache[name] = obj

    def _get_element(self, name):
        """Retrieve a cached element, raising a clear error if missing."""
        if name not in self._element_cache:
            raise KeyError(
                f"Element '{name}' not found in PowerFactory "
                f"project '{self.project_name}'. Available: "
                f"{list(self._element_cache.keys())[:20]}..."
            )
        return self._element_cache[name]

    def _run_calculation(self):
        """Execute the active calculation command."""
        if self.pf == 'rms':
            self._calc_cmd.Execute()      # ComInc (initialise)
            self._sim_cmd.Execute()       # ComSim (simulate)
        else:
            self._calc_cmd.Execute()
