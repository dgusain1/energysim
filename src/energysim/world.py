# -*- coding: utf-8 -*-
"""
World cosimulation canvas for energysim.

The implementation lives in :mod:`energysim._world` as mixin classes
(registration, connections, coupling, runner, validation, scenario); this
module composes them into the public :class:`world` class.
"""

from __future__ import annotations

from .utils import parse_unit
from ._world._common import (  # noqa: F401  (re-exported for backwards compatibility)
    logger, ITERATIVE_REMOVED_MSG, _COUPLINGS, _EXTRAPOLATIONS, UNSAFE_SIM_TYPES, OPTION_KEYS,
    _MODEL_ALIAS_KEYS, _CONTAINER_TYPES, _WORLD_KEYS, RES_FILENAME_REMOVED_MSG, _REMOVED_WORLD_KEYS,
    _normalize_mode, _to_fraction, _filter_kwargs, _is_adapter_like, _enable_verbose_logging,
)
from ._world import (
    RegistryMixin, ConnectionsMixin, UnitsMixin, CouplingMixin, RunnerMixin, ValidationMixin, ScenarioMixin,
    EventsMixin,
)
from ._world.events import _init_event_state
from ._world.unit_resolution import coerce_unit_system, UNIT_POLICIES


class world(RegistryMixin, ConnectionsMixin, UnitsMixin, CouplingMixin, EventsMixin, RunnerMixin,
            ValidationMixin, ScenarioMixin):
    r"""
    Initialization
    ---------------
    ``world`` is your cosimulation canvas. It creates a ``world`` object on
    which simulators can be freely added. ``world`` accepts the
    following parameters :
        1. ``start_time`` : (0 by default).
        2. ``stop_time`` : (1000 by default). ``stop_time - start_time`` must
           be an integer multiple of the macro step, otherwise ``simulate()``
           raises ``ValueError`` (validated with a relative tolerance of 1e-9).
        3. ``logging`` : (False by default). When True, INFO messages of the
           ``'energysim'`` logger are printed to stderr.
        4. ``t_macro`` : (60 by default). Users can specify instance of information exchange.
           ``0`` means "least common multiple of all simulator step sizes".  It is the
           maximum (and default) macro step.
           ``t_macro_min`` : (default ``t_macro``: fixed step).  Set it smaller to let
           event hints of the simulators shrink the macro step to ``[t_macro_min, t_macro]``
           (event-driven / variable macro step, see :meth:`schedule`, :meth:`add_trigger`).
           With a variable step, ``stop_time - start_time`` need not be a multiple of ``t_macro``.
        5. ``coupling`` : ``'jacobi'`` (default) or ``'gauss-seidel'``.
        6. ``start_datetime`` : optional wall-clock start (str / datetime); results are
           then indexed by datetime instead of seconds.
        7. ``seed`` : optional random seed, stored in the results metadata.
        8. ``out_dir`` : optional directory; every run then writes
           ``<out_dir>/<YYYYmmdd-HHMMSS>_<hash8>/`` (Parquet/CSV per simulator,
           ``meta.json``, ``scenario.yaml``).  By default nothing is written.
        9. ``unit_system`` : ``None`` / ``'default'`` / ``'si'`` / overrides dict
           (``{'active_power': 'MW'}``) / ``{'name': 'si', 'overrides': {...}}`` /
           :class:`~energysim.units.UnitSystem`.  It supplies the default unit of
           every variable whose name denotes a quantity but which declares no unit
           (``P_cmd`` -> active power), and the units shown in results.  See
           :meth:`unit_table` / :meth:`connection_units`.
       10. ``unit_policy`` : ``'lenient'`` (default) passes values whose unit is
           unspecified through unscaled (with a warning); ``'strict'`` makes an
           unspecified unit on any connected variable an error at ``add_connections``.
       11. ``power_unit`` : shorthand for the unit of power-like variable names
           (``p``, ``P_cmd``, ``power``...) of every simulator that does not declare
           its own; ``None`` (default) uses the unit system's ``active_power`` unit.

    Unit resolution per connected variable: simulator ``units={var: unit}`` (and
    ``power_unit`` for power-like names) -> the component's own ``units`` ->
    ``from_unit``/``to_unit`` on the connection -> name suffix (``p_mw``) ->
    the unit system's default for the quantity the name denotes -> unspecified
    (never scaled).  Values are converted into the receiver's unit.

    >>> my_world = world(start_time=0, stop_time=1000, logging=True, t_macro=60)

    Specifying Simulators
    ---------------------
    After initializing the world cosimulation object, you can add simulators
    to the world by ``add_simulator()``. This is done by specifying
    the following :
        1. ``sim_type`` : 'fmu', 'powerflow', 'csv', 'signal', ...
        2. ``sim_name`` : It is essential that you assign the simulator a unique name.
        3. ``model`` (or legacy ``sim_loc``) : model object or location.
        4. ``outputs`` : specify the outputs that need to be recorded during simulation from the simulator.
        5. ``inputs`` : specify the inputs of the simulator.
        6. ``step_size`` : step size for simulator.

    >>> my_world.add_simulator(sim_type='fmu', sim_name='FMU1',
    >>> model=/path/to/sim, inputs=[], outputs=['var1','var2'], step_size=1)

    Connections between simulators
    -------------------------------
    >>> connections = {'sim1.output_variable1' : 'sim2.input_variable1',
    >>>    'sim3.output_variable2' : 'sim4.input_variable2',
    >>>    'sim1.output_variable3' : 'sim2.input_variable3',}
    >>> my_world.add_connections(connections)

    Initializing simulator variables
    --------------------------------
    >>> initializations = {'sim_name1' : (['sim_variables'], [values]),
    >>>                    'sim_name2' : (['sim_variables'], [values])}
    >>> my_world.options({'init' : initializations})

    Initial values are applied at the start of every run, either before
    or after ``adapter.init()`` depending on the adapter's
    ``SET_PARAMS_BEFORE_INIT`` flag.

    Simulate
    --------
    >>> my_world.simulate(pbar=True, record_all=False)

    ``simulate()`` can be called repeatedly; each call starts a fresh run
    (``configure()`` + ``init()`` on every adapter) and ``cleanup()`` is
    always called on every adapter, even on errors.

    Results
    -------
    >>> res = my_world.simulate()          # energysim.results.Results
    >>> res['sim1.var1']                   # pandas Series indexed by time
    >>> res.to_frame(); res.summary(); res.kpis(); res.save('out')

    Package Info
    ------------
    Author : Digvijay Gusain

    Email : digvijay.gusain29@gmail.com

    """

    #: rows buffered per simulator before they are converted to typed column chunks
    _results_chunk_rows = 65536

    def __init__(self, start_time=0, stop_time=1000, logging=False, t_macro=60,
                 coupling='jacobi', extrapolation='zero-order',
                 power_unit=None, start_datetime=None, seed=None, out_dir=None, unit_system=None,
                 unit_policy='lenient', t_macro_min=None):
        self.start_time = start_time
        self.stop_time = stop_time
        self.logging = logging
        if logging:
            _enable_verbose_logging()
        self.exchange = t_macro
        if t_macro_min is not None and not (float(t_macro_min) > 0):
            raise ValueError(f"t_macro_min must be > 0, got {t_macro_min!r}")
        self.t_macro_min = t_macro_min
        self._current_dt = None
        _init_event_state(self)
        if power_unit:
            parse_unit(power_unit)
        self.power_unit = power_unit
        self.modify_signal = False
        self.modify_dict = {}
        self.init_dict = {}
        self._parameter_set = {}
        self.record_all = None
        self.G = None
        self.sim_dict = {}
        self.simulator_connections = {}
        self._parsed_connections = []
        self.simulator_dict = {}
        if start_datetime is not None:
            import pandas as pd
            try:
                pd.Timestamp(start_datetime)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid start_datetime {start_datetime!r}: {exc}") from None
        self.start_datetime = start_datetime
        self.seed = seed
        self.out_dir = out_dir
        self.last_results = None
        self.unit_system = coerce_unit_system(unit_system)
        if unit_policy not in UNIT_POLICIES:
            raise ValueError(f"unit_policy must be one of {UNIT_POLICIES}, got {unit_policy!r}")
        self.unit_policy = unit_policy
        self._unit_warned = set()
        if logging:
            self._log_unit_system()
        if coupling == 'iterative':
            raise ValueError(ITERATIVE_REMOVED_MSG)
        if coupling not in _COUPLINGS:
            raise ValueError(f"coupling must be 'jacobi' or 'gauss-seidel', got '{coupling}'")
        self.coupling = coupling
        if extrapolation not in _EXTRAPOLATIONS:
            raise ValueError(f"extrapolation must be 'zero-order' or 'linear', got '{extrapolation}'")
        self.extrapolation = extrapolation
        self._exchange_history = {}   # (sim_name, var_name) → deque(maxlen=2) of (time, value)
        self.macro_tstep = None
        self._total_steps = None
        # Execution order (populated by _compute_execution_order)
        self._execution_order = None
        self._dep_graph = None
        self._cycle_sccs = []  # strongly connected components with cycles


# PEP 8 alias
World = world
