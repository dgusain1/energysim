Working with external simulators
================================

One of the key features of ``energysim`` is easy integration of custom
Python simulators.  Any simulator that implements the
``SimulatorAdapter`` interface can participate in a co-simulation.

Quick start
-----------

1. Create a Python file (e.g. ``my_sim.py``) containing a class called
   ``external_simulator`` that inherits from ``SimulatorAdapter``.
2. Add it to the world::

       my_world.add_simulator(
           sim_type='external',
           sim_name='my_sim',
           sim_loc='/path/to/directory/containing/my_sim.py',
           inputs=['cmd'],
           outputs=['result'],
           step_size=60)

   The ``sim_name`` must match the filename (without ``.py``).


Template
--------

.. literalinclude:: external_simulator.py
   :language: python
   :caption: external_simulator.py


The ``SimulatorAdapter`` interface
----------------------------------

Every external simulator must implement these **abstract methods**:

``step(time)``
    Advance the model by one micro-step at the given ``time``.  This is
    called repeatedly by ``advance()`` at intervals of ``step_size``.

``get_value(parameters, time)``
    Return a **list** of values corresponding to the variable names in
    ``parameters``.  The ``world`` calls this to read outputs.

``set_value(parameters, values)``
    Set the variables named in ``parameters`` to the given ``values``.
    Both arguments are lists of equal length.  The ``world`` calls this
    to push inputs before each macro step.

The following methods have sensible **defaults** but can be overridden:

``init()``
    Called once before the first time step.  Use it to establish
    connections, load data, or start up the model.

``cleanup()``
    Called once after the last time step.  Release resources here.

``advance(start_time, stop_time)``
    The default implementation loops ``step()`` from *start_time* to
    *stop_time* at ``step_size`` intervals using index-based time
    computation to avoid floating-point drift.  Override this only if
    your simulator has its own internal solver that should handle the
    full interval in one call.

``set_parameters(params)``
    Apply a ``{name: value}`` dictionary of initial values.  The default
    delegates to ``set_value()``.

``advance_with_recording(start_time, stop_time, outputs)``
    Advance the model *and* collect intermediate results at each
    micro-step.  Returns a dictionary ``{var_name: [values...]}``.  The
    default calls ``advance()`` then ``get_value()``.  Override this if
    your simulator can provide higher-fidelity intermediate results
    (e.g. from an adaptive solver).  Used when ``record_all=True``.

``save_state() -> Any``
    Save a snapshot of the model state.  Used by iterative coupling to
    roll back if convergence is not achieved.  Return any picklable
    object that captures the full state.  Default returns ``None``
    (no-op).

``restore_state(state)``
    Restore a previously saved state.  ``state`` is whatever was
    returned by ``save_state()``.  Default is a no-op.

``get_available_variables() -> dict``
    Return a dictionary of known variable names grouped by category
    (e.g. ``{'inputs': [...], 'outputs': [...]}``) for connection
    validation.  Default returns an empty dict.


Supporting iterative coupling
-----------------------------

If your external simulator will be used with ``coupling='iterative'``,
you **should** implement ``save_state()`` and ``restore_state()`` so
the coordinator can roll back the simulator when convergence is not
achieved::

    class external_simulator(SimulatorAdapter):

        def __init__(self, inputs=None, outputs=None, **kwargs):
            self.inputs = inputs or []
            self.outputs = outputs or []
            self.temperature = 20.0
            self.pressure = 101325.0

        def save_state(self):
            return {
                'temperature': self.temperature,
                'pressure': self.pressure,
            }

        def restore_state(self, state):
            self.temperature = state['temperature']
            self.pressure = state['pressure']

        # ... step, get_value, set_value as usual


Example: a simple battery model
--------------------------------

.. code-block:: python

    from energysim.base import SimulatorAdapter

    class external_simulator(SimulatorAdapter):

        def __init__(self, inputs=None, outputs=None, **kwargs):
            self.inputs = inputs or []
            self.outputs = outputs or []
            self.soc = 0.5      # initial state of charge
            self.capacity = 100 # kWh
            self.power = 0.0    # kW (positive = charging)

        def init(self):
            pass

        def step(self, time):
            dt = self.step_size  # set by energysim
            self.soc += (self.power * dt / 3600) / self.capacity
            self.soc = max(0.0, min(1.0, self.soc))

        def get_value(self, parameters, time):
            mapping = {'soc': self.soc, 'power': self.power}
            return [mapping.get(p, 0.0) for p in parameters]

        def set_value(self, parameters, values):
            for p, v in zip(parameters, values):
                if p == 'power':
                    self.power = v

        def cleanup(self):
            pass

        # Optional: support iterative coupling
        def save_state(self):
            return {'soc': self.soc, 'power': self.power}

        def restore_state(self, state):
            self.soc = state['soc']
            self.power = state['power']

        # Optional: support variable introspection
        def get_available_variables(self):
            return {
                'inputs': self.inputs,
                'outputs': self.outputs,
            }


Example: recording intermediate results
-----------------------------------------

If your simulator has an internal adaptive solver and you want to
capture its intermediate time points when ``record_all=True``::

    def advance_with_recording(self, start_time, stop_time, outputs):
        results = {var: [] for var in outputs}
        times = self.solver.integrate(start_time, stop_time)
        for t in times:
            vals = self.get_value(outputs, t)
            for var, val in zip(outputs, vals):
                results[var].append(val)
        return results
