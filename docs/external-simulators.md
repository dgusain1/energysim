# External Simulators

User-defined Python simulators can be integrated into `energysim`
co-simulations by subclassing `SimulatorAdapter` from `energysim.base`.

---

## Quick start

1. Create a Python file named `<sim_name>.py` in a directory of your choice.
2. Define a class called `external_simulator` that inherits from `SimulatorAdapter`.
3. Implement the required methods: `step()`, `get_value()`, `set_value()`.
4. Add the simulator to the `world` object with `sim_type='external'`.

```python
my_world.add_simulator(
    sim_type='external',
    sim_name='my_sim',        # must match filename (without .py)
    sim_loc='/path/to/dir',   # directory containing my_sim.py
    inputs=['cmd'],
    outputs=['result'],
    step_size=60)
```

---

## Template

```python
from energysim.base import SimulatorAdapter


class external_simulator(SimulatorAdapter):
    """Template for an energysim external simulator."""

    def __init__(self, **kwargs):
        # kwargs contains: sim_name, inputs, outputs, step_size
        self.sim_name = kwargs.get('sim_name', 'external')
        self.inputs = kwargs.get('inputs', [])
        self.outputs = kwargs.get('outputs', [])
        self.step_size = kwargs.get('step_size', 1)

        # Internal state
        self.state = {var: 0.0 for var in self.outputs}

    def init(self):
        """Called once before the first time step."""
        pass

    def step(self, time):
        """Advance the model by one micro-step (step_size seconds)."""
        pass

    def get_value(self, parameters, time):
        """Return current values for the requested output variables."""
        return [self.state.get(p, 0.0) for p in parameters]

    def set_value(self, parameters, values):
        """Set input values received from other simulators."""
        for p, v in zip(parameters, values):
            self.state[p] = v

    def cleanup(self):
        """Called once after the last time step."""
        pass

    # Optional: support iterative coupling
    def save_state(self):
        """Return a snapshot of the current state."""
        return None

    def restore_state(self, state):
        """Restore state from a previous snapshot."""
        pass

    # Optional: support variable introspection
    def get_available_variables(self):
        """Return known variable names for validation."""
        return {}
```

---

## SimulatorAdapter interface

### Abstract methods (must implement)

#### `step(time)`

Advance the model by one internal time-step (`step_size` seconds).
Called repeatedly by `advance()`. The `time` argument is the current
simulation time in seconds.

#### `get_value(parameters, time)`

Return a list of current values for the requested variables.
`parameters` is a list of variable name strings, `time` is the
current simulation time. The return value must be a list of the same
length as `parameters`.

#### `set_value(parameters, values)`

Set input values. `parameters` is a list of variable name strings,
`values` is a list of corresponding values.

### Default methods (override when needed)

#### `init()`

Called once before the first time step. Default is a no-op. Override
to load data, open connections, or initialise solvers.

#### `cleanup()`

Called once after the last time step. Default is a no-op. Override
to release resources, close files, or disconnect from external servers.

#### `advance(start_time, stop_time)`

Advance the model from `start_time` to `stop_time` by repeatedly
calling `step()`. The default implementation uses index-based time
computation to avoid floating-point drift:

```python
n_steps = round((stop_time - start_time) / self.step_size)
for i in range(n_steps):
    t = start_time + i * self.step_size
    self.step(t)
```

Override this if your model has its own solver (e.g. adaptive ODE
solver) that can handle the full interval.

#### `set_parameters(params)`

Set model parameters before the first step. Default delegates to
`set_value()`.

#### `advance_with_recording(start_time, stop_time, outputs)`

Advance the model and collect output values at each micro-step. Used
when `record_all=True`. Default calls `advance()` then `get_value()`.

#### `save_state() -> Any`

Snapshot the model state for iterative coupling. Default returns `None`.
When iterative coupling is used, the coordinator calls this before each
iteration pass and `restore_state()` if convergence is not achieved.

#### `restore_state(state)`

Restore a previously saved state. Default is a no-op.

#### `get_available_variables() -> dict`

Return known variable names grouped by category (e.g. `{'inputs': [...], 'outputs': [...]}`). Used by connection validation to warn about misspelled variable names.

---

## Supporting iterative coupling

To support iterative coupling (`coupling='iterative'`), implement
`save_state()` and `restore_state()`:

```python
def save_state(self):
    """Return a snapshot of all internal state."""
    return {
        'soc': self.soc,
        'temperature': self.temperature,
        'power': self.power,
    }

def restore_state(self, state):
    """Restore state from a previous snapshot."""
    self.soc = state['soc']
    self.temperature = state['temperature']
    self.power = state['power']
```

!!! note
    If `save_state()` returns `None`, the coordinator cannot roll back
    the simulator and iterative coupling degrades to Gauss-Seidel for
    that simulator.

---

## Example: battery model

```python
from energysim.base import SimulatorAdapter


class external_simulator(SimulatorAdapter):
    """Simple battery model with Coulomb counting."""

    def __init__(self, **kwargs):
        self.sim_name = kwargs.get('sim_name', 'battery')
        self.inputs = kwargs.get('inputs', ['power'])
        self.outputs = kwargs.get('outputs', ['soc'])
        self.step_size = kwargs.get('step_size', 1)

        # Battery parameters
        self.capacity_kwh = 10.0
        self.max_power = 5.0    # kW
        self.efficiency = 0.95

        # State
        self.soc = 0.5
        self.power = 0.0

    def init(self):
        pass

    def step(self, time):
        dt = self.step_size
        p = max(-self.max_power, min(self.max_power, self.power))
        eta = self.efficiency if p > 0 else 1 / self.efficiency
        self.soc -= (p * eta * dt / 3600) / self.capacity_kwh
        self.soc = max(0.05, min(0.95, self.soc))

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
```

---

## Example: recording intermediate results

If your simulator has an internal adaptive solver and you want to
capture its intermediate time points when `record_all=True`:

```python
def advance_with_recording(self, start_time, stop_time, outputs):
    results = {var: [] for var in outputs}
    times = self.solver.integrate(start_time, stop_time)
    for t in times:
        vals = self.get_value(outputs, t)
        for var, val in zip(outputs, vals):
            results[var].append(val)
    return results
```
