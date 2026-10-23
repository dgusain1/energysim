# Building Your Own Vectorized Fleet Container

`energysim` makes it straightforward to author custom high-performance vectorized fleet containers for any large-scale asset class — such as EV fleets with stochastic arrival/departure schedules, water heaters, industrial electrolysis cells, or cold storage facilities.

There are two primary approaches:
1. **Subclassing `VectorizedContainerAdapter`** (Recommended for reusable, robust domain models).
2. **Using `VectorizedFunctionAdapter`** (Quickest for prototyping with a pure NumPy function).

---

## Approach 1: Subclassing `VectorizedContainerAdapter`

By subclassing `VectorizedContainerAdapter`, you define vector states, inputs, aliases, and a fully vectorized `step()` method executed via NumPy C-arrays.

### Step-by-Step Guide: EV Fleet with Commute Logic

Let's build a `VectorizedEVFleet` where $N$ electric vehicles plug in during evening hours, charge at home, and drive to work in the morning:

```python
from typing import Optional, Union
import numpy as np
from energysim.container import VectorizedContainerAdapter


class VectorizedEVFleet(VectorizedContainerAdapter):
    """
    Vectorized Electric Vehicle (EV) Fleet Container.

    Models N electric vehicles with randomized battery capacities,
    plug-in status flags, commute consumption, and charging setpoints.
    """

    def __init__(
        self,
        name: str = "ev_fleet",
        size: int = 1000,
        battery_kwh: Union[float, np.ndarray] = 60.0,
        max_charger_kw: Union[float, np.ndarray] = 7.4,
        initial_soc: Union[float, np.ndarray] = 0.5,
        step_size: float = 900.0,
        **kwargs,
    ):
        super().__init__(name=name, size=size, step_size=step_size)

        # 1. Initialize parameter vectors (size N)
        if isinstance(battery_kwh, (int, float)):
            self.battery_capacity = np.full(self.size, float(battery_kwh), dtype=np.float64)
        else:
            self.battery_capacity = np.asarray(battery_kwh, dtype=np.float64)

        if isinstance(max_charger_kw, (int, float)):
            self.charger_limit = np.full(self.size, float(max_charger_kw), dtype=np.float64)
        else:
            self.charger_limit = np.asarray(max_charger_kw, dtype=np.float64)

        # 2. Register states (mapped to length-N numpy arrays)
        self.register_state("soc", initial_soc)
        self.register_state("actual_p_kw", 0.0)
        self.register_state("plugged_in", 1.0)  # 1.0 = connected, 0.0 = driving

        # 3. Register inputs
        self.register_input("charge_price_signal", 0.15)  # $/kWh
        self.register_input("grid_power_limit_kw", 1e6)   # Substation constraint

        # 4. Register convenient aliases
        self.register_alias("p_kw", "actual_p_kw")
        self.register_alias("power", "actual_p_kw")

    def step(self, time: float, step_size: Optional[float] = None) -> None:
        """Advance all N EVs simultaneously in C-speed vectorized operations."""
        dt_hr = (step_size or self.step_size) / 3600.0
        hr_of_day = (time / 3600.0) % 24.0

        soc = self.states["soc"]
        plugged = self.states["plugged_in"]

        # Vectorized commute schedule:
        # EVs unplug and drive between 7:30 and 8:30 AM
        driving_morning = (hr_of_day >= 7.5) & (hr_of_day < 8.5)
        # EVs return and plug in between 17:30 and 18:30 PM
        home_evening = (hr_of_day >= 17.5) | (hr_of_day < 7.5)

        plugged[driving_morning] = 0.0
        plugged[home_evening] = 1.0

        # Energy consumed during commute: 3 kWh drive loss
        commute_mask = driving_morning & (soc > 0.1)
        soc[commute_mask] -= (3.0 / self.battery_capacity[commute_mask]) * dt_hr

        # Charge logic for plugged-in vehicles wanting to reach 90% SoC
        needs_charge = (plugged == 1.0) & (soc < 0.90)
        headroom_kwh = np.maximum(0.0, (0.90 - soc) * self.battery_capacity)
        max_p_headroom_kw = headroom_kwh / dt_hr

        charge_p = np.zeros(self.size, dtype=np.float64)
        charge_p[needs_charge] = np.minimum(self.charger_limit[needs_charge], max_p_headroom_kw[needs_charge])

        # Update SoC
        soc[needs_charge] += (charge_p[needs_charge] * 0.95 * dt_hr) / self.battery_capacity[needs_charge]
        self.states["soc"][:] = np.clip(soc, 0.05, 1.0)
        self.states["actual_p_kw"][:] = charge_p
```

### Simulating the Custom EV Fleet

You can integrate your custom container directly into `energysim`:

```python
from energysim import world

# Instantiate fleet of 10,000 EVs
N = 10000
ev_fleet = VectorizedEVFleet(name="ev_fleet", size=N, battery_kwh=65.0)

sim = world(start_time=0, stop_time=86400, t_macro=900)
sim.add_model("evs", model=ev_fleet, outputs=[
    "total_actual_p_kw", "mean_soc", "min_soc", "max_soc"
])

sim.simulate(dashboard=False)
results = sim.results(dashboard=False)

print("Peak aggregate EV charging demand:", results["evs"]["total_actual_p_kw"].max() / 1000.0, "MW")
```

---

## Approach 2: Using `VectorizedFunctionAdapter`

If you prefer not to write a class, you can pass a pure vectorized NumPy function directly to `VectorizedFunctionAdapter`:

```python
import numpy as np
from energysim import world
from energysim.container import VectorizedFunctionAdapter


def my_vectorized_update(time, states, inputs, dt):
    """
    Vectorized state update function.
    states: dict mapping var_name -> 1D numpy array
    inputs: dict mapping var_name -> 1D numpy array
    Returns: dict of updated state arrays
    """
    dt_hr = dt / 3600.0
    soc = states["soc"]
    p_cmd = inputs["p_setpoint"]  # Length-N array or scalar broadcast

    # Update states in single vectorized operation
    new_soc = np.clip(soc + (p_cmd * 0.92 * dt_hr) / 50.0, 0.0, 1.0)
    return {"soc": new_soc, "actual_p": p_cmd}


sim = world(start_time=0, stop_time=3600, t_macro=900)
custom_container = VectorizedFunctionAdapter(
    name="quick_fleet",
    size=1000,
    update_func=my_vectorized_update,
    initial_states={"soc": 0.5, "actual_p": 0.0},
    inputs=["p_setpoint"],
    step_size=900.0,
)

sim.add_model("quick_fleet", model=custom_container, outputs=["total_actual_p", "mean_soc"])
sim.simulate(dashboard=False)
```

---

## Performance Best Practices

1. **Avoid Python Loops**: Never write `for i in range(self.size)` inside `step()`. Use boolean masks (`mask = soc < 0.2`) and NumPy array slices (`soc[mask] += ...`).
2. **Pre-allocate Arrays**: In `__init__`, pre-allocate temporary working buffers if needed, or use in-place assignment (`self.states['soc'][:] = ...`) to prevent memory allocations.
3. **Exploit In-Place Slicing**: Assigning `arr[:] = new_val` preserves existing C-contiguous array buffers, avoiding garbage collector overhead.
