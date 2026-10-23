from __future__ import annotations

import pandas as pd
import numpy as np
import warnings

from .base import SimulatorAdapter, SimulatorVariableError


class csv_simulator(SimulatorAdapter):

    def __init__(self, sim_name, sim_loc, step_size=900, outputs=None, delimiter=','):
        self.sim_name = sim_name
        self.df = pd.read_csv(sim_loc, delimiter=delimiter)
        self.step_size = step_size
        self.outputs = outputs or []
        if 'time' not in self.df.columns:
            raise ValueError(
                f"CSV simulator '{sim_name}': no 'time' column in {sim_loc!r}. "
                f"Please convert the csv file to the required format."
            )
        self.time_array = self.df.time.to_numpy()
        if len(self.time_array) > 1:
            diffs = np.diff(self.time_array)
            if not np.allclose(diffs, diffs[0], rtol=1e-4, atol=1e-6):
                raise ValueError(
                    f"CSV simulator '{sim_name}': energysim can only read csv files with fixed "
                    f"time intervals; {sim_loc!r} does not have a fixed interval."
                )
        missing = [o for o in self.outputs if o not in self.df.columns]
        if missing:
            raise SimulatorVariableError(
                f"CSV simulator '{sim_name}': column(s) {missing} not found. "
                f"Available columns: {[c for c in self.df.columns if c != 'time']}"
            )

    def init(self):
        pass

    def set_value(self, parameters, values):
        # CSV simulators are read-only — warn users if a connection writes here
        warnings.warn(
            f"CSV simulator is read-only. set_value({parameters}) was called"
            f" but values were discarded. Check your connections.",
            UserWarning, stacklevel=2
        )

    def get_value(self, parameters, time):
        # np.searchsorted is O(log N) and handles end-of-array without crashing
        idx = np.searchsorted(self.time_array, time, side='right') - 1
        idx = max(0, min(int(idx), len(self.time_array) - 1))
        vals = []
        for ele in parameters:
            if ele not in self.df.columns:
                raise SimulatorVariableError(
                    f"CSV simulator '{self.sim_name}': column '{ele}' not found. "
                    f"Available columns: {[c for c in self.df.columns if c != 'time']}"
                )
            vals.append(self.df.at[idx, ele])
        return vals

    def step(self, time):
        pass

    def get_available_variables(self):
        """Return CSV column names as available output variables."""
        cols = [c for c in self.df.columns if c != 'time']
        return {'inputs': [], 'outputs': cols}
