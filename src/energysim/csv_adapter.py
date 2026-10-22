import pandas as pd
import numpy as np
import warnings

from .base import SimulatorAdapter


class csv_simulator(SimulatorAdapter):

    def __init__(self, sim_name, sim_loc, step_size=900, outputs=None, delimiter=','):
        # initalize sim object here
        self.df = pd.read_csv(sim_loc, delimiter=delimiter)
        self.step_size = step_size
        self.outputs = outputs or []
        # analyse the df, and calculate step size
        assert 'time' in self.df.columns, (
            'No time column in csv file. Please convert csv file'
            ' to required format. CSV not added.'
        )
        self.time_array = self.df.time.to_numpy()
        if len(self.time_array) > 1:
            diffs = np.diff(self.time_array)
            assert np.allclose(diffs, diffs[0], rtol=1e-4, atol=1e-6), (
                'energysim can only read csv with fixed time intervals.'
                ' Current file does not have time stamps with fixed'
                ' interval. Cant add csv simulator.'
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
        # this should return a list of values from simulator as a list corresponding to parameters
        # np.searchsorted is O(log N) and handles end-of-array without crashing
        idx = np.searchsorted(self.time_array, time, side='right') - 1
        idx = max(0, min(int(idx), len(self.time_array) - 1))
        return [self.df.at[idx, ele] for ele in parameters]

    def step(self, time):
        # use the time variable (if needed) to step the simulator to t=time

        # return is not required. remove the pass command afterwards.
        pass

    def get_available_variables(self):
        """Return CSV column names as available output variables."""
        cols = [c for c in self.df.columns if c != 'time']
        return {'inputs': [], 'outputs': cols}
