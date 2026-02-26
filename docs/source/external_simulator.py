# Template for user-defined external simulators.
#
# Copy this file, rename it to match your sim_name, and implement the
# abstract methods.  The ``world`` orchestrator will call:
#   init() -> advance() (which loops step()) -> cleanup()
# automatically.
#
# Optional methods for advanced features:
#   save_state() / restore_state()  -- for iterative coupling
#   advance_with_recording()        -- for record_all=True
#   get_available_variables()       -- for connection validation

from energysim.base import SimulatorAdapter


class external_simulator(SimulatorAdapter):
    """Minimal external simulator template."""

    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs = inputs or []
        self.outputs = outputs or []
        # Initialise your model state here

    def init(self):
        # Called once before the first time step.
        # Establish connections, load data, start up model.
        pass

    def step(self, time):
        # Advance the model by one micro-step at ``time``.
        pass

    def get_value(self, parameters, time):
        # Return a list of values corresponding to ``parameters``.
        # Example: return [self.var1, self.var2]
        pass

    def set_value(self, parameters, values):
        # Set the variables named in ``parameters`` to ``values``.
        # Both are lists of equal length.
        pass

    def cleanup(self):
        # Release any resources.
        pass

    # --- Optional: support iterative coupling ---

    def save_state(self):
        # Return a snapshot of your model's state.
        # Example: return {'var1': self.var1, 'var2': self.var2}
        return None

    def restore_state(self, state):
        # Restore a previously saved state.
        # Example:
        #   self.var1 = state['var1']
        #   self.var2 = state['var2']
        pass

    # --- Optional: support connection validation ---

    def get_available_variables(self):
        # Return known variable names for validation.
        return {
            'inputs': self.inputs,
            'outputs': self.outputs,
        }
