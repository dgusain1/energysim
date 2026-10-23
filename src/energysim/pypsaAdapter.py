"""
PyPSA adapter for energysim.

LIMITATION NOTE (Pattern B - Single-Snapshot Mode):
This adapter operates in Pattern B (Single Snapshot Online Dispatch/Power Flow).
At each co-simulation macro time step, the PyPSA network evaluates a single snapshot.
Multi-period lookahead time series optimization (e.g., continuous multi-day LOPF
with inter-temporal storage/SOC linkages) is evaluated sequentially snapshot-by-snapshot
within the co-simulation loop rather than solving the entire multi-period time-horizon
simultaneously.
"""
from __future__ import annotations


import logging

from .base import SimulatorAdapter, SimulatorElementNotFoundError


class pypsa_adapter(SimulatorAdapter):
    """
    Adapter for PyPSA networks in energysim.

    Design Characteristic / Limitation (Pattern B):
    Operates on a single active snapshot at each macro time step. Multi-snapshot
    lookahead horizons are evaluated sequentially step-by-step.
    """

    def __init__(self, network_name, net_loc=None, inputs=None, outputs=None, mode='ac', pf=None, logger_level='CRITICAL', model=None):
        import pypsa  # lazy import: pypsa is optional and has heavy dependencies
        pypsa.pf.logger.setLevel(getattr(logging, logger_level.upper(), logging.CRITICAL))
        self.network_name = network_name
        target = model if model is not None else net_loc
        if target is None:
            raise ValueError(f"No network or model specified for PyPSA simulator '{network_name}'.")

        from .utils import resolve_model_reference
        target = resolve_model_reference(target)

        # Standardized powerflow mode: strictly 'ac' or 'dc'
        raw_mode = (mode if (mode != 'ac' or pf is None) else pf) or 'ac'
        raw_mode = raw_mode.lower()
        if raw_mode in ('ac', 'pf'):
            self.mode = 'ac'
        elif raw_mode in ('dc', 'dcpf', 'lpf'):
            self.mode = 'dc'
        else:
            raise ValueError(
                f"Unsupported calculation mode '{raw_mode}'. "
                f"energysim only supports 'ac' and 'dc' power flow modes."
            )
        self.pf = self.mode  # backward compatibility alias

        if hasattr(target, 'snapshots') and hasattr(target, 'buses'):
            self.network = target
            self.net_loc = getattr(target, 'name', None) or network_name
        elif isinstance(target, str):
            self.net_loc = target
            self.network = pypsa.Network()
            if target.endswith('.nc'):
                self.network.import_from_netcdf(target)
            else:
                self.network.import_from_csv_folder(target)
        elif callable(target):
            self.network = target()
            self.net_loc = getattr(self.network, 'name', None) or network_name
        else:
            self.network = target
            self.net_loc = network_name

        assert len(self.network.snapshots) == 1, (
            "Pattern B limitation: Only one snapshot is supported per co-simulation step."
        )
        inputs = inputs or []
        outputs = outputs or []
        self.new_inputs, self.new_outputs = self.process_powerflow_ipop(self.network, inputs, outputs)
        self.outputs = outputs

    def init(self):
        self.step(0)

    def set_value(self, parameters, values):
        for parameter, value in zip(parameters, values):
            ele_name, input_variable = parameter.split('.')
            assert input_variable in ['P', 'Q'], "Powerflow input variable not valid. Use P, Q to  define variables."
            if ele_name in list(self.network.generators.index):
                adder, residual = 'generators', input_variable.lower() + '_set'
            elif ele_name in list(self.network.loads.index):
                adder, residual = 'loads', input_variable.lower() + '_set'
            elif ele_name in list(self.network.storage_units.index):
                adder, residual = 'storage_units', input_variable.lower() + '_set'
            else:
                raise SimulatorElementNotFoundError(
                    f"Could not find '{ele_name}' as a component in network '{self.network_name}'."
                    f" Make sure element names are correctly specified."
                )

            getattr(self.network, adder).at[ele_name, residual] = value

    def get_value(self, parameters, time):
        temp_parameter_list = [x.split('.') for x in parameters]
        temp_output = []
        for ele_name, output_variable in temp_parameter_list:
            x = None  # only valid for bus voltage queries; prevents NameError
            if output_variable.lower() == 'v':
                x = 'v_mag_pu'
            elif output_variable.lower() == 'va':
                x = 'v_ang'
            if ele_name in list(self.network.generators.index):
                adder, residual = 'generators_t', output_variable.lower()
            elif ele_name in list(self.network.loads.index):
                adder, residual = 'loads_t', output_variable.lower()
            elif ele_name in list(self.network.buses.index):
                if x is None:
                    raise ValueError(
                        f"output variable '{output_variable}' is not valid for bus '{ele_name}'. "
                        "Use 'V' (voltage magnitude) or 'Va' (voltage angle)."
                    )
                adder, residual = 'buses_t', x
            elif ele_name in list(self.network.storage_units.index):
                adder, residual = 'storage_units_t', output_variable.lower()
            else:
                raise SimulatorElementNotFoundError(
                    f"Could not find '{ele_name}' as a component in network '{self.network_name}'."
                    f" Make sure element names are correctly specified."
                )

            temp_var = getattr(getattr(self.network, adder), residual).at[list(
                getattr(getattr(self.network, adder), residual).index)[0], str(ele_name)]
            temp_output.append(temp_var)

        return temp_output

    def getOutput(self):
        return [
            getattr(getattr(self.network, adder), residual).at[
                list(getattr(
                    getattr(self.network, adder), residual
                ).index)[0], str(ele)
            ]
            for ele, adder, residual in self.new_outputs
        ]

    def setInput(self, inputValues):
        pass

    def step(self, time):
        if self.mode == 'ac':
            self.network.lpf()
            self.network.pf(use_seed=True)
        else:
            self.network.lpf()

    def process_powerflow_ipop(self, network, inputs, outputs):
        new_inputs = []
        new_outputs = []
        for item in inputs:
            ele_name, input_variable = item.split('.')
            assert input_variable in ['P', 'Q'], "Powerflow input variable not valid. Use P, Q to  define variables."
            # check in generators
            if ele_name in list(network.generators.index):
                adder, residual = 'generators', input_variable.lower() + '_set'
            elif ele_name in list(network.loads.index):
                adder, residual = 'loads', input_variable.lower() + '_set'
            elif ele_name in list(network.storage_units.index):
                adder, residual = 'storage_units', input_variable.lower() + '_set'
            else:
                raise SimulatorElementNotFoundError(
                    f"Only Generator, load, and storage P, Q inputs are supported."
                    f" Could not find '{ele_name}' in network '{self.network_name}'."
                )
            new_inputs.append((ele_name, adder, residual))
#                print('done input analysing')

        for item in outputs:
            ele_name, output_variable = item.split('.')
            assert output_variable in [
                'P', 'Q', 'V', 'Va'], "Powerflow output variable not valid. Use P, Q, V, Va to  define variables."

            if output_variable == 'V':
                x = 'v_mag_pu'
            elif output_variable == 'Va':
                x = 'v_ang'

            if ele_name in list(network.generators.index):
                adder, residual = 'generators_t', output_variable.lower()
            elif ele_name in list(network.loads.index):
                adder, residual = 'loads_t', output_variable.lower()
            elif ele_name in list(network.buses.index):
                adder, residual = 'buses_t', x
            elif ele_name in list(network.storage_units.index):
                adder, residual = 'storage_units_t', output_variable.lower()  # was: input_variable (from prior loop)
            else:
                raise SimulatorElementNotFoundError(
                    f"Only Generator, load, storage, and bus P, Q outputs are supported."
                    f" Could not find '{ele_name}' in network '{self.network_name}'."
                )
            new_outputs.append((ele_name, adder, residual))
#                print('done output analysing')

        return new_inputs, new_outputs

    def cleanup(self):
        del self.network

    # Backward compatibility alias
    cleanUp = cleanup
