# -*- coding: utf-8 -*-
"""Pre-flight validation of a world."""

from __future__ import annotations

import networkx as nx

from ..base import (
    ValidationError, ValidationReport,
)
from ._common import (ITERATIVE_REMOVED_MSG, _COUPLINGS)


class ValidationMixin:
    """Pre-flight validation of a world."""

    def validate(self, raise_on_error: bool = False) -> ValidationReport:
        """
        Pre-flight semantic and graph validation of the co-simulation world.

        Checks:
          1. At least one simulator registered.
          2. Step size > 0 for all simulators and divisibility with the macro step;
             the simulation span must be an integer multiple of the macro step.
          3. Connection endpoints exist and are valid.
          4. Read-only simulators (CSV, signal) are not targets of inputs.
          5. Adapters with variable introspection are verified for variable name mismatches.
          6. Algebraic loops / cycles are identified with coupling strategy advice.

        Parameters
        ----------
        raise_on_error : bool, default False
            If True and validation errors are found, raises ValidationError.

        Returns
        -------
        ValidationReport
            Detailed report containing is_valid, errors, and warnings.
        """
        errors = []
        warnings_list = []

        # 1. Simulators checks
        if not self.simulator_dict:
            errors.append("No simulators registered in the world.")
            report = ValidationReport(is_valid=False, errors=errors, warnings=warnings_list)
            if raise_on_error:
                raise ValidationError(str(report))
            return report

        if self.stop_time < self.start_time:
            errors.append(f"stop_time ({self.stop_time}) must be >= start_time ({self.start_time}).")

        if self.coupling not in _COUPLINGS:
            errors.append(ITERATIVE_REMOVED_MSG if self.coupling == 'iterative'
                          else f"Unknown coupling '{self.coupling}'.")

        step_ok = True
        for sim_name, entry in self.simulator_dict.items():
            if entry.step_size is None or entry.step_size <= 0:
                errors.append(f"Simulator '{sim_name}' has invalid step_size: {entry.step_size} <= 0.")
                step_ok = False

        macro = None
        if self.exchange is not None and self.exchange < 0:
            errors.append(f"Macro step size t_macro ({self.exchange}) must be > 0.")
        elif step_ok:
            try:
                macro = self._macro_step()
            except ValueError as exc:
                errors.append(str(exc))

        if macro is not None:
            for sim_name, entry in self.simulator_dict.items():
                ratio = macro / entry.step_size
                if abs(ratio - round(ratio)) > 1e-6:
                    warnings_list.append(
                        f"Simulator '{sim_name}' step_size ({entry.step_size}s) does not divide "
                        f"macro step t_macro ({macro}s) evenly (ratio={ratio:.4f}). "
                        f"This may lead to time misalignment."
                    )
            if self.stop_time >= self.start_time:
                try:
                    self._compute_total_steps(macro)
                except ValueError as exc:
                    errors.append(str(exc))

        # 2. Connection checks
        read_only_types = {'csv', 'signal'}
        for conn in self._parsed_connections:
            output_, input_ = conn[0], conn[1]

            # check outputs
            if isinstance(output_, tuple) and isinstance(output_[0], tuple):
                outputs_to_check = output_
            else:
                outputs_to_check = [output_]

            for ou_sim, ou_var, ou_key in outputs_to_check:
                if ou_sim not in self.simulator_dict:
                    errors.append(f"Source simulator '{ou_sim}' in connection '{ou_key}' is not registered.")
                else:
                    adapter = self.simulator_dict[ou_sim].adapter
                    avail = adapter.get_available_variables()
                    all_vars = set(avail.get('inputs', [])) | set(avail.get('outputs', []))
                    if all_vars and ou_var not in all_vars:
                        warnings_list.append(
                            f"Output variable '{ou_var}' in simulator '{ou_sim}' not found in known variables."
                        )

            # check inputs
            if isinstance(input_, list):
                inputs_to_check = input_
            else:
                inputs_to_check = [input_]

            for in_sim, in_var in inputs_to_check:
                if in_sim not in self.simulator_dict:
                    errors.append(f"Target simulator '{in_sim}' in connection to '{in_sim}.{in_var}' is not registered.")
                else:
                    entry = self.simulator_dict[in_sim]
                    if entry.sim_type in read_only_types:
                        warnings_list.append(
                            f"Simulator '{in_sim}' is read-only (type='{entry.sim_type}'). "
                            f"Targeted input '{in_var}' will be ignored."
                        )
                    adapter = entry.adapter
                    avail = adapter.get_available_variables()
                    all_vars = set(avail.get('inputs', [])) | set(avail.get('outputs', []))
                    if all_vars and in_var not in all_vars:
                        warnings_list.append(
                            f"Input variable '{in_var}' in simulator '{in_sim}' not found in known variables."
                        )

        # 3. Topology & Coupling advice
        if self._dep_graph is not None and not nx.is_directed_acyclic_graph(self._dep_graph):
            sccs = [scc for scc in nx.strongly_connected_components(self._dep_graph) if len(scc) > 1]
            if self.coupling == 'gauss-seidel':
                how = ("Gauss-Seidel cannot resolve cyclic feedback without delay; inside the loop a "
                       "one macro-step (Jacobi) delay is used.")
            else:
                how = "Jacobi coupling resolves this by introducing a one macro-step delay."
            warnings_list.append(
                f"Algebraic feedback loop detected among simulators {sccs}. {how} "
                f"To reduce the coupling error, reduce t_macro or merge the coupled components "
                f"into a single simulator."
            )

        is_valid = len(errors) == 0
        report = ValidationReport(is_valid=is_valid, errors=errors, warnings=warnings_list)
        if raise_on_error and not is_valid:
            raise ValidationError(str(report))
        return report

