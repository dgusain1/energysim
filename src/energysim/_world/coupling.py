# -*- coding: utf-8 -*-
"""Coupling strategies (Jacobi / Gauss-Seidel), exchange and extrapolation."""

from __future__ import annotations

from collections import deque



class CouplingMixin:
    """Coupling strategies (Jacobi / Gauss-Seidel), exchange and extrapolation."""

    # ------------------------------------------------------------------
    # Coupling strategies
    # ------------------------------------------------------------------

    def _step_jacobi(self, exec_order, time, local_stop_time, record_all):
        """Jacobi coupling: exchange all, then step all."""
        self.exchange_variable_values(time)
        tmp_dict = {}
        for sim_name in exec_order:
            entry = self.simulator_dict[sim_name]
            tmp_dict[sim_name] = self.step_simulator(sim_name, entry, time, record_all, local_stop_time)
        return tmp_dict

    def _step_gauss_seidel(self, exec_order, time, local_stop_time, record_all):
        """Gauss-Seidel coupling: step in topological order,
        exchanging each simulator's outputs before the next one steps."""
        # For simulators in cyclic SCCs, exchange their inputs first (Jacobi within SCC)
        scc_members = set()
        for scc in self._cycle_sccs:
            scc_members.update(scc)

        # Initial exchange for cyclic simulators (they need stale values to start)
        if scc_members:
            self._exchange_for_sims(time, target_sims=scc_members)

        tmp_dict = {}
        for sim_name in exec_order:
            # For non-cyclic sims: exchange inputs from already-stepped upstream sims
            if sim_name not in scc_members:
                self._exchange_inputs_for(sim_name, time)
            entry = self.simulator_dict[sim_name]
            tmp_dict[sim_name] = self.step_simulator(sim_name, entry, time, record_all, local_stop_time)
        return tmp_dict

    # ------------------------------------------------------------------
    # Gauss-Seidel exchange helpers
    # ------------------------------------------------------------------

    def _exchange_inputs_for(self, target_sim, t):
        """Exchange only those connections whose input side is *target_sim*."""
        for conn in self._parsed_connections:
            output_, input_ = conn[0], conn[1]
            scale = getattr(conn, 'scale', 1.0)
            offset = getattr(conn, 'offset', 0.0)

            # Determine which input sims this connection targets
            if isinstance(input_, list):
                matching = [(in_sim, in_var) for in_sim, in_var in input_
                            if in_sim == target_sim]
                if not matching:
                    continue
            else:
                in_sim_name, in_sim_var = input_
                if in_sim_name != target_sim:
                    continue
                matching = [(in_sim_name, in_sim_var)]

            # Gather output value(s)
            tmp_var = self._gather_output_value(output_, t, getattr(conn, 'source_scales', None))
            per_target = getattr(conn, 'target_conversions', None)
            if per_target:
                targets = input_ if isinstance(input_, list) else [input_]
                for (in_sim_name, in_sim_var), (sc, off) in zip(targets, per_target):
                    if in_sim_name == target_sim:
                        self.simulator_dict[in_sim_name].adapter.set_value([in_sim_var], [tmp_var[0] * sc + off])
                continue
            if scale != 1.0 or offset != 0.0:
                tmp_var = [tmp_var[0] * scale + offset]

            # Set to matching inputs only
            for in_sim_name, in_sim_var in matching:
                self.simulator_dict[in_sim_name].adapter.set_value([in_sim_var], tmp_var)

    def _exchange_for_sims(self, t, target_sims):
        """Exchange connections whose input side is in target_sims set."""
        for sim_name in target_sims:
            self._exchange_inputs_for(sim_name, t)

    def _gather_output_value(self, output_, t, source_scales=None):
        """Read output value(s) from a parsed connection's output side.

        For fan-in connections the outputs are summed; *source_scales* (if
        given) first converts every source into the first source's unit.
        """
        if isinstance(output_, tuple) and isinstance(output_[0], tuple):
            # Fan-in: sum multiple outputs
            tmps = []
            for i, (ou_sim_name, ou_sim_var, ou_key) in enumerate(output_):
                tmp1 = self.simulator_dict[ou_sim_name].adapter.get_value([ou_sim_var], t)
                tmp = self.alter_signal(ou_key, tmp1)
                tmps.append(tmp[0] * source_scales[i] if source_scales else tmp[0])
            return [sum(tmps)]
        else:
            ou_sim_name, ou_sim_var, ou_key = output_
            tmp_var1 = self.simulator_dict[ou_sim_name].adapter.get_value([ou_sim_var], t)
            return self.alter_signal(ou_key, tmp_var1)

    def alter_signal(self, op_, tmp):
        '''
        Applies the ``modify_signal`` option (``[gain]`` or ``[gain, offset]``) to output *op_*.
        '''
        temp_var = tmp[0]
        mod = self.modify_dict.get(op_) if self.modify_dict else None
        if mod is None:
            return [temp_var]
        if len(mod) == 1:
            return [temp_var * mod[0]]
        if len(mod) == 2:
            return [temp_var * mod[0] + mod[1]]
        raise ValueError(f"Invalid modify_signal entry for '{op_}': {mod!r} (expected [gain] or [gain, offset]).")

    def exchange_variable_values(self, t):
        """
        Helper function for exchanging the variable values between simulators at particular time.
        Supports zero-order hold (default) and linear extrapolation, plus per-connection scaling.
        """
        for conn in self._parsed_connections:
            output_, input_ = conn[0], conn[1]
            scale = getattr(conn, 'scale', 1.0)
            offset = getattr(conn, 'offset', 0.0)

            # --- gather output value(s) ---
            tmp_var = self._gather_output_value(output_, t, getattr(conn, 'source_scales', None))

            per_target = getattr(conn, 'target_conversions', None)
            if per_target:
                # fan-out to receivers with different units: extrapolate the raw
                # value, then convert into each receiver's unit
                if self.extrapolation == 'linear':
                    tmp_var = self._apply_extrapolation(output_, t, tmp_var)
                for (in_sim_name, in_sim_var), (sc, off) in zip(input_, per_target):
                    self.simulator_dict[in_sim_name].adapter.set_value([in_sim_var], [tmp_var[0] * sc + off])
                continue

            # --- apply scale and offset ---
            if scale != 1.0 or offset != 0.0:
                tmp_var = [tmp_var[0] * scale + offset]

            # --- apply linear extrapolation if enabled ---
            if self.extrapolation == 'linear':
                tmp_var = self._apply_extrapolation(output_, t, tmp_var)

            # --- distribute to input(s) ---
            if isinstance(input_, list):
                for in_sim_name, in_sim_var in input_:
                    self.simulator_dict[in_sim_name].adapter.set_value([in_sim_var], tmp_var)
            else:
                in_sim_name, in_sim_var = input_
                self.simulator_dict[in_sim_name].adapter.set_value([in_sim_var], tmp_var)

    def _apply_extrapolation(self, output_, t, current_value):
        """Apply linear extrapolation using history of exchanged values."""
        # Build a hashable key for this output
        if isinstance(output_, tuple) and isinstance(output_[0], tuple):
            hist_key = tuple((s, v) for s, v, _ in output_)
        else:
            hist_key = (output_[0], output_[1])

        # Update history
        if hist_key not in self._exchange_history:
            self._exchange_history[hist_key] = deque(maxlen=2)
        self._exchange_history[hist_key].append((t, current_value[0]))

        history = self._exchange_history[hist_key]
        if len(history) >= 2:
            t_prev, v_prev = history[0]
            t_curr, v_curr = history[1]
            dt = t_curr - t_prev
            if abs(dt) > 1e-15:
                slope = (v_curr - v_prev) / dt
                # Extrapolate forward by one macro-step
                dt_next = getattr(self, '_current_dt', None) or self.macro_tstep
                v_extrap = v_curr + slope * dt_next
                return [v_extrap]
        # Fall back to zero-order hold if not enough history
        return current_value

    def get_step_time(self, step_size, local_stop_time, current_time, final_tStep, int_start_time):
        """
        Helper function to obtain the variable time step for specified FMUs.
        """
        if current_time - int_start_time < 0.01 or local_stop_time - current_time < 0.01:
            new_step_size = 0.001
        else:
            new_step_size = final_tStep - 0.01
        return new_step_size

