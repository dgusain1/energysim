# -*- coding: utf-8 -*-
"""Connection parsing, validation, units and the dependency graph."""

from __future__ import annotations

import re
import warnings
import networkx as nx

from ..units import get_unit_conversion, is_known_unit, parse_unit
from ..base import (
    ConnectionSpecError,
    ParsedConnection,
)
from ._common import (logger)


def known_variables(avail):
    """Union of ``inputs``, ``outputs`` and (when present) ``all`` of ``get_available_variables()``."""
    avail = avail or {}
    return set(avail.get('inputs', [])) | set(avail.get('outputs', [])) | set(avail.get('all', []) or [])


def variable_is_known(var_name, all_vars):
    """True if *var_name* (optionally indexed ``x[3]`` / aggregated ``total_x``) is in *all_vars*."""
    if var_name in all_vars:
        return True
    base_var = re.sub(r'\[\d+\]$', '', var_name)       # 'soc[0]' -> 'soc'
    if base_var in all_vars:
        return True
    agg_base = re.sub(r'^(total_|mean_|sum_|min_|max_)', '', var_name)   # 'total_soc' -> 'soc'
    return agg_base in all_vars or re.sub(r'\[\d+\]$', '', agg_base) in all_vars


class ConnectionsMixin:
    """Connection parsing, validation, units and the dependency graph."""

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def add_connections(self, connections=None):
        '''
        Adds connections between simulators. Connections are
        specified as dictionary with keys as sending variables,
        and values as receiving variables (or a list of dicts with
        ``from``/``to``/``scale``/``offset``/``from_unit``/``to_unit``).

        Units: the unit of each end is resolved (simulator ``units`` /
        ``power_unit`` for power-like names -> the component's ``units`` ->
        ``from_unit``/``to_unit`` on the connection -> name suffix such as
        ``_mw`` / ``_degc`` -> the world's unit-system default for the
        quantity the name denotes) and the value is converted into the
        **receiver's** unit.  An end whose unit stays unspecified is passed
        through unscaled with a warning (an error with
        ``unit_policy='strict'``).  Unknown units and incompatible dimensions
        of explicitly known units raise ``ValueError``; incompatible *guessed*
        units are passed through with a warning.  A ``scale`` given without
        any unit key disables automatic conversion.  Inspect the result with
        :meth:`connection_units` / :meth:`conversion_lines`.
        '''
        if connections is None:
            connections = {}

        if not self.simulator_dict:
            raise ValueError("Cannot make connections with no simulators defined.")
        if not isinstance(connections, (dict, list)):
            raise TypeError(f"connections must be a dict or a list of dicts, got {type(connections).__name__}.")
        self.simulator_connections = connections

        if isinstance(connections, list):
            raw_connections = []
            for item in connections:
                if not isinstance(item, dict):
                    raise ConnectionSpecError(f"Connection list items must be dicts, got {item!r}.")
                raw_connections.append(item)
        else:
            raw_connections = []
            for k, v in connections.items():
                if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], dict):
                    item = {'from': k, 'to': v[0]}
                    item.update({kk: vv for kk, vv in v[1].items() if kk not in ('to', 'target')})
                    raw_connections.append(item)
                elif isinstance(v, dict):
                    item = {'from': k, 'to': v.get('to') or v.get('target')}
                    item.update({kk: vv for kk, vv in v.items() if kk not in ('to', 'target')})
                    raw_connections.append(item)
                else:
                    raw_connections.append({'from': k, 'to': v})

        # Pre-parse connections once so exchange_variable_values() doesn't
        # re-split strings on every time step.
        parsed_connections = []
        for item in raw_connections:
            output_key = item.get('from') or item.get('source')
            input_key = item.get('to') or item.get('target')
            user_scale = float(item.get('scale', 1.0))
            user_offset = float(item.get('offset', 0.0))
            from_unit = item.get('from_unit') or item.get('unit')
            to_unit = item.get('to_unit') or item.get('unit')

            # --- parse output side ---
            if isinstance(output_key, str):
                ou_sim, ou_var = self._split_var(output_key)
                parsed_output = (ou_sim, ou_var, output_key)
                first_ou_sim, first_ou_var = ou_sim, ou_var
            elif isinstance(output_key, (tuple, list)) and output_key:
                parsed_output = tuple(
                    (self._split_var(k)[0], self._split_var(k)[1], k)
                    for k in output_key
                )
                first_ou_sim, first_ou_var = parsed_output[0][0], parsed_output[0][1]
            else:
                raise ConnectionSpecError(f"Invalid connection source {output_key!r}.")
            # --- parse input side ---
            if isinstance(input_key, str):
                in_sim, in_var = self._split_var(input_key)
                parsed_input = (in_sim, in_var)
                first_in_sim, first_in_var = in_sim, in_var
            elif isinstance(input_key, (list, tuple)) and input_key:
                parsed_input = [
                    (self._split_var(k)[0], self._split_var(k)[1])
                    for k in input_key
                ]
                first_in_sim, first_in_var = parsed_input[0][0], parsed_input[0][1]
            else:
                raise ConnectionSpecError(f"Invalid connection target {input_key!r} (from {output_key!r}).")

            explicit_units = any(k in item for k in ('from_unit', 'to_unit', 'unit'))
            sources = ([(o[0], o[1]) for o in parsed_output] if isinstance(parsed_output[0], tuple)
                       else [(first_ou_sim, first_ou_var)])
            targets = parsed_input if isinstance(parsed_input, list) else [parsed_input]
            if 'scale' in item and not explicit_units:
                # A manual scale without units overrides automatic conversion.
                infos = [{'from_unit': None, 'to_unit': None, 'from_source': 'manual',
                          'to_source': 'manual', 'scale': 1.0, 'offset': 0.0, 'note': None}
                         for _ in targets]
                source_scales = None
            else:
                try:
                    for u in {from_unit, to_unit} - {None, ''}:
                        parse_unit(u)       # connection-level units are always validated
                    infos = [self._conversion_for(sources[0], tgt, from_unit, to_unit) for tgt in targets]
                    source_scales = self._fan_in_scales(sources, from_unit)
                except ValueError as exc:
                    raise ValueError(f"Connection {output_key!r} -> {input_key!r}: {exc}") from None
                if self.unit_policy == 'strict':
                    ends = [(sources[0], infos[0]['from_unit'])] + [(t, i['to_unit']) for t, i in zip(targets, infos)]
                    missing = sorted({f"{sv[0]}.{sv[1]}" for sv, u in ends if u is None})
                    if missing:
                        raise ValueError(
                            f"Connection {output_key!r} -> {input_key!r}: unit of {', '.join(missing)} is "
                            f"unspecified (unit_policy='strict'). Declare it via the simulator's "
                            f"units={{...}}, the component's units, or from_unit/to_unit on the connection.")

            first = infos[0]
            conversions = [(i['scale'] * user_scale, i['offset'] * user_scale + user_offset) for i in infos]
            target_conversions = conversions if len(set(conversions)) > 1 else None
            conn = ParsedConnection(
                parsed_output, parsed_input, conversions[0][0], conversions[0][1],
                first['from_unit'], first['to_unit'], user_scale=user_scale, user_offset=user_offset,
                from_source=first['from_source'], to_source=first['to_source'],
                unit_scale=first['scale'], unit_offset=first['offset'],
                target_conversions=target_conversions, source_scales=source_scales,
                explicit_units=explicit_units)
            conn._unit_info = infos
            conn._spec_units = (item.get('from_unit') or item.get('unit'),
                                item.get('to_unit') or item.get('unit'))
            parsed_connections.append(conn)
        self._parsed_connections = parsed_connections

        # Run validation checks
        self._validate_connections()

        # Build dependency graph and compute execution order
        self._build_dependency_graph()
        self._compute_execution_order()

        logger.info("Added %i connections between simulators to the World", len(self._parsed_connections))
        self._log_conversions('debug')
        warned = getattr(self, '_unit_warned', None)
        if warned is None:
            warned = self._unit_warned = set()
        for msg in self._unit_warnings():
            if msg not in warned:
                warned.add(msg)
                logger.warning(msg)

    def _fan_in_scales(self, sources, from_unit):
        """Factors converting each fan-in source into the first source's unit (None if all equal)."""
        if len(sources) < 2:
            return None
        u0, _ = self.resolve_variable_unit(sources[0][0], sources[0][1], from_unit)
        scales = [1.0]
        for sim, var in sources[1:]:
            u, _ = self.resolve_variable_unit(sim, var, from_unit)
            sc = 1.0
            if u0 and u and u != u0 and is_known_unit(u) and is_known_unit(u0):
                sc, off = get_unit_conversion(u, u0)
                if off:
                    raise ValueError(f"fan-in sum of {u!r} and {u0!r} needs an offset conversion; "
                                     f"declare a common unit.")
            scales.append(sc)
        return None if all(x == 1.0 for x in scales) else scales

    # ------------------------------------------------------------------
    # Connection validation
    # ------------------------------------------------------------------

    @staticmethod
    def _split_var(dotted_name):
        """Split 'sim_name.var.sub' → ('sim_name', 'var.sub')."""
        if not isinstance(dotted_name, str):
            raise ConnectionSpecError(f"Connection endpoint must be a string, got {dotted_name!r}.")
        parts = dotted_name.split('.', 1)
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ConnectionSpecError(
                f"Connection string '{dotted_name}' is malformed — expected "
                f"format 'simulator_name.variable_name'."
            )
        return parts[0], parts[1]

    def _validate_connections(self):
        """Validate all parsed connections for correctness."""
        known_sims = set(self.simulator_dict.keys())
        read_only_types = {'csv', 'signal'}

        for conn in self._parsed_connections:
            output_, input_ = conn[0], conn[1]
            # --- validate output side ---
            if isinstance(output_, tuple) and isinstance(output_[0], tuple):
                # fan-in
                for ou_sim_name, ou_sim_var, ou_key in output_:
                    self._validate_sim_exists(ou_sim_name, known_sims, ou_key)
                    self._validate_variable(ou_sim_name, ou_sim_var, direction='output')
            else:
                ou_sim_name, ou_sim_var, ou_key = output_
                self._validate_sim_exists(ou_sim_name, known_sims, ou_key)
                self._validate_variable(ou_sim_name, ou_sim_var, direction='output')

            # --- validate input side ---
            if isinstance(input_, list):
                for in_sim_name, in_sim_var in input_:
                    self._validate_sim_exists(in_sim_name, known_sims,
                                              f"{in_sim_name}.{in_sim_var}")
                    self._validate_variable(in_sim_name, in_sim_var, direction='input')
                    self._warn_if_read_only(in_sim_name, in_sim_var, read_only_types)
            else:
                in_sim_name, in_sim_var = input_
                self._validate_sim_exists(in_sim_name, known_sims,
                                          f"{in_sim_name}.{in_sim_var}")
                self._validate_variable(in_sim_name, in_sim_var, direction='input')
                self._warn_if_read_only(in_sim_name, in_sim_var, read_only_types)

    def _validate_sim_exists(self, sim_name, known_sims, connection_str):
        """Raise ConnectionSpecError if simulator doesn't exist."""
        if sim_name not in known_sims:
            raise ConnectionSpecError(
                f"Simulator '{sim_name}' referenced in connection "
                f"'{connection_str}' does not exist. "
                f"Available simulators: {sorted(known_sims)}"
            )

    def _validate_variable(self, sim_name, var_name, direction='output'):
        """Emit a warning if a variable name doesn't match known variables."""
        entry = self.simulator_dict[sim_name]
        adapter = entry.adapter
        get_vars = getattr(adapter, 'get_available_variables', None)
        avail = get_vars() if callable(get_vars) else {}
        all_vars = known_variables(avail)
        if not all_vars:
            # Adapter doesn't expose variable metadata — skip validation
            return
        if not variable_is_known(var_name, all_vars):
            warnings.warn(
                f"Variable '{var_name}' not found in simulator '{sim_name}' "
                f"declared variables. Connection may fail at runtime. "
                f"Known variables: {sorted(all_vars)[:20]}{'...' if len(all_vars) > 20 else ''}",
                UserWarning, stacklevel=4
            )

    def _warn_if_read_only(self, sim_name, var_name, read_only_types):
        """Warn if connecting to a read-only simulator."""
        entry = self.simulator_dict[sim_name]
        if entry.sim_type in read_only_types:
            warnings.warn(
                f"Simulator '{sim_name}' is read-only (type='{entry.sim_type}'). "
                f"Connection to '{var_name}' will be silently ignored.",
                UserWarning, stacklevel=4
            )

    # ------------------------------------------------------------------
    # Dependency graph & topological ordering
    # ------------------------------------------------------------------

    def _build_dependency_graph(self):
        """Build a directed graph of simulator dependencies from connections."""
        self._dep_graph = nx.DiGraph()
        # Add all simulators as nodes (including unconnected ones)
        for sim_name in self.simulator_dict:
            self._dep_graph.add_node(sim_name)

        for conn in self._parsed_connections:
            output_, input_ = conn[0], conn[1]
            if isinstance(output_, tuple) and isinstance(output_[0], tuple):
                ou_sims = [p[0] for p in output_]
            else:
                ou_sims = [output_[0]]

            if isinstance(input_, list):
                in_sims = [p[0] for p in input_]
            else:
                in_sims = [input_[0]]

            for o_sim in ou_sims:
                for i_sim in in_sims:
                    self._dep_graph.add_edge(o_sim, i_sim)

    def _compute_execution_order(self):
        """Compute execution order via topological sort.

        For DAGs, produces a dependency-respecting order.
        For cyclic graphs, uses condensation (SCCs collapsed to super-nodes)
        to get a partial topological order, with simulators inside each SCC
        kept in registration order. Emits a warning for each detected cycle.
        """
        G = self._dep_graph
        if nx.is_directed_acyclic_graph(G):
            self._execution_order = list(nx.topological_sort(G))
            self._cycle_sccs = []
        else:
            # Find all strongly connected components (cycles)
            sccs = list(nx.strongly_connected_components(G))
            self._cycle_sccs = [scc for scc in sccs if len(scc) > 1]
            for scc in self._cycle_sccs:
                warnings.warn(
                    f"Algebraic loop detected involving simulators: {sorted(scc)}. "
                    f"Using Jacobi coupling (one-step delay) for these cyclic dependencies. "
                    f"Reduce t_macro or merge the coupled components to limit the delay error.",
                    UserWarning, stacklevel=3
                )
            condensation = nx.condensation(G)
            topo_order_condensed = list(nx.topological_sort(condensation))
            reg_order = list(self.simulator_dict.keys())
            self._execution_order = []
            for super_node in topo_order_condensed:
                members = condensation.nodes[super_node]['members']
                sorted_members = sorted(members, key=lambda x: reg_order.index(x))
                self._execution_order.extend(sorted_members)

        logger.info("Execution order: %s", self._execution_order)

    def get_fmu_name(self, name):
        if type(name).__name__ == 'str':
            return name.split('.')[0]
        if type(name).__name__ == 'tuple':
            return [x.split('.')[0] for x in name]

    def plot(self, plot_edge_labels=False, node_size=300, node_color='r'):
        """
        Plots approximate graph diagram for the energysim network.
        """
        import matplotlib.pyplot as plt
        # Build graph if not already built
        if self._dep_graph is None:
            self._build_dependency_graph()
        self.G = self._dep_graph.copy()
        # Add edge labels for visualization
        conns = self.simulator_connections if isinstance(self.simulator_connections, dict) else {}
        for key, value in conns.items():
            if isinstance(key, str) and isinstance(value, str):
                n1 = key.split('.')[0]
                n2 = value.split('.')[0]
                self.G.edges[n1, n2]['name'] = key + ' to ' + value
            elif isinstance(key, str) and isinstance(value, tuple):
                n1 = key.split('.')[0]
                for v in value:
                    if isinstance(v, str):
                        n2 = v.split('.')[0]
                        self.G.edges[n1, n2]['name'] = key + ' to ' + v
            elif isinstance(key, tuple) and isinstance(value, str):
                n2 = value.split('.')[0]
                for k in key:
                    n1 = k.split('.')[0]
                    self.G.edges[n1, n2]['name'] = k + ' to ' + value
        pos = nx.spring_layout(self.G)
        nx.draw_networkx_nodes(self.G, pos, node_size=node_size, node_color=node_color, alpha=0.8)
        nx.draw_networkx_edges(self.G, pos, alpha=0.8, edge_color='b')
        nx.draw_networkx_labels(self.G, pos)
        if plot_edge_labels:
            edge_labels = nx.get_edge_attributes(self.G, 'name')
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)
        plt.tick_params(axis="x", which="both", bottom=False, top=False)
        plt.tick_params(axis="y", which="both", bottom=False, top=False)
        plt.tight_layout()
        plt.show()

