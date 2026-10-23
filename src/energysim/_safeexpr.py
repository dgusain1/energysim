# -*- coding: utf-8 -*-
"""
Safe expression evaluation and structured signal specifications.

String signals (e.g. loaded from YAML) are **never** passed to ``eval``.
They are parsed with :mod:`ast` and only a small, whitelisted subset of
Python expression syntax is accepted:

* numeric literals (int / float), the variable ``t`` and the constants
  ``pi`` and ``e``;
* binary operators ``+ - * / ** %`` and unary ``+ -``;
* comparisons (``< <= > >= == !=``, chained), ``and`` / ``or`` / ``not``;
* conditional expressions ``a if cond else b``;
* calls to ``sin cos tan exp log sqrt abs min max floor ceil``;
* a top-level list / tuple literal of the above (parsed here, but ``world.add_signal`` rejects
  signals returning more than one value: a signal has the single output ``y``).

An optional ``lambda t:`` prefix is accepted for backwards compatibility
with 2.x scenario files (``"lambda t: 5*t"``); it is stripped and the body
is parsed with the same restricted grammar.

Structured specifications are also supported (see :func:`build_signal`)::

    {type: constant, value: 5}
    {type: sine, amplitude: 1, frequency: 0.01, phase: 0, offset: 0}   # or period:
    {type: step, time: 100, before: 0, after: 1}
    {type: ramp, start_time: 0, end_time: 100, start_value: 0, end_value: 10}
    {type: piecewise, breakpoints: [[0, 0], [60, 5], [120, 2]], interpolation: linear}
    {type: expression, expr: "3*sin(t)"}

Every signal callable returned here yields a list (``[value]`` or the
list of values for a list expression), as required by the signal adapter.
Structured ``step`` / ``ramp`` / ``piecewise`` signals expose their
discontinuity times as ``.breakpoints`` (used as event hints).

:func:`compile_condition` compiles boolean trigger conditions over simulator
variables with the same whitelist, e.g. ``"grid.Bus1.vm_pu < 0.95"``:
dotted attribute-like names (``sim.var`` / ``sim.elem.attr``) and quoted
names (``"'fmu.der(x)' > 0"``) are variable references.
"""
from __future__ import annotations

import ast
import bisect
import math
import operator
import re
from typing import Any, Callable, Dict, List

__all__ = [
    "UnsafeExpressionError",
    "compile_expression",
    "compile_condition",
    "build_signal",
]


class UnsafeExpressionError(ValueError):
    """Raised when an expression uses syntax outside the safe whitelist."""


_MAX_EXPR_LEN = 2000
_MAX_POW_EXPONENT = 1000

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

_CMP_OPS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}

_FUNCS: Dict[str, Callable[..., Any]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "abs": abs,
    "min": min,
    "max": max,
    "floor": math.floor,
    "ceil": math.ceil,
}

_CONSTS = {"pi": math.pi, "e": math.e}

_LAMBDA_PREFIX = re.compile(r"^\s*lambda\s+t\s*:\s*")


def _safe_pow(a, b):
    if abs(b) > _MAX_POW_EXPONENT:
        raise UnsafeExpressionError(f"Exponent {b!r} too large in signal expression.")
    # Float power: huge results raise OverflowError instead of building
    # gigantic integers.
    return operator.pow(float(a), float(b))


def _dotted(node: ast.AST):
    """``a.b.c`` attribute chain on a plain name -> ``'a.b.c'`` (else None)."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _check(node: ast.AST, src: str, top: bool = False, names=None) -> None:
    """Recursively verify that *node* only contains whitelisted syntax.

    List / tuple literals are only accepted as the top-level node.  With
    *names* (a list) the expression is a condition: free names, dotted
    attribute chains and quoted strings are variable references collected
    into *names*.
    """
    if isinstance(node, ast.Expression):
        _check(node.body, src, top=True, names=names)
    elif names is not None and isinstance(node, ast.Constant) and isinstance(node.value, str):
        if node.value not in names:
            names.append(node.value)
    elif names is not None and isinstance(node, ast.Attribute):
        dotted = _dotted(node)
        if dotted is None:
            raise UnsafeExpressionError(f"Attribute access on an expression is not allowed in {src!r}.")
        if dotted not in names:
            names.append(dotted)
    elif names is not None and isinstance(node, ast.Name) and node.id != "t" and node.id not in _CONSTS:
        if node.id not in names:
            names.append(node.id)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpressionError(
                f"Only numeric literals are allowed in signal expressions, got {node.value!r} in {src!r}."
            )
    elif isinstance(node, ast.Name):
        if node.id != "t" and node.id not in _CONSTS:
            raise UnsafeExpressionError(
                f"Unknown name '{node.id}' in signal expression {src!r}. "
                f"Allowed: t, {', '.join(sorted(_CONSTS))}."
            )
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BIN_OPS and not isinstance(node.op, ast.Pow):
            raise UnsafeExpressionError(f"Operator {type(node.op).__name__} not allowed in {src!r}.")
        _check(node.left, src, names=names)
        _check(node.right, src, names=names)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPS:
            raise UnsafeExpressionError(f"Operator {type(node.op).__name__} not allowed in {src!r}.")
        _check(node.operand, src, names=names)
    elif isinstance(node, ast.BoolOp):
        for v in node.values:
            _check(v, src, names=names)
    elif isinstance(node, ast.Compare):
        for op in node.ops:
            if type(op) not in _CMP_OPS:
                raise UnsafeExpressionError(f"Comparison {type(op).__name__} not allowed in {src!r}.")
        _check(node.left, src, names=names)
        for c in node.comparators:
            _check(c, src, names=names)
    elif isinstance(node, ast.IfExp):
        _check(node.test, src, names=names)
        _check(node.body, src, names=names)
        _check(node.orelse, src, names=names)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise UnsafeExpressionError(
                f"Function call not allowed in signal expression {src!r}. "
                f"Allowed functions: {', '.join(sorted(_FUNCS))}."
            )
        if node.keywords:
            raise UnsafeExpressionError(f"Keyword arguments not allowed in {src!r}.")
        for a in node.args:
            if isinstance(a, ast.Starred):
                raise UnsafeExpressionError(f"Star-arguments not allowed in {src!r}.")
            _check(a, src, names=names)
    elif isinstance(node, (ast.List, ast.Tuple)) and top:
        for elt in node.elts:
            if isinstance(elt, ast.Starred):
                raise UnsafeExpressionError(f"Star-expressions not allowed in {src!r}.")
            _check(elt, src, names=names)
    else:
        raise UnsafeExpressionError(
            f"Syntax element '{type(node).__name__}' is not allowed in signal expression {src!r}."
        )


def _eval(node: ast.AST, t: float, env=None) -> Any:
    if env is not None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return env[node.value]
        if isinstance(node, ast.Attribute):
            return env[_dotted(node)]
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return t if node.id == "t" else _CONSTS[node.id]
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, t, env)
        right = _eval(node.right, t, env)
        if isinstance(node.op, ast.Pow):
            return _safe_pow(left, right)
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        return _UNARY_OPS[type(node.op)](_eval(node.operand, t, env))
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = True
            for v in node.values:
                result = _eval(v, t, env)
                if not result:
                    return result
            return result
        result = False
        for v in node.values:
            result = _eval(v, t, env)
            if result:
                return result
        return result
    if isinstance(node, ast.Compare):
        left = _eval(node.left, t, env)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, t, env)
            if not _CMP_OPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return _eval(node.body, t, env) if _eval(node.test, t, env) else _eval(node.orelse, t, env)
    if isinstance(node, ast.Call):
        return _FUNCS[node.func.id](*[_eval(a, t, env) for a in node.args])
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e, t, env) for e in node.elts]
    raise UnsafeExpressionError(f"Unsupported node {type(node).__name__}")  # pragma: no cover


class _CompiledExpression:
    """Callable wrapper ``f(t) -> list`` around a validated AST."""

    def __init__(self, source: str, tree: ast.Expression):
        self.source = source
        self._tree = tree

    def __call__(self, t: float) -> List[Any]:
        val = _eval(self._tree.body, t)
        return val if isinstance(val, list) else [val]

    def __repr__(self) -> str:
        return f"<safe signal expression {self.source!r}>"


def compile_expression(expr: str) -> Callable[[float], List[Any]]:
    """Parse *expr* with the safe whitelist and return ``f(t) -> list``.

    Raises :class:`UnsafeExpressionError` (a ``ValueError``) when the
    expression contains anything outside the whitelist.  Nothing in the
    expression is executed during compilation.
    """
    if not isinstance(expr, str):
        raise TypeError(f"Signal expression must be a string, got {type(expr).__name__}.")
    if len(expr) > _MAX_EXPR_LEN:
        raise UnsafeExpressionError("Signal expression is too long.")
    body = _LAMBDA_PREFIX.sub("", expr, count=1).strip()
    if not body:
        raise UnsafeExpressionError("Empty signal expression.")
    try:
        tree = ast.parse(body, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Invalid signal expression {expr!r}: {exc.msg}") from None
    _check(tree, expr)
    return _CompiledExpression(expr, tree)


class _CompiledCondition:
    """Callable ``f(values, t=0.0) -> bool`` over ``{'sim.var': value}``."""

    def __init__(self, source: str, tree: ast.Expression, names: List[str]):
        self.source = source
        self._tree = tree
        #: variable references in order of appearance (``'sim.var'`` ...)
        self.variables = list(names)

    def __call__(self, values: Dict[str, Any], t: float = 0.0) -> bool:
        missing = [n for n in self.variables if n not in values]
        if missing:
            raise KeyError(f"Condition {self.source!r}: no value for {missing}.")
        return bool(_eval(self._tree.body, t, dict(values)))

    def __repr__(self) -> str:
        return f"<safe condition {self.source!r}>"


def compile_condition(expr: str) -> _CompiledCondition:
    """Compile a boolean condition over simulator variables (safe whitelist, never ``eval``-ed).

    ``"grid.Bus1.vm_pu < 0.95"``, ``"bess.SoC >= 0.9 and t > 3600"``,
    ``"'fmu.der(x)' > 0"`` (quote names that are not attribute-like).
    """
    if not isinstance(expr, str):
        raise TypeError(f"Condition must be a string, got {type(expr).__name__}.")
    if len(expr) > _MAX_EXPR_LEN:
        raise UnsafeExpressionError("Condition expression is too long.")
    body = expr.strip()
    if not body:
        raise UnsafeExpressionError("Empty condition expression.")
    try:
        tree = ast.parse(body, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Invalid condition {expr!r}: {exc.msg}") from None
    if isinstance(tree.body, (ast.List, ast.Tuple)):
        raise UnsafeExpressionError(f"A condition must be a single expression, got a list: {expr!r}.")
    names: List[str] = []
    _check(tree, expr, names=names)
    return _CompiledCondition(expr, tree, names)


# ---------------------------------------------------------------------------
# Structured signal specs
# ---------------------------------------------------------------------------

def _num(spec: Dict[str, Any], *keys: str, default: Any = None) -> float:
    for k in keys:
        if k in spec and spec[k] is not None:
            try:
                return float(spec[k])
            except (TypeError, ValueError):
                raise ValueError(f"Signal spec field '{k}' must be numeric, got {spec[k]!r}.") from None
    if default is None:
        raise ValueError(f"Signal spec of type '{spec.get('type')}' requires field '{keys[0]}'.")
    return float(default)


class _StructuredSignal:
    def __init__(self, spec: Dict[str, Any], fn: Callable[[float], float], breakpoints=()):
        self.spec = dict(spec)
        self._fn = fn
        #: times at which the signal is discontinuous / changes slope (event hints)
        self.breakpoints = sorted(float(b) for b in breakpoints if math.isfinite(float(b)))

    def __call__(self, t: float) -> List[float]:
        return [self._fn(t)]

    def __repr__(self) -> str:
        return f"<signal {self.spec!r}>"


def _piecewise(spec: Dict[str, Any]) -> Callable[[float], float]:
    bps = spec.get("breakpoints") or spec.get("points")
    if not bps:
        raise ValueError("Piecewise signal requires a non-empty 'breakpoints' list of [time, value] pairs.")
    try:
        pts = sorted((float(p[0]), float(p[1])) for p in bps)
    except (TypeError, ValueError, IndexError):
        raise ValueError("Piecewise 'breakpoints' must be a list of [time, value] pairs.") from None
    times = [p[0] for p in pts]
    if len(set(times)) != len(times):
        raise ValueError("Piecewise 'breakpoints' must have unique times.")
    values = [p[1] for p in pts]
    interp = str(spec.get("interpolation", "linear")).lower()
    if interp not in ("linear", "previous", "hold", "step", "zoh"):
        raise ValueError(f"Unknown piecewise interpolation '{interp}' (use 'linear' or 'previous').")

    def fn(t: float) -> float:
        if t <= times[0]:
            return values[0]
        if t >= times[-1]:
            return values[-1]
        i = bisect.bisect_right(times, t) - 1
        if interp != "linear":
            return values[i]
        t0, t1 = times[i], times[i + 1]
        v0, v1 = values[i], values[i + 1]
        return v0 + (v1 - v0) * (t - t0) / (t1 - t0)

    fn.times = list(times)
    return fn


def build_signal(spec: Any) -> Callable[[float], List[Any]]:
    """Build a safe signal callable from a spec.

    *spec* may be a number (constant), a string (safe expression) or a
    dict with a ``type`` key (``constant``, ``sine``, ``step``, ``ramp``,
    ``piecewise`` or ``expression``).  Callables are returned unchanged.
    """
    if callable(spec):
        return spec
    if isinstance(spec, bool):
        raise TypeError("A boolean is not a valid signal specification.")
    if isinstance(spec, (int, float)):
        value = float(spec)
        return _StructuredSignal({"type": "constant", "value": value}, lambda t: value)
    if isinstance(spec, str):
        return compile_expression(spec)
    if not isinstance(spec, dict):
        raise TypeError(
            f"Unsupported signal specification of type {type(spec).__name__}; "
            f"expected a callable, number, expression string or dict."
        )

    kind = str(spec.get("type", "")).lower()
    bps: List[float] = []
    if kind == "constant":
        value = _num(spec, "value")
        fn = lambda t: value  # noqa: E731
    elif kind in ("sine", "sin", "sinusoid"):
        amp = _num(spec, "amplitude", default=1.0)
        off = _num(spec, "offset", default=0.0)
        phase = _num(spec, "phase", default=0.0)
        if "period" in spec:
            period = _num(spec, "period")
            if period <= 0:
                raise ValueError("Sine signal 'period' must be > 0.")
            freq = 1.0 / period
        else:
            freq = _num(spec, "frequency", default=1.0)
        fn = lambda t: off + amp * math.sin(2.0 * math.pi * freq * t + phase)  # noqa: E731
    elif kind == "step":
        t_step = _num(spec, "time", "step_time", "t")
        before = _num(spec, "before", "initial", default=0.0)
        after = _num(spec, "after", "final", default=1.0)
        fn = lambda t: after if t >= t_step else before  # noqa: E731
        bps = [t_step]
    elif kind == "ramp":
        t0 = _num(spec, "start_time", default=0.0)
        v0 = _num(spec, "start_value", default=0.0)
        if "slope" in spec:
            slope = _num(spec, "slope")
            t1 = _num(spec, "end_time", default=float("inf"))
        else:
            t1 = _num(spec, "end_time")
            v1 = _num(spec, "end_value")
            if t1 <= t0:
                raise ValueError("Ramp signal 'end_time' must be greater than 'start_time'.")
            slope = (v1 - v0) / (t1 - t0)

        def fn(t: float) -> float:
            if t <= t0:
                return v0
            return v0 + slope * (min(t, t1) - t0)
        bps = [t0, t1]
    elif kind == "piecewise":
        fn = _piecewise(spec)
        bps = fn.times
    elif kind in ("expression", "expr"):
        expr = spec.get("expr") or spec.get("expression")
        return compile_expression(str(expr) if expr is not None else "")
    else:
        raise ValueError(
            f"Unknown signal type '{spec.get('type')}'. "
            f"Supported: constant, sine, step, ramp, piecewise, expression."
        )
    return _StructuredSignal(spec, fn, bps)
