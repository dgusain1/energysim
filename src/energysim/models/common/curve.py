# -*- coding: utf-8 -*-
"""
Empirical Curve and Lookup Table Utilities for energysim Models.

Provides 1D and 2D interpolation with automatic bounds clamping,
scaling, offset, and extrapolation guards to guarantee numerical stability.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


class Curve1D:
    """
    1D empirical lookup curve with clamping and extrapolation protection.

    Parameters
    ----------
    x : Sequence[float]
        Monotonically increasing independent variable coordinates (e.g. load ratio, temperature, SoC).
    y : Sequence[float]
        Dependent variable values (e.g. heat rate, efficiency, OCV).
    clamp : bool, default True
        If True, queries outside [min(x), max(x)] are clamped to y[0] and y[-1]
        instead of extrapolating unphysically.
    min_val : float, optional
        Hard minimum floor for output values (e.g. efficiency cannot be negative).
    max_val : float, optional
        Hard maximum ceiling for output values (e.g. efficiency cannot exceed 1.0).
    name : str, optional
        Descriptive label for the curve (for diagnostics and plotting).
    """

    def __init__(
        self,
        x: Sequence[float],
        y: Sequence[float],
        clamp: bool = True,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        name: str = "",
    ):
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if len(x_arr) != len(y_arr):
            raise ValueError(f"Length mismatch: x has {len(x_arr)} points, y has {len(y_arr)} points.")
        if len(x_arr) < 2:
            raise ValueError("Curve1D requires at least 2 points.")

        # Ensure x is sorted
        order = np.argsort(x_arr)
        self.x = x_arr[order]
        self.y = y_arr[order]
        self.clamp = clamp
        self.min_val = min_val
        self.max_val = max_val
        self.name = name

    def evaluate(self, val: Union[float, Sequence[float]]) -> Union[float, np.ndarray]:
        """Interpolate the curve at val."""
        is_scalar = np.isscalar(val)
        q = np.atleast_1d(val).astype(float)

        if self.clamp:
            q_eval = np.clip(q, self.x[0], self.x[-1])
        else:
            q_eval = q

        result = np.interp(q_eval, self.x, self.y)

        if self.min_val is not None:
            result = np.maximum(result, self.min_val)
        if self.max_val is not None:
            result = np.minimum(result, self.max_val)

        return float(result[0]) if is_scalar else result

    def __call__(self, val: Union[float, Sequence[float]]) -> Union[float, np.ndarray]:
        return self.evaluate(val)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize curve points and configuration to dictionary."""
        return {
            "name": self.name,
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "clamp": self.clamp,
            "min_val": self.min_val,
            "max_val": self.max_val,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Curve1D:
        """Create Curve1D from dictionary."""
        return cls(
            x=data["x"],
            y=data["y"],
            clamp=data.get("clamp", True),
            min_val=data.get("min_val"),
            max_val=data.get("max_val"),
            name=data.get("name", ""),
        )


class Surface2D:
    """
    2D bilinear lookup surface for bivariate characteristics.
    (e.g. COP = f(T_sink, T_source), Heat Rate = f(Load, T_ambient)).

    Parameters
    ----------
    x : Sequence[float]
        1D grid values along dimension 1 (e.g. T_source), length M.
    y : Sequence[float]
        1D grid values along dimension 2 (e.g. T_sink), length N.
    z : Sequence[Sequence[float]] or 2D array of shape (M, N)
        Grid values Z[i, j] at (x[i], y[j]).
    clamp : bool, default True
        If True, coordinates outside bounds are clamped to grid limits.
    min_val : float, optional
        Minimum value floor for the output.
    max_val : float, optional
        Maximum value ceiling for the output.
    name : str, optional
        Descriptive label.
    """

    def __init__(
        self,
        x: Sequence[float],
        y: Sequence[float],
        z: Sequence[Sequence[float]],
        clamp: bool = True,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        name: str = "",
    ):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.z = np.asarray(z, dtype=float)

        if self.z.shape != (len(self.x), len(self.y)):
            raise ValueError(
                f"Shape mismatch: z has shape {self.z.shape}, expected ({len(self.x)}, {len(self.y)})."
            )

        self.clamp = clamp
        self.min_val = min_val
        self.max_val = max_val
        self.name = name

    def evaluate(self, x_val: float, y_val: float) -> float:
        """Bilinear interpolation at (x_val, y_val)."""
        x_q = float(x_val)
        y_q = float(y_val)

        if self.clamp:
            x_q = min(max(x_q, self.x[0]), self.x[-1])
            y_q = min(max(y_q, self.y[0]), self.y[-1])

        # Find bounding indices
        i = np.searchsorted(self.x, x_q) - 1
        i = max(0, min(i, len(self.x) - 2))
        j = np.searchsorted(self.y, y_q) - 1
        j = max(0, min(j, len(self.y) - 2))

        x0, x1 = self.x[i], self.x[i + 1]
        y0, y1 = self.y[j], self.y[j + 1]

        xd = (x_q - x0) / (x1 - x0) if x1 != x0 else 0.0
        yd = (y_q - y0) / (y1 - y0) if y1 != y0 else 0.0

        z00 = self.z[i, j]
        z10 = self.z[i + 1, j]
        z01 = self.z[i, j + 1]
        z11 = self.z[i + 1, j + 1]

        # Bilinear interpolation formula
        c0 = z00 * (1.0 - xd) + z10 * xd
        c1 = z01 * (1.0 - xd) + z11 * xd
        result = c0 * (1.0 - yd) + c1 * yd

        if self.min_val is not None:
            result = max(result, self.min_val)
        if self.max_val is not None:
            result = min(result, self.max_val)

        return float(result)

    def __call__(self, x_val: float, y_val: float) -> float:
        return self.evaluate(x_val, y_val)
