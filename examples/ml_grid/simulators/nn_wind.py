# -*- coding: utf-8 -*-
"""Neural-network wind generation predictor — external simulator for energysim.

Loads pre-trained weights from wind_weights.npz and evaluates
a small feedforward NN: [4] -> [16 ReLU] -> [8 ReLU] -> [1].
"""

from energysim.base import SimulatorAdapter
import numpy as np
from pathlib import Path


class external_simulator(SimulatorAdapter):
    """Predicts wind generation (MW) from time-of-day and weather features."""

    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs = inputs or []
        self.outputs = outputs or []

        # Load pre-trained weights
        w = np.load(Path(__file__).resolve().parent.parent / "wind_weights.npz")
        self.W1, self.b1 = w["W1"], w["b1"]
        self.W2, self.b2 = w["W2"], w["b2"]
        self.W3, self.b3 = w["W3"], w["b3"]
        self.input_mean = w["input_mean"]
        self.input_std = w["input_std"]

        # State / I/O variables
        self.hour_sin = 0.0
        self.hour_cos = 1.0
        self.wind_speed = 5.0
        self.temperature = 10.0
        self.P_gen = 0.0

    def _forward(self, x):
        x = (x - self.input_mean) / (self.input_std + 1e-8)
        a1 = np.maximum(0, x @ self.W1 + self.b1)
        a2 = np.maximum(0, a1 @ self.W2 + self.b2)
        return float((a2 @ self.W3 + self.b3)[0])

    def step(self, time):
        x = np.array([[self.hour_sin, self.hour_cos,
                        self.wind_speed, self.temperature]])
        self.P_gen = max(0.0, self._forward(x))

    def get_value(self, parameters, time):
        mapping = {"P_gen": self.P_gen}
        return [mapping.get(p, 0.0) for p in parameters]

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            if hasattr(self, p):
                setattr(self, p, v)

    def init(self):
        self.step(0)

    def cleanup(self):
        pass
