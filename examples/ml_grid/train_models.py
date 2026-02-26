#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train small neural networks (pure numpy) for load/generation forecasting.

Creates four models (feedforward NN, architecture [4]→[16 ReLU]→[8 ReLU]→[1]):
  - residential_load  : (hour_sin, hour_cos, temperature, is_weekend) → load_mw
  - commercial_load   : (hour_sin, hour_cos, temperature, is_weekend) → load_mw
  - pv_output         : (hour_sin, hour_cos, cloud_cover, temperature) → gen_mw
  - wind_output       : (hour_sin, hour_cos, wind_speed, temperature) → gen_mw

Trained with MSE loss and vanilla gradient descent (500 epochs).
Weights are saved as .npz files in the same directory as this script.
"""

import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
np.random.seed(42)


# ── Synthetic data generators ────────────────────────────────────────────

def _make_features(n=2000):
    """Random time-of-day, temperature, cloud, wind features."""
    hour = np.random.uniform(0, 24, n)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    temperature = 8 + 5 * np.sin(2 * np.pi * (hour - 14) / 24) + np.random.normal(0, 1, n)
    cloud_cover = np.clip(
        0.3 + 0.3 * np.sin(np.pi * hour / 12) + np.random.normal(0, 0.1, n), 0, 1)
    wind_speed = np.clip(
        5 + 3 * np.sin(2 * np.pi * hour / 24 + 1.5) + np.random.normal(0, 1.5, n), 0, 15)
    is_weekend = np.random.choice([0.0, 1.0], n, p=[5 / 7, 2 / 7])
    return hour, hour_sin, hour_cos, temperature, cloud_cover, wind_speed, is_weekend


def _residential_target(hour, temperature, is_weekend):
    """Residential load in MW — morning & evening peaks, cold-boost."""
    base = 0.002
    morning = 0.003 * np.exp(-0.5 * ((hour - 8) / 1.5) ** 2)
    evening = 0.004 * np.exp(-0.5 * ((hour - 19) / 2.0) ** 2)
    cold_boost = np.clip((10 - temperature) * 0.0003, 0, 0.002)
    weekend_factor = 1.0 + 0.15 * is_weekend
    noise = np.random.normal(0, 0.0003, len(hour))
    return np.clip((base + morning + evening + cold_boost) * weekend_factor + noise, 0, None)


def _commercial_target(hour, temperature, is_weekend):
    """Commercial load in MW — business-hours peak, low on weekends."""
    business = np.where((hour >= 9) & (hour <= 17), 0.005, 0.001)
    ramp_up = np.where((hour >= 7) & (hour < 9), 0.001 + 0.002 * (hour - 7), 0.0)
    ramp_down = np.where((hour > 17) & (hour <= 19), 0.005 - 0.002 * (hour - 17), 0.0)
    weekend_factor = np.where(is_weekend > 0.5, 0.15, 1.0)
    noise = np.random.normal(0, 0.0003, len(hour))
    return np.clip((business + ramp_up + ramp_down) * weekend_factor + noise, 0, None)


def _pv_target(hour, cloud_cover, temperature):
    """PV generation in MW — solar bell curve modulated by clouds."""
    solar = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    solar = 0.010 * solar ** 1.5
    cloud_factor = 1.0 - 0.8 * cloud_cover
    temp_factor = 1.0 - 0.004 * np.clip(temperature - 25, 0, None)
    noise = np.random.normal(0, 0.0005, len(hour))
    return np.clip(solar * cloud_factor * temp_factor + noise, 0, 0.010)


def _wind_target(hour, wind_speed, temperature):
    """Wind generation in MW — cubic power curve with cut-in/rated/cut-out."""
    cut_in, rated, cut_out = 3.0, 12.0, 25.0
    ws = np.clip(wind_speed, 0, cut_out)
    power = np.where(ws < cut_in, 0.0,
                     np.where(ws < rated,
                              0.015 * ((ws - cut_in) / (rated - cut_in)) ** 3,
                              0.015))
    power = np.where(ws >= cut_out, 0.0, power)
    noise = np.random.normal(0, 0.0005, len(hour))
    return np.clip(power + noise, 0, 0.015)


# ── Neural network training (pure numpy) ────────────────────────────────

def train_nn(X, y, hidden1=16, hidden2=8, epochs=500, lr=0.001):
    """Train a 2-hidden-layer feedforward NN with ReLU activations."""
    n_in = X.shape[1]
    # Xavier initialisation
    W1 = np.random.randn(n_in, hidden1) * np.sqrt(2 / n_in)
    b1 = np.zeros(hidden1)
    W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2 / hidden1)
    b2 = np.zeros(hidden2)
    W3 = np.random.randn(hidden2, 1) * np.sqrt(2 / hidden2)
    b3 = np.zeros(1)

    for epoch in range(epochs):
        # Forward pass
        z1 = X @ W1 + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2
        a2 = np.maximum(0, z2)
        y_pred = a2 @ W3 + b3

        loss = np.mean((y_pred - y) ** 2)

        # Backward pass
        m = len(y)
        dl = 2 * (y_pred - y) / m
        dW3 = a2.T @ dl
        db3 = dl.sum(axis=0)
        da2 = dl @ W3.T
        da2[z2 <= 0] = 0
        dW2 = a1.T @ da2
        db2 = da2.sum(axis=0)
        da1 = da2 @ W2.T
        da1[z1 <= 0] = 0
        dW1 = X.T @ da1
        db1 = da1.sum(axis=0)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        W3 -= lr * dW3
        b3 -= lr * db3

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d}: loss = {loss:.8f}")

    return W1, b1, W2, b2, W3, b3


def _normalise(X):
    """Zero-mean, unit-variance normalisation.  Returns (X_norm, mean, std)."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    return (X - mu) / sigma, mu, sigma


def _train_and_save(name, X_raw, y_raw):
    """Normalise, train, and save weights to <name>.npz."""
    X, mu, sigma = _normalise(X_raw)
    y = y_raw.reshape(-1, 1)
    print(f"\nTraining {name} ...")
    W1, b1, W2, b2, W3, b3 = train_nn(X, y)
    path = HERE / f"{name}.npz"
    np.savez(path, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3,
             input_mean=mu, input_std=sigma)
    print(f"  Saved -> {path.name}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    hour, hour_sin, hour_cos, temperature, cloud_cover, wind_speed, is_weekend = (
        _make_features(2000))

    # Residential load
    X_res = np.column_stack([hour_sin, hour_cos, temperature, is_weekend])
    y_res = _residential_target(hour, temperature, is_weekend)
    _train_and_save("residential_load_weights", X_res, y_res)

    # Commercial load
    X_com = np.column_stack([hour_sin, hour_cos, temperature, is_weekend])
    y_com = _commercial_target(hour, temperature, is_weekend)
    _train_and_save("commercial_load_weights", X_com, y_com)

    # PV output
    X_pv = np.column_stack([hour_sin, hour_cos, cloud_cover, temperature])
    y_pv = _pv_target(hour, cloud_cover, temperature)
    _train_and_save("pv_weights", X_pv, y_pv)

    # Wind output
    X_wind = np.column_stack([hour_sin, hour_cos, wind_speed, temperature])
    y_wind = _wind_target(hour, wind_speed, temperature)
    _train_and_save("wind_weights", X_wind, y_wind)

    print("\nAll models trained and saved.")


if __name__ == "__main__":
    main()
