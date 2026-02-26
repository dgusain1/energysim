#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Forecast analysis & visualization — prediction windows, accuracy, stochastics.

This script demonstrates six types of energy-system forecast analysis plots:

  1. Prediction Horizon: MPC forecast vs actual at selected time steps
  2. Stochastic Fan Chart: ensemble scenarios with confidence bands
  3. Forecast Error vs Lead Time: RMSE and MAE degradation
  4. Bottom-up Load Decomposition: stacked area of all agents
  5. Battery Electrochemistry Dashboard: SoC, voltage, temperature, resistance
  6. Forecast vs Realised Overlay: full-day comparison with shaded error

Works standalone with synthetic data.  If co-simulation results exist
(forecast_log.json + CSV results), it uses those instead.

Usage:
    python analyze_forecasts.py          # standalone (synthetic)
    python analyze_forecasts.py --live   # use co-sim results
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

HERE = Path(__file__).resolve().parent


# ════════════════════════════════════════════════════════════
#  Data loading
# ════════════════════════════════════════════════════════════

def load_cosim_results():
    """Load co-sim CSVs and dispatcher forecast log if available."""
    log_path = HERE / "forecast_log.json"
    res_dir = HERE / "res"

    forecast_log = None
    results = {}

    if log_path.exists():
        with open(log_path) as f:
            forecast_log = json.load(f)

    if res_dir.exists():
        for csv in res_dir.glob("*.csv"):
            df = pd.read_csv(csv)
            results[csv.stem] = df

    return forecast_log, results


def generate_synthetic_data(n_steps=96, dt=900):
    """Generate realistic synthetic profiles for standalone demo."""
    np.random.seed(42)
    t = np.arange(n_steps) * dt
    h = t / 3600.0

    # Temperature
    T = 6.0 + 5.0 * np.sin(2 * np.pi * (h - 5.0) / 24.0)
    T += np.random.normal(0, 0.3, n_steps)

    # Solar irradiance
    solar = np.where(
        (h >= 6) & (h <= 18),
        850.0 * np.sin(np.pi * (h - 6.0) / 12.0),
        0.0
    )
    solar = np.maximum(0, solar + np.random.normal(0, 30, n_steps))

    # Electricity price
    price = 0.15 + 0.12 * np.sin(2 * np.pi * (h - 18.0) / 24.0)
    price += np.where((h >= 7) & (h <= 9), 0.04, 0.0)
    price += np.random.normal(0, 0.005, n_steps)

    # Agent loads (kW)
    P_h1 = 1.0 + 0.8 * np.sin(np.pi * np.maximum(0, h - 6) / 16.0)
    P_h1 += np.random.normal(0, 0.15, n_steps)
    P_h2 = 0.5 + 0.4 * np.sin(np.pi * np.maximum(0, h - 7) / 15.0)
    P_h2 += np.random.normal(0, 0.1, n_steps)
    P_mall = np.where((h >= 8) & (h <= 22), 60 + 20 * np.sin(np.pi * (h - 8) / 14), 5)
    P_mall = P_mall + np.random.normal(0, 3, n_steps)
    P_super = np.where((h >= 6) & (h <= 23), 40 + 10 * np.sin(np.pi * (h - 6) / 17), 32)
    P_super = P_super + np.random.normal(0, 2, n_steps)
    P_pv = np.maximum(0, 5.0 * solar / 1000.0 * 0.85)

    # Battery SoC, voltage, temperature, resistance
    soc = 0.5 * np.ones(n_steps)
    V_term = 51.8 * np.ones(n_steps)
    T_cell = 25.0 * np.ones(n_steps)
    R_int = 0.012 * np.ones(n_steps)
    for i in range(1, n_steps):
        P_grid = P_h1[i] + P_h2[i] + P_mall[i] + P_super[i] - P_pv[i]
        if price[i] < 0.12 and soc[i - 1] < 0.90:
            soc[i] = soc[i - 1] + 0.02
        elif price[i] > 0.22 and soc[i - 1] > 0.15:
            soc[i] = soc[i - 1] - 0.025
        else:
            soc[i] = soc[i - 1] - 0.002
        soc[i] = np.clip(soc[i], 0.05, 0.95)
        V_term[i] = 14 * (3.43 + 1.68 * soc[i] - 4.21 * soc[i]**2
                          + 6.71 * soc[i]**3 - 5.13 * soc[i]**4
                          + 1.55 * soc[i]**5)
        T_cell[i] = T_cell[i-1] + 0.05 * (P_grid / 100)**2 - 0.02 * (T_cell[i-1] - T[i])
        R_int[i] = 0.012 * (1 + 0.3 * np.exp(-8 * soc[i])) * np.exp(4000 * (1 / (T_cell[i] + 273.15) - 1 / 298.15))

    # Forecasts (with increasing error vs lead time)
    forecast_log = []
    for i in range(n_steps):
        fc_load, fc_pv, fc_price = [], [], []
        for k in range(min(16, n_steps - i)):
            noise_load = np.random.normal(0, 2 * (k + 1))
            noise_pv = np.random.normal(0, 0.1 * (k + 1))
            noise_price = np.random.normal(0, 0.003 * (k + 1))
            actual_load = P_h1[i + k] + P_h2[i + k] + P_mall[i + k] + P_super[i + k]
            fc_load.append(round(actual_load + noise_load, 3))
            fc_pv.append(round(float(P_pv[i + k] + noise_pv), 3))
            fc_price.append(round(float(price[i + k] + noise_price), 4))

        actual_total = P_h1[i] + P_h2[i] + P_mall[i] + P_super[i]
        forecast_log.append({
            'time': int(t[i]),
            'hour': round(float(h[i]), 2),
            'actual_load': round(float(actual_total), 3),
            'actual_pv': round(float(P_pv[i]), 3),
            'actual_price': round(float(price[i]), 4),
            'forecast_load': fc_load[:4],
            'forecast_pv': fc_pv[:4],
            'forecast_price': fc_price[:4],
            'SoC': round(float(soc[i]), 4),
        })

    data = {
        'hours': h, 'T': T, 'solar': solar, 'price': price,
        'P_h1': P_h1, 'P_h2': P_h2, 'P_mall': P_mall, 'P_super': P_super,
        'P_pv': P_pv, 'soc': soc, 'V_term': V_term,
        'T_cell': T_cell, 'R_int': R_int,
    }
    return forecast_log, data


# ════════════════════════════════════════════════════════════
#  Plot 1: Prediction Horizon — forecast vs actual at key steps
# ════════════════════════════════════════════════════════════

def plot_prediction_horizon(forecast_log, ax=None):
    """Show MPC prediction window at 3 selected time stamps."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # Pick 3 time stamps: morning, midday, evening
    n = len(forecast_log)
    idx_picks = [n // 6, n // 2, 3 * n // 4]
    colors = ['#2196F3', '#FF9800', '#E91E63']

    # Actual load trace
    hours_all = [e['hour'] for e in forecast_log]
    load_all = [e['actual_load'] for e in forecast_log]
    ax.plot(hours_all, load_all, 'k-', lw=1.5, label='Actual load', zorder=5)

    for idx, color in zip(idx_picks, colors):
        entry = forecast_log[idx]
        h0 = entry['hour']
        fc = entry['forecast_load']
        h_fc = [h0 + k * 0.25 for k in range(len(fc))]

        ax.plot(h_fc, fc, 'o--', color=color, lw=2, markersize=5,
                label=f'Forecast @ {h0:.1f}h', zorder=4)

        # Shade prediction window
        ax.axvspan(h_fc[0], h_fc[-1], alpha=0.07, color=color)

    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Community load [kW]')
    ax.set_title('MPC Prediction Windows — Forecast vs Actual')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(0, 24)
    ax.grid(True, alpha=0.3)


# ════════════════════════════════════════════════════════════
#  Plot 2: Stochastic Fan Chart — ensemble scenarios
# ════════════════════════════════════════════════════════════

def plot_stochastic_fan_chart(forecast_log, data, ax=None):
    """Generate stochastic scenarios from forecast + uncertainty model."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    hours = data['hours']
    P_total = data['P_h1'] + data['P_h2'] + data['P_mall'] + data['P_super']

    # Generate ensemble scenarios
    n_scenarios = 50
    np.random.seed(123)
    scenarios = np.zeros((n_scenarios, len(hours)))
    for s in range(n_scenarios):
        noise = np.cumsum(np.random.normal(0, 1.5, len(hours)))  # correlated noise
        seasonal = 3.0 * np.sin(2 * np.pi * hours / 24 + np.random.uniform(0, 0.5))
        scenarios[s] = P_total + noise + seasonal

    # Percentile bands
    p10 = np.percentile(scenarios, 10, axis=0)
    p25 = np.percentile(scenarios, 25, axis=0)
    p50 = np.percentile(scenarios, 50, axis=0)
    p75 = np.percentile(scenarios, 75, axis=0)
    p90 = np.percentile(scenarios, 90, axis=0)

    ax.fill_between(hours, p10, p90, alpha=0.15, color='#2196F3', label='P10–P90')
    ax.fill_between(hours, p25, p75, alpha=0.25, color='#2196F3', label='P25–P75')
    ax.plot(hours, p50, '--', color='#1565C0', lw=1, label='Median scenario')
    ax.plot(hours, P_total, 'k-', lw=2, label='Realised', zorder=5)

    # Plot a few individual scenarios
    for s in range(5):
        ax.plot(hours, scenarios[s], '-', alpha=0.2, color='grey', lw=0.5)

    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Community load [kW]')
    ax.set_title('Stochastic Scenarios — Fan Chart with Confidence Bands')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 24)
    ax.grid(True, alpha=0.3)


# ════════════════════════════════════════════════════════════
#  Plot 3: Forecast Error vs Lead Time
# ════════════════════════════════════════════════════════════

def plot_error_vs_lead_time(forecast_log, ax=None):
    """RMSE and MAE degradation with increasing forecast lead time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    n_leads = 4
    errors_by_lead = {k: [] for k in range(n_leads)}

    for i, entry in enumerate(forecast_log):
        fc = entry['forecast_load']
        for k in range(min(n_leads, len(fc))):
            j = i + k
            if j < len(forecast_log):
                actual = forecast_log[j]['actual_load']
                errors_by_lead[k].append(fc[k] - actual)

    leads = []
    rmse_vals = []
    mae_vals = []
    for k in range(n_leads):
        errs = np.array(errors_by_lead[k])
        if len(errs) > 0:
            leads.append((k + 1) * 15)   # minutes
            rmse_vals.append(np.sqrt(np.mean(errs**2)))
            mae_vals.append(np.mean(np.abs(errs)))

    ax2 = ax.twinx()
    b1 = ax.bar(np.array(leads) - 1.5, rmse_vals, width=3, color='#2196F3',
                alpha=0.8, label='RMSE')
    b2 = ax2.bar(np.array(leads) + 1.5, mae_vals, width=3, color='#FF9800',
                 alpha=0.8, label='MAE')

    ax.set_xlabel('Forecast lead time [min]')
    ax.set_ylabel('RMSE [kW]', color='#2196F3')
    ax2.set_ylabel('MAE [kW]', color='#FF9800')
    ax.set_title('Forecast Error Degradation vs Lead Time')
    ax.set_xticks(leads)

    lines = [b1, b2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')


# ════════════════════════════════════════════════════════════
#  Plot 4: Bottom-up Load Decomposition (stacked area)
# ════════════════════════════════════════════════════════════

def plot_load_decomposition(data, ax=None):
    """Stacked area chart of all agent loads."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    hours = data['hours']
    labels = ['House 1 (EV+PV)', 'House 2', 'Mall', 'Supermarket']
    loads = [np.maximum(0, data['P_h1']),
             np.maximum(0, data['P_h2']),
             np.maximum(0, data['P_mall']),
             np.maximum(0, data['P_super'])]
    colors = ['#4CAF50', '#8BC34A', '#FF9800', '#F44336']

    ax.stackplot(hours, *loads, labels=labels, colors=colors, alpha=0.8)
    ax.plot(hours, data['P_pv'], 'gold', lw=2, label='PV generation')
    ax.plot(hours, sum(loads) - data['P_pv'], 'k--', lw=1.5, label='Net demand')

    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Power [kW]')
    ax.set_title('Bottom-up Load Decomposition — All Community Agents')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(0, 24)
    ax.grid(True, alpha=0.3)


# ════════════════════════════════════════════════════════════
#  Plot 5: Battery Electrochemistry Dashboard
# ════════════════════════════════════════════════════════════

def plot_battery_dashboard(data, fig=None):
    """4-panel battery dashboard: SoC, voltage, temperature, resistance."""
    if fig is None:
        fig = plt.figure(figsize=(12, 8))

    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    hours = data['hours']

    # SoC
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(hours, data['soc'], alpha=0.3, color='#2196F3')
    ax1.plot(hours, data['soc'], '#1565C0', lw=2)
    ax1.axhline(0.95, ls='--', color='red', alpha=0.5, label='SoC limits')
    ax1.axhline(0.05, ls='--', color='red', alpha=0.5)
    ax1.set_ylabel('SoC [-]')
    ax1.set_title('State of Charge')
    ax1.set_xlim(0, 24)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Terminal voltage
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(hours, data['V_term'], '#4CAF50', lw=2)
    ax2.set_ylabel('V_terminal [V]')
    ax2.set_title('Pack Terminal Voltage (14s NMC622)')
    ax2.set_xlim(0, 24)
    ax2.grid(True, alpha=0.3)

    # Cell temperature
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(hours, data['T_cell'], '#F44336', lw=2)
    ax3.axhline(45, ls='--', color='red', alpha=0.5, label='Thermal limit')
    ax3.set_xlabel('Hour of day')
    ax3.set_ylabel('T_cell [°C]')
    ax3.set_title('Cell Temperature (lumped thermal)')
    ax3.set_xlim(0, 24)
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # Internal resistance
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(hours, data['R_int'] * 1000, '#9C27B0', lw=2)
    ax4.set_xlabel('Hour of day')
    ax4.set_ylabel('R_internal [mΩ]')
    ax4.set_title('Cell Internal Resistance (Arrhenius + SoC)')
    ax4.set_xlim(0, 24)
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Battery Electrochemistry Dashboard', fontsize=14, y=0.98)


# ════════════════════════════════════════════════════════════
#  Plot 6: Forecast vs Realised Overlay
# ════════════════════════════════════════════════════════════

def plot_forecast_vs_realised(forecast_log, ax=None):
    """Full-day forecast (1-step-ahead) vs actual with error shading."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    hours = [e['hour'] for e in forecast_log]
    actuals = [e['actual_load'] for e in forecast_log]
    # 1-step-ahead forecast (k=0 is the nowcast, k=1 is 15-min ahead)
    forecasts = []
    for i in range(len(forecast_log)):
        if i > 0 and len(forecast_log[i - 1]['forecast_load']) > 1:
            forecasts.append(forecast_log[i - 1]['forecast_load'][1])
        else:
            forecasts.append(actuals[i])

    actuals = np.array(actuals)
    forecasts = np.array(forecasts)
    errors = forecasts - actuals

    ax.plot(hours, actuals, 'k-', lw=2, label='Actual')
    ax.plot(hours, forecasts, '#2196F3', lw=1.5, ls='--', label='1-step forecast')
    ax.fill_between(hours,
                    np.minimum(actuals, forecasts),
                    np.maximum(actuals, forecasts),
                    alpha=0.2, color='#F44336', label='Error band')

    # Add RMSE annotation
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    ax.text(0.98, 0.95, f'RMSE: {rmse:.1f} kW\nMAE: {mae:.1f} kW',
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Community load [kW]')
    ax.set_title('Forecast vs Realised — 15-min-ahead Prediction')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 24)
    ax.grid(True, alpha=0.3)


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

def main(use_live=False):
    """Generate all analysis plots."""
    forecast_log, data = None, None

    if use_live:
        forecast_log, results = load_cosim_results()
        if forecast_log and results:
            print("[analyze] Using co-simulation results.")
            # Build data dict from CSVs
            data = {}
            for name, df in results.items():
                if 'time' in df.columns:
                    data['hours'] = df['time'].values / 3600.0
                for col in df.columns:
                    if col != 'time':
                        data[col] = df[col].values
            # Map column names to expected keys
            if 'P_h1' not in data:
                for key_map in [
                    ('household_2p_ev_pv', 'P_net', 'P_h1'),
                    ('household_1p', 'P_net', 'P_h2'),
                    ('mall', 'P_net', 'P_mall'),
                    ('supermarket', 'P_net', 'P_super'),
                    ('household_2p_ev_pv', 'P_pv', 'P_pv'),
                    ('community_battery', 'SoC', 'soc'),
                    ('community_battery', 'V_terminal', 'V_term'),
                    ('community_battery', 'T_cell', 'T_cell'),
                    ('community_battery', 'R_internal', 'R_int'),
                ]:
                    sim, col, key = key_map
                    if sim in results and col in results[sim].columns:
                        vals = results[sim][col].values
                        # Convert MW to kW for load values
                        if key in ('P_h1', 'P_h2', 'P_mall', 'P_super', 'P_pv'):
                            vals = vals * 1000.0
                        data[key] = vals
        else:
            print("[analyze] Co-sim results not found, using synthetic data.")
            forecast_log, data = generate_synthetic_data()
    else:
        print("[analyze] Generating synthetic demo data.")
        forecast_log, data = generate_synthetic_data()

    if forecast_log is None or data is None:
        forecast_log, data = generate_synthetic_data()

    # Ensure all required keys exist
    for key in ['hours', 'P_h1', 'P_h2', 'P_mall', 'P_super', 'P_pv',
                'soc', 'V_term', 'T_cell', 'R_int']:
        if key not in data:
            print(f"[warning] Missing '{key}' — filling with zeros.")
            data[key] = np.zeros_like(data.get('hours', np.arange(96)))

    # ── Create figure canvas ──
    fig = plt.figure(figsize=(18, 22))
    gs_top = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3,
                               top=0.95, bottom=0.38)

    ax1 = fig.add_subplot(gs_top[0, 0])
    ax2 = fig.add_subplot(gs_top[0, 1])
    ax3 = fig.add_subplot(gs_top[1, 0])
    ax4 = fig.add_subplot(gs_top[1, 1])
    ax5 = fig.add_subplot(gs_top[2, :])

    plot_prediction_horizon(forecast_log, ax=ax1)
    plot_stochastic_fan_chart(forecast_log, data, ax=ax2)
    plot_error_vs_lead_time(forecast_log, ax=ax3)
    plot_load_decomposition(data, ax=ax4)
    plot_forecast_vs_realised(forecast_log, ax=ax5)

    # Battery dashboard in bottom third
    gs_batt = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3,
                                top=0.33, bottom=0.03)
    ax_b1 = fig.add_subplot(gs_batt[0, 0])
    ax_b2 = fig.add_subplot(gs_batt[0, 1])
    ax_b3 = fig.add_subplot(gs_batt[1, 0])
    ax_b4 = fig.add_subplot(gs_batt[1, 1])

    hours = data['hours']
    ax_b1.fill_between(hours, data['soc'], alpha=0.3, color='#2196F3')
    ax_b1.plot(hours, data['soc'], '#1565C0', lw=2)
    ax_b1.axhline(0.95, ls='--', color='red', alpha=0.5)
    ax_b1.axhline(0.05, ls='--', color='red', alpha=0.5)
    ax_b1.set_ylabel('SoC [-]')
    ax_b1.set_title('State of Charge')
    ax_b1.set_xlim(0, 24); ax_b1.grid(True, alpha=0.3)

    ax_b2.plot(hours, data['V_term'], '#4CAF50', lw=2)
    ax_b2.set_ylabel('V_terminal [V]')
    ax_b2.set_title('Pack Terminal Voltage')
    ax_b2.set_xlim(0, 24); ax_b2.grid(True, alpha=0.3)

    ax_b3.plot(hours, data['T_cell'], '#F44336', lw=2)
    ax_b3.axhline(45, ls='--', color='red', alpha=0.5)
    ax_b3.set_xlabel('Hour'); ax_b3.set_ylabel('T_cell [°C]')
    ax_b3.set_title('Cell Temperature')
    ax_b3.set_xlim(0, 24); ax_b3.grid(True, alpha=0.3)

    ax_b4.plot(hours, data['R_int'] * 1000, '#9C27B0', lw=2)
    ax_b4.set_xlabel('Hour'); ax_b4.set_ylabel('R_internal [mΩ]')
    ax_b4.set_title('Internal Resistance')
    ax_b4.set_xlim(0, 24); ax_b4.grid(True, alpha=0.3)

    fig.suptitle('Community Microgrid — Forecast Analysis & Battery Dashboard',
                 fontsize=16, y=0.98, fontweight='bold')

    out_path = HERE / "analysis_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[analyze] Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true',
                        help='Use co-simulation results instead of synthetic data')
    args = parser.parse_args()
    main(use_live=args.live)
