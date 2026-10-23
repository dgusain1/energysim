# -*- coding: utf-8 -*-
"""
Operating State Machine and Dynamics Tracker for Industrial Equipment.

Tracks startup trajectories, cold/warm/hot categories, minimum up/downtime,
and physical ramp-rate constraints.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional


class OperatingState(str, Enum):
    OFF = "OFF"
    STANDBY = "STANDBY"
    PURGE = "PURGE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    COOLDOWN = "COOLDOWN"
    FAULT = "FAULT"


class OperatingStateMachine:
    """
    Manages equipment operating states, startup timers, and ramp limits.

    Parameters
    ----------
    ramp_up_rate_min : float
        Maximum ramp-up rate in units/min (e.g. MW/min).
    ramp_down_rate_min : float
        Maximum ramp-down rate in units/min (e.g. MW/min).
    min_uptime_sec : float, default 0.0
        Minimum continuous operating time in seconds once started.
    min_downtime_sec : float, default 0.0
        Minimum continuous offline time in seconds before restart is permitted.
    cold_start_time_hr : float, default 48.0
        Hours offline after which the unit enters a Cold start state.
    warm_start_time_hr : float, default 8.0
        Hours offline after which the unit enters a Warm start state.
    cold_start_duration_sec : float, default 1800.0
        Duration in seconds required to complete a Cold startup.
    warm_start_duration_sec : float, default 900.0
        Duration in seconds required to complete a Warm startup.
    hot_start_duration_sec : float, default 300.0
        Duration in seconds required to complete a Hot startup.
    """

    def __init__(
        self,
        ramp_up_rate_min: float = 1e6,
        ramp_down_rate_min: float = 1e6,
        min_uptime_sec: float = 0.0,
        min_downtime_sec: float = 0.0,
        cold_start_time_hr: float = 48.0,
        warm_start_time_hr: float = 8.0,
        cold_start_duration_sec: float = 1800.0,
        warm_start_duration_sec: float = 900.0,
        hot_start_duration_sec: float = 300.0,
    ):
        self.ramp_up_rate_min = ramp_up_rate_min
        self.ramp_down_rate_min = ramp_down_rate_min
        self.min_uptime_sec = min_uptime_sec
        self.min_downtime_sec = min_downtime_sec

        self.cold_start_time_hr = cold_start_time_hr
        self.warm_start_time_hr = warm_start_time_hr
        self.cold_start_duration_sec = cold_start_duration_sec
        self.warm_start_duration_sec = warm_start_duration_sec
        self.hot_start_duration_sec = hot_start_duration_sec

        # Dynamic state
        self.state: OperatingState = OperatingState.OFF
        self.time_in_state_sec: float = 0.0
        self.time_offline_sec: float = 1e6  # Start assuming fully cold
        self.current_startup_category: str = "COLD"
        self.startup_timer_sec: float = 0.0

    def transition(self, new_state: OperatingState) -> bool:
        """Change state; resets ``time_in_state_sec`` only on an actual change.

        Returns True if the state changed.
        """
        if new_state == self.state:
            return False
        self.state = new_state
        self.time_in_state_sec = 0.0
        return True

    def can_stop(self) -> bool:
        """True if the unit has run at least ``min_uptime_sec`` in a running state."""
        return self.time_in_state_sec >= self.min_uptime_sec

    def can_start(self) -> bool:
        """True if the unit has been offline at least ``min_downtime_sec``."""
        return self.time_offline_sec >= self.min_downtime_sec

    def update_time(self, dt_sec: float) -> None:
        """Advance internal timers by dt_sec."""
        self.time_in_state_sec += dt_sec
        if self.state in (OperatingState.OFF, OperatingState.COOLDOWN, OperatingState.STANDBY):
            self.time_offline_sec += dt_sec
        else:
            self.time_offline_sec = 0.0

    def apply_ramp(self, current: float, target: float, dt_sec: float) -> float:
        """
        Ramp current value towards target value respecting ramp_up and ramp_down rates.
        """
        dt_min = dt_sec / 60.0
        max_up = self.ramp_up_rate_min * dt_min
        max_down = self.ramp_down_rate_min * dt_min

        if target > current:
            delta = min(target - current, max_up)
            return current + delta
        elif target < current:
            delta = min(current - target, max_down)
            return current - delta
        return current

    def determine_startup_category(self) -> str:
        """Return 'COLD', 'WARM', or 'HOT' based on time offline."""
        hrs_off = self.time_offline_sec / 3600.0
        if hrs_off >= self.cold_start_time_hr:
            return "COLD"
        elif hrs_off >= self.warm_start_time_hr:
            return "WARM"
        return "HOT"

    def get_startup_duration(self, category: Optional[str] = None) -> float:
        """Return duration in seconds for given or current startup category."""
        cat = category or self.current_startup_category
        if cat == "COLD":
            return self.cold_start_duration_sec
        elif cat == "WARM":
            return self.warm_start_duration_sec
        return self.hot_start_duration_sec
