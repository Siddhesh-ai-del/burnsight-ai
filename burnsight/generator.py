"""Arrhenius-driven synthetic burn-in telemetry generator.

Physics model
-------------
Degradation progress of a unit follows a thermally accelerated kinetics:

    progress(t) = min(1, (t * AF(T)) / t_end) ** p        p = 0.8

where AF(T) is the Arrhenius acceleration factor relative to the 125 C
burn-in baseline (MIL-STD-883 Method 1015 stress range):

    AF(T) = exp( (Ea / kB) * (1/T_ref - 1/T) ),  Ea = 0.7 eV

Higher lane temperature does not change the terminal state of a unit; it
changes how early it gets there. That is exactly the signal early burn-in
screening has to work with.

Ground truth
------------
- severity == DEFECTIVE  -> exceeds a spec limit at 168h (true infant mortality)
- severity == BORDERLINE -> passes 168h but sits >= 80% toward a limit
- severity == NORMAL     -> comfortably inside every limit at 168h
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from burnsight.config import (
    BATCH_COLUMNS,
    DEVICE_TYPE,
    DEFECT_MODES,
    PARAM_CHANNELS,
    SAMPLE_HOURS,
    SEVERITY_LEVELS,
    SPEC_LIMITS,
    channel_column,
)

BOLTZMANN_EV = 8.617333262e-5
ACTIVATION_ENERGY_EV = 0.7
BASELINE_TEMP_C = 125.0  # MIL-STD-883 burn-in baseline
MIN_TEMP_C, MAX_TEMP_C = 125.0, 150.0
KINETICS_EXPONENT = 0.8

# (mean, sd) intrinsic baseline per channel, measured at burn-in temperature
_BASELINE_PARAMS = {
    "rds_on_mohm": (100.0, 1.5),
    "igss_na": (5.0, 0.5),
    "vgsth_v": (4.0, 0.05),
}
# (mean, sd) benign drift accumulated over the full 168h cycle
_NORMAL_DRIFT = {
    "rds_on_mohm": (4.0, 1.5),
    "igss_na": (2.0, 1.0),
    "vgsth_v": (0.02, 0.02),
}
# measurement noise (sd) per channel
_NOISE_SD = {"rds_on_mohm": 0.4, "igss_na": 0.5, "vgsth_v": 0.02}
_BASELINE_FLOOR = {"rds_on_mohm": 60.0, "igss_na": 0.5, "vgsth_v": 3.2}
# borderlines are planted at 85-95% of the way to their channel's USL;
# defects are planted 5-25% beyond the USL.
_BORDERLINE_POS = (0.85, 0.95)
_DEFECT_OVERSHOOT = (1.05, 1.25)


def arrhenius_rate(
    temp_c: float,
    ea_ev: float = ACTIVATION_ENERGY_EV,
    ref_temp_c: float = BASELINE_TEMP_C,
) -> np.ndarray | float:
    """Arrhenius thermal acceleration factor relative to ``ref_temp_c``.

    AF(125 C) == 1.0 by definition; AF grows monotonically with temperature.
    """
    temp_k = np.asarray(temp_c, dtype=float) + 273.15
    ref_k = ref_temp_c + 273.15
    return np.exp((ea_ev / BOLTZMANN_EV) * (1.0 / ref_k - 1.0 / temp_k))


def progress_curve(
    acceleration: np.ndarray, hours: Sequence[float] = SAMPLE_HOURS
) -> np.ndarray:
    """Damage fraction at each hour for every unit: (n_units, n_hours).

    ``min(t * AF / 168h, 1) ** 0.8`` — shared by the generator (synthesis)
    and the forecaster (trajectory plotting) so one kinetics law is used.
    """
    elapsed = np.asarray(hours, dtype=float)
    effective = (
        np.asarray(acceleration, dtype=float)[:, None]
        * elapsed[None, :]
        / SAMPLE_HOURS[-1]
    )
    return np.minimum(effective, 1.0) ** KINETICS_EXPONENT


def _validate(
    n_units: int, defect_count: int, borderline_count: int, seed: int
) -> None:
    if (
        not isinstance(n_units, (int, np.integer))
        or isinstance(n_units, bool)
        or n_units <= 0
    ):
        raise ValueError("n_units must be a positive integer")
    if not isinstance(defect_count, (int, np.integer)) or defect_count < 0:
        raise ValueError("defect_count must be a non-negative integer")
    if not isinstance(borderline_count, (int, np.integer)) or borderline_count < 0:
        raise ValueError("borderline_count must be a non-negative integer")
    if defect_count + borderline_count > n_units:
        raise ValueError("defect_count + borderline_count cannot exceed n_units")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")


def _assign_severity(
    rng: np.random.Generator, n_units: int, defect_count: int, borderline_count: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Plant defects first, then borderlines among the remaining units."""
    severity = np.full(n_units, SEVERITY_LEVELS[0], dtype=object)
    defect_idx = np.sort(rng.choice(n_units, size=defect_count, replace=False))
    remaining = np.setdiff1d(np.arange(n_units), defect_idx, assume_unique=True)
    borderline_idx = np.sort(
        rng.choice(remaining, size=borderline_count, replace=False)
    )
    severity[defect_idx] = SEVERITY_LEVELS[2]
    severity[borderline_idx] = SEVERITY_LEVELS[1]
    return severity, {"defect": defect_idx, "borderline": borderline_idx}


def _lane_temperatures(
    rng: np.random.Generator, n_units: int, temp_mean_c: float, temp_std_c: float
) -> np.ndarray:
    """One clipped lane temperature per unit."""
    drawn = rng.normal(temp_mean_c, temp_std_c, n_units)
    return np.clip(drawn, MIN_TEMP_C, MAX_TEMP_C)


def _sample_population(
    rng: np.random.Generator, n_units: int
) -> tuple[np.ndarray, np.ndarray]:
    """Intrinsic baseline (floored) and benign 168 h drift, per channel."""
    baseline = np.column_stack(
        [
            np.maximum(rng.normal(*_BASELINE_PARAMS[ch], n_units), _BASELINE_FLOOR[ch])
            for ch in PARAM_CHANNELS
        ]
    )
    drift = np.column_stack(
        [rng.normal(*_NORMAL_DRIFT[ch], n_units) for ch in PARAM_CHANNELS]
    )
    return baseline, drift


def _plant_targets(
    rng: np.random.Generator,
    baseline: np.ndarray,
    drift: np.ndarray,
    planted: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(drift, defect_mode)`` with the planted failure targets applied."""
    n_units = drift.shape[0]
    modes = list(DEFECT_MODES)
    defect_modes = {
        int(idx): modes[i % len(modes)] for i, idx in enumerate(planted["defect"])
    }
    mode_names = np.array(
        [defect_modes.get(i, "NONE") for i in range(n_units)], dtype=object
    )

    for idx in planted["defect"]:
        channel = DEFECT_MODES[defect_modes[int(idx)]][0]
        j = PARAM_CHANNELS.index(channel)
        _lsl, usl = SPEC_LIMITS[channel]
        target = rng.uniform(*_DEFECT_OVERSHOOT) * usl
        drift[idx, j] = target - baseline[idx, j]

    for i, idx in enumerate(planted["borderline"]):
        channel = PARAM_CHANNELS[i % len(PARAM_CHANNELS)]
        j = PARAM_CHANNELS.index(channel)
        lsl, usl = SPEC_LIMITS[channel]
        target = lsl + rng.uniform(*_BORDERLINE_POS) * (usl - lsl)
        drift[idx, j] = target - baseline[idx, j]
    return drift, mode_names


def _trajectories(
    rng: np.random.Generator,
    baseline: np.ndarray,
    drift: np.ndarray,
    temp_c: np.ndarray,
) -> np.ndarray:
    """Layer noise over the Arrhenius-scaled drift: (n_units, n_ch, n_hours)."""
    noise = (
        rng.normal(size=(len(temp_c), len(PARAM_CHANNELS), len(SAMPLE_HOURS)))
        * np.array([_NOISE_SD[ch] for ch in PARAM_CHANNELS])[:, None]
    )
    progress = progress_curve(arrhenius_rate(temp_c))[:, None, :]
    return baseline[:, :, None] + drift[:, :, None] * progress + noise


def generate_batch(
    n_units: int = 100,
    defect_count: int = 3,
    borderline_count: int = 5,
    seed: int = 42,
    temp_mean_c: float = 125.0,
    temp_std_c: float = 3.0,
) -> pd.DataFrame:
    """Generate one deterministic burn-in batch with planted failure modes."""
    _validate(n_units, defect_count, borderline_count, seed)
    rng = np.random.default_rng(seed)

    temp_c = _lane_temperatures(rng, n_units, temp_mean_c, temp_std_c)
    severity, planted = _assign_severity(rng, n_units, defect_count, borderline_count)
    baseline, drift = _sample_population(rng, n_units)
    drift, defect_mode = _plant_targets(rng, baseline, drift, planted)
    trajectories = _trajectories(rng, baseline, drift, temp_c)

    data: dict[str, object] = {
        "unit_id": [f"U{j:03d}" for j in range(n_units)],
        "device_type": DEVICE_TYPE,
        "temp_c": temp_c,
        "severity": severity,
        "defect_mode": defect_mode,
    }
    for j, channel in enumerate(PARAM_CHANNELS):
        for h, hour in enumerate(SAMPLE_HOURS):
            data[channel_column(channel, hour)] = trajectories[:, j, h]
    return pd.DataFrame(data, columns=BATCH_COLUMNS)
