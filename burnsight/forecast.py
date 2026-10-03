"""MVP-4: 168h degradation predictor — physics features + XGBoost regressor.

The model extrapolates each measurement channel to its 168h terminal value
from early telemetry only (0h / 24h / 96h). Three ingredient groups:

- **levels** at 0h, 24h, 96h — where the unit is;
- **slopes** 0h->24h and 24h->96h — how fast it is moving;
- **Arrhenius physics** — the lot temperature, its acceleration factor, the
  effective stress-hours by 96h, and ``{ch}_extrap_terminal``: the closed-form
  kinetics extrapolation ``value_0 + drift_96 / progress(96h, AF)``.

Rather than regressing the skewed terminal value directly (which shrinks rare
extreme units back toward the population mean), the booster is trained on the
**residual over the physics extrapolation**: prediction = ``extrap_terminal +
GBDT(features)``. The physics term carries the magnitude, the model learns the
correction — the classic physics-informed hybrid.

The 168h columns are never inputs: `test_terminal_columns_do_not_leak_into_predictions`
zeroes them and the predictions must not move.

Trajectory plotting reuses the generator's single kinetics law
(``progress_curve``) so synthesis and forecast share one physics model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from burnsight.config import PARAM_CHANNELS, channel_column
from burnsight.generator import arrhenius_rate, progress_curve

DEFAULT_TRAJECTORY_HOURS: tuple[int, ...] = (0, 24, 96, 120, 168)

DEFAULT_XGB_PARAMS: dict[str, object] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 1,  # single thread keeps training bit-deterministic
}

_GLOBAL_FEATURES = ("temp_c", "arrhenius_af", "stress_hours_96")


def _channel_features(channel: str) -> tuple[str, ...]:
    return (
        f"{channel}_0h",
        f"{channel}_24h",
        f"{channel}_96h",
        f"{channel}_slope_0_24",
        f"{channel}_slope_24_96",
        f"{channel}_drift_0_96",
        f"{channel}_extrap_terminal",
    )


DEFAULT_FEATURE_COLUMNS: tuple[str, ...] = _GLOBAL_FEATURES + tuple(
    name for ch in PARAM_CHANNELS for name in _channel_features(ch)
)


@dataclass(frozen=True)
class ForecastModel:
    """Fitted per-channel regressors plus the exact feature contract."""

    models: dict[str, XGBRegressor]
    feature_columns: tuple[str, ...]
    n_units: int = field(default=0)
    # Per-channel std of training residuals: the forecast confidence input
    # consumed by triage (MVP-5) to widen escalation when the model is unsure.
    error_std: dict[str, float] = field(default_factory=dict)


def build_features(batch: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix from early telemetry only — never from 168h columns."""
    temperature = batch["temp_c"].to_numpy(dtype=float)
    acceleration = np.asarray(arrhenius_rate(temperature), dtype=float)
    progress_96 = progress_curve(acceleration, hours=(96,))[:, 0]

    data: dict[str, object] = {
        "temp_c": batch["temp_c"].to_numpy(dtype=float),
        "arrhenius_af": acceleration,
        "stress_hours_96": 96.0 * acceleration,
    }
    for channel in PARAM_CHANNELS:
        level_0 = batch[channel_column(channel, 0)].to_numpy(dtype=float)
        level_24 = batch[channel_column(channel, 24)].to_numpy(dtype=float)
        level_96 = batch[channel_column(channel, 96)].to_numpy(dtype=float)
        drift_96 = level_96 - level_0
        data[f"{channel}_0h"] = level_0
        data[f"{channel}_24h"] = level_24
        data[f"{channel}_96h"] = level_96
        data[f"{channel}_slope_0_24"] = (level_24 - level_0) / 24.0
        data[f"{channel}_slope_24_96"] = (level_96 - level_24) / 72.0
        data[f"{channel}_drift_0_96"] = drift_96
        data[f"{channel}_extrap_terminal"] = level_0 + drift_96 / progress_96

    return pd.DataFrame(data, columns=DEFAULT_FEATURE_COLUMNS)


def train_forecast_model(train_batch: pd.DataFrame) -> ForecastModel:
    """Fit one XGBoost residual regressor per channel on a labeled lot.

    Targets are ``terminal - extrap_terminal`` so the booster only has to
    learn the correction over physics, not the full skewed magnitude.
    """
    features = build_features(train_batch)
    models: dict[str, XGBRegressor] = {}
    error_std: dict[str, float] = {}
    for channel in PARAM_CHANNELS:
        residual = (
            train_batch[channel_column(channel, 168)]
            - features[f"{channel}_extrap_terminal"]
        )
        regressor = XGBRegressor(**DEFAULT_XGB_PARAMS)
        regressor.fit(features, residual)
        models[channel] = regressor
        fitted = regressor.predict(features)
        error_std[channel] = float(np.std(residual - fitted, ddof=1))
    return ForecastModel(
        models=models,
        feature_columns=tuple(features.columns),
        n_units=len(train_batch),
        error_std=error_std,
    )


def predict_terminal(model: ForecastModel, batch: pd.DataFrame) -> pd.DataFrame:
    """Predicted 168h value per channel, one row per input unit."""
    features = build_features(batch)[list(model.feature_columns)]
    prediction: dict[str, object] = {"unit_id": batch["unit_id"].to_numpy()}
    for channel in PARAM_CHANNELS:
        correction = model.models[channel].predict(features)
        prediction[f"{channel}_168h_pred"] = (
            features[f"{channel}_extrap_terminal"].to_numpy() + correction
        )
    return pd.DataFrame(prediction)


def forecast_trajectory(
    model: ForecastModel,
    batch: pd.DataFrame,
    hours: Sequence[float] = DEFAULT_TRAJECTORY_HOURS,
) -> pd.DataFrame:
    """0h baseline through predicted 168h terminal for every unit and hour.

    Interpolation follows the shared kinetics law, so the curve passes
    through the measured baseline at t=0 and the model's terminal at t=168h.
    Long-wide shape: one row per (unit, hour), unit-major order.
    """
    terminal = predict_terminal(model, batch)
    acceleration = np.asarray(
        arrhenius_rate(batch["temp_c"].to_numpy(dtype=float)), dtype=float
    )
    fractions = progress_curve(acceleration, hours=hours)  # (n_units, n_hours)

    baseline = batch[[channel_column(ch, 0) for ch in PARAM_CHANNELS]].to_numpy(
        dtype=float
    )
    predicted = terminal[[f"{ch}_168h_pred" for ch in PARAM_CHANNELS]].to_numpy(
        dtype=float
    )

    # (n_units, n_hours, n_channels): baseline + fraction of predicted drift
    values = (
        baseline[:, None, :]
        + (predicted - baseline)[:, None, :] * fractions[:, :, None]
    )

    n_units, n_hours = values.shape[0], len(hours)
    data: dict[str, object] = {
        "unit_id": np.repeat(batch["unit_id"].to_numpy(), n_hours),
        "hour": np.tile(np.asarray(hours), n_units),
    }
    for index, channel in enumerate(PARAM_CHANNELS):
        data[channel] = values[:, :, index].reshape(-1)
    return pd.DataFrame(data, columns=["unit_id", "hour", *PARAM_CHANNELS])
