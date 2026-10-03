"""Population outlier screening: robust MAD z-scores + IsolationForest.

Stage 1 of the pipeline answers one question per batch: which units are
statistically out-of-family versus their lot?

Two signals are combined (union, biased toward flagging — a false alarm
costs a review, an escape costs a mission):

- **MAD z-scores on the T=0h baseline** catch units born out of family.
  Pre-stress baseline spread alone cannot see units that merely degrade
  fast: the planted defect modes start statistically normal at 0h.
- **MAD z-scores on 0h->24h drift** catch units whose early degradation
  kinetics are abnormal — this is where infant mortality shows itself.
- **IsolationForest** over both feature blocks adds a non-linear
  multivariate check (axis-aligned splits catch combinations that
  per-channel z-scores miss).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from burnsight.config import PARAM_CHANNELS, channel_column

# 0.75 quantile of the standard normal: makes MAD z-scores consistent
# with standard-deviation z-scores for Gaussian populations.
MAD_SCALE = 0.6744897501960817
DEFAULT_MAD_THRESHOLD = 4.0
DEFAULT_CONTAMINATION = 0.05
DEFAULT_SEED = 42


def robust_zscores(values: np.ndarray) -> np.ndarray:
    """Median Absolute Deviation z-scores, safe for zero-variance inputs."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0:
        return MAD_SCALE * (values - median) / mad
    std = float(np.std(values))
    if std > 0:
        return (values - median) / std
    return np.zeros_like(values, dtype=float)


def baseline_features(batch: pd.DataFrame) -> pd.DataFrame:
    """T=0h pre-stress readings per channel."""
    return pd.DataFrame(
        {ch: batch[channel_column(ch, 0)] for ch in PARAM_CHANNELS},
        index=batch.index,
    )


def drift_features(batch: pd.DataFrame) -> pd.DataFrame:
    """0h->24h parametric drift per channel (early kinetics signal)."""
    return pd.DataFrame(
        {
            ch: batch[channel_column(ch, 24)] - batch[channel_column(ch, 0)]
            for ch in PARAM_CHANNELS
        },
        index=batch.index,
    )


def _channel_z(frame: pd.DataFrame) -> np.ndarray:
    """Per-unit matrix of channel z-scores: shape (n_units, n_channels)."""
    return np.column_stack(
        [robust_zscores(frame[ch].to_numpy(dtype=float)) for ch in PARAM_CHANNELS]
    )


def screen_population(
    batch: pd.DataFrame,
    mad_threshold: float = DEFAULT_MAD_THRESHOLD,
    contamination: float = DEFAULT_CONTAMINATION,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Flag out-of-family units. One row per input unit, input order kept.

    ``out_of_family = mad_baseline OR mad_drift OR iforest`` — the union,
    so a unit only escapes screening when every signal agrees it is normal.
    """
    baseline = baseline_features(batch)
    drift = drift_features(batch)

    baseline_z = np.abs(_channel_z(baseline)).max(axis=1)
    drift_z = np.abs(_channel_z(drift)).max(axis=1)
    mad_baseline = baseline_z > mad_threshold
    mad_drift = drift_z > mad_threshold

    features = np.column_stack(
        [baseline.to_numpy(dtype=float), drift.to_numpy(dtype=float)]
    )
    predictions = IsolationForest(
        contamination=contamination, random_state=seed
    ).fit_predict(features)
    iforest = predictions == -1

    return pd.DataFrame(
        {
            "unit_id": batch["unit_id"].to_numpy(),
            "mad_baseline": mad_baseline,
            "mad_drift": mad_drift,
            "iforest": iforest,
            "out_of_family": mad_baseline | mad_drift | iforest,
            "max_abs_z_baseline": baseline_z,
            "max_abs_z_drift": drift_z,
        }
    )
