"""MVP-6: SHAP attribution + physics reason codes for flagged decisions.

Every Yellow/Red unit gets an explanation built from two complementary
pieces:

- **Reason code** (rule-based, from the fixed config registry): the triage
  worst channel inverts to its defect mode (``defect_mode_for_channel``) and
  then to a MIL-STD-883 style code — ``ERR_MOSFET_GATE_DEGRADE``,
  ``ERR_MOSFET_RDS_DRIFT``, ``ERR_MOSFET_VGSTH_SHIFT``.
- **SHAP attribution** (shap 0.52, locked primary): attributions explain the
  *total* terminal prediction — physics extrapolation plus GBDT correction —
  in physical units, so the drifting channel's features dominate a Red
  unit's explanation and the top-ranked feature names the driver.

Scope: flagged units only (SHAP runtime). Background = the lot's own Green
units (subsampled), i.e. "normal for this batch" — the comparison an
operator actually cares about. A fixed seed keeps audit output reproducible.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import shap

from burnsight.config import defect_mode_for_channel, reason_code
from burnsight.forecast import ForecastModel, build_features

MAX_BACKGROUND_ROWS = 32
DEFAULT_SEED = 42
_BACKGROUND_FLOOR = 1e-12  # avoids 0/0 when a unit sits exactly at baseline

EXPLAIN_COLUMNS = (
    "unit_id",
    "color",
    "reason_code",
    "defect_mode",
    "worst_channel",
    "rank",
    "feature",
    "shap_value",
    "feature_value",
    "weight",
)


def _total_predictor(model: ForecastModel, channel: str):
    """f(features) = physics extrapolation + GBDT correction for one channel."""
    regressor = model.models[channel]
    extrapolation = f"{channel}_extrap_terminal"

    def predict_total(matrix: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(
            np.atleast_2d(np.asarray(matrix, dtype=float)),
            columns=model.feature_columns,
        )
        return frame[extrapolation].to_numpy() + regressor.predict(frame)

    return predict_total


def _background_rows(features: pd.DataFrame, triage: pd.DataFrame) -> np.ndarray:
    """Green units of this lot as SHAP baseline, subsampled for runtime."""
    green_ids = set(triage.loc[triage.color == "GREEN", "unit_id"])
    if green_ids:
        baseline = features[features["unit_id"].isin(green_ids)]
    else:
        baseline = features
    step = max(1, math.ceil(len(baseline) / MAX_BACKGROUND_ROWS))
    return baseline.drop(columns="unit_id").iloc[::step].to_numpy(dtype=float)


def _instance(
    features: pd.DataFrame, unit_id: str, feature_columns: list[str]
) -> np.ndarray:
    """One unit's feature row as a float array."""
    return features.loc[features.unit_id == unit_id, list(feature_columns)].to_numpy(
        dtype=float
    )[0]


def _explain_unit(
    explainer: shap.Explainer,
    decision: pd.Series,
    instance: np.ndarray,
    feature_columns: list[str],
) -> list[dict]:
    """Attribution rows for one flagged unit, ranked by |SHAP value|.

    shap 0.52 iterates axis 0: a single instance must be a
    ``(1, n_features)`` batch — a bare 1-D row would be read as
    ``n_features`` scalar instances.
    """
    channel = decision.worst_channel
    values = np.asarray(explainer(instance.reshape(1, -1)).values)[0]
    mode = defect_mode_for_channel(channel)
    code = reason_code(mode)
    total = float(np.abs(values).sum())

    rows: list[dict] = []
    for rank, index in enumerate(np.argsort(-np.abs(values)), start=1):
        magnitude = abs(float(values[index]))
        rows.append(
            {
                "unit_id": decision.unit_id,
                "color": decision.color,
                "reason_code": code,
                "defect_mode": mode,
                "worst_channel": channel,
                "rank": int(rank),
                "feature": feature_columns[index],
                "shap_value": float(values[index]),
                "feature_value": float(instance[index]),
                "weight": magnitude / max(total, _BACKGROUND_FLOOR),
            }
        )
    return rows


def explain_batch(
    model: ForecastModel,
    batch: pd.DataFrame,
    triage: pd.DataFrame,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Attributions + reason code for every Yellow/Red unit in ``triage``.

    One row per (unit, feature), ranked by |SHAP value| within the unit;
    ``weight`` normalizes |SHAP| per unit. Green units are not explained.
    """
    flagged = triage[triage.color != "GREEN"].reset_index(drop=True)
    if flagged.empty:
        return pd.DataFrame(columns=EXPLAIN_COLUMNS)

    features = build_features(batch).copy()
    features.insert(0, "unit_id", batch["unit_id"].to_numpy())
    background = _background_rows(features, triage)

    rows: list[dict] = []
    explainers: dict[str, shap.Explainer] = {}
    for _, decision in flagged.iterrows():
        channel = decision.worst_channel
        if channel not in explainers:
            explainers[channel] = shap.Explainer(
                _total_predictor(model, channel), background, seed=seed
            )
        instance = _instance(features, decision.unit_id, model.feature_columns)
        rows.extend(
            _explain_unit(
                explainers[channel], decision, instance, model.feature_columns
            )
        )
    return pd.DataFrame(rows, columns=EXPLAIN_COLUMNS)
