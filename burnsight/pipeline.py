"""MVP-7: dashboard pipeline — screen -> forecast -> triage -> explain -> chart.

``run_triage`` turns validated upload records into the JSON payload the
dashboard renders: badge table for every unit, and for each Yellow/Red unit
the MVP-6 explanation plus a driver-trajectory chart bundle (measured
0/24/96 h points, 0-168 h forecast curve, spec limits).

The reference forecast model is trained once per process on four synthetic
reference lots (seeds 1-4) and cached — uploads are scored, never trained on.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from burnsight.audit import append_audit, build_audit_record
from burnsight.config import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    SPEC_LIMITS,
    channel_column,
)
from burnsight.explain import explain_batch
from burnsight.forecast import (
    forecast_trajectory,
    predict_terminal,
    train_forecast_model,
)
from burnsight.generator import generate_batch
from burnsight.screen import screen_population
from burnsight.triage import LossMatrix, triage_batch

REFERENCE_SEEDS = (1, 2, 3, 4)
TRAJECTORY_HOURS = (0, 24, 96, 120, 144, 168)
MEASURED_HOURS = (0, 24, 96)
MAX_EXPLANATION_FEATURES = 8
TRIAGE_COLORS = ("GREEN", "YELLOW", "RED")


@lru_cache(maxsize=1)
def reference_model():
    """Forecast model trained once per process on reference lots 1-4."""
    train = pd.concat(
        [
            generate_batch(n_units=100, defect_count=3, borderline_count=5, seed=seed)
            for seed in REFERENCE_SEEDS
        ],
        ignore_index=True,
    )
    return train_forecast_model(train)


def run_triage(records: list[dict], alpha: float = DEFAULT_ALPHA) -> dict:
    """Full MVP pipeline over one uploaded batch, shaped for the dashboard."""
    batch = pd.DataFrame.from_records(records)
    model = reference_model()
    matrix = LossMatrix(alpha=alpha, beta=DEFAULT_BETA)
    decisions = triage_batch(
        predict_terminal(model, batch),
        screen_population(batch),
        model.error_std,
        matrix=matrix,
    )
    explanation = explain_batch(model, batch, decisions)
    trajectory = forecast_trajectory(model, batch, hours=TRAJECTORY_HOURS)
    counts = decisions.color.value_counts()
    record = build_audit_record(records, model, matrix, decisions, explanation)
    append_audit(record)  # one JSONL line per analysis run (MVP-8)
    return {
        "n_units": int(len(decisions)),
        "counts": {color: int(counts.get(color, 0)) for color in TRIAGE_COLORS},
        "alpha": float(matrix.alpha),
        "beta": float(matrix.beta),
        "escalation_threshold": float(matrix.escalation_threshold),
        "audit_id": record["audit_id"],
        "input_digest": record["input_digest"],
        "units": _unit_rows(decisions),
        "flagged": _flagged_details(decisions, explanation, trajectory, batch),
    }


def _unit_rows(decisions: pd.DataFrame) -> list[dict]:
    """Badge-table rows: one entry per unit in upload order."""
    return [
        {
            "unit_id": str(row.unit_id),
            "color": row.color,
            "worst_channel": row.worst_channel,
            "predicted_168h": float(row.predicted_168h),
            "headroom": float(row.headroom),
            "risk": float(row.risk),
            "reasons": row.reasons,
        }
        for row in decisions.itertuples(index=False)
    ]


def _flagged_details(
    decisions: pd.DataFrame,
    explanation: pd.DataFrame,
    trajectory: pd.DataFrame,
    batch: pd.DataFrame,
) -> list[dict]:
    """Expandable detail for every Yellow/Red unit: reason code + SHAP + chart."""
    flagged = decisions[decisions.color != "GREEN"]
    details = []
    for row in flagged.itertuples(index=False):
        unit_explanation = explanation[explanation.unit_id == row.unit_id].sort_values(
            "rank"
        )
        details.append(
            {
                "unit_id": str(row.unit_id),
                "color": row.color,
                "worst_channel": row.worst_channel,
                "predicted_168h": float(row.predicted_168h),
                "reasons": row.reasons,
                "reason_code": str(unit_explanation["reason_code"].iloc[0]),
                "defect_mode": str(unit_explanation["defect_mode"].iloc[0]),
                "explanation": _explanation_rows(unit_explanation),
                "chart": _chart_payload(row, trajectory, batch),
            }
        )
    return details


def _explanation_rows(unit_explanation: pd.DataFrame) -> list[dict]:
    """Top SHAP features, already rank-ordered by |attribution|."""
    top = unit_explanation.head(MAX_EXPLANATION_FEATURES)
    return [
        {
            "feature": str(item.feature),
            "shap_value": float(item.shap_value),
            "feature_value": float(item.feature_value),
            "weight": float(item.weight),
        }
        for item in top.itertuples(index=False)
    ]


def _chart_payload(row, trajectory: pd.DataFrame, batch: pd.DataFrame) -> dict:
    """Driver channel: measured points, forecast curve, spec limit lines."""
    channel = row.worst_channel
    curve = trajectory[trajectory.unit_id == row.unit_id].sort_values("hour")
    telemetry = batch[batch.unit_id == row.unit_id].iloc[0]
    lsl, usl = SPEC_LIMITS[channel]
    return {
        "channel": channel,
        "hours": [int(hour) for hour in curve["hour"]],
        "forecast": [float(value) for value in curve[channel]],
        "measured": {
            str(hour): float(telemetry[channel_column(channel, hour)])
            for hour in MEASURED_HOURS
        },
        "predicted_168h": float(row.predicted_168h),
        "limits": {"lsl": float(lsl), "usl": float(usl)},
    }
