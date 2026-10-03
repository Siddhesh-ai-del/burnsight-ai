"""MVP-5: asymmetric risk triage — Green/Yellow/Red from three evidence lines.

Decision rule (in order)::

    RED    = a channel's forecast clears a spec limit by > red_sigma of model
             error (confident failure; the limit check covers both USL and LSL)
    YELLOW = otherwise, any of:
             - population screening flagged the unit out of family (MVP-3),
             - >= 80% of the window toward the failure limit is consumed
               (NEAR_LIMIT_GUARD — the project's own borderline definition),
             - statistical risk exceeds the loss matrix threshold
               p* = beta / (alpha + beta) (0.091 for alpha=10beta), which is
               where forecast uncertainty and the asymmetric cost of a miss
               force an escalation a symmetric matrix would let pass.
    GREEN  = none of the above — and never for anything with a reason attached.

Failure-side note: all three registered defect modes approach their limit from
*inside and above* (leakage up, RDS(on) up, V(th) up), so the near-limit guard
measures headroom to the USL. Red still checks the LSL, catching any unit
confidently below spec regardless of direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from burnsight.config import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    NEAR_LIMIT_GUARD,
    PARAM_CHANNELS,
    RED_SIGMA,
    SEVERITY_LEVELS,
    SPEC_LIMITS,
    TRIAGE_COLORS,
)

REASON_BEYOND_LIMIT = "BEYOND_LIMIT"
REASON_OUT_OF_FAMILY = "OUT_OF_FAMILY"
REASON_NEAR_LIMIT = "NEAR_LIMIT"
REASON_ESCALATED = "ESCALATED_BY_LOSS_MATRIX"

OUTPUT_COLUMNS = (
    "unit_id",
    "color",
    "worst_channel",
    "predicted_168h",
    "headroom",
    "risk",
    "reasons",
)


@dataclass(frozen=True)
class LossMatrix:
    """Cost of (triage color, ground truth): alpha for a miss, beta for a review."""

    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("alpha and beta must be positive costs")
        if self.alpha < self.beta:
            raise ValueError(
                "asymmetric policy: missing a failing unit must cost at "
                "least as much as a false alarm"
            )

    @property
    def escalation_threshold(self) -> float:
        """p* = beta/(alpha+beta): escalate whenever evidence risk exceeds it."""
        return self.beta / (self.alpha + self.beta)

    def loss(self, color: str, truth: str) -> float:
        if color not in TRIAGE_COLORS:
            raise ValueError(f"unknown triage color: {color!r}")
        if truth not in SEVERITY_LEVELS:
            raise ValueError(f"unknown severity: {truth!r}")
        if truth == "DEFECTIVE":
            return self.alpha if color == "GREEN" else 0.0  # caught by review
        return 0.0 if color == "GREEN" else self.beta  # passing unit reviewed


DEFAULT_LOSS_MATRIX = LossMatrix()


def expected_loss(
    decisions: pd.DataFrame,
    ground: pd.DataFrame,
    matrix: LossMatrix = DEFAULT_LOSS_MATRIX,
) -> float:
    """Total loss of a triage frame against ground truth (unit_id, severity)."""
    merged = decisions[["unit_id", "color"]].merge(ground, on="unit_id", how="left")
    if merged["severity"].isna().any():
        missing = merged.loc[merged["severity"].isna(), "unit_id"].tolist()
        raise ValueError(f"missing ground truth for units: {missing}")
    return float(
        sum(
            matrix.loss(color, severity)
            for color, severity in zip(merged["color"], merged["severity"])
        )
    )


def _phi(values: np.ndarray) -> np.ndarray:
    """Standard normal CDF without pulling in scipy."""
    return np.fromiter(
        (0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in values),
        dtype=float,
        count=len(values),
    )


def triage_batch(
    forecast: pd.DataFrame,
    screen_result: pd.DataFrame | None,
    error_std: Mapping[str, float],
    matrix: LossMatrix = DEFAULT_LOSS_MATRIX,
    guard_fraction: float = NEAR_LIMIT_GUARD,
    red_sigma: float = RED_SIGMA,
) -> pd.DataFrame:
    """Color every unit Green/Yellow/Red. One row per forecast row, order kept.

    ``error_std`` must cover every channel (missing keys are rejected) — the
    escalation logic is only as honest as its confidence input.
    """
    missing = [ch for ch in PARAM_CHANNELS if ch not in error_std]
    if missing:
        raise ValueError(f"error_std missing channel(s): {missing}")

    frame = forecast.reset_index(drop=True).copy()
    if screen_result is not None:
        flags = screen_result[["unit_id", "out_of_family"]]
        frame = frame.merge(flags, on="unit_id", how="left")
        frame["out_of_family"] = frame["out_of_family"].fillna(False).astype(bool)
    else:
        frame["out_of_family"] = False

    predictions = {
        ch: frame[f"{ch}_168h_pred"].to_numpy(dtype=float) for ch in PARAM_CHANNELS
    }
    # Failure-side headroom (fraction of window left before the USL) and
    # two-sided statistical risk of landing outside either spec limit.
    headroom: dict[str, np.ndarray] = {}
    risk: dict[str, np.ndarray] = {}
    beyond: dict[str, np.ndarray] = {}
    for channel in PARAM_CHANNELS:
        lsl, usl = SPEC_LIMITS[channel]
        sigma = float(error_std[channel])
        distance_upper = usl - predictions[channel]
        distance_lower = predictions[channel] - lsl
        headroom[channel] = distance_upper / (usl - lsl)
        risk[channel] = np.maximum(
            _phi(-distance_upper / sigma), _phi(-distance_lower / sigma)
        )
        beyond[channel] = (distance_upper < -red_sigma * sigma) | (
            distance_lower < -red_sigma * sigma
        )

    headroom_matrix = np.vstack([headroom[ch] for ch in PARAM_CHANNELS])
    worst_index = np.argmin(headroom_matrix, axis=0)
    risk_vector = np.max(np.vstack([risk[ch] for ch in PARAM_CHANNELS]), axis=0)
    beyond_vector = np.logical_or.reduce([beyond[ch] for ch in PARAM_CHANNELS])
    flagged = frame["out_of_family"].to_numpy(dtype=bool)

    colors: list[str] = []
    reasons_out: list[str] = []
    for index in range(len(frame)):
        worst = PARAM_CHANNELS[worst_index[index]]
        reasons: list[str] = []
        if beyond_vector[index]:
            reasons.append(REASON_BEYOND_LIMIT)
        if flagged[index]:
            reasons.append(REASON_OUT_OF_FAMILY)
        if headroom_matrix[worst_index[index], index] <= 1.0 - guard_fraction:
            reasons.append(REASON_NEAR_LIMIT)
        if risk_vector[index] > matrix.escalation_threshold:
            reasons.append(REASON_ESCALATED)

        if beyond_vector[index]:
            colors.append("RED")
        elif reasons:
            colors.append("YELLOW")
        else:
            colors.append("GREEN")
        reasons_out.append("|".join(reasons))

    return pd.DataFrame(
        {
            "unit_id": frame["unit_id"].to_numpy(),
            "color": colors,
            "worst_channel": [PARAM_CHANNELS[i] for i in worst_index],
            "predicted_168h": [
                predictions[PARAM_CHANNELS[i]][n] for n, i in enumerate(worst_index)
            ],
            "headroom": headroom_matrix[worst_index, np.arange(len(frame))],
            "risk": risk_vector,
            "reasons": reasons_out,
        },
        columns=OUTPUT_COLUMNS,
    )
