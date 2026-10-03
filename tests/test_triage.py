"""MVP-5: asymmetric risk triage — alpha=10*beta loss matrix over evidence.

Three evidence channels combine into a Green/Yellow/Red call:

- MVP-3 outlier flag (population screening),
- MVP-4 terminal forecast vs spec limits (USL/LSL),
- forecast confidence (per-channel model error sigma).

The loss matrix is not decoration: its escalation threshold
``p* = beta/(alpha+beta)`` decides when statistical risk alone forces an
escalation, so alpha=10beta must flip borderline/uncertain cases from what a
symmetric matrix would call Green. Thresholds are anchored to the spec: the
near-limit guard (0.8) is the project's own borderline definition (>=80%
toward a limit), and red requires the prediction to clear the limit by
2 sigma of model error — none are tuned to the demo batch.
"""

import numpy as np
import pandas as pd
import pytest

from burnsight.config import PARAM_CHANNELS, SPEC_LIMITS
from burnsight.forecast import predict_terminal, train_forecast_model
from burnsight.generator import generate_batch
from burnsight.screen import screen_population
from burnsight.triage import (
    REASON_BEYOND_LIMIT,
    REASON_ESCALATED,
    REASON_NEAR_LIMIT,
    REASON_OUT_OF_FAMILY,
    LossMatrix,
    expected_loss,
    triage_batch,
)

TRAIN = pd.concat(
    [
        generate_batch(n_units=100, defect_count=3, borderline_count=5, seed=s)
        for s in (1, 2, 3, 4)
    ],
    ignore_index=True,
)
TEST = generate_batch(n_units=100, defect_count=3, borderline_count=5, seed=42)
MODEL = train_forecast_model(TRAIN)
FORECAST = predict_terminal(MODEL, TEST)
SCREEN = screen_population(TEST)
TRIAGE = triage_batch(FORECAST, SCREEN, MODEL.error_std)

DEFECT_IDS = set(TEST.loc[TEST.severity == "DEFECTIVE", "unit_id"])
BORDERLINE_IDS = set(TEST.loc[TEST.severity == "BORDERLINE", "unit_id"])
NORMAL_IDS = set(TEST.loc[TEST.severity == "NORMAL", "unit_id"])


def _subset(uid: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast = FORECAST[FORECAST.unit_id == uid].reset_index(drop=True)
    screen = SCREEN.loc[SCREEN.unit_id == uid, ["unit_id", "out_of_family"]]
    return forecast, screen.reset_index(drop=True)


def _with_terminal(forecast: pd.DataFrame, **overrides: float) -> pd.DataFrame:
    result = forecast.copy()
    for channel, value in overrides.items():
        result[f"{channel}_168h_pred"] = value
    return result


def _single(uid: str, *, screen_flag: bool | None = None, **overrides: float):
    forecast, screen = _subset(uid)
    if screen_flag is not None:
        screen["out_of_family"] = screen_flag
    return _with_terminal(forecast, **overrides), screen


class TestLossMatrix:
    def test_default_is_alpha_10_beta(self):
        matrix = LossMatrix()
        assert matrix.alpha == 10.0
        assert matrix.beta == 1.0
        assert matrix.alpha == 10 * matrix.beta

    def test_matrix_costs(self):
        matrix = LossMatrix()
        assert matrix.loss("GREEN", "DEFECTIVE") == 10.0  # the miss
        assert matrix.loss("YELLOW", "NORMAL") == 1.0  # the false alarm
        assert matrix.loss("RED", "BORDERLINE") == 1.0  # scrapping a pass
        assert matrix.loss("GREEN", "NORMAL") == 0.0
        assert matrix.loss("GREEN", "BORDERLINE") == 0.0  # passes 168h by def
        assert matrix.loss("YELLOW", "DEFECTIVE") == 0.0  # caught in review
        assert matrix.loss("RED", "DEFECTIVE") == 0.0

    def test_configurable_alpha_beta(self):
        matrix = LossMatrix(alpha=25.0, beta=5.0)
        assert matrix.loss("GREEN", "DEFECTIVE") == 25.0
        assert matrix.loss("YELLOW", "NORMAL") == 5.0
        assert matrix.escalation_threshold == pytest.approx(5.0 / 30.0)

    def test_escalation_threshold_moves_with_asymmetry(self):
        assert LossMatrix(alpha=10.0, beta=1.0).escalation_threshold == pytest.approx(
            1 / 11
        )
        assert LossMatrix(alpha=1.0, beta=1.0).escalation_threshold == pytest.approx(
            0.5
        )

    def test_unknown_color_or_truth_rejected(self):
        matrix = LossMatrix()
        with pytest.raises(ValueError):
            matrix.loss("AMBER", "NORMAL")
        with pytest.raises(ValueError):
            matrix.loss("GREEN", "ALIEN")

    def test_invalid_parameters_rejected(self):
        with pytest.raises(ValueError):
            LossMatrix(alpha=0.0)
        with pytest.raises(ValueError):
            LossMatrix(alpha=1.0, beta=2.0)  # asymmetric means FN costs more

    def test_expected_loss_of_all_green_batch(self):
        decisions = TRIAGE.copy()
        decisions["color"] = "GREEN"
        assert expected_loss(decisions, TEST[["unit_id", "severity"]]) == pytest.approx(
            30.0
        )

    def test_expected_loss_rejects_missing_ground_truth(self):
        with pytest.raises(ValueError, match="missing ground truth"):
            expected_loss(TRIAGE, TEST.iloc[3:][["unit_id", "severity"]])


class TestRules:
    def test_clean_unit_is_green(self):
        uid = sorted(NORMAL_IDS)[0]
        forecast, screen = _single(uid, screen_flag=False)
        row = triage_batch(forecast, screen, MODEL.error_std)
        assert row.iloc[0].color == "GREEN"
        assert row.iloc[0].reasons == ""

    def test_confident_beyond_limit_is_red_without_screen_flag(self):
        uid = sorted(NORMAL_IDS)[0]
        forecast, screen = _single(uid, screen_flag=False, rds_on_mohm=175.0)
        row = triage_batch(forecast, screen, MODEL.error_std)
        assert row.iloc[0].color == "RED"
        assert REASON_BEYOND_LIMIT in row.iloc[0].reasons

    def test_below_lsl_is_red_too(self):
        uid = sorted(NORMAL_IDS)[0]
        forecast, screen = _single(uid, screen_flag=False, vgsth_v=2.5)
        row = triage_batch(forecast, screen, MODEL.error_std)
        assert row.iloc[0].color == "RED"

    def test_straddling_forecast_escalates_yellow_never_green(self):
        """Predicted 149.5 +/- 1.9 (2*sigma): may or may not fail -> Yellow."""
        uid = sorted(NORMAL_IDS)[0]
        forecast, screen = _single(uid, screen_flag=False, rds_on_mohm=149.5)
        row = triage_batch(forecast, screen, MODEL.error_std)
        assert row.iloc[0].color == "YELLOW"
        assert row.iloc[0].color != "GREEN"

    def test_near_limit_in_family_is_yellow(self):
        """Headroom 15% of window: spec guard (80% consumed) trips it."""
        uid = sorted(NORMAL_IDS)[0]
        forecast, screen = _single(uid, screen_flag=False, rds_on_mohm=139.5)
        row = triage_batch(forecast, screen, MODEL.error_std)
        assert row.iloc[0].color == "YELLOW"
        assert REASON_NEAR_LIMIT in row.iloc[0].reasons

    def test_out_of_family_flag_forces_yellow_despite_safe_forecast(self):
        uid = sorted(NORMAL_IDS)[0]
        forecast, screen = _single(uid, screen_flag=True)
        row = triage_batch(forecast, screen, MODEL.error_std)
        assert row.iloc[0].color == "YELLOW"
        assert REASON_OUT_OF_FAMILY in row.iloc[0].reasons

    def test_uncertain_forecast_escalates_via_loss_matrix(self):
        """Wide sigma mid-window: p(fail)=0.35 > 1/11 -> Yellow, not Green."""
        uid = sorted(NORMAL_IDS)[0]
        forecast, screen = _single(uid, screen_flag=False, rds_on_mohm=135.0)
        wide = dict(MODEL.error_std)
        wide["rds_on_mohm"] = 40.0
        row = triage_batch(forecast, screen, wide)
        assert row.iloc[0].color == "YELLOW"
        assert REASON_ESCALATED in row.iloc[0].reasons

    def test_screen_is_optional(self):
        uid = sorted(DEFECT_IDS)[0]
        forecast, _ = _subset(uid)
        row = triage_batch(forecast, None, MODEL.error_std)
        assert row.iloc[0].color == "RED"

    def test_missing_error_std_channel_rejected(self):
        forecast, screen = _subset(sorted(NORMAL_IDS)[0])
        partial = {k: v for k, v in MODEL.error_std.items() if k != "igss_na"}
        with pytest.raises(ValueError):
            triage_batch(forecast, screen, partial)

    def test_output_schema(self):
        assert list(TRIAGE.columns) == [
            "unit_id",
            "color",
            "worst_channel",
            "predicted_168h",
            "headroom",
            "risk",
            "reasons",
        ]
        assert len(TRIAGE) == 100
        assert set(TRIAGE.color) <= {"GREEN", "YELLOW", "RED"}
        assert not TRIAGE.isna().any().any()
        assert TRIAGE.risk.between(0.0, 1.0).all()


class TestFalseNegativeSafety:
    """The pitch promise: no failing unit ever calls home Green."""

    def test_no_defect_is_green(self):
        assert DEFECT_IDS & set(TRIAGE.loc[TRIAGE.color == "GREEN", "unit_id"]) == set()

    def test_all_defects_are_red(self):
        red = set(TRIAGE.loc[TRIAGE.color == "RED", "unit_id"])
        assert DEFECT_IDS <= red, f"missed: {DEFECT_IDS - red}"

    def test_borderlines_never_green(self):
        green = set(TRIAGE.loc[TRIAGE.color == "GREEN", "unit_id"])
        assert BORDERLINE_IDS & green == set()

    def test_no_normal_is_red(self):
        red = set(TRIAGE.loc[TRIAGE.color == "RED", "unit_id"])
        assert NORMAL_IDS & red == set()

    def test_full_batch_split(self):
        counts = TRIAGE.color.value_counts()
        assert counts.get("RED", 0) == 3
        assert counts.get("YELLOW", 0) == 5
        assert counts.get("GREEN", 0) == 92

    def test_every_escalation_carries_a_reason(self):
        escalated = TRIAGE[TRIAGE.color != "GREEN"]
        assert (escalated.reasons != "").all()
        greens = TRIAGE[TRIAGE.color == "GREEN"]
        assert (greens.reasons == "").all()


class TestAsymmetricBoundary:
    def test_alpha_10_beta_flips_the_decision_a_symmetric_matrix_would_miss(self):
        """Same evidence, two matrices: only the asymmetric one escalates."""
        uid = sorted(NORMAL_IDS)[0]
        forecast, screen = _single(uid, screen_flag=False, rds_on_mohm=130.0)
        uncertain = dict(MODEL.error_std)
        uncertain["rds_on_mohm"] = 40.0  # p(fail) lands between 1/11 and 1/2

        asymmetric = triage_batch(
            forecast, screen, uncertain, matrix=LossMatrix(alpha=10.0, beta=1.0)
        )
        symmetric = triage_batch(
            forecast, screen, uncertain, matrix=LossMatrix(alpha=1.0, beta=1.0)
        )

        assert asymmetric.iloc[0].color == "YELLOW"
        assert symmetric.iloc[0].color == "GREEN"

    def test_real_triage_beats_the_all_green_baseline(self):
        baseline = TRIAGE.copy()
        baseline["color"] = "GREEN"
        ours = expected_loss(TRIAGE, TEST[["unit_id", "severity"]])
        theirs = expected_loss(baseline, TEST[["unit_id", "severity"]])
        assert ours < theirs
        assert ours == pytest.approx(5.0)  # five borderline reviews x beta
