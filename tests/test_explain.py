"""MVP-6: SHAP attribution + physics reason codes for every flagged unit.

The core assertion is not "an explanation object exists" — it is that a Red
unit's explanation *names its planted defect driver*: U026 (gate oxide
degradation) must be explained by igss leakage features under code
ERR_MOSFET_GATE_DEGRADE, U044 by RDS(on) drift features under
ERR_MOSFET_RDS_DRIFT. Codes come from the fixed config registry; attribution
weights are normalized per unit.

SHAP 0.52 is locked as the primary (probe OK in Phase 0) — no fallback
re-litigation. Explainable function is the *total* terminal prediction
(physics extrapolation + GBDT correction), so attributions carry physical
units and the drift features can dominate.
"""

import numpy as np
import pandas as pd
import pytest

from burnsight.config import DEFECT_MODES, reason_code
from burnsight.explain import EXPLAIN_COLUMNS, explain_batch
from burnsight.forecast import build_features, predict_terminal, train_forecast_model
from burnsight.generator import generate_batch
from burnsight.screen import screen_population
from burnsight.triage import triage_batch

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
EXPLAIN = explain_batch(MODEL, TEST, TRIAGE)

EXPECTED_REDRIVERS = {
    "U026": ("GATE_OXIDE_DEGRADE", "ERR_MOSFET_GATE_DEGRADE", "igss_na"),
    "U044": ("RDS_DRIFT", "ERR_MOSFET_RDS_DRIFT", "rds_on_mohm"),
    "U076": ("THRESHOLD_SHIFT", "ERR_MOSFET_VGSTH_SHIFT", "vgsth_v"),
}


class TestReasonCodes:
    def test_red_units_name_the_planted_driver(self):
        for uid, (mode, code, channel) in EXPECTED_REDRIVERS.items():
            unit = EXPLAIN[EXPLAIN.unit_id == uid]
            assert len(unit) > 0, uid
            assert set(unit.defect_mode) == {mode}, uid
            assert set(unit.reason_code) == {code}, uid
            assert set(unit.worst_channel) == {channel}, uid

    def test_codes_align_with_triage_worst_channel(self):
        flagged = TRIAGE[TRIAGE.color != "GREEN"][["unit_id", "worst_channel"]]
        merged = EXPLAIN.merge(flagged, on="unit_id", suffixes=("", "_triage"))
        assert (merged.worst_channel == merged.worst_channel_triage).all()

    def test_yellow_units_get_codes_too(self):
        yellow_ids = set(TRIAGE.loc[TRIAGE.color == "YELLOW", "unit_id"])
        explained = set(EXPLAIN.loc[EXPLAIN.color == "YELLOW", "unit_id"])
        assert yellow_ids == explained
        for uid in yellow_ids:
            assert EXPLAIN.loc[EXPLAIN.unit_id == uid, "reason_code"].iloc[0] in {
                code for _mode, (_channel, code) in DEFECT_MODES.items()
            }

    def test_codes_come_from_the_fixed_registry(self):
        registry = {reason_code(mode) for mode in DEFECT_MODES}
        assert set(EXPLAIN.reason_code) <= registry
        assert set(EXPLAIN.defect_mode) <= set(DEFECT_MODES)


class TestAttributionNamesTheDriver:
    @pytest.mark.parametrize(
        "uid, prefix",
        [("U026", "igss_na"), ("U044", "rds_on_mohm"), ("U076", "vgsth_v")],
    )
    def test_top_ranked_feature_is_the_driving_channel(self, uid, prefix):
        top = EXPLAIN[(EXPLAIN.unit_id == uid) & (EXPLAIN["rank"] == 1)].iloc[0]
        assert top.feature.startswith(prefix), f"{uid}: {top.feature}"

    def test_weights_are_normalized_per_unit(self):
        sums = EXPLAIN.groupby("unit_id").weight.sum()
        assert np.allclose(sums.to_numpy(), 1.0, atol=1e-6)

    def test_rank_orders_by_absolute_attribution(self):
        for uid in EXPECTED_REDRIVERS:
            unit = EXPLAIN[EXPLAIN.unit_id == uid].sort_values("rank")
            magnitudes = np.abs(unit.shap_value.to_numpy())
            assert np.all(np.diff(magnitudes) <= 1e-12), uid
            assert list(unit["rank"]) == list(range(1, len(unit) + 1)), uid

    def test_feature_values_match_the_units_telemetry(self):
        features = build_features(TEST)
        for uid, (_mode, _code, _channel) in EXPECTED_REDRIVERS.items():
            top = EXPLAIN[(EXPLAIN.unit_id == uid) & (EXPLAIN["rank"] == 1)].iloc[0]
            expected = features.loc[TEST.unit_id == uid, top.feature].iloc[0]
            assert top.feature_value == pytest.approx(expected)

    def test_attributions_are_finite_and_signed(self):
        assert np.isfinite(EXPLAIN.shap_value).all()
        assert np.isfinite(EXPLAIN.weight).all()
        assert (EXPLAIN.weight >= 0).all() and (EXPLAIN.weight <= 1).all()
        assert (EXPLAIN.shap_value.abs() > 0).any()


class TestScopeAndContract:
    def test_explains_exactly_the_flagged_units(self):
        flagged = set(TRIAGE.loc[TRIAGE.color != "GREEN", "unit_id"])
        assert set(EXPLAIN.unit_id) == flagged

    def test_output_columns(self):
        assert list(EXPLAIN.columns) == list(EXPLAIN_COLUMNS)
        assert not EXPLAIN.isna().any().any()

    def test_all_green_triage_explains_nothing(self):
        greens = TRIAGE[TRIAGE.color == "GREEN"]
        nothing = explain_batch(MODEL, TEST, greens)
        assert len(nothing) == 0
        assert list(nothing.columns) == list(EXPLAIN_COLUMNS)

    def test_explanation_is_reproducible(self):
        subset = pd.concat(
            [
                TRIAGE[TRIAGE.unit_id == "U026"],
                TRIAGE[TRIAGE.color == "GREEN"].head(2),
            ],
            ignore_index=True,
        )
        first = explain_batch(MODEL, TEST, subset)
        second = explain_batch(MODEL, TEST, subset)
        pd.testing.assert_frame_equal(first, second)

    def test_works_when_triage_contains_no_green_units(self):
        flagged_only = TRIAGE[TRIAGE.unit_id == "U026"]
        result = explain_batch(MODEL, TEST, flagged_only)
        assert set(result.unit_id) == {"U026"}
        assert set(result.reason_code) == {"ERR_MOSFET_GATE_DEGRADE"}
