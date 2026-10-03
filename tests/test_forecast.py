"""MVP-4: 168h degradation predictor — features, accuracy, trajectory.

Accuracy tests run on a held-out synthetic lot (seed 42) that the model
never saw during training (seeds 1-4). Thresholds are set against the
spec windows: rds window 70 milliohm, igss window 100 nanoamp, vgsth
window 2.0 volt — every MAE must stay under ~5% of its window, which a
mean-predictor baseline could never do.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from burnsight.config import PARAM_CHANNELS, SPEC_LIMITS, channel_column
from burnsight.forecast import (
    DEFAULT_FEATURE_COLUMNS,
    build_features,
    forecast_trajectory,
    predict_terminal,
    train_forecast_model,
)
from burnsight.generator import generate_batch

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"

# Training pool: four lots the model fits on. Held-out: the golden seed-42 lot.
TRAIN_BATCH = pd.concat(
    [
        generate_batch(n_units=100, defect_count=3, borderline_count=5, seed=s)
        for s in (1, 2, 3, 4)
    ],
    ignore_index=True,
)
TEST_BATCH = generate_batch(n_units=100, defect_count=3, borderline_count=5, seed=42)

MODEL = train_forecast_model(TRAIN_BATCH)

# MAE thresholds: <= ~5% of each spec window (see module docstring).
MAE_LIMITS = {"rds_on_mohm": 3.0, "igss_na": 3.0, "vgsth_v": 0.12}


class TestFeatures:
    def test_features_cover_every_unit_and_are_finite(self):
        features = build_features(TEST_BATCH)
        assert len(features) == len(TEST_BATCH)
        assert np.isfinite(features.to_numpy(dtype=float)).all()

    def test_features_contain_no_terminal_columns(self):
        columns = list(build_features(TEST_BATCH).columns)
        assert all("168" not in c for c in columns), columns
        assert columns == list(DEFAULT_FEATURE_COLUMNS)

    def test_features_include_slopes_and_arrhenius_block(self):
        columns = set(build_features(TEST_BATCH).columns)
        assert "temp_c" in columns
        assert "arrhenius_af" in columns
        assert "stress_hours_96" in columns
        for ch in PARAM_CHANNELS:
            assert f"{ch}_slope_0_24" in columns
            assert f"{ch}_slope_24_96" in columns
            assert f"{ch}_drift_0_96" in columns
            assert f"{ch}_extrap_terminal" in columns

    def test_slope_features_match_hand_computation(self):
        features = build_features(TEST_BATCH)
        ch = "rds_on_mohm"
        expected = (TEST_BATCH[f"{ch}_24h"] - TEST_BATCH[f"{ch}_0h"]) / 24.0
        assert np.allclose(features[f"{ch}_slope_0_24"], expected)


class TestHeldOutAccuracy:
    @pytest.mark.parametrize("channel", PARAM_CHANNELS)
    def test_terminal_mae_within_threshold(self, channel):
        predictions = predict_terminal(MODEL, TEST_BATCH)
        actual = TEST_BATCH[f"{channel}_168h"]
        mae = float(np.mean(np.abs(predictions[f"{channel}_168h_pred"] - actual)))
        assert mae < MAE_LIMITS[channel], f"{channel} MAE={mae:.3f}"

    def test_predictions_finite_and_aligned(self):
        predictions = predict_terminal(MODEL, TEST_BATCH)
        assert list(predictions.unit_id) == list(TEST_BATCH.unit_id)
        assert np.isfinite(
            predictions[[f"{ch}_168h_pred" for ch in PARAM_CHANNELS]].to_numpy()
        ).all()

    def test_all_defects_predicted_beyond_their_spec_limit(self):
        """The terminal prediction must cross the line triage will judge on."""
        predictions = predict_terminal(MODEL, TEST_BATCH)
        defects = TEST_BATCH[TEST_BATCH.severity == "DEFECTIVE"]
        for _, row in defects.iterrows():
            mode_channel = {
                "GATE_OXIDE_DEGRADE": "igss_na",
                "RDS_DRIFT": "rds_on_mohm",
                "THRESHOLD_SHIFT": "vgsth_v",
            }[row["defect_mode"]]
            _lsl, usl = SPEC_LIMITS[mode_channel]
            predicted = predictions.loc[
                predictions.unit_id == row["unit_id"],
                f"{mode_channel}_168h_pred",
            ].iloc[0]
            assert predicted > usl, f"{row['unit_id']} predicted {predicted} <= {usl}"

    def test_terminal_columns_do_not_leak_into_predictions(self):
        """Zeroing every 168h column must not change a single prediction."""
        corrupted = TEST_BATCH.copy()
        for ch in PARAM_CHANNELS:
            corrupted[channel_column(ch, 168)] = 99999.0
        base = predict_terminal(MODEL, TEST_BATCH)
        poisoned = predict_terminal(MODEL, corrupted)
        for ch in PARAM_CHANNELS:
            assert np.allclose(base[f"{ch}_168h_pred"], poisoned[f"{ch}_168h_pred"]), ch

    def test_training_is_deterministic(self):
        second = train_forecast_model(TRAIN_BATCH)
        a = predict_terminal(MODEL, TEST_BATCH)
        b = predict_terminal(second, TEST_BATCH)
        for ch in PARAM_CHANNELS:
            assert np.allclose(a[f"{ch}_168h_pred"], b[f"{ch}_168h_pred"])


class TestTrajectory:
    def test_grid_and_shape(self):
        hours = (0, 24, 96, 120, 168)
        trajectory = forecast_trajectory(MODEL, TEST_BATCH, hours=hours)
        assert len(trajectory) == len(TEST_BATCH) * len(hours)
        assert sorted(trajectory.hour.unique()) == list(hours)
        assert list(trajectory.columns) == ["unit_id", "hour", *PARAM_CHANNELS]

    def test_starts_at_measured_baseline(self):
        trajectory = forecast_trajectory(MODEL, TEST_BATCH, hours=(0, 24, 96, 168))
        at_zero = trajectory[trajectory.hour == 0].reset_index(drop=True)
        for ch in PARAM_CHANNELS:
            assert np.allclose(at_zero[ch], TEST_BATCH[f"{ch}_0h"])

    def test_ends_at_predicted_terminal(self):
        terminal = predict_terminal(MODEL, TEST_BATCH)
        trajectory = forecast_trajectory(MODEL, TEST_BATCH, hours=(0, 24, 96, 168))
        at_end = trajectory[trajectory.hour == 168].reset_index(drop=True)
        for ch in PARAM_CHANNELS:
            assert np.allclose(at_end[ch], terminal[f"{ch}_168h_pred"], atol=1e-9)

    def test_trajectory_values_finite(self):
        trajectory = forecast_trajectory(MODEL, TEST_BATCH, hours=(0, 48, 96, 144, 168))
        assert np.isfinite(trajectory[list(PARAM_CHANNELS)].to_numpy()).all()
