"""MVP-3: population outlier screening — MAD z-scores + IsolationForest.

Physics note driving the test design: at T=0h the lot is a pre-stress
baseline, so planted defects look statistically normal there — only their
0h->24h drift kinetics separate them. The baseline-only test below pins
that fact so the drift signal stays a documented requirement, not an
implementation accident.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from burnsight.api import app
from burnsight.config import PARAM_CHANNELS, channel_column
from burnsight.generator import generate_batch
from burnsight.screen import (
    baseline_features,
    drift_features,
    robust_zscores,
    screen_population,
)

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
client = TestClient(app)

BATCH = generate_batch(n_units=100, defect_count=3, borderline_count=5, seed=42)
DEFECT_IDS = set(BATCH.loc[BATCH.severity == "DEFECTIVE", "unit_id"])
BORDERLINE_IDS = set(BATCH.loc[BATCH.severity == "BORDERLINE", "unit_id"])
NORMAL_IDS = set(BATCH.loc[BATCH.severity == "NORMAL", "unit_id"])


class TestRobustZ:
    def test_constant_column_gives_finite_zeros(self):
        z = robust_zscores(np.full(10, 5.0))
        assert np.isfinite(z).all()
        assert (z == 0).all()

    def test_single_extreme_value_gets_large_z(self):
        values = np.array([10.0, 10.2, 9.8, 10.1, 9.9, 10.3, 42.0])
        z = robust_zscores(values)
        assert abs(z[-1]) > 5
        assert (np.abs(z[:-1]) < 4).all()

    def test_normal_population_stays_below_four_sigma(self):
        rng = np.random.default_rng(0)
        z = robust_zscores(rng.normal(0, 1, 500))
        assert np.abs(z).max() < 4

    def test_zero_mad_with_variance_falls_back_to_std(self):
        # [1,1,1,2]: MAD collapses to 0 while spread exists -> std fallback
        z = robust_zscores(np.array([1.0, 1.0, 1.0, 2.0]))
        assert np.isfinite(z).all()
        assert abs(z[-1]) > 1


class TestFeatures:
    def test_baseline_features_are_zero_hour_columns(self):
        frame = baseline_features(BATCH)
        for ch in PARAM_CHANNELS:
            assert np.allclose(frame[ch], BATCH[f"{ch}_0h"])

    def test_drift_features_are_0h_to_24h_deltas(self):
        frame = drift_features(BATCH)
        for ch in PARAM_CHANNELS:
            expected = BATCH[f"{ch}_24h"] - BATCH[f"{ch}_0h"]
            assert np.allclose(frame[ch], expected)


class TestScreeningBehavior:
    def test_baseline_alone_cannot_see_planted_defects(self):
        """T=0h is pre-stress: baseline spread alone must show ~zero recall.

        This documents why the 0h->24h drift signal exists in Stage 1.
        """
        z = np.column_stack(
            [
                robust_zscores(baseline_features(BATCH)[ch].to_numpy())
                for ch in PARAM_CHANNELS
            ]
        )
        flagged = set(BATCH.loc[np.abs(z).max(axis=1) > 4.0, "unit_id"])
        assert flagged & DEFECT_IDS == set(), "defects must be invisible at T=0h"

    def test_all_planted_defects_flagged(self):
        result = screen_population(BATCH)
        flagged = set(result.loc[result.out_of_family, "unit_id"])
        assert DEFECT_IDS <= flagged, f"missed: {DEFECT_IDS - flagged}"

    def test_borderline_units_flagged_too(self):
        """Borderlines have kinetically abnormal early drift: out-of-family by definition."""
        result = screen_population(BATCH)
        flagged = set(result.loc[result.out_of_family, "unit_id"])
        assert BORDERLINE_IDS <= flagged, f"missed: {BORDERLINE_IDS - flagged}"

    def test_no_normal_units_flagged(self):
        result = screen_population(BATCH)
        flagged = set(result.loc[result.out_of_family, "unit_id"])
        false_positives = flagged & NORMAL_IDS
        assert len(false_positives) <= 2, f"FPR too high: {false_positives}"

    def test_drift_signal_alone_catches_every_defect(self):
        result = screen_population(BATCH)
        drift_flagged = set(result.loc[result.mad_drift, "unit_id"])
        assert DEFECT_IDS <= drift_flagged

    def test_screening_is_deterministic(self):
        first = screen_population(BATCH)
        second = screen_population(BATCH)
        pd.testing.assert_frame_equal(first, second)

    def test_output_schema_complete_and_finite(self):
        result = screen_population(BATCH)
        assert len(result) == len(BATCH)
        assert list(result.unit_id) == list(BATCH.unit_id)
        for column in ("mad_baseline", "mad_drift", "iforest", "out_of_family"):
            assert result[column].dtype == bool, column
        assert not result.isna().any().any()
        assert np.isfinite(result.max_abs_z_drift).all()
        assert np.isfinite(result.max_abs_z_baseline).all()

    def test_flagged_rows_carry_their_signals(self):
        result = screen_population(BATCH)
        flagged = result[result.out_of_family]
        assert (flagged.mad_baseline | flagged.mad_drift | flagged.iforest).all()

    def test_works_on_early_only_batch_without_labels(self):
        early = pd.read_csv(SAMPLES / "batch_early_96h.csv")
        result = screen_population(early)
        assert len(result) == 100
        assert (
            set(result.loc[result.out_of_family, "unit_id"]) & DEFECT_IDS == DEFECT_IDS
        )


class TestScreenEndpoint:
    def test_screen_golden_flags_planted_defects(self):
        response = client.post(
            "/api/screen",
            files={
                "file": (
                    "batch_golden.csv",
                    (SAMPLES / "batch_golden.csv").read_bytes(),
                    "text/csv",
                )
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["n_units"] == 100
        flagged = set(body["flagged_units"])
        assert DEFECT_IDS <= flagged
        assert body["n_flagged"] == len(flagged)
        assert body["n_clean"] == 100 - body["n_flagged"]
        assert 0 < body["n_flagged"] < 20

    def test_screen_early_batch_accepted(self):
        response = client.post(
            "/api/screen",
            files={
                "file": (
                    "batch_early_96h.csv",
                    (SAMPLES / "batch_early_96h.csv").read_bytes(),
                    "text/csv",
                )
            },
        )
        assert response.status_code == 200
        assert response.json()["n_units"] == 100

    def test_screen_empty_rejected(self):
        response = client.post(
            "/api/screen",
            files={"file": ("batch_empty.csv", b"", "text/csv")},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "EMPTY_INPUT"
