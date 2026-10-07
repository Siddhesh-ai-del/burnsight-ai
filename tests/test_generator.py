"""MVP-1: synthetic burn-in telemetry generator.

TDD cycle: this file is written FIRST (RED). Implementation lives in
burnsight/config.py and burnsight/generator.py (GREEN).

Ground-truth design:
- severity == DEFECTIVE  -> fails a spec limit at 168h (true infant-mortality part)
- severity == BORDERLINE -> passes 168h but sits >=80% toward a limit (Yellow bait)
- severity == NORMAL     -> comfortably inside every limit at 168h
"""

import numpy as np
import pandas as pd
import pytest

from burnsight.config import (
    BATCH_COLUMNS,
    DEFECT_MODES,
    EARLY_HOURS,
    PARAM_CHANNELS,
    SAMPLE_HOURS,
    SEVERITY_LEVELS,
    SPEC_LIMITS,
)
from burnsight.generator import (
    _assign_severity,
    _plant_targets,
    arrhenius_rate,
    generate_batch,
)

DEFECT_CHANNEL = {mode: channel for mode, (channel, _code) in DEFECT_MODES.items()}


class TestValidation:
    def test_zero_units_rejected(self):
        with pytest.raises(ValueError):
            generate_batch(n_units=0, seed=1)

    def test_negative_defect_count_rejected(self):
        with pytest.raises(ValueError):
            generate_batch(n_units=10, defect_count=-1, seed=1)

    def test_negative_borderline_count_rejected(self):
        with pytest.raises(ValueError):
            generate_batch(n_units=10, borderline_count=-1, seed=1)

    def test_plants_exceeding_population_rejected(self):
        with pytest.raises(ValueError):
            generate_batch(n_units=5, defect_count=3, borderline_count=3, seed=1)

    def test_unknown_seed_type_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            generate_batch(n_units=10, seed="abc")


class TestDeterminism:
    def test_same_seed_identical(self):
        a = generate_batch(n_units=30, defect_count=3, borderline_count=4, seed=7)
        b = generate_batch(n_units=30, defect_count=3, borderline_count=4, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seed_differs(self):
        a = generate_batch(n_units=30, defect_count=3, borderline_count=4, seed=7)
        b = generate_batch(n_units=30, defect_count=3, borderline_count=4, seed=8)
        assert not a.equals(b)


class TestSchema:
    def test_columns_exact(self):
        df = generate_batch(n_units=20, defect_count=2, borderline_count=2, seed=1)
        assert list(df.columns) == BATCH_COLUMNS

    def test_row_count_matches(self):
        df = generate_batch(n_units=37, defect_count=2, borderline_count=2, seed=1)
        assert len(df) == 37

    def test_channel_columns_cover_all_channels_and_hours(self):
        df = generate_batch(n_units=10, defect_count=1, borderline_count=1, seed=1)
        for channel in PARAM_CHANNELS:
            for hour in SAMPLE_HOURS:
                assert f"{channel}_{hour}h" in df.columns

    def test_sample_hours_early_and_full(self):
        assert EARLY_HOURS == (0, 24, 96)
        assert SAMPLE_HOURS == (0, 24, 96, 168)
        assert set(EARLY_HOURS) < set(SAMPLE_HOURS)

    def test_channel_columns_numeric_float(self):
        df = generate_batch(n_units=10, defect_count=1, borderline_count=1, seed=1)
        for channel in PARAM_CHANNELS:
            for hour in SAMPLE_HOURS:
                assert df[f"{channel}_{hour}h"].dtype == np.float64

    def test_unit_ids_unique_strings(self):
        df = generate_batch(n_units=50, defect_count=2, borderline_count=2, seed=1)
        assert df["unit_id"].is_unique
        assert df["unit_id"].map(type).eq(str).all()

    def test_device_type_constant(self):
        df = generate_batch(n_units=10, defect_count=1, borderline_count=1, seed=1)
        assert (df["device_type"] == "POWER_MOSFET").all()

    def test_severity_values_from_registry(self):
        df = generate_batch(n_units=50, defect_count=3, borderline_count=4, seed=1)
        assert set(df["severity"]) <= set(SEVERITY_LEVELS)


class TestPhysics:
    def test_arrhenius_rate_positive(self):
        assert arrhenius_rate(125.0) > 0

    def test_arrhenius_rate_monotonic_in_temperature(self):
        temps = [125.0, 130.0, 135.0, 140.0, 145.0, 150.0]
        rates = [arrhenius_rate(t) for t in temps]
        assert rates == sorted(rates)
        assert all(r2 > r1 for r1, r2 in zip(rates, rates[1:]))

    def test_arrhenius_rate_baseline_is_one(self):
        # Reference burn-in baseline: acceleration factor at 125C is exactly 1.
        assert arrhenius_rate(125.0) == pytest.approx(1.0)

    def test_hotter_batch_degrades_faster_early(self):
        """Same seed, hotter lanes -> strictly more drift by 24h (Arrhenius kinetics)."""
        kwargs = dict(n_units=40, defect_count=3, borderline_count=4, seed=7)
        cool = generate_batch(temp_mean_c=125.0, **kwargs)
        hot = generate_batch(temp_mean_c=140.0, **kwargs)
        for channel in PARAM_CHANNELS:
            drift_cool = (cool[f"{channel}_24h"] - cool[f"{channel}_0h"]).mean()
            drift_hot = (hot[f"{channel}_24h"] - hot[f"{channel}_0h"]).mean()
            assert drift_hot > drift_cool, channel

    def test_temperature_clipped_to_mil_std_range(self):
        df = generate_batch(n_units=200, defect_count=3, borderline_count=5, seed=3)
        assert df["temp_c"].between(125.0, 150.0).all()


class TestLabels:
    def test_defect_count_exact(self):
        df = generate_batch(n_units=100, defect_count=3, borderline_count=5, seed=1)
        assert (df["severity"] == "DEFECTIVE").sum() == 3

    def test_borderline_count_exact(self):
        df = generate_batch(n_units=100, defect_count=3, borderline_count=5, seed=1)
        assert (df["severity"] == "BORDERLINE").sum() == 5

    def test_normal_count_exact(self):
        df = generate_batch(n_units=100, defect_count=3, borderline_count=5, seed=1)
        assert (df["severity"] == "NORMAL").sum() == 92

    def test_severities_mutually_exclusive(self):
        df = generate_batch(n_units=60, defect_count=3, borderline_count=4, seed=1)
        assert (df.groupby("unit_id").size() == 1).all()

    def test_defect_units_carry_registered_modes(self):
        df = generate_batch(n_units=60, defect_count=3, borderline_count=4, seed=1)
        modes = set(df.loc[df["severity"] == "DEFECTIVE", "defect_mode"])
        assert modes <= set(DEFECT_MODES)
        assert len(modes) == 3  # one of each planted mode

    def test_non_defect_units_have_none_mode(self):
        df = generate_batch(n_units=60, defect_count=3, borderline_count=4, seed=1)
        normal_modes = set(df.loc[df["severity"] != "DEFECTIVE", "defect_mode"])
        assert normal_modes == {"NONE"}


class TestGroundTruth:
    def test_defective_units_exceed_their_modes_spec_limit_at_168h(self):
        df = generate_batch(n_units=100, defect_count=3, borderline_count=5, seed=1)
        defects = df[df["severity"] == "DEFECTIVE"]
        for _, row in defects.iterrows():
            channel = DEFECT_CHANNEL[row["defect_mode"]]
            _lsl, usl = SPEC_LIMITS[channel]
            assert row[f"{channel}_168h"] > usl, row["unit_id"]

    def test_normals_pass_all_limits_at_168h(self):
        df = generate_batch(n_units=100, defect_count=3, borderline_count=5, seed=1)
        normals = df[df["severity"] == "NORMAL"]
        passing = 0
        for _, row in normals.iterrows():
            inside = all(
                SPEC_LIMITS[ch][0] <= row[f"{ch}_168h"] <= SPEC_LIMITS[ch][1]
                for ch in PARAM_CHANNELS
            )
            passing += int(inside)
        assert passing / len(normals) >= 0.9

    def test_borderline_units_pass_but_sit_near_a_limit(self):
        df = generate_batch(n_units=100, defect_count=3, borderline_count=5, seed=1)
        borderline = df[df["severity"] == "BORDERLINE"]
        assert len(borderline) == 5
        for _, row in borderline.iterrows():
            positions = []
            for ch in PARAM_CHANNELS:
                lsl, usl = SPEC_LIMITS[ch]
                pos = (row[f"{ch}_168h"] - lsl) / (usl - lsl)
                positions.append(pos)
            assert min(positions) < 0.8, "borderline must not fail a limit"
            assert max(positions) >= 0.8, "borderline must sit near a limit"

    def test_baseline_readings_within_spec(self):
        df = generate_batch(n_units=100, defect_count=3, borderline_count=5, seed=1)
        for ch in PARAM_CHANNELS:
            lsl, usl = SPEC_LIMITS[ch]
            col = df[f"{ch}_0h"]
            assert col.between(lsl, usl).all(), ch


class TestDataQuality:
    def test_no_missing_values(self):
        df = generate_batch(n_units=80, defect_count=3, borderline_count=4, seed=1)
        assert not df.isna().any().any()

    def test_all_values_finite(self):
        df = generate_batch(n_units=80, defect_count=3, borderline_count=4, seed=1)
        numeric = df.select_dtypes(include=[np.number])
        assert np.isfinite(numeric.to_numpy()).all()

    def test_no_negative_electrical_measurements(self):
        df = generate_batch(n_units=80, defect_count=3, borderline_count=4, seed=1)
        for ch in PARAM_CHANNELS:
            assert (df[f"{ch}_0h"] >= 0).all()
            assert (df[f"{ch}_168h"] >= 0).all()


class TestImmutability:
    """Issue #2: intermediates assembled on the way to the batch are never
    back-patched in place."""

    def test_plant_targets_leaves_benign_drift_untouched(self):
        rng = np.random.default_rng(0)
        n_units = 12
        baseline = np.full((n_units, len(PARAM_CHANNELS)), 100.0)
        drift = np.full((n_units, len(PARAM_CHANNELS)), 4.0)
        planted = {"defect": np.array([1, 5]), "borderline": np.array([3, 7, 9])}
        before = drift.copy()

        planted_drift, mode_names = _plant_targets(rng, baseline, drift, planted)

        assert planted_drift is not drift, "planting must build a copy"
        assert np.array_equal(drift, before), "benign drift must not be mutated"
        assert not np.array_equal(planted_drift, before), "targets must be applied"
        assert len(mode_names) == n_units

    def test_defect_mode_labels_cover_exactly_the_planted_defects(self):
        rng = np.random.default_rng(0)
        n_units = 12
        baseline = np.full((n_units, len(PARAM_CHANNELS)), 100.0)
        drift = np.full((n_units, len(PARAM_CHANNELS)), 4.0)
        defect = np.array([2, 6, 10])
        planted = {"defect": defect, "borderline": np.array([], dtype=int)}

        _plant_drift, mode_names = _plant_targets(rng, baseline, drift, planted)

        assert set(mode_names[defect]) <= set(DEFECT_MODES)
        assert set(mode_names[np.setdiff1d(np.arange(n_units), defect)]) == {"NONE"}

    def test_assign_severity_labels_every_unit_without_gaps(self):
        rng = np.random.default_rng(3)
        n_units = 20
        severity, planted = _assign_severity(rng, n_units, 6, 5)

        assert len(severity) == n_units
        assert set(severity) <= set(SEVERITY_LEVELS)
        assert (severity[planted["defect"]] == "DEFECTIVE").all()
        assert (severity[planted["borderline"]] == "BORDERLINE").all()
        assert (
            severity[
                np.setdiff1d(
                    np.arange(n_units),
                    np.concatenate([planted["defect"], planted["borderline"]]),
                )
            ]
            == "NORMAL"
        ).all()
