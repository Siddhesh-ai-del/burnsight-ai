"""MVP-1: config contract tests — spec limits, defect registry, schema helpers."""

import pytest

from burnsight.config import (
    BATCH_COLUMNS,
    DEFECT_MODES,
    EARLY_HOURS,
    PARAM_CHANNELS,
    SAMPLE_HOURS,
    SEVERITY_LEVELS,
    SPEC_LIMITS,
    channel_column,
    dominant_channel,
    reason_code,
)


class TestRegistry:
    def test_every_defect_mode_has_channel_and_reason_code(self):
        for mode, (channel, code) in DEFECT_MODES.items():
            assert channel in PARAM_CHANNELS
            assert code.startswith("ERR_MOSFET_")
            assert code == code.upper()
            assert " " not in code

    def test_reason_code_lookup(self):
        assert reason_code("GATE_OXIDE_DEGRADE") == "ERR_MOSFET_GATE_DEGRADE"
        assert reason_code("RDS_DRIFT") == "ERR_MOSFET_RDS_DRIFT"
        assert reason_code("THRESHOLD_SHIFT") == "ERR_MOSFET_VGSTH_SHIFT"

    def test_reason_code_none_for_passing_unit(self):
        assert reason_code("NONE") is None

    def test_reason_code_unknown_mode_raises(self):
        with pytest.raises(KeyError):
            reason_code("MADE_UP_MODE")

    def test_dominant_channel_lookup(self):
        assert dominant_channel("GATE_OXIDE_DEGRADE") == "igss_na"
        assert dominant_channel("NONE") is None


class TestSpecLimits:
    def test_limits_ordered_and_finite(self):
        for channel, (lsl, usl) in SPEC_LIMITS.items():
            assert lsl < usl, channel

    def test_channels_have_limits(self):
        assert set(PARAM_CHANNELS) == set(SPEC_LIMITS)

    def test_severity_levels(self):
        assert SEVERITY_LEVELS == ("NORMAL", "BORDERLINE", "DEFECTIVE")


class TestSchemaHelpers:
    def test_channel_column_naming(self):
        assert channel_column("rds_on_mohm", 96) == "rds_on_mohm_96h"

    def test_batch_columns_structure(self):
        expected = ["unit_id", "device_type", "temp_c"]
        for ch in PARAM_CHANNELS:
            expected += [f"{ch}_{h}h" for h in SAMPLE_HOURS]
        expected += ["severity", "defect_mode"]
        assert BATCH_COLUMNS == expected

    def test_early_hours_prefix_of_sample_hours(self):
        assert list(EARLY_HOURS) == list(SAMPLE_HOURS)[:-1]
