"""Physical and policy constants for BurnSight.

Single source of truth for the telemetry schema, specification limits,
planted defect modes, and reason-code mapping. Every downstream module
(outlier screening, forecasting, triage, explainability, audit) imports
from here so a spec change happens in exactly one place.
"""

from __future__ import annotations

# --- Device and sampling -----------------------------------------------------

DEVICE_TYPE = "POWER_MOSFET"
SAMPLE_HOURS = (0, 24, 96, 168)
EARLY_HOURS = (0, 24, 96)  # observations available before the 168h terminal state
MIN_TEMP_C, MAX_TEMP_C = 125.0, 150.0  # MIL-STD-883 Method 1015 stress range

# Canonical MOSFET screening parameters (MIL-STD-883 tolerance windows).
PARAM_CHANNELS = ("rds_on_mohm", "igss_na", "vgsth_v")
CHANNEL_UNITS = {
    "rds_on_mohm": "milliohm",
    "igss_na": "nanoamp",
    "vgsth_v": "volt",
}

# --- Specification limits (LSL, USL) at burn-in measurement temperature ------

SPEC_LIMITS: dict[str, tuple[float, float]] = {
    "rds_on_mohm": (80.0, 150.0),
    "igss_na": (0.0, 100.0),
    "vgsth_v": (3.0, 5.0),
}

# --- Planted failure modes ---------------------------------------------------
# defect_mode -> (dominant drifting channel, MIL-STD-883 style reason code)

DEFECT_MODES: dict[str, tuple[str, str]] = {
    "GATE_OXIDE_DEGRADE": ("igss_na", "ERR_MOSFET_GATE_DEGRADE"),
    "RDS_DRIFT": ("rds_on_mohm", "ERR_MOSFET_RDS_DRIFT"),
    "THRESHOLD_SHIFT": ("vgsth_v", "ERR_MOSFET_VGSTH_SHIFT"),
}

SEVERITY_LEVELS = ("NORMAL", "BORDERLINE", "DEFECTIVE")

# --- Triage policy (asymmetric loss matrix) ----------------------------------
# Alpha = cost of missing a failing unit (FN), beta = cost of reviewing a
# passing unit (FP). Project policy: a miss costs 10x a false alarm, which
# biases every threshold toward escalating rather than silently passing.

DEFAULT_ALPHA = 10.0
DEFAULT_BETA = 1.0
TRIAGE_COLORS = ("GREEN", "YELLOW", "RED")

# Near-limit guard: escalate when >=80% of the spec window toward a limit is
# consumed — the same borderline definition the generator plants by.
NEAR_LIMIT_GUARD = 0.8

# Red requires the forecast to clear a limit by 2 sigma of model error, so a
# straddling (uncertain) forecast degrades to Yellow, never to a confident Red.
RED_SIGMA = 2.0


def reason_code(defect_mode: str) -> str | None:
    """Return the reason code for a planted defect mode, or None for NONE."""
    if defect_mode == "NONE":
        return None
    return DEFECT_MODES[defect_mode][1]


def dominant_channel(defect_mode: str) -> str | None:
    """Return the channel driven by a defect mode, or None for NONE."""
    if defect_mode == "NONE":
        return None
    return DEFECT_MODES[defect_mode][0]


# Inverse lookup: which defect mode a driving channel points at. Used by
# explainability (MVP-6) to name the suspected physics failure for the
# channel a triage decision hangs on.
CHANNEL_DEFECT_MODE: dict[str, str] = {
    channel: mode for mode, (channel, _code) in DEFECT_MODES.items()
}


def defect_mode_for_channel(channel: str) -> str:
    """Suspected defect mode when ``channel`` drives a triage decision."""
    return CHANNEL_DEFECT_MODE[channel]


def channel_column(channel: str, hour: int) -> str:
    """Telemetry column name for a channel at a sample hour."""
    return f"{channel}_{hour}h"


# --- Canonical batch schema (order matters for CSV round-trips) --------------

CHANNEL_COLUMNS = [channel_column(ch, h) for ch in PARAM_CHANNELS for h in SAMPLE_HOURS]
BATCH_COLUMNS = [
    "unit_id",
    "device_type",
    "temp_c",
    *CHANNEL_COLUMNS,
    "severity",
    "defect_mode",
]
LABEL_COLUMNS = ("severity", "defect_mode")
