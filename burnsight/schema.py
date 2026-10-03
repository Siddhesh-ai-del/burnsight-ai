"""Pydantic contract for one telemetry record (one component).

The model is the single validation gate for ingested batches: required
early-hour channels (0h/24h/96h), optional terminal state (168h) and
ground-truth labels (present only in evaluation files, absent in real
uploads).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from burnsight.config import DEVICE_TYPE, MAX_TEMP_C, MIN_TEMP_C

Severity = Literal["NORMAL", "BORDERLINE", "DEFECTIVE"]
DefectMode = Literal["NONE", "GATE_OXIDE_DEGRADE", "RDS_DRIFT", "THRESHOLD_SHIFT"]


class UnitRecord(BaseModel):
    """One component's burn-in telemetry."""

    unit_id: str = Field(min_length=1)
    device_type: str = Field(default=DEVICE_TYPE, min_length=1)
    temp_c: float = Field(ge=MIN_TEMP_C, le=MAX_TEMP_C)

    # Telemetry channels in canonical BATCH_COLUMNS order: each channel's
    # early observations (0h/24h/96h, required) followed by its terminal
    # state (168h, optional — only present in labeled/evaluation files).
    rds_on_mohm_0h: float = Field(ge=0.0)
    rds_on_mohm_24h: float = Field(ge=0.0)
    rds_on_mohm_96h: float = Field(ge=0.0)
    rds_on_mohm_168h: float | None = Field(default=None, ge=0.0)

    igss_na_0h: float = Field(ge=0.0)
    igss_na_24h: float = Field(ge=0.0)
    igss_na_96h: float = Field(ge=0.0)
    igss_na_168h: float | None = Field(default=None, ge=0.0)

    vgsth_v_0h: float = Field(ge=0.0)
    vgsth_v_24h: float = Field(ge=0.0)
    vgsth_v_96h: float = Field(ge=0.0)
    vgsth_v_168h: float | None = Field(default=None, ge=0.0)

    severity: Severity | None = None
    defect_mode: DefectMode | None = None
