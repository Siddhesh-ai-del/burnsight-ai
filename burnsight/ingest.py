"""Batch parsing and row-level validation for CSV and JSON telemetry.

Contract: ``parse_batch(raw)`` returns ``(records, errors)``. ``errors`` is
empty if and only if the batch is valid; an invalid batch never yields
partial records (all-or-nothing, so a QA batch cannot be half-accepted).

Error objects are machine-readable dicts: ``code`` always present, plus
``record`` (1-based data-record index), ``field``, and ``message`` where
applicable. Codes: EMPTY_INPUT, INVALID_ENCODING, INVALID_JSON,
INVALID_JSON_SHAPE, MISSING_COLUMN, INVALID_VALUE, DUPLICATE_UNIT.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from pydantic import ValidationError

from burnsight.config import EARLY_HOURS, PARAM_CHANNELS, channel_column
from burnsight.schema import UnitRecord

REQUIRED_COLUMNS = [
    "unit_id",
    "temp_c",
    *[channel_column(ch, h) for ch in PARAM_CHANNELS for h in EARLY_HOURS],
]


def _error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **extra}


def _decode(
    raw: bytes | str | list | dict | None,
) -> tuple[str | list | None, dict | None]:
    """Normalize input to text or an in-memory record list; return (payload, error)."""
    if raw is None:
        return None, _error("EMPTY_INPUT", "no input provided")
    if isinstance(raw, list):
        return raw, None
    if isinstance(raw, dict):
        return None, _error(
            "INVALID_JSON_SHAPE", "expected a JSON array of records, got an object"
        )
    if isinstance(raw, bytes):
        if not raw.strip():
            return None, _error("EMPTY_INPUT", "input is empty")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return None, _error("INVALID_ENCODING", "input is not valid UTF-8")
    elif isinstance(raw, str):
        text = raw
    else:
        return None, _error(
            "INVALID_JSON_SHAPE", f"unsupported input type {type(raw).__name__}"
        )
    if not text.strip():
        return None, _error("EMPTY_INPUT", "input is empty")
    return text, None


def _rows_from_json(text: str) -> tuple[list[dict] | None, dict | None]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, _error("INVALID_JSON", f"body is not valid JSON: {exc.msg}")
    if not isinstance(payload, list):
        return None, _error("INVALID_JSON_SHAPE", "expected a JSON array of records")
    if not payload:
        return None, _error("EMPTY_INPUT", "record array is empty")
    if not all(isinstance(row, dict) for row in payload):
        return None, _error("INVALID_JSON_SHAPE", "every array entry must be an object")
    return payload, None


def _rows_from_csv(text: str) -> tuple[list[dict] | None, list[dict] | None]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return None, [_error("EMPTY_INPUT", "CSV has no header row")]
    missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
    if missing:
        return None, [
            _error("MISSING_COLUMN", f"required column '{c}' is missing", field=c)
            for c in missing
        ]
    rows = [
        {k: v for k, v in row.items() if k is not None}
        for row in reader
        if any(str(v or "").strip() for v in row.values() if v is not None)
    ]
    if not rows:
        return None, [_error("EMPTY_INPUT", "CSV contains no data rows")]
    return rows, None


def _rows_from_text(text: str) -> tuple[list[dict] | None, dict | list | None]:
    stripped = text.lstrip()
    if stripped.startswith("["):
        return _rows_from_json(text)
    if stripped.startswith("{"):
        return None, _error(
            "INVALID_JSON_SHAPE", "expected a JSON array of records, got an object"
        )
    return _rows_from_csv(text)


def _validate_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    errors: list[dict] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        try:
            record = UnitRecord.model_validate(row).model_dump(exclude_none=True)
        except ValidationError as exc:
            for detail in exc.errors():
                field = str(detail["loc"][0]) if detail["loc"] else "unknown"
                code = (
                    "MISSING_COLUMN" if detail["type"] == "missing" else "INVALID_VALUE"
                )
                errors.append(_error(code, detail["msg"], record=index, field=field))
            continue
        unit_id = record["unit_id"]
        if unit_id in seen_ids:
            errors.append(
                _error(
                    "DUPLICATE_UNIT",
                    f"duplicate unit_id '{unit_id}'",
                    record=index,
                    field="unit_id",
                )
            )
            continue
        seen_ids.add(unit_id)
        records.append(record)
    if errors:
        return [], errors
    return records, []


def parse_batch(raw: bytes | str | list | dict | None) -> tuple[list[dict], list[dict]]:
    """Parse and validate one batch. Returns ``(records, errors)``; all-or-nothing."""
    payload, error = _decode(raw)
    if error is not None:
        return [], [error]
    if isinstance(payload, list):
        if not payload:
            return [], [_error("EMPTY_INPUT", "record array is empty")]
        if not all(isinstance(row, dict) for row in payload):
            return [], [
                _error("INVALID_JSON_SHAPE", "every array entry must be an object")
            ]
        rows = payload
    else:
        rows, structural = _rows_from_text(payload)
        if structural is not None:
            if isinstance(structural, dict):
                return [], [structural]
            return [], structural
    return _validate_rows(rows)
