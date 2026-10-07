"""MVP-2: CSV/JSON batch ingestion — parsing and validation (RED first)."""

from pathlib import Path

import pytest

from burnsight.config import BATCH_COLUMNS, EARLY_HOURS, PARAM_CHANNELS, channel_column
from burnsight.generator import generate_batch
from burnsight.ingest import parse_batch

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


def _read(name: str) -> bytes:
    return (SAMPLES / name).read_bytes()


class TestValidInput:
    def test_golden_csv_parses_to_100_clean_records(self):
        records, errors = parse_batch(_read("batch_golden.csv"))
        assert errors == []
        assert len(records) == 100
        assert list(records[0]) == BATCH_COLUMNS

    def test_golden_json_parses_to_100_clean_records(self):
        records, errors = parse_batch(_read("batch_golden.json"))
        assert errors == []
        assert len(records) == 100

    def test_csv_round_trips_against_generator_output(self):
        records, errors = parse_batch(_read("batch_golden.csv"))
        assert errors == []
        df = generate_batch(n_units=100, defect_count=3, borderline_count=5, seed=42)
        for record, (_, row) in zip(records, df.iterrows()):
            for column in BATCH_COLUMNS:
                expected = row[column]
                if isinstance(expected, str):
                    assert record[column] == expected
                else:
                    assert record[column] == pytest.approx(float(expected))

    def test_early_only_csv_accepted_without_terminal_columns(self):
        records, errors = parse_batch(_read("batch_early_96h.csv"))
        assert errors == []
        assert len(records) == 100
        assert all(rec.get("severity") is None for rec in records)
        assert all(rec.get("rds_on_mohm_168h") is None for rec in records)

    def test_in_memory_json_list_accepted(self):
        payload = [
            {
                "unit_id": f"U{i:03d}",
                "device_type": "POWER_MOSFET",
                "temp_c": 125.0,
                **{
                    channel_column(ch, h): 50.0
                    for ch in PARAM_CHANNELS
                    for h in EARLY_HOURS
                },
            }
            for i in range(3)
        ]
        records, errors = parse_batch(payload)
        assert errors == []
        assert len(records) == 3


class TestStructuralErrors:
    def test_empty_input_rejected(self):
        _records, errors = parse_batch(b"")
        assert [e["code"] for e in errors] == ["EMPTY_INPUT"]

    def test_header_only_csv_rejected_as_empty(self):
        header = ",".join(BATCH_COLUMNS)
        _records, errors = parse_batch((header + "\n").encode())
        assert [e["code"] for e in errors] == ["EMPTY_INPUT"]

    def test_non_utf8_rejected(self):
        _records, errors = parse_batch(b"\xff\xfe\x00garbage\x00")
        assert [e["code"] for e in errors] == ["INVALID_ENCODING"]

    def test_missing_column_names_the_field(self):
        _records, errors = parse_batch(_read("batch_missing_column.csv"))
        assert len(errors) == 1
        assert errors[0]["code"] == "MISSING_COLUMN"
        assert errors[0]["field"] == "rds_on_mohm_24h"

    def test_json_object_shape_rejected(self):
        _records, errors = parse_batch(b'{"units": []}')
        assert [e["code"] for e in errors] == ["INVALID_JSON_SHAPE"]

    def test_malformed_json_rejected(self):
        _records, errors = parse_batch(b"[not valid json")
        assert [e["code"] for e in errors] == ["INVALID_JSON"]

    def test_invalid_rows_reject_the_whole_batch(self):
        records, errors = parse_batch(_read("batch_bad_values.csv"))
        assert records == []
        by_record = {e["record"]: e for e in errors}
        assert set(by_record) == {5, 10}
        assert by_record[5]["field"] == "temp_c"
        assert by_record[10]["field"] == "igss_na_24h"
        assert all(e["code"] == "INVALID_VALUE" for e in errors)

    def test_duplicate_unit_ids_rejected(self):
        payload = [
            {
                "unit_id": "U001",
                "device_type": "POWER_MOSFET",
                "temp_c": 125.0,
                **{
                    channel_column(ch, h): 50.0
                    for ch in PARAM_CHANNELS
                    for h in EARLY_HOURS
                },
            }
            for _ in range(2)
        ]
        _records, errors = parse_batch(payload)
        assert [e["code"] for e in errors] == ["DUPLICATE_UNIT"]
        assert errors[0]["field"] == "unit_id"

    @pytest.mark.parametrize(
        "unit_id",
        [
            "<img src=x onerror=alert(1)>",
            'U001"><script>alert(1)</script>',
            "U001'; DROP TABLE units;--",
            "U001/../secret",
        ],
        ids=["img-tag", "script-tag", "sql-ish", "path-traversal"],
    )
    def test_unit_id_rejects_markup_and_control_characters(self, unit_id):
        """XSS defense-in-depth (issue #6): unit_id is rendered into the
        dashboard, so ingest rejects anything outside a safe identifier
        charset instead of relying on the UI to escape it."""
        rows = [
            {
                "unit_id": unit_id,
                "device_type": "POWER_MOSFET",
                "temp_c": 125.0,
                **{
                    channel_column(ch, h): 50.0
                    for ch in PARAM_CHANNELS
                    for h in EARLY_HOURS
                },
            }
        ]
        records, errors = parse_batch(rows)
        assert records == []
        assert errors[0]["code"] == "INVALID_VALUE"
        assert errors[0]["field"] == "unit_id"

    def test_unit_id_accepts_safe_identifier_charset(self):
        rows = [
            {
                "unit_id": f"U-001_{i}.A",
                "device_type": "POWER_MOSFET",
                "temp_c": 125.0,
                **{
                    channel_column(ch, h): 50.0
                    for ch in PARAM_CHANNELS
                    for h in EARLY_HOURS
                },
            }
            for i in range(2)
        ]
        records, errors = parse_batch(rows)
        assert errors == []
        assert len(records) == 2

    def test_unknown_severity_value_rejected(self):
        rows = [
            {
                "unit_id": "U001",
                "device_type": "POWER_MOSFET",
                "temp_c": 125.0,
                **{
                    channel_column(ch, h): 50.0
                    for ch in PARAM_CHANNELS
                    for h in EARLY_HOURS
                },
                "severity": "PURPLE",
                "defect_mode": "NONE",
            }
        ]
        _records, errors = parse_batch(rows)
        assert errors[0]["code"] == "INVALID_VALUE"
        assert errors[0]["field"] == "severity"

    def test_non_dict_json_entry_rejected(self):
        _records, errors = parse_batch(b'["not-a-record"]')
        assert [e["code"] for e in errors] == ["INVALID_JSON_SHAPE"]

    def test_none_input_rejected(self):
        _records, errors = parse_batch(None)
        assert [e["code"] for e in errors] == ["EMPTY_INPUT"]

    def test_in_memory_dict_rejected(self):
        _records, errors = parse_batch({"units": []})
        assert [e["code"] for e in errors] == ["INVALID_JSON_SHAPE"]

    def test_unsupported_input_type_rejected(self):
        _records, errors = parse_batch(42)
        assert [e["code"] for e in errors] == ["INVALID_JSON_SHAPE"]

    def test_whitespace_only_string_rejected(self):
        _records, errors = parse_batch("   \n  ")
        assert [e["code"] for e in errors] == ["EMPTY_INPUT"]

    def test_empty_json_array_bytes_rejected(self):
        _records, errors = parse_batch(b"[]")
        assert [e["code"] for e in errors] == ["EMPTY_INPUT"]

    def test_in_memory_list_of_non_objects_rejected(self):
        _records, errors = parse_batch([1, 2])
        assert [e["code"] for e in errors] == ["INVALID_JSON_SHAPE"]

    def test_plain_string_csv_accepted(self):
        text = (SAMPLES / "batch_early_96h.csv").read_text()
        records, errors = parse_batch(text)
        assert errors == []
        assert len(records) == 100


class TestSchemaParity:
    def test_severity_literal_matches_config_registry(self):
        from typing import get_args

        from burnsight.schema import DefectMode, Severity
        from burnsight.config import DEFECT_MODES, SEVERITY_LEVELS

        assert set(get_args(Severity)) == set(SEVERITY_LEVELS)
        assert set(get_args(DefectMode)) == set(DEFECT_MODES) | {"NONE"}
