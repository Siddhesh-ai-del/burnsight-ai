"""MVP-2: POST /api/batch integration tests (RED first)."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from burnsight.api import app
from burnsight.config import BATCH_COLUMNS

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
client = TestClient(app)


def _upload(name: str, part_name: str = "file"):
    return client.post(
        "/api/batch",
        files={part_name: (name, (SAMPLES / name).read_bytes(), "text/csv")},
    )


class TestHappyPath:
    def test_health_endpoint(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_multipart_golden_csv_returns_validated_json(self):
        response = _upload("batch_golden.csv")
        assert response.status_code == 200
        body = response.json()
        assert body["n_units"] == 100
        assert body["columns"] == BATCH_COLUMNS
        assert body["labels_present"] is True
        assert len(body["units"]) == 100

    def test_multipart_golden_json_file_accepted(self):
        response = _upload("batch_golden.json")
        assert response.status_code == 200
        assert response.json()["n_units"] == 100

    def test_json_array_body_accepted(self):
        record = {
            "unit_id": "U001",
            "device_type": "POWER_MOSFET",
            "temp_c": 130.0,
            "rds_on_mohm_0h": 99.0,
            "rds_on_mohm_24h": 100.0,
            "rds_on_mohm_96h": 101.0,
            "igss_na_0h": 5.0,
            "igss_na_24h": 5.2,
            "igss_na_96h": 5.4,
            "vgsth_v_0h": 4.0,
            "vgsth_v_24h": 4.0,
            "vgsth_v_96h": 4.01,
        }
        response = client.post("/api/batch", json=[record])
        assert response.status_code == 200
        body = response.json()
        assert body["n_units"] == 1
        assert body["labels_present"] is False

    def test_raw_csv_body_accepted(self):
        raw = (SAMPLES / "batch_early_96h.csv").read_bytes()
        response = client.post(
            "/api/batch", content=raw, headers={"content-type": "text/csv"}
        )
        assert response.status_code == 200
        assert response.json()["n_units"] == 100


class TestRejections:
    def test_malformed_values_rejected_with_record_and_field(self):
        response = _upload("batch_bad_values.csv")
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["error"] == "VALIDATION_FAILED"
        errors = {e["record"]: e for e in detail["errors"]}
        assert errors[5]["field"] == "temp_c"
        assert errors[10]["field"] == "igss_na_24h"
        assert errors[5]["message"]

    def test_missing_column_rejected_with_field_name(self):
        response = _upload("batch_missing_column.csv")
        assert response.status_code == 422
        errors = response.json()["detail"]["errors"]
        assert errors[0]["code"] == "MISSING_COLUMN"
        assert errors[0]["field"] == "rds_on_mohm_24h"

    def test_empty_file_rejected_400(self):
        response = _upload("batch_empty.csv")
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "EMPTY_INPUT"

    def test_unsupported_media_type_rejected_415(self):
        response = client.post(
            "/api/batch", content=b"hello", headers={"content-type": "text/plain"}
        )
        assert response.status_code == 415
        assert response.json()["detail"]["error"] == "UNSUPPORTED_MEDIA_TYPE"

    def test_multipart_without_file_part_rejected(self):
        response = client.post(
            "/api/batch", files={"wrong": ("x.txt", b"data", "text/plain")}
        )
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "MISSING_FILE"

    def test_json_object_body_rejected(self):
        response = client.post("/api/batch", json={"units": []})
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "INVALID_JSON_SHAPE"

    def test_empty_json_array_rejected(self):
        response = client.post("/api/batch", json=[])
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "EMPTY_INPUT"

    def test_invalid_json_body_rejected(self):
        response = client.post(
            "/api/batch",
            content=b"{not valid json",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "INVALID_JSON"
