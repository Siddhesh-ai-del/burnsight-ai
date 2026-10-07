"""MVP-8: audit log, certificate, and end-to-end export.

TDD targets from PLAN.md:
- audit record contains all required fields (input digest, model/policy
  version, decision, explanation, timestamp),
- audit is append-only JSONL,
- certificate renders for a Red unit with correct details,
- e2e test covers the full pipeline (ingest -> triage -> explain -> export)
  in one call.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from burnsight.audit import (
    AUDIT_ENV,
    AuditLogError,
    append_audit,
    build_audit_record,
    certificate_lines,
    certificate_pdf,
    load_audit,
)
from burnsight.api import app
from burnsight.config import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    MODEL_VERSION,
    NEAR_LIMIT_GUARD,
    RED_SIGMA,
)
from burnsight.explain import explain_batch
from burnsight.forecast import predict_terminal
from burnsight.pipeline import reference_model
from burnsight.screen import screen_population
from burnsight.triage import LossMatrix, triage_batch

GOLDEN = (
    Path(__file__).resolve().parent.parent / "data" / "samples" / "batch_golden.csv"
)
client = TestClient(app)

REQUIRED_FIELDS = {
    "audit_id",
    "timestamp",
    "input_digest",
    "model",
    "policy",
    "counts",
    "decisions",
    "explanations",
}
GOLDEN_FLAGGED = {"U024", "U026", "U044", "U045", "U070", "U074", "U076", "U078"}


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return pd.read_csv(GOLDEN).to_dict("records")


@pytest.fixture(scope="module")
def record(records) -> dict:
    batch = pd.DataFrame.from_records(records)
    model = reference_model()
    matrix = LossMatrix(alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA)
    decisions = triage_batch(
        predict_terminal(model, batch),
        screen_population(batch),
        model.error_std,
        matrix=matrix,
    )
    explanation = explain_batch(model, batch, decisions)
    return build_audit_record(records, model, matrix, decisions, explanation)


def canonical(records: list[dict]) -> str:
    return json.dumps(records, sort_keys=True, separators=(",", ":"), default=str)


class TestAuditRecord:
    def test_record_contains_all_required_fields(self, record):
        assert REQUIRED_FIELDS <= set(record)

    def test_timestamp_is_utc_iso8601(self, record):
        parsed = datetime.fromisoformat(record["timestamp"])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset().total_seconds() == 0

    def test_input_digest_is_sha256_of_canonical_records(self, record, records):
        expected = hashlib.sha256(canonical(records).encode()).hexdigest()
        assert record["input_digest"] == {
            "algorithm": "sha256",
            "value": expected,
            "n_units": len(records),
        }
        # key order inside records must not matter...
        reordered = [{key: row[key] for key in reversed(list(row))} for row in records]
        assert hashlib.sha256(canonical(reordered).encode()).hexdigest() == expected
        # ...but content must
        changed = [dict(records[0], temp_c=999.0), *records[1:]]
        assert hashlib.sha256(canonical(changed).encode()).hexdigest() != expected

    def test_model_and_policy_versions_recorded(self, record):
        model = record["model"]
        assert model["version"] == MODEL_VERSION
        assert model["n_units_trained"] == 400
        assert set(model["error_std"]) == {"rds_on_mohm", "igss_na", "vgsth_v"}
        policy = record["policy"]
        assert policy["alpha"] == DEFAULT_ALPHA
        assert policy["beta"] == DEFAULT_BETA
        assert policy["guard_fraction"] == NEAR_LIMIT_GUARD
        assert policy["red_sigma"] == RED_SIGMA
        assert policy["escalation_threshold"] == pytest.approx(
            DEFAULT_BETA / (DEFAULT_ALPHA + DEFAULT_BETA)
        )

    def test_decisions_cover_every_unit(self, record):
        assert len(record["decisions"]) == 100
        assert record["counts"] == {"GREEN": 92, "YELLOW": 5, "RED": 3}
        assert {d["color"] for d in record["decisions"]} == {"GREEN", "YELLOW", "RED"}

    def test_explanations_cover_flagged_units_with_ranked_features(self, record):
        assert {e["unit_id"] for e in record["explanations"]} == GOLDEN_FLAGGED
        u026 = next(e for e in record["explanations"] if e["unit_id"] == "U026")
        assert u026["reason_code"] == "ERR_MOSFET_GATE_DEGRADE"
        assert u026["defect_mode"] == "GATE_OXIDE_DEGRADE"
        weights = [f["weight"] for f in u026["features"]]
        assert len(weights) == 24
        assert weights == sorted(weights, reverse=True)


class TestAppendOnlyJsonl:
    def test_append_only_jsonl(self, tmp_path, record):
        path = tmp_path / "audit.jsonl"
        append_audit(record, path=path)
        prefix = path.read_bytes()
        second = dict(record, audit_id="second-run-id")
        append_audit(second, path=path)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        # append-only: the first record is a byte-exact prefix of the file
        assert path.read_bytes().startswith(prefix)
        assert json.loads(lines[0]) == record
        assert json.loads(lines[1])["audit_id"] == "second-run-id"

    def test_load_audit_roundtrip_and_misses(self, tmp_path, record):
        path = tmp_path / "audit.jsonl"
        assert load_audit(record["audit_id"], path=path) is None
        append_audit(record, path=path)
        assert load_audit(record["audit_id"], path=path) == record
        assert load_audit("no-such-id", path=path) is None
        assert load_audit("any-id", path=tmp_path / "missing.jsonl") is None


class TestAuditLogRobustness:
    """Issue #8: concurrent appends must serialize, and a damaged line must
    fail loudly with file + line context instead of an anonymous 500."""

    def test_append_acquires_exclusive_file_lock(self, tmp_path, record, monkeypatch):
        import fcntl

        calls: list[int] = []
        real_flock = fcntl.flock

        def spy(fd: int, operation: int) -> None:
            calls.append(operation)
            real_flock(fd, operation)

        monkeypatch.setattr(fcntl, "flock", spy)
        path = tmp_path / "audit.jsonl"
        append_audit(record, path=path)
        assert fcntl.LOCK_EX in calls, "append must take a write lock"

    def test_lock_is_released_once_append_returns(self, tmp_path, record):
        import fcntl

        path = tmp_path / "audit.jsonl"
        append_audit(record, path=path)
        with path.open("a", encoding="utf-8") as probe:
            # Succeeds only when no lock was left behind.
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(probe.fileno(), fcntl.LOCK_UN)

    def test_concurrent_appends_keep_every_line_valid_json(self, tmp_path, record):
        from concurrent.futures import ThreadPoolExecutor

        path = tmp_path / "audit.jsonl"
        big = dict(record, payload="x" * 60_000)
        jobs = [dict(big, audit_id=f"run-{i}") for i in range(24)]

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda item: append_audit(item, path=path), jobs))

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(jobs)
        parsed = [json.loads(line) for line in lines]
        assert {item["audit_id"] for item in parsed} == {
            item["audit_id"] for item in jobs
        }
        assert all(item["payload"] == "x" * 60_000 for item in parsed)

    def test_blank_lines_between_records_are_ignored(self, tmp_path, record):
        path = tmp_path / "audit.jsonl"
        path.write_text(
            json.dumps(record)
            + "\n\n\n"
            + json.dumps(dict(record, audit_id="later"))
            + "\n",
            encoding="utf-8",
        )
        assert load_audit("later", path=path)["audit_id"] == "later"

    def test_malformed_line_raises_error_naming_file_and_line(self, tmp_path, record):
        path = tmp_path / "audit.jsonl"
        first = dict(record, audit_id="first")
        third = dict(record, audit_id="third")
        path.write_text(
            json.dumps(first) + "\n" + '{"audit_id": "truncated\n' + json.dumps(third),
            encoding="utf-8",
        )
        # A hit *before* the damage still resolves.
        assert load_audit("first", path=path)["audit_id"] == "first"
        # Scanning past the damage is loud and locates the problem.
        with pytest.raises(AuditLogError) as excinfo:
            load_audit("third", path=path)
        message = str(excinfo.value)
        assert str(path) in message
        assert "line 2" in message
        assert excinfo.value.__cause__ is not None

    def test_api_reports_corrupt_audit_log_as_explicit_500(
        self, monkeypatch, tmp_path, record
    ):
        path = tmp_path / "corrupt.jsonl"
        path.write_text(
            json.dumps(record) + "\n" + "not json at all\n", encoding="utf-8"
        )
        monkeypatch.setenv(AUDIT_ENV, str(path))
        response = client.get(f"/api/audit/{record['audit_id']}")
        # the target record is on line 1, so it is still served
        assert response.status_code == 200

        later = dict(record, audit_id="after-the-damage")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(later) + "\n")
        damaged = client.get("/api/audit/after-the-damage")
        assert damaged.status_code == 500
        assert damaged.json()["detail"]["error"] == "AUDIT_LOG_CORRUPT"
        assert str(path) in damaged.json()["detail"]["message"]


class TestCertificate:
    def test_certificate_lines_for_red_unit(self, record):
        text = "\n".join(certificate_lines(record, "U026"))
        decision = next(d for d in record["decisions"] if d["unit_id"] == "U026")
        explanation = next(e for e in record["explanations"] if e["unit_id"] == "U026")
        assert "U026" in text
        assert "RED" in text
        assert decision["worst_channel"] in text
        assert "GATE_OXIDE_DEGRADE" in text
        assert "ERR_MOSFET_GATE_DEGRADE" in text
        assert f"{decision['predicted_168h']:.2f}" in text
        assert decision["reasons"] in text
        # the explanation names the driver
        assert explanation["features"][0]["feature"] in text
        assert f"{explanation['features'][0]['weight']:.1%}" in text
        # provenance
        assert record["input_digest"]["value"] in text
        assert record["model"]["version"] in text
        assert record["audit_id"] in text
        assert record["timestamp"] in text

    def test_certificate_rejects_unflagged_unit(self, record):
        with pytest.raises(ValueError, match="U000"):
            certificate_lines(record, "U000")

    def test_certificate_pdf_renders_pdf_magic(self, record):
        data = certificate_pdf(record, "U026")
        assert data[:5] == b"%PDF-"
        assert len(data) > 1000


class TestExportEndpoints:
    @pytest.fixture(scope="module")
    def triage_body(self) -> dict:
        response = client.post(
            "/api/triage",
            files={"file": ("batch_golden.csv", GOLDEN.read_bytes(), "text/csv")},
        )
        assert response.status_code == 200, response.text
        return response.json()

    def test_triage_response_carries_audit_provenance(self, triage_body):
        assert triage_body["audit_id"]
        assert triage_body["input_digest"]["algorithm"] == "sha256"
        stored = load_audit(triage_body["audit_id"])
        assert stored is not None, "triage must append an audit record"
        assert stored["counts"] == triage_body["counts"]
        assert stored["input_digest"] == triage_body["input_digest"]
        assert len(stored["decisions"]) == triage_body["n_units"]

    def test_audit_json_endpoint(self, triage_body):
        response = client.get(f"/api/audit/{triage_body['audit_id']}")
        assert response.status_code == 200
        assert response.json()["audit_id"] == triage_body["audit_id"]
        missing = client.get("/api/audit/does-not-exist")
        assert missing.status_code == 404
        assert missing.json()["detail"]["error"] == "AUDIT_NOT_FOUND"

    def test_certificate_endpoint_serves_pdf(self, triage_body):
        response = client.get(f"/api/certificate/{triage_body['audit_id']}/U026")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert "certificate-U026.pdf" in response.headers["content-disposition"]
        assert response.content[:5] == b"%PDF-"

        green = client.get(f"/api/certificate/{triage_body['audit_id']}/U000")
        assert green.status_code == 404
        assert green.json()["detail"]["error"] == "UNIT_NOT_CERTIFIED"

        unknown_audit = client.get("/api/certificate/nope/U026")
        assert unknown_audit.status_code == 404
        assert unknown_audit.json()["detail"]["error"] == "AUDIT_NOT_FOUND"


class TestEndToEnd:
    def test_e2e_ingest_triage_explain_export(self, monkeypatch, tmp_path):
        """One call runs ingest -> triage -> explain -> export; the
        certificate is then derived from the stored audit record alone."""
        log = tmp_path / "e2e-audit.jsonl"
        monkeypatch.setenv(AUDIT_ENV, str(log))

        upload = client.post(
            "/api/triage",
            files={"file": ("batch_golden.csv", GOLDEN.read_bytes(), "text/csv")},
        )
        assert upload.status_code == 200, upload.text
        body = upload.json()

        lines = log.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1, "exactly one audit record per run"
        stored = json.loads(lines[0])
        assert REQUIRED_FIELDS <= set(stored)
        assert stored["audit_id"] == body["audit_id"]
        assert stored["counts"] == {"GREEN": 92, "YELLOW": 5, "RED": 3}
        assert any(e["unit_id"] == "U026" for e in stored["explanations"])

        certificate = client.get(f"/api/certificate/{stored['audit_id']}/U026")
        assert certificate.status_code == 200
        assert certificate.headers["content-type"] == "application/pdf"
        assert certificate.content[:5] == b"%PDF-"
        # details on the certificate come from the *stored log copy*
        assert "ERR_MOSFET_GATE_DEGRADE" in "\n".join(certificate_lines(stored, "U026"))
