"""FastAPI service: batch telemetry ingestion.

``POST /api/batch`` accepts three payload styles:
- ``multipart/form-data`` with a ``file`` part (CSV or JSON file)
- ``application/json`` with a JSON array of records
- ``text/csv`` raw CSV body

Rejected batches return machine-readable errors: 400 for client/input
problems, 413 when the body exceeds ``MAX_BODY_BYTES``, 415 for
unsupported media types, 422 for schema/value failures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from burnsight.audit import AuditLogError, certificate_pdf, load_audit
from burnsight.config import BATCH_COLUMNS, DEFAULT_ALPHA, DEFAULT_BETA
from burnsight.ingest import parse_batch
from burnsight.pipeline import run_triage
from burnsight.screen import screen_population

app = FastAPI(title="BurnSight", version="0.0.1")

_FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "dist"

# Byte budget for any single upload. Bodies are streamed against this cap
# so an accidental multi-gigabyte file cannot exhaust memory on the
# single-operator workstation the service is scoped to (issue #3).
MAX_BODY_BYTES = 50 * 1024 * 1024

_INPUT_ERROR_CODES = {
    "EMPTY_INPUT",
    "INVALID_ENCODING",
    "INVALID_JSON",
    "INVALID_JSON_SHAPE",
}


def _detail(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"error": code, "message": message, **extra}


def _too_large(limit: int) -> HTTPException:
    return HTTPException(
        413,
        _detail(
            "PAYLOAD_TOO_LARGE",
            f"request body exceeds the {limit}-byte limit; split the batch "
            "into smaller files and retry",
        ),
    )


async def _read_body(request: Request) -> bytes:
    """Read the whole request body, raising 413 as soon as it exceeds the budget.

    The declared ``Content-Length`` is checked first (cheap reject before
    a single byte is buffered), then the stream itself is metered so a
    chunked upload with no declared length cannot slip past the cap.
    """
    limit = MAX_BODY_BYTES
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise _too_large(limit)
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise _too_large(limit)
        chunks.append(chunk)
    return b"".join(chunks)


def _replay(request: Request, body: bytes) -> Request:
    """A fresh ``Request`` over the same scope carrying an already-read body.

    Starlette's form parser consumes the receive channel, so multipart
    parsing runs against a replay of the bounded bytes instead of the
    (now spent) original stream.
    """

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(request.scope, receive)


async def _payload_from_request(request: Request) -> bytes | list:
    """Extract the batch payload from the request, or raise the HTTP error."""
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if ctype == "multipart/form-data":
        form = await _replay(request, await _read_body(request)).form()
        part = form.get("file")
        if not hasattr(part, "read"):
            raise HTTPException(
                400,
                _detail("MISSING_FILE", "multipart body must contain a 'file' part"),
            )
        return await part.read()
    if ctype not in ("application/json", "text/csv", "application/octet-stream"):
        raise HTTPException(
            415,
            _detail(
                "UNSUPPORTED_MEDIA_TYPE",
                f"content-type '{ctype or 'missing'}' is not supported; "
                "use multipart/form-data, application/json, or text/csv",
            ),
        )
    body = await _read_body(request)
    if ctype == "application/json":
        try:
            parsed = json.loads(body)
        except ValueError:
            raise HTTPException(
                400, _detail("INVALID_JSON", "request body is not valid JSON")
            ) from None
        if not isinstance(parsed, list):
            raise HTTPException(
                400,
                _detail("INVALID_JSON_SHAPE", "expected a JSON array of records"),
            )
        return parsed
    return body


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


async def _validated_records(request: Request) -> list[dict]:
    """Parse the request payload into validated records, or raise the HTTP error."""
    payload = await _payload_from_request(request)
    records, errors = parse_batch(payload)
    if errors:
        if len(errors) == 1 and errors[0]["code"] in _INPUT_ERROR_CODES:
            first = errors[0]
            raise HTTPException(400, _detail(first["code"], first["message"]))
        raise HTTPException(
            422,
            {
                "error": "VALIDATION_FAILED",
                "message": f"batch rejected: {len(errors)} problem(s)",
                "errors": errors,
            },
        )
    return records


@app.post("/api/batch")
async def ingest_batch(request: Request) -> dict[str, Any]:
    records = await _validated_records(request)
    return {
        "n_units": len(records),
        "columns": list(BATCH_COLUMNS),
        "labels_present": any("severity" in record for record in records),
        "units": records,
    }


@app.post("/api/screen")
async def screen_batch(request: Request) -> dict[str, Any]:
    """Stage 1: population outlier screening over one uploaded batch."""
    records = await _validated_records(request)
    result = screen_population(pd.DataFrame.from_records(records))
    flagged = result[result.out_of_family]
    return {
        "n_units": len(result),
        "n_flagged": int(len(flagged)),
        "n_clean": int(len(result) - len(flagged)),
        "flagged_units": [str(uid) for uid in flagged["unit_id"]],
        "units": [
            {
                "unit_id": str(row.unit_id),
                "mad_baseline": bool(row.mad_baseline),
                "mad_drift": bool(row.mad_drift),
                "iforest": bool(row.iforest),
                "out_of_family": bool(row.out_of_family),
                "max_abs_z_baseline": float(row.max_abs_z_baseline),
                "max_abs_z_drift": float(row.max_abs_z_drift),
            }
            for row in result.itertuples(index=False)
        ],
    }


@app.get("/")
async def dashboard() -> FileResponse:
    """Built React dashboard shell (Vite output served from frontend/dist)."""
    return FileResponse(_FRONTEND / "index.html", media_type="text/html")


@app.post("/api/triage")
async def triage_endpoint(
    request: Request,
    alpha: float = Query(DEFAULT_ALPHA, ge=DEFAULT_BETA),
) -> dict[str, Any]:
    """Full pipeline: screen -> forecast -> risk triage -> SHAP explanation.

    ``alpha`` is the miss cost of the MVP-5 loss matrix (beta fixed at
    DEFAULT_BETA): the slider value re-runs triage server-side rather than
    being echoed back.
    """
    records = await _validated_records(request)
    return run_triage(records, alpha=alpha)


def _stored_audit(audit_id: str) -> dict:
    """Load a stored audit record or raise the matching HTTP error.

    A damaged log line is surfaced as an explicit 500 rather than an
    unhandled crash, so the operator learns which file to repair.
    """
    try:
        record = load_audit(audit_id)
    except AuditLogError as exc:
        raise HTTPException(500, _detail("AUDIT_LOG_CORRUPT", str(exc))) from None
    if record is None:
        raise HTTPException(
            404, _detail("AUDIT_NOT_FOUND", f"no audit record {audit_id!r}")
        )
    return record


@app.get("/api/audit/{audit_id}")
async def get_audit(audit_id: str) -> dict[str, Any]:
    """Machine-readable audit record for one analysis run (JSONL lookup)."""
    return _stored_audit(audit_id)


@app.get("/api/certificate/{audit_id}/{unit_id}")
async def get_certificate(audit_id: str, unit_id: str) -> Response:
    """PDF certificate for a flagged unit, rendered from the stored log."""
    record = _stored_audit(audit_id)
    try:
        pdf = certificate_pdf(record, unit_id)
    except ValueError as exc:
        raise HTTPException(404, _detail("UNIT_NOT_CERTIFIED", str(exc))) from None
    safe_id = "".join(c for c in unit_id if c.isalnum() or c in "-_")
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="certificate-{safe_id}.pdf"'
        },
    )


app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")
