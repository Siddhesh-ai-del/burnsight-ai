"""FastAPI service: batch telemetry ingestion.

``POST /api/batch`` accepts three payload styles:
- ``multipart/form-data`` with a ``file`` part (CSV or JSON file)
- ``application/json`` with a JSON array of records
- ``text/csv`` raw CSV body

Rejected batches return machine-readable errors: 400 for client/input
problems, 415 for unsupported media types, 422 for schema/value failures.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request

from burnsight.config import BATCH_COLUMNS
from burnsight.ingest import parse_batch

app = FastAPI(title="BurnSight", version="0.0.1")

_INPUT_ERROR_CODES = {
    "EMPTY_INPUT",
    "INVALID_ENCODING",
    "INVALID_JSON",
    "INVALID_JSON_SHAPE",
}


def _detail(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"error": code, "message": message, **extra}


async def _payload_from_request(request: Request) -> bytes | list:
    """Extract the batch payload from the request, or raise the HTTP error."""
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if ctype == "multipart/form-data":
        form = await request.form()
        part = form.get("file")
        if not hasattr(part, "read"):
            raise HTTPException(
                400,
                _detail("MISSING_FILE", "multipart body must contain a 'file' part"),
            )
        return await part.read()
    if ctype == "application/json":
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(
                400, _detail("INVALID_JSON", "request body is not valid JSON")
            ) from None
        if not isinstance(body, list):
            raise HTTPException(
                400,
                _detail("INVALID_JSON_SHAPE", "expected a JSON array of records"),
            )
        return body
    if ctype in ("text/csv", "application/octet-stream"):
        return await request.body()
    raise HTTPException(
        415,
        _detail(
            "UNSUPPORTED_MEDIA_TYPE",
            f"content-type '{ctype or 'missing'}' is not supported; "
            "use multipart/form-data, application/json, or text/csv",
        ),
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


@app.post("/api/batch")
async def ingest_batch(request: Request) -> dict[str, Any]:
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
    return {
        "n_units": len(records),
        "columns": list(BATCH_COLUMNS),
        "labels_present": any("severity" in record for record in records),
        "units": records,
    }
