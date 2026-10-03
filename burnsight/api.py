"""FastAPI service: batch telemetry ingestion.

``POST /api/batch`` accepts three payload styles:
- ``multipart/form-data`` with a ``file`` part (CSV or JSON file)
- ``application/json`` with a JSON array of records
- ``text/csv`` raw CSV body

Rejected batches return machine-readable errors: 400 for client/input
problems, 415 for unsupported media types, 422 for schema/value failures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from burnsight.config import BATCH_COLUMNS, DEFAULT_ALPHA, DEFAULT_BETA
from burnsight.ingest import parse_batch
from burnsight.pipeline import run_triage
from burnsight.screen import screen_population

app = FastAPI(title="BurnSight", version="0.0.1")

_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

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
    """Static dashboard shell: plain HTML + vendored Chart.js, no build step."""
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


app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")
