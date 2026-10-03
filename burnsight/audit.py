"""MVP-8: append-only JSON audit log and PDF screening certificates.

Every ``POST /api/triage`` run appends exactly **one** JSONL record holding
the required audit fields — input digest (sha256 over canonical records),
model and policy version, per-unit decisions, flagged-unit explanations, and
an ISO-8601 UTC timestamp. The log is append-only: existing bytes are never
rewritten (verified byte-exact in tests).

A certificate is *derived from the stored record alone* (no model re-run),
so what a judge downloads provably matches what was logged.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from burnsight.config import (
    MODEL_VERSION,
    NEAR_LIMIT_GUARD,
    RED_SIGMA,
    SPEC_LIMITS,
)
from burnsight.triage import LossMatrix

AUDIT_ENV = "BURNSIGHT_AUDIT_PATH"
DEFAULT_AUDIT_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "audit" / "audit.jsonl"
)
TRIAGE_COLORS = ("GREEN", "YELLOW", "RED")
CERTIFICATE_FEATURES = 5


def audit_path() -> Path:
    """Where the JSONL log lives — env override keeps tests/demo hermetic."""
    override = os.environ.get(AUDIT_ENV)
    return Path(override) if override else DEFAULT_AUDIT_PATH


def input_digest(records: list[dict]) -> dict:
    """sha256 over canonical records: key order irrelevant, content decisive."""
    canonical = json.dumps(
        records, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return {
        "algorithm": "sha256",
        "value": hashlib.sha256(canonical).hexdigest(),
        "n_units": len(records),
    }


def build_audit_record(
    records: list[dict],
    model,
    matrix: LossMatrix,
    decisions: pd.DataFrame,
    explanation: pd.DataFrame,
) -> dict:
    """Assemble the JSON-native audit record for one analysis run."""
    counts = decisions.color.value_counts()
    return {
        "audit_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_digest": input_digest(records),
        "model": {
            "name": "xgboost-residual",
            "version": MODEL_VERSION,
            "n_units_trained": int(model.n_units),
            "error_std": {ch: float(v) for ch, v in model.error_std.items()},
        },
        "policy": {
            "alpha": float(matrix.alpha),
            "beta": float(matrix.beta),
            "guard_fraction": float(NEAR_LIMIT_GUARD),
            "red_sigma": float(RED_SIGMA),
            "escalation_threshold": float(matrix.escalation_threshold),
        },
        "counts": {color: int(counts.get(color, 0)) for color in TRIAGE_COLORS},
        "decisions": [
            {
                "unit_id": str(row.unit_id),
                "color": row.color,
                "worst_channel": row.worst_channel,
                "predicted_168h": float(row.predicted_168h),
                "headroom": float(row.headroom),
                "risk": float(row.risk),
                "reasons": row.reasons,
            }
            for row in decisions.itertuples(index=False)
        ],
        "explanations": _explanation_records(explanation),
    }


def _explanation_records(explanation: pd.DataFrame) -> list[dict]:
    """Flagged-unit explanations with features rank-ordered by |SHAP|."""
    records = []
    for unit_id, unit in explanation.groupby("unit_id", sort=False):
        unit = unit.sort_values("rank")
        head = unit.iloc[0]
        records.append(
            {
                "unit_id": str(unit_id),
                "color": str(head.color),
                "reason_code": str(head.reason_code),
                "defect_mode": str(head.defect_mode),
                "worst_channel": str(head.worst_channel),
                "features": [
                    {
                        "feature": str(item.feature),
                        "shap_value": float(item.shap_value),
                        "feature_value": float(item.feature_value),
                        "weight": float(item.weight),
                    }
                    for item in unit.itertuples(index=False)
                ],
            }
        )
    return records


def append_audit(record: dict, path: Path | None = None) -> Path:
    """Append exactly one JSON line. Existing bytes are never modified."""
    target = Path(path) if path is not None else audit_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return target


def load_audit(audit_id: str, path: Path | None = None) -> dict | None:
    """Look up one record by id; None when the id or the log is missing."""
    target = Path(path) if path is not None else audit_path()
    if not target.exists():
        return None
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                candidate = json.loads(line)
                if candidate.get("audit_id") == audit_id:
                    return candidate
    return None


def certificate_lines(record: dict, unit_id: str) -> list[str]:
    """Plain-text certificate content — the testable source of truth.

    Raises ValueError for units without a flagged decision (no certificate
    for Green units: nothing was certified against a suspicion).
    """
    decision = next((d for d in record["decisions"] if d["unit_id"] == unit_id), None)
    explanation = next(
        (e for e in record["explanations"] if e["unit_id"] == unit_id), None
    )
    if decision is None or explanation is None:
        raise ValueError(
            f"unit {unit_id!r} has no flagged decision in audit {record['audit_id']}"
        )
    lsl, usl = SPEC_LIMITS[explanation["worst_channel"]]
    policy = record["policy"]
    model = record["model"]
    lines = [
        "BurnSight AI - Burn-In Screening Certificate",
        "=" * 54,
        f"Unit:             {unit_id}",
        f"Verdict:          {decision['color']}",
        f"Defect mode:      {explanation['defect_mode']}",
        f"Reason code:      {explanation['reason_code']}",
        f"Driver channel:   {explanation['worst_channel']}",
        f"Forecast 168 h:   {decision['predicted_168h']:.2f} (spec {lsl}..{usl})",
        f"Headroom:         {decision['headroom']:.1%} of window",
        f"Risk:             {decision['risk']:.3e}",
        f"Triage reasons:   {decision['reasons']}",
        "",
        "Model and policy",
        f"Model:            {model['name']} {model['version']}"
        f" (trained on {model['n_units_trained']} units)",
        f"Policy:           alpha={policy['alpha']} beta={policy['beta']}"
        f" p*={policy['escalation_threshold']:.4f}",
        f"                  guard={policy['guard_fraction']}"
        f" red_sigma={policy['red_sigma']}",
        "",
        "Top explanations (SHAP)",
    ]
    for index, feature in enumerate(
        explanation["features"][:CERTIFICATE_FEATURES], start=1
    ):
        lines.append(
            f"{index}. {feature['feature']:<28} {feature['shap_value']:+10.2f}"
            f"  weight {feature['weight']:.1%}"
        )
    lines += [
        "",
        f"Input digest:     sha256:{record['input_digest']['value']}",
        f"Audit id:         {record['audit_id']}",
        f"Issued (UTC):     {record['timestamp']}",
        "",
        "Issued by BurnSight MVP (SIH 2026 PS 26170) from a synthetic Arrhenius lot.",
        "Machine-readable provenance: JSONL audit log entry with this audit id.",
    ]
    return lines


def certificate_pdf(record: dict, unit_id: str) -> bytes:
    """Render certificate_lines as a ReportLab PDF (locked primary, 5.0.1)."""
    lines = certificate_lines(record, unit_id)
    styles = getSampleStyleSheet()
    story: list = [
        Paragraph(escape(lines[0]), styles["Title"]),
        Spacer(1, 12),
    ]
    for line in lines[1:]:
        story.append(Paragraph(escape(line) or "&nbsp;", styles["Code"]))
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title=f"BurnSight certificate {unit_id}",
        author="BurnSight AI",
    )
    document.build(story)
    return buffer.getvalue()
