"""MVP-7: dashboard endpoints — static serving, badge payload, slider, smoke flow.

TDD targets from PLAN.md:
- endpoints serve HTML/JS with correct content type,
- batch results endpoint returns badge data,
- slider value affects triage server-side (not just echoed — a real
  recomputation must change a unit's escalation reasons),
- smoke test drives upload -> results flow end-to-end via httpx.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from burnsight.api import app

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
GOLDEN = SAMPLES / "batch_golden.csv"
client = TestClient(app)


def _local_assets() -> list[str]:
    """Asset paths (/static/...) referenced by the built shell."""
    html = client.get("/").text
    return re.findall(r'(?:src|href)="(/static/[^"]+)"', html)


@pytest.fixture(scope="module")
def payload() -> dict:
    """One full /api/triage run over the golden lot, shared by the class."""
    response = client.post(
        "/api/triage",
        files={"file": ("batch_golden.csv", GOLDEN.read_bytes(), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


class TestStaticServing:
    def test_root_serves_html(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "BurnSight" in response.text
        assert '<div id="root">' in response.text  # React mount point

    def test_js_bundle_served_with_javascript_content_type(self):
        js_assets = [p for p in _local_assets() if p.endswith(".js")]
        assert js_assets, "built shell must reference a local JS bundle"
        for path in js_assets:
            response = client.get(path)
            assert response.status_code == 200, path
            assert "javascript" in response.headers["content-type"]

    def test_css_served_with_css_content_type(self):
        css_assets = [p for p in _local_assets() if p.endswith(".css")]
        assert css_assets, "built shell must reference a local stylesheet"
        for path in css_assets:
            response = client.get(path)
            assert response.status_code == 200, path
            assert response.headers["content-type"].startswith("text/css")

    def test_no_cdn_dependency(self):
        """All assets ship from /static — no remote script/style URLs."""
        html = client.get("/").text
        assert not re.search(r'(?:src|href)="https?://', html)


class TestTriageEndpoint:
    def test_badge_data_for_every_unit(self, payload):
        assert payload["n_units"] == 100
        assert payload["counts"] == {"GREEN": 92, "YELLOW": 5, "RED": 3}
        assert len(payload["units"]) == 100
        assert {u["color"] for u in payload["units"]} <= {"GREEN", "YELLOW", "RED"}

    def test_flagged_details_carry_explanation_and_chart(self, payload):
        flagged = {d["unit_id"]: d for d in payload["flagged"]}
        assert len(flagged) == 8
        u026 = flagged["U026"]
        assert u026["color"] == "RED"
        assert u026["reason_code"] == "ERR_MOSFET_GATE_DEGRADE"
        assert u026["defect_mode"] == "GATE_OXIDE_DEGRADE"
        explanation = u026["explanation"]
        assert 0 < len(explanation) <= 8
        weights = [f["weight"] for f in explanation]
        assert weights == sorted(weights, reverse=True)
        assert explanation[0]["feature"].startswith("igss_na")

    def test_chart_has_measured_points_forecast_and_limits(self, payload):
        u026 = next(d for d in payload["flagged"] if d["unit_id"] == "U026")
        chart = u026["chart"]
        assert chart["channel"] == "igss_na"
        assert chart["hours"] == [0, 24, 96, 120, 144, 168]
        assert len(chart["forecast"]) == 6
        assert chart["forecast"][-1] == pytest.approx(
            u026["chart"]["predicted_168h"], rel=1e-6
        )
        assert chart["limits"] == {"lsl": 0.0, "usl": 100.0}
        # measured points must be the uploaded telemetry, not model output
        uploaded = pd.read_csv(GOLDEN).query("unit_id == 'U026'").iloc[0]
        assert chart["measured"]["0"] == pytest.approx(uploaded["igss_na_0h"])
        assert chart["measured"]["24"] == pytest.approx(uploaded["igss_na_24h"])
        assert chart["measured"]["96"] == pytest.approx(uploaded["igss_na_96h"])

    def test_empty_upload_rejected(self):
        response = client.post(
            "/api/triage", files={"file": ("empty.csv", b"", "text/csv")}
        )
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "EMPTY_INPUT"


class TestSliderAffectsServerSideTriage:
    def test_alpha_reaches_server_and_sets_threshold(self):
        for alpha, expected in [(10.0, 1 / 11), (49.0, 0.02), (1.0, 0.5)]:
            response = client.post(
                f"/api/triage?alpha={alpha}",
                files={"file": ("batch_golden.csv", GOLDEN.read_bytes(), "text/csv")},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["alpha"] == alpha
            assert body["escalation_threshold"] == pytest.approx(expected)

    def test_extreme_alpha_recomputes_escalation_reasons(self):
        """A real recompute, not an echo.

        On golden data every risk is exactly 0 or 1 (MVP-5 finding), so no
        alpha can move a reason. This crafts a unit whose forecast lands at
        *intermediate* risk (flat RDS at 149.85 -> risk ~8e-3, inside
        (0, p*)): default alpha must not escalate it, while an alpha derived
        from the server's own risk value must — proving the slider value
        re-runs triage server-side.
        """
        golden = pd.read_csv(GOLDEN)
        flat = golden["unit_id"] == "U000"
        for hour in (0, 24, 96):
            golden.loc[flat, f"rds_on_mohm_{hour}h"] = 149.85
        crafted = golden.to_csv(index=False).encode()

        def post(alpha: float) -> dict:
            response = client.post(
                f"/api/triage?alpha={alpha}",
                files={"file": ("crafted.csv", crafted, "text/csv")},
            )
            assert response.status_code == 200, response.text
            return response.json()

        default = post(10.0)
        unit = next(u for u in default["units"] if u["unit_id"] == "U000")
        assert unit["color"] == "YELLOW"  # guard/outlier rules still decide
        assert 0 < unit["risk"] < default["escalation_threshold"]
        assert "ESCALATED_BY_LOSS_MATRIX" not in unit["reasons"]

        alpha_needed = 2.0 / unit["risk"]  # p* = 1/(alpha+1) < risk/2 < risk
        assert alpha_needed > 10.0  # slider can actually reach it
        escalated = post(alpha_needed)
        unit = next(u for u in escalated["units"] if u["unit_id"] == "U000")
        assert "ESCALATED_BY_LOSS_MATRIX" in unit["reasons"]
        assert escalated["escalation_threshold"] < unit["risk"]

    def test_invalid_alpha_rejected(self):
        response = client.post(
            f"/api/triage?alpha=0.5",
            files={"file": ("batch_golden.csv", GOLDEN.read_bytes(), "text/csv")},
        )
        assert response.status_code == 422


class TestHttpxSmokeFlow:
    def test_upload_to_results_flow_like_a_browser(self):
        home = client.get("/")
        assert home.status_code == 200
        assert '<div id="root">' in home.text
        assert _local_assets()

        upload = client.post(
            "/api/triage",
            files={"file": ("batch_golden.csv", GOLDEN.read_bytes(), "text/csv")},
            params={"alpha": 10.0},
        )
        assert upload.status_code == 200
        body = upload.json()
        assert body["counts"]["RED"] == 3
        red = next(u for u in body["units"] if u["color"] == "RED")
        detail = next(d for d in body["flagged"] if d["unit_id"] == red["unit_id"])
        assert detail["explanation"][0]["feature"].startswith(
            detail["worst_channel"].split("_")[0]
        )
        assert len(detail["chart"]["forecast"]) == 6
