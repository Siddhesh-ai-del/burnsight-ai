/* BurnSight dashboard — plain JS, no build tooling.
   Flow: upload -> POST /api/triage?alpha=... -> badges, expandable detail,
   trajectory chart, SHAP explanation panel. */
"use strict";

const $ = (id) => document.getElementById(id);
let latest = null;
let chart = null;

function alphaValue() {
  return Math.pow(10, Number($("alphaSlider").value));
}

function updateSliderLabels() {
  const exponent = Number($("alphaSlider").value);
  const alpha = Math.pow(10, exponent);
  $("alphaOut").innerHTML = `&alpha; = 10<sup>${exponent}</sup>`;
  $("pStar").textContent = (1 / (alpha + 1)).toFixed(exponent >= 4 ? 8 : 3);
}

async function analyze(event) {
  if (event) event.preventDefault();
  const file = $("fileInput").files[0];
  if (!file) return;
  const status = $("status");
  status.hidden = false;
  status.textContent = "Running screen → forecast → triage → SHAP … (≈5 s)";
  $("analyzeBtn").disabled = true;
  try {
    const body = new FormData();
    body.append("file", file);
    const response = await fetch(`/api/triage?alpha=${alphaValue()}`, {
      method: "POST",
      body,
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = data.detail || data;
      throw new Error(detail.message || JSON.stringify(detail));
    }
    latest = data;
    render(data);
    status.textContent =
      `Analyzed ${data.n_units} units · escalation threshold p* = ` +
      `${data.escalation_threshold.toExponential(2)}`;
  } catch (error) {
    status.textContent = `Error: ${error.message}`;
  } finally {
    $("analyzeBtn").disabled = false;
  }
}

function render(data) {
  $("summary").hidden = false;
  $("content").hidden = false;
  $("nGreen").textContent = data.counts.GREEN;
  $("nYellow").textContent = data.counts.YELLOW;
  $("nRed").textContent = data.counts.RED;
  $("batchMeta").textContent =
    `${data.n_units} units · alpha = ${data.alpha} · p* = ` +
    `${data.escalation_threshold.toExponential(2)}`;

  const links = $("auditLinks");
  links.textContent = "";
  const label = document.createElement("span");
  label.textContent = `audit ${data.audit_id.slice(0, 8)} `;
  const auditLink = document.createElement("a");
  auditLink.href = `/api/audit/${data.audit_id}`;
  auditLink.target = "_blank";
  auditLink.rel = "noopener";
  auditLink.textContent = "JSON ↗";
  links.append(label, auditLink);

  const detailById = Object.fromEntries(data.flagged.map((d) => [d.unit_id, d]));
  const body = $("unitBody");
  body.innerHTML = "";
  for (const unit of data.units) {
    const row = document.createElement("tr");
    const expandable = Boolean(detailById[unit.unit_id]);
    row.className = `unit ${unit.color}`;
    const chevron = expandable ? '<span class="chev">▸</span>' : "";
    row.innerHTML =
      `<td class="uid">${unit.unit_id} ${chevron}</td>` +
      `<td><span class="badge ${unit.color}">${unit.color}</span></td>` +
      `<td>${unit.worst_channel}</td>` +
      `<td class="num">${unit.predicted_168h.toFixed(2)}</td>` +
      `<td class="num">${formatRisk(unit.risk)}</td>` +
      `<td class="why-cell">${unit.reasons || "—"}</td>`;
    if (expandable) {
      row.addEventListener("click", () => showDetail(detailById[unit.unit_id]));
    }
    body.appendChild(row);
  }
}

function formatRisk(risk) {
  if (risk === 0) return "0";
  if (risk < 1e-3) return risk.toExponential(1);
  return risk.toFixed(3);
}

function showDetail(detail) {
  $("detail").hidden = false;
  $("detailTitle").textContent = `Explain Decision — ${detail.unit_id}`;
  $("detailBadge").className = `badge ${detail.color}`;
  $("detailBadge").textContent = detail.color;
  $("detailCode").textContent =
    `${detail.reason_code} · ${detail.defect_mode}`;
  const certificate = $("certLink");
  certificate.href = `/api/certificate/${latest.audit_id}/${detail.unit_id}`;
  certificate.hidden = false;
  $("detailWhy").textContent =
    `Reasons: ${detail.reasons || "—"} · driver: ${detail.worst_channel} · ` +
    `forecast 168 h = ${detail.predicted_168h.toFixed(2)}`;

  const list = $("shapList");
  list.innerHTML = "";
  for (const feature of detail.explanation) {
    const pct = feature.weight * 100;
    const li = document.createElement("li");
    const bar = document.createElement("span");
    bar.className = "bar";
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(pct, 1)}%`;
    bar.appendChild(fill);
    const name = document.createElement("span");
    name.className = "feat";
    name.textContent = feature.feature;
    const value = document.createElement("span");
    value.className = "val";
    value.textContent =
      `${feature.shap_value >= 0 ? "+" : ""}${feature.shap_value.toFixed(2)}` +
      ` (${pct.toFixed(1)}%)`;
    li.append(name, bar, value);
    list.appendChild(li);
  }
  drawChart(detail.chart);
  $("detail").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function drawChart(spec) {
  const forecast = spec.hours.map((hour, index) => ({
    x: hour,
    y: spec.forecast[index],
  }));
  const measured = Object.entries(spec.measured).map(([hour, value]) => ({
    x: Number(hour),
    y: value,
  }));
  const usl = [
    { x: 0, y: spec.limits.usl },
    { x: 168, y: spec.limits.usl },
  ];
  const lsl = [
    { x: 0, y: spec.limits.lsl },
    { x: 168, y: spec.limits.lsl },
  ];
  if (chart) chart.destroy();
  chart = new Chart($("trajChart"), {
    type: "line",
    data: {
      datasets: [
        {
          label: `${spec.channel} forecast`,
          data: forecast,
          borderColor: "#4f9cf9",
          backgroundColor: "#4f9cf9",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.25,
          fill: false,
        },
        {
          label: "measured (0/24/96 h)",
          data: measured,
          borderColor: "#2fbf71",
          backgroundColor: "#2fbf71",
          showLine: false,
          pointRadius: 5,
        },
        {
          label: `USL ${spec.limits.usl}`,
          data: usl,
          borderColor: "#e5484d",
          borderDash: [6, 4],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
        },
        {
          label: `LSL ${spec.limits.lsl}`,
          data: lsl,
          borderColor: "#e5484d",
          borderDash: [6, 4],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: {
          type: "linear",
          min: 0,
          max: 168,
          title: { display: true, text: "burn-in hour" },
        },
        y: {
          title: { display: true, text: spec.channel },
        },
      },
      plugins: { legend: { position: "bottom" } },
    },
  });
}

document.addEventListener("DOMContentLoaded", () => {
  updateSliderLabels();
  $("uploadForm").addEventListener("submit", analyze);
  $("alphaSlider").addEventListener("input", updateSliderLabels);
  // The slider re-runs triage server-side so alpha is never a dead control.
  $("alphaSlider").addEventListener("change", () => {
    if (latest) analyze();
  });
});
