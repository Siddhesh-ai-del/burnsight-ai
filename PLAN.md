# BurnSight — One-Day MVP Sprint Plan

**Goal:** a working, demoable MVP of ISRO SIH 2026 PS 26170 (AI-Driven Anomaly
Detection in Component Burn-In & Screening) in one non-stop day.
**Rule of the day: done beats fancy.**

**Repo:** https://github.com/Siddhesh-ai-del/burnsight-ai

## The loop (per ticket)

```
┌── SPRINT START — set timer 90 min ──────────────┐
│ 1. Pick ONE ticket (MVP-n) — the only one       │
│ 2. /tdd <ticket description>                    │
│    → RED test first → GREEN code                │
│ 3. It works? /code-review  (fix CRITICAL only)  │
│ 4. ✅ DEMO IT — to a human or to camera.         │
│    Tick the box. (This is your dopamine shot.)  │
│ 5. /checkpoint   (your future self thanks you)  │
├── TIMER OFF — stand up, water, 5 min ───────────┤
│ 6. Next ticket? → loop again                     │
└─────────────────────────────────────────────────┘
```

- One ticket in flight at a time. Strictly sequential.
- `/code-review`: fix CRITICAL only; log HIGH/MED as GitHub issues.
- DEMO = show a human or record the camera. Then tick the box.
- `/checkpoint` at every ticket boundary: commit + push + record state.

## Scope (locked)

**In scope:** synthetic Arrhenius-based telemetry generator; CSV/JSON ingestion
API; MAD + IsolationForest population screening; 168h forecast with Arrhenius
normalization; Green/Yellow/Red asymmetric triage (α=10β); SHAP (or fallback)
explanations + physics reason codes; FastAPI + static HTML/Chart.js dashboard;
JSON audit log + downloadable certificate.

**Out of scope (do not build, no matter how tempting):** thermal-IR fusion,
PINN training, PDF cryptographic signing, NASA dataset download, hardware
integration, auth/multi-tenancy, distributed infrastructure.

## Pre-agreed cut list (only if a ticket overruns)

| Pressure | Cut to |
|---|---|
| SHAP unavailable or too slow | permutation importance + rule-based reason codes |
| XGBoost unavailable | `sklearn.ensemble.HistGradientBoostingRegressor` |
| ReportLab unavailable | styled HTML certificate |
| Dashboard time overrun | table + badges only; chart next ticket |

Nothing else gets cut, because nothing else is in scope.

## Dependency decisions (recorded at Phase 0)

**Core (all REQUIRED for every ticket): ALL OK on Python 3.14.7**
`pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, fastapi 0.142.2, uvicorn, pydantic 2.13.5, pytest 9.1.1, httpx, python-multipart, jinja2, pytest-cov` — no blocker, no fallback needed.

**Optional probes (decided at Phase 0, applied consistently — not re-litigated mid-sprint):**

| Capability | Primary | Fallback (LOCKED) | Probe result |
|---|---|---|---|
| Explainability | SHAP | permutation importance + reason codes | **AVAILABLE (shap 0.52)** — primary locked for MVP-6 |
| Forecast model | XGBoost | sklearn `HistGradientBoostingRegressor` | **AVAILABLE (xgboost 3.4.1, CPU)** — primary locked for MVP-4; installed from cached wheel without the unused CUDA/nvidia-nccl dependency |
| Certificate | ReportLab PDF | styled HTML certificate | **DECIDED at MVP-8 start: PDF via reportlab 5.0.1 (locked primary)** — HTML cut NOT taken; rendered and demoed |

Decision rule (locked): if the probe says FAIL, or the package costs more time than it saves, the fallback above is used with no further discussion.

## Decisions log

**D1 — Stage-1 screening uses baseline + 0h→24h drift signals (MVP-3).**
The spec's "MAD at T=0h" only measures lot-to-lot baseline spread: 0h is the
pre-stress baseline, and every planted defect mode starts statistically normal
there. This is proven by test, not asserted: `test_baseline_alone_cannot_see_planted_defects`
shows baseline-only recall on planted defects is **0/3**. Stage 1 therefore
flags `out_of_family = MAD(baseline) OR MAD(drift 0h→24h) OR IsolationForest`
— the union, biased toward flagging (a false alarm costs a review; an escape
costs a mission). Measured on the seed-42 lot: 3/3 defects + 5/5 borderlines
flagged, **0/92 normal units flagged**, defect z-scores 11.9–31.0 against a
4.0 threshold.

---

## Tickets (8 × ≤90 min)

### MVP-1 — Synthetic Telemetry Generator (≤75 min)
- [x] **MVP-1**: deterministic batch generator producing component telemetry at
  0h/24h/96h with 168h terminal labels; Arrhenius rate law drives drift;
  three planted defect modes (gate degradation, RDS(on) drift, leakage);
  fixed column schema; seeded RNG reproducibility.
- **TDD targets:** schema invariants; same seed → identical batch; drift rate
  scales with Arrhenius factor; planted defects labeled and detectable by
  simple z-score sanity check.
- **DEMO:** print a 100-component batch showing 3 planted defective units.

### MVP-2 — CSV/JSON Batch Ingestion API (≤75 min)
- [x] **MVP-2**: `POST /api/batch` accepts CSV and JSON payloads, pydantic
  schema validation, rejects malformed rows with machine-readable errors;
  golden sample fixtures under `data/samples/`.
- **TDD targets:** valid CSV → 200 + normalized records; valid JSON → 200;
  missing column → 422 with field name; out-of-range value → 422; empty file →
  400; round-trip equality with generator output.
- **DEMO:** `curl` a CSV file in, get validated JSON back.

### MVP-3 — Population Outlier Detector (≤90 min)
- [x] **MVP-3**: MAD z-score screening + IsolationForest at T=0h →
  `out_of_family` flag per component; batch-level summary counts.
- **TDD targets:** planted outliers flagged; normals not flagged (no false
  alarms on clean population); deterministic given seed; MAD handles zero-
  variance channel without divide-by-zero.
- **DEMO:** 3 planted units flagged out of 100, 97 clean.

### MVP-4 — 168h Degradation Predictor (≤90 min)
- [x] **MVP-4**: early-cycle features (0h/24h/96h levels, slopes, accelerations,
  Arrhenius-normalized temperature) → regressor predicts 168h terminal values
  for each channel; returns full 0→168h trajectory for plotting.
- **TDD targets:** MAE below threshold on held-out synthetic batch; forecast is
  a pure function of features (no leakage of labels); trajectory length and
  time points fixed; prediction respects physical direction of drift for
  planted defects.
- **DEMO:** predict RED unit's 168h VCE vs its planted actual value.

### MVP-5 — Asymmetric Risk Triage (≤75 min)
- [x] **MVP-5**: loss matrix (α=10β, configurable) + triage rules combining
  outlier flag, forecast vs USL/LSL, forecast uncertainty → Green/Yellow/Red.
- **TDD targets:** borderline/uncertain cases escalate to Yellow (never
  silently Green); known-bad unit → Red; clean unit → Green; α/β change moves
  the decision boundary in the expected direction; 100% coverage of loss-matrix
  logic.
- **DEMO:** full batch triages into a 92/5/3-style split; show one borderline
  case escalating to Yellow.

### MVP-6 — Explainability & Reason Codes (≤75 min)
- [x] **MVP-6**: per-flagged-component feature attribution (SHAP if installed,
  else permutation importance) + physics-grounded reason codes
  (e.g. `ERR_MOSFET_GATE_DEGRADE`) mapped to the defect mode; explanation
  attached to every Yellow/Red decision.
- **TDD targets:** explanation names the planted defect driver for Red units;
  Green units get an empty/low-complexity explanation; reason codes are stable
  strings from a fixed registry; attribution scores sum/normalize sensibly.
- **DEMO:** "Explain Decision" on Red #34: driver + reason code.

### MVP-7 — Interactive Dashboard (≤90 min)
- [x] **MVP-7**: FastAPI serves a static dashboard: upload CSV → batch table
  with Green/Yellow/Red badges, trajectory chart (measured + forecast + spec
  limits) via Chart.js, expandable explanation panel, triage threshold slider.
- **TDD targets:** endpoints serve HTML/JS with correct content type; batch
  results endpoint returns badge data; slider value affects triage server-side;
  smoke test drives upload → results flow end-to-end via httpx.
- **DEMO:** full browser click-through: upload → badges → expand Red → chart +
  explanation.

### MVP-8 — Audit Export & Ship (≤90 min)
- [x] **MVP-8**: machine-readable JSON audit log (input digest, model/policy
  version, decision, explanation, timestamp) + downloadable certificate
  (ReportLab PDF if available, else styled HTML) + end-to-end smoke test
  (ingest → triage → explain → export) + honest README with real usage.
- **TDD targets:** audit record contains all required fields; audit is
  append-only JSONL; certificate renders for a Red unit with correct details;
  e2e test covers the full pipeline in one call.
- **DEMO:** run the full 90-second judge pitch end-to-end. Tick the last box.

---

**D2 — Triage guard measures USL headroom; Red checks both bounds (MVP-5).**
All three registered defect modes approach their limit from inside and above
(leakage ↑, RDS(on) ↑, Vth ↑), so the 80%-toward-limit guard scores headroom
to the USL. Measuring proximity to the *nearest* limit instead would misfire:
healthy leakage sits ~93% of its window above LSL=0 and would flag every
normal unit. The Red rule still checks both USL and LSL (a unit confidently
below spec is Red regardless of direction), and the statistical risk term is
two-sided — only the proximity guard is failure-side. If a downward defect
mode is ever registered, its channel's guard side flips in one place.

## Progress log

| Ticket | State | Notes |
|---|---|---|
| Phase 0 (repo + env) | complete | repo live, core deps verified, community profile green |
| MVP-1 | complete | RED→GREEN, 44 tests, 100% coverage, 0 CRITICAL in review, issues #1 (HIGH) #2 (MED) logged |
| MVP-2 | complete | RED→GREEN, 80 tests, 99% coverage, 0 CRITICAL; MEDIUM (upload size cap) = issue #3; shap probe OK |
| MVP-3 | complete | RED→GREEN, 98 tests, 99% coverage (screen 100%), 0 CRITICAL, 0 new findings; decision D1 logged; demo: 8 flagged / 92 clean / 0 FP |
| MVP-4 | complete | RED→GREEN (2 in-cycle bugs: feature naming, residual-learning fix), 113 tests, 99% coverage (forecast 100%), 0 CRITICAL, 0 new findings; held-out MAE 0.68/0.61/0.023 vs limits 3.0/3.0/0.12; probe complete: reportlab OK |
| MVP-5 | complete | RED→GREEN, 139 tests, 99% coverage (triage 100%), 0 CRITICAL; HIGH (triage_batch 88 lines) = issue #4; honest 92/5/3 split achieved with spec-anchored thresholds (no tuning); A/B test proves α=10β flips uncertain units to Yellow |
| MVP-6 | complete | RED→GREEN (2 in-cycle fixes: bg unit_id cast, shap (1,p) row convention), 155 tests, 99% coverage (explain 100%), 0 CRITICAL; HIGH (explain_batch 58 lines) = issue #5; SHAP top-ranked feature names planted driver for all 3 Reds, codes from config registry, reproducible via shap seed |
| MVP-7 | complete | RED→GREEN (1 test rewrite: golden-data risks are all 0/1 → crafted intermediate-risk unit for the slider test), 167 tests, 99% coverage (api 100%, pipeline 100%), 0 CRITICAL; XSS via row.innerHTML unit_id = issue #6 (HIGH); Chart.js vendored (no CDN/build), static dashboard + /api/triage with real alpha recompute |
| MVP-8 | complete | RED→GREEN (first try, 0 in-cycle fixes), 182 tests, 99% coverage (audit 100%, api 100%, pipeline 100%, config 100%), 0 CRITICAL; findings: certificate_lines 57 lines = issue #7 (HIGH), JSONL lock/malformed-line = issue #8 (MED); **cut-list decision: certificate = ReportLab PDF (locked primary), HTML cut NOT taken**; honest README shipped with screenshots; 8/8 boxes ticked |

## Frontend redesign (Phases 0–6) — locked plan status

Architecture: Vite 7 + React 19 + TS SPA under `frontend/`, built into
`frontend/dist` (committed — uvicorn/pytest run without Node). ObsidianUI
component sets; the vanilla `style.css`/`app.js` dashboard is removed. Zero
backend-logic changes (static-dir pointer in `api.py` only); **182 tests
green at every phase boundary**.

| Phase | Commit | State | Notes |
|---|---|---|---|
| 0 toolchain | `fefea0d` | complete | Vite+React served by FastAPI |
| 1 tokens/fonts/shell | `c15d9e9` | complete | design tokens, vendored fonts, mission-console shell |
| 2 ObsidianUI | `fa19e17` | complete | all six sets wired; upload → triage → drawer flow verified live |
| 3 liquid glass | *this commit* | complete, **gate aborted default** | refraction proven on 4 surfaces; FPS gate failed → fallback default, `?glass=on` opt-in (below) |
| 4 data presentation | — | pending | readouts, recharts trajectory, signed SHAP bars, certificate/audit |
| 5 a11y & states | — | pending | skeleton loading, keyboard expansion, focus, error/empty |
| 6 ship | — | pending | full pytest, Playwright E2E, README refresh, `/code-review`, `ecc:security-audit` |

### Phase 3 decision record — liquid glass ships disabled by default (option A)

**Success criterion status:** *visible real refraction on exactly 4 chrome
surfaces* — met as **capability evidence**: bend quantified numerically on
all four surfaces in Chrome 155 (header rim +15–17 px tapering to ~1 px at
centre; button −2.56 px → 0 across the lens; thumb +0.79 px rim; drawer rim
stripe spacing distorted from 17 px nominal). Evidence:
`.playwright-mcp/p3-cleanbars.png` (header), `p3-striped.png`
(button+thumb), `p3-drawer-on.png` (drawer rim), clean states
`p3-mat-on.png` / `p3-mat-off.png`.

**FPS gate (mandatory) — all four surfaces FAILED on real GPU**
(ANGLE/Radeon 740M; SwiftShader cross-check in parentheses):

| Config | FPS | Gate ≥55 & ≥90% of baseline | Result |
|---|---|---|---|
| `glass=off` baseline | 60.0 (60.2) | — | — |
| header only | 35.8 (18.6) | ≥55 | ✗ |
| button only | 37.0 (40.2) | ≥55 | ✗ |
| thumb only | 35.9 (40.1) | ≥55 | ✗ |
| drawer open | 28.4 (16.8) | ≥55 | ✗ |
| all three idle | 13.9 (6.2) | ≥55 | ✗ |

Rationale is measured, not assumed: a minimal `saturate() url(#…)` filter
over the same header region runs at 57.1 fps (cost = the library's
13-primitive displacement graph, not `backdrop-filter: url()` itself);
freezing all ambient animation does not rescue any surface (header 30.3,
button 39.8, thumb 27.1, drawer 28.7, all-on 12.2); an 80×80 thumb region
costs about the same as the header's 1397×278 region (cost is
graph-complexity-bound, not area-bound). Decision **A (plan-literal abort)**:
`.glass-fallback` renders on all four surfaces by default; glass stays
available for demos/evaluation via **`?glass=on`** (also
`?glass=header,thumb` for per-surface); `?glass=off` is a no-op kept for
stable URLs; `prefers-reduced-motion` still force-kills the lens.

**v0.1.0 API divergence finding:** the plan's perf knob does not exist on
the material path — `@samasante/liquid-glass@0.1.0` only dispatches to
`GlassMaterial` when `filterResolution` is *absent*, so passing
`filterResolution` demotes the surface to `GlassDOM` (in-place bend, which
smears children). GlassDOM itself defaults `filterResolution = 1`, so the
"cap at 1" invariant holds, but material surfaces have **no resolution
lever** at all. Copy mode (`refract`) is only visually correct for the
drawer (its backdrop *is* `Background`); header/button/thumb need the live
page backdrop and are material-only — no plan-compliant path to a passing
surface remained after the animation-freeze control failed.

## Post-MVP backlog (open GitHub issues)

All review findings were logged, not fixed in-sprint (CRITICAL-only fix rule;
0 CRITICAL was found at any ticket):

| # | Severity | Finding |
|---|---|---|
| [#1](https://github.com/Siddhesh-ai-del/burnsight-ai/issues/1) | HIGH | `generate_batch` 62 lines |
| [#2](https://github.com/Siddhesh-ai-del/burnsight-ai/issues/2) | MED | generator in-place mutation |
| [#3](https://github.com/Siddhesh-ai-del/burnsight-ai/issues/3) | MEDIUM | upload body size cap |
| [#4](https://github.com/Siddhesh-ai-del/burnsight-ai/issues/4) | HIGH | `triage_batch` 88 lines |
| [#5](https://github.com/Siddhesh-ai-del/burnsight-ai/issues/5) | HIGH | `explain_batch` 58 lines |
| [#6](https://github.com/Siddhesh-ai-del/burnsight-ai/issues/6) | HIGH | dashboard `row.innerHTML` interpolates unvalidated `unit_id` (XSS) |
| [#7](https://github.com/Siddhesh-ai-del/burnsight-ai/issues/7) | HIGH | `certificate_lines` 57 lines |
| [#8](https://github.com/Siddhesh-ai-del/burnsight-ai/issues/8) | MED | audit JSONL: no file lock on append; malformed line bricks export endpoints |

The four length findings (#1/#4/#5/#7) are one refactor pass: extract the
shared helpers each function repeats.
