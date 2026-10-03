# BurnSight

![License](https://img.shields.io/badge/License-Apache_2.0-0F172A?style=for-the-badge&labelColor=0F172A&logo=apache&logoColor=white)
![Status](https://img.shields.io/badge/Status-pre--release-0F172A?style=for-the-badge&labelColor=0F172A&logo=rocket&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Linux-0F172A?style=for-the-badge&labelColor=0F172A&logo=linux&logoColor=white)
![Language](https://img.shields.io/badge/Language-Python-0F172A?style=for-the-badge&labelColor=0F172A&logo=python&logoColor=white)

**Early burn-in telemetry screening that forecasts 168-hour component health and triages Green/Yellow/Red risk before the burn-in cycle ends.**

## Overview

BurnSight is a software-only screening platform for high-reliability microelectronics qualification. It ingests early-burn-in parametric telemetry (0h, 24h, 96h), detects out-of-family components against the batch population, forecasts terminal 168-hour parameter states, and assigns asymmetric risk triage with explainable, auditable reasoning.

| Principle | What it means in practice |
|---|---|
| **Physics-grounded** | Arrhenius temperature normalization and semiconductor failure-mode reasoning bound every forecast; MIL-STD-883 terminology is used throughout. |
| **Asymmetric by design** | The risk engine penalizes False Negatives (escaped defects) tenfold against False Positives (yield loss), matching space-grade zero-defect requirements. |
| **Explainable and auditable** | Every decision carries feature attribution, a physics-grounded reason code, and a machine-readable audit record. |
| **Software-only** | CSV and JSON telemetry in, decisions and certificates out. No test hardware, no chamber integration. |

## Why BurnSight

Standard burn-in screening stresses components for 168 continuous hours before aerospace payload assembly, but qualification decisions still rely on static pass/fail thresholds applied at the end of the cycle. Two failure modes dominate:

- **Catastrophic escape** — a part stays inside static specification limits at 168h despite abnormal drift kinetics during early hours, then degrades in orbit.
- **Yield and capacity loss** — every part runs the full 168 hours, and benign process variation triggers false alarms under rigid limits, discarding expensive space-grade components.

BurnSight targets the first hours of the cycle: population outlier screening, terminal-value forecasting, asymmetric triage, and audit-grade explanations — so QA teams can reject early, extend selectively, and document every decision.

## Planned capabilities

- Multi-format batch ingestion (CSV and JSON telemetry) with schema validation
- Population outlier detection at T=0h (Median Absolute Deviation and Isolation Forest)
- 168-hour degradation forecasting from early-cycle features with Arrhenius normalization
- Asymmetric Green/Yellow/Red risk triage driven by a configurable loss matrix
- Feature attribution with physics-grounded failure reason codes
- Interactive dashboard with trajectory plots and threshold controls
- Machine-readable audit logs and downloadable compliance certificates

> None of these are shipped yet. Items above reflect the current build plan; see Project status for phase-by-phase tracking.

## Project status

BurnSight is **pre-release**. Public APIs, data formats, and model behavior may change without notice until 0.1.0.

### Roadmap

| Phase | Scope | Status |
|---|---|---|
| Phase 0 | Repository bootstrap, environment verification, sprint plan | In progress |
| Phase 1 | Telemetry generator and batch ingestion API | Planned |
| Phase 2 | Population outlier detection and 168h degradation forecasting | Planned |
| Phase 3 | Asymmetric risk triage and explainability | Planned |
| Phase 4 | Dashboard, audit export, and end-to-end demonstration | Planned |

## Installation

Not yet available. BurnSight has no released version; installation instructions will be published with 0.1.0.

## Architecture

BurnSight is intended as a single Python service. A FastAPI application exposes ingestion and decision endpoints; behind them, a linear pipeline validates telemetry, computes population statistics, extracts early-cycle features, forecasts terminal states, applies the asymmetric triage policy, and attaches explanations before writing an audit record. A static, server-rendered dashboard served by the same process presents batch results. The design deliberately avoids distributed infrastructure: one process, one dataset file, one audit trail.

## Repository layout

```
burnsight-ai/
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── SUPPORT.md
├── LICENSE
├── docs/
│   └── research-brief.md
└── .github/
    ├── CODEOWNERS
    ├── PULL_REQUEST_TEMPLATE.md
    └── ISSUE_TEMPLATE/
        ├── config.yml
        ├── bug_report.yml
        └── feature_request.yml
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for bug reports, feature proposals, and the test-driven development and coverage standards this repository requires.

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting. Please do not open public issues for security reports.

## Support

See [SUPPORT.md](SUPPORT.md) for where to get help.

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE).
