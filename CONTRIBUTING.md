# Contributing to BurnSight

Thank you for your interest in BurnSight. This document explains how to report bugs, propose features, and submit changes.

## Reporting bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml). Before filing, search existing issues for duplicates. Include the exact input that reproduces the problem and the observed versus expected behavior. Redact any telemetry you are not authorized to share — burn-in data is frequently export-controlled or proprietary.

## Proposing features

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml). Lead with the problem the feature solves, not the implementation. For each proposal, state which principle it advances: physics-grounded reasoning, asymmetric risk handling, explainability and auditability, or software-only operation.

## Development workflow

1. Fork the repository and create a branch from `main`.
2. Follow strict test-driven development: write a failing test first, implement minimally to pass, then refactor while tests stay green.
3. Run the full test suite before opening a pull request.
4. Use conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `perf:`, `ci:`).
5. Open a pull request using the provided template and complete its checklist.

## Coverage standards

| Code type | Minimum coverage |
|---|---|
| Standard code | 80% |
| Risk triage and loss-matrix calculations | 100% |
| Input validation boundaries | 100% |

Tests are written before implementation. A pull request that lowers coverage without explicit justification will not be merged.

## Code review

Every pull request receives at least one review. Critical issues must be resolved before merge; high and medium findings are either addressed or recorded as follow-up issues.

## Code of conduct

By participating, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).
