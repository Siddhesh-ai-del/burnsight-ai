# Security Policy

## Reporting a vulnerability

Please report security vulnerabilities privately using GitHub's **private vulnerability reporting** for this repository (Security tab → Report a vulnerability). Do not open a public issue for security reports.

Include, where possible:

- The affected component and version or commit
- A description of the issue and its impact
- Reproduction steps or a proof of concept
- Any suggested remediation

You will receive an acknowledgement within a reasonable period of time. Confirmed vulnerabilities will be addressed before public disclosure, and reporters will be credited unless they request otherwise.

## Scope

In scope: BurnSight application code, its ingestion endpoints, and any authentication or authorization logic added in future releases.

Out of scope: reports that require access to proprietary customer telemetry to demonstrate, denial-of-service against local development deployments, and vulnerabilities in third-party dependencies (these should be reported upstream).

## User guarantees

BurnSight is pre-release software. Until 0.1.0 is published:

- There is **no** uptime, availability, or compatibility guarantee.
- No authentication or multi-tenant access control is provided; the service is intended for trusted, single-operator deployment.
- Telemetry submitted to the service is processed in memory and written only to the local audit store; do not deploy it on shared infrastructure without a review of your own data-handling obligations.

## Supported versions

| Version | Supported |
|---|---|
| pre-release (`main`) | Best effort |
| 0.x | Planned |
