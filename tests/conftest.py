"""Session-wide test wiring.

Keeps the runtime audit log (append-only JSONL, a shipping artifact) out of
the repository: every test run gets a throwaway path via BURNSIGHT_AUDIT_PATH.
"""

import os

import pytest

from burnsight.audit import AUDIT_ENV


@pytest.fixture(scope="session", autouse=True)
def isolated_audit_log(tmp_path_factory):
    path = tmp_path_factory.mktemp("audit") / "audit.jsonl"
    previous = os.environ.get(AUDIT_ENV)
    os.environ[AUDIT_ENV] = str(path)
    yield path
    if previous is None:
        os.environ.pop(AUDIT_ENV, None)
    else:
        os.environ[AUDIT_ENV] = previous
