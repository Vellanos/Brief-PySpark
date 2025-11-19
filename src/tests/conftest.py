# File: src/tests/conftest.py
from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

@dataclass
class _Counters:
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    xfailed: int = 0
    xpassed: int = 0
    errors: int = 0

_cnt = _Counters()
_t0 = None

@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    global _t0
    _t0 = time.time()
    print("\n=== Test session start ===")

def _status_emoji(report) -> str:
    if report.skipped:
        return "⏭️ "
    if report.passed and report.when == "call":
        return "✅"
    if report.failed and report.when == "call":
        return "❌"
    if report.failed and report.when != "call":
        return "💥"
    return "•"

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    print(f"\n→ {item.nodeid}")
    outcome = yield
    return outcome

def pytest_runtest_logreport(report: pytest.TestReport):
    if report.when != "call" and not report.failed:
        return
    emoji = _status_emoji(report)
    dur = getattr(report, "duration", 0.0)
    print(f"{emoji} {report.nodeid} [{dur:.2f}s]")

    if report.when == "call":
        if report.passed:
            _cnt.passed += 1
        elif report.failed:
            _cnt.failed += 1
        elif report.skipped:
            _cnt.skipped += 1
    else:
        if report.failed:
            _cnt.errors += 1

def pytest_sessionfinish(session, exitstatus):
    dt = time.time() - _t0 if _t0 else 0.0
    print("\n=== TEST SUMMARY ===")
    print(f"Passed :   {_cnt.passed}")
    print(f"Failed :   {_cnt.failed}")
    print(f"Errors :   {_cnt.errors}")
    print(f"Skipped:   {_cnt.skipped}")
    print(f"Duration:  {dt:.2f}s")
    print("====================\n")
