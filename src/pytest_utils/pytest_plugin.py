# Adapted from
# https://github.com/ucsb-gradescope-tools/pytest_utils/blob/master/pytest_utils/pytest_plugin.py
#
# Turns a pytest run into Gradescope's results.json. Load it with
#     pytest -p src.pytest_utils.pytest_plugin ...
# Every test is reported, including ones that were skipped or errored during
# setup: those score 0 and carry the reason, so a student sees "no model.pt
# submitted" rather than a test that silently vanished from the list.

import json

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item):
    x = yield
    x._result.max_score = getattr(item._obj, "max_score", 0)
    x._result.visibility = getattr(item._obj, "visibility", "visible")


def _describe(report):
    """A short, student-readable reason for a non-passing outcome."""
    longrepr = report.longrepr
    if isinstance(longrepr, tuple):            # skip: (path, lineno, reason)
        return str(longrepr[2])
    try:
        return str(longrepr.chain[0][0].reprentries[0])
    except Exception:
        text = str(longrepr)
        return text if len(text) <= 4000 else text[-4000:]


def pytest_terminal_summary(terminalreporter, exitstatus):
    json_results = {
        "stdout_visibility": "visible",
        "tests": [],
    }

    reports = []
    for category in ("passed", "failed", "error", "skipped"):
        reports.extend(terminalreporter.stats.get(category, []))

    seen = set()
    for r in reports:
        if not hasattr(r, "location"):       # collection reports carry no test
            continue
        name = r.location[2]
        key = (r.location[0], name)
        if key in seen:                       # one line per test, not per phase
            continue
        seen.add(key)

        max_score = getattr(r, "max_score", 0)
        visibility = getattr(r, "visibility", "visible")
        if r.outcome == "passed" and r.when == "call":
            score, status, output = max_score, "passed", ""
        elif r.outcome == "skipped":
            score, status, output = 0, "failed", "skipped: " + _describe(r)
        else:
            score, status, output = 0, "failed", _describe(r)

        json_results["tests"].append(
            {
                "score": score,
                "status": status,
                "max_score": max_score,
                "name": name,
                "output": output,
                "visibility": visibility,
            }
        )

    with open("results.json", "w") as results:
        results.write(json.dumps(json_results, indent=4))
