"""Questions that fail again and again: counted with denominators, linked to their cases."""

from __future__ import annotations

import html
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("sqlalchemy")

from tests.test_cloud_checks import pin
from tests.test_cloud_monitor import _bad
from tests.test_cloud_runs import a_run, client_for
from tests.test_cloud_web import sign_in
from tests.test_eval_compare import fake_run
from tests.test_shared_records import TEXT, case
from vectraudit.cloud import quotas
from vectraudit.cloud.db import transaction
from vectraudit.cloud.lead import recurring
from vectraudit.cloud.ledger import failed_questions
from vectraudit.cloud.models import Run, utcnow
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.policy import parse_policy
from vectraudit.integrations.shared_records import project_record

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
P = "/api/v1/workspaces/acme/projects/search"


def decision(
    hours, regressed=(), critical=(), safety=(), dataset="d", trigger="release", listed=True
):
    summary = {"failed_questions": None}
    if listed:
        summary["failed_questions"] = {
            "regressed": list(regressed),
            "critical": list(critical),
            "safety": list(safety),
        }
    return {
        "at": T0 + timedelta(hours=hours),
        "check_id": f"c{hours}",
        "dataset_sha256": dataset,
        "trigger": trigger,
        "summary": summary if listed else {},
    }


def test_recurrence_counts_with_denominators_streaks_and_kinds():
    newest_first = [
        decision(5, regressed=["q1", "q2"], critical=["q1"]),
        decision(4, regressed=["q1"], trigger="monitor"),
        decision(3, regressed=["q3"]),
        decision(2, regressed=["q2", "q3"], safety=["q2"]),
        decision(1, listed=False),  # before questions were recorded: not a pass
        decision(0, regressed=["q1", "q2", "q3"], dataset="old"),  # another benchmark
    ]
    found = recurring(newest_first)
    assert found["dataset_sha256"] == "d"
    assert found["checks"] == 4
    assert found["without_questions"] == 1
    by_id = {q["query_id"]: q for q in found["questions"]}
    assert list(by_id) == ["q1", "q2", "q3"]  # still failing first, then most failures
    q1 = by_id["q1"]
    assert (q1["failures"], q1["checks"], q1["streak"], q1["still_failing"]) == (2, 4, 2, True)
    assert q1["kinds"] == {"regressed": 2, "critical": 1, "safety": 0}  # one failure per check
    assert q1["triggers"] == {"release": 1, "monitor": 1}
    assert q1["last_check_id"] == "c5"
    assert q1["first_failed_at"] == T0 + timedelta(hours=4)
    assert by_id["q2"]["streak"] == 1
    assert by_id["q3"]["still_failing"] is False
    assert by_id["q3"]["failures"] == 2
    assert recurring(newest_first, min_failures=3)["questions"] == []
    assert recurring([])["questions"] == []


def test_failed_questions_come_from_regressions_critical_and_safety_rules():
    base = fake_run([1.0] * 20)
    cand = fake_run([1.0] * 18 + [0.0, 0.0], run_id="c")
    cand["queries"][3].update(forbidden_ids=["secret"], permission_checked=True)
    base["queries"][3].update(forbidden_ids=[], permission_checked=True)
    for run in (base, cand):
        for q in run["queries"]:
            q["repeat_ids"] = [q["ids"]]
        run["timing"] = {"repeats": 1}
    rules = parse_policy(
        {
            "metrics": {"ndcg": {"max_drop": 0.5}},
            "critical_queries": [{"query_id": "q19", "doc_id": "a", "max_rank": 1}],
        }
    )
    found = failed_questions(compare_runs(base, cand, rules))
    assert found == {"regressed": ["q18", "q19"], "critical": ["q19"], "safety": ["q3"]}


def test_hosted_recurring_problems_link_cases_and_survive_retention(world, cloud_app) -> None:
    client = client_for(world, "acme_runner")
    client.upload_run(a_run("base"))
    pin(world, "acme_engineer", "base")
    for name, bad in (("r1", True), ("r2", False), ("r3", True)):
        client.upload_run(_bad(name) if bad else a_run(name, label=name))
        client.create_check(name)
    client.upload_record(project_record("case", case(), share_query_text=True))
    found = world.client.get(f"{P}/recurring", headers=world.as_("acme_viewer")).json()
    assert found["checks"] == 3
    by_id = {q["query_id"]: q for q in found["questions"]}
    assert len(by_id) == 20  # every question regressed in the two failing checks
    q1 = by_id["q1"]
    assert (q1["failures"], q1["checks"], q1["still_failing"], q1["streak"]) == (2, 3, True, 1)
    (linked,) = q1["cases"]
    assert linked["question_text"] == TEXT
    assert linked["status"] == "open"
    assert by_id["q0"]["cases"] == []

    with transaction(cloud_app.state.sessions) as s:
        for run in s.query(Run).filter(Run.run_id.in_(["r1", "r2", "r3"])):
            run.uploaded_at = utcnow() - timedelta(days=40)
    with transaction(cloud_app.state.sessions) as s:
        quotas.purge_expired(s)
    later = world.client.get(f"{P}/recurring", headers=world.as_("acme_viewer")).json()
    assert {q["query_id"] for q in later["questions"]} == set(by_id)
    assert world.client.get(f"{P}/recurring", headers=world.as_("globex_owner")).status_code == 404


def test_without_shared_text_only_ids_are_shown(world, cloud_app) -> None:
    client = client_for(world, "acme_runner")
    client.upload_run(a_run("base"))
    pin(world, "acme_engineer", "base")
    for name in ("r1", "r2"):
        client.upload_run(_bad(name))
        client.create_check(name)
    client.upload_record(project_record("case", case()))
    found = world.client.get(f"{P}/recurring", headers=world.as_("acme_viewer")).json()
    q1 = next(q for q in found["questions"] if q["query_id"] == "q1")
    assert q1["cases"][0]["question_text"] is None
    browser = sign_in(cloud_app, world.tokens["acme_viewer"])
    page = browser.get("/app/acme/search").text
    assert "Recurring problems" in page
    assert "All recurring problems (15 more)" in page
    full = browser.get("/app/acme/search/recurring").text
    assert full.count('class="mono"') >= 20
    assert html.escape(TEXT) not in full
    assert "None shared" in full
    assert "/app/acme/search/records/" in full
    outsider = sign_in(cloud_app, world.tokens["globex_owner"])
    assert outsider.get("/app/acme/search/recurring").status_code == 404


def test_shared_text_appears_on_the_page_when_the_case_shared_it(world, cloud_app) -> None:
    client = client_for(world, "acme_runner")
    client.upload_run(a_run("base"))
    pin(world, "acme_engineer", "base")
    for name in ("r1", "r2"):
        client.upload_run(_bad(name))
        client.create_check(name)
    client.upload_record(project_record("case", case(), share_query_text=True))
    browser = sign_in(cloud_app, world.tokens["acme_viewer"])
    page = browser.get("/app/acme/search/recurring").text
    assert html.escape(TEXT) in page  # the fixture's text carries markup; it is escaped
    assert "<script>" not in page
