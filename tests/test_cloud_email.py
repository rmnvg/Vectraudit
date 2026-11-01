"""Email delivery: the same small event as Slack, once per event, through the operator's SMTP."""

from __future__ import annotations

import re
import smtplib
from email import message_from_bytes
from email.policy import default as email_policy
from typing import ClassVar

import pytest

pytest.importorskip("sqlalchemy")

from tests.test_cloud_checks import pin
from tests.test_cloud_monitor import _bad
from tests.test_cloud_runs import a_run, client_for
from tests.test_cloud_web import sign_in
from tests.test_cloud_webhooks import HOOKS, Recorder, deliver, public_dns  # noqa: F401
from vectraudit.cloud.db import transaction
from vectraudit.cloud.models import WebhookDelivery
from vectraudit.cloud.settings import Settings, SettingsError, SmtpSettings

SMTP = SmtpSettings(
    host="smtp.example.com", sender="vectraudit@example.com", username="u", password="p"
)


def add_email(world, address="mailto:release-team@example.com", who="acme_owner"):
    return world.client.post(
        HOOKS,
        json={"url": address, "style": "email", "events": ["decision"]},
        headers=world.as_(who),
    )


def failing_check(world):
    client = client_for(world, "acme_runner")
    client.upload_run(a_run("base"))
    pin(world, "acme_engineer", "base")
    worse = _bad("worse")
    worse["queries"][0]["text"] = "PRIVATE_QUESTION_TEXT"
    client.upload_run(worse)
    return client.create_check("worse")


@pytest.mark.parametrize(
    "address",
    [
        "release-team@example.com",
        "mailto:a@b",
        "mailto:one@example.com,two@example.com",
        "mailto:Team <team@example.com>",
        "mailto:team@example.com?cc=boss@example.com",
        "mailto:team@example.com\nBcc: everyone@example.com",
    ],
)
def test_only_one_plain_address_is_accepted(world, address) -> None:
    refused = add_email(world, address)
    assert refused.status_code == 422
    assert refused.json()["error_code"] == "invalid_webhook_url"


def test_a_decision_is_emailed_once_with_no_evidence(world, cloud_app) -> None:
    created = add_email(world)
    assert created.status_code == 201
    listed = world.client.get(HOOKS, headers=world.as_("acme_engineer")).json()
    assert listed[0]["url"] == "mailto:r…@example.com"  # addresses are not listed in full
    check = failing_check(world)
    mailer = Recorder()
    assert deliver(cloud_app, Recorder(), send_email=mailer) == {
        "delivered": 1,
        "retrying": 0,
        "failed": 0,
    }
    assert deliver(cloud_app, Recorder(), send_email=mailer)["delivered"] == 0
    ((url, body, headers),) = mailer.calls
    assert (url, headers["To"]) == (
        "mailto:release-team@example.com",
        "release-team@example.com",
    )
    message = message_from_bytes(body, policy=email_policy)
    assert message["Subject"].startswith("[Vectraudit] acme/search: nightly @ abc123 is blocked")
    assert message["X-Vectraudit-Event-Key"] == f"decision:{check['id']}"
    text = message.get_content()
    assert f"https://va.example.com/app/acme/search/decisions/{check['id']}" in text
    assert "PRIVATE_QUESTION_TEXT" not in body.decode()
    assert "q0" not in body.decode()


def test_without_a_mail_server_deliveries_wait_with_a_clear_error(world, cloud_app) -> None:
    add_email(world)
    failing_check(world)
    assert deliver(cloud_app, Recorder()) == {"delivered": 0, "retrying": 1, "failed": 0}
    with transaction(cloud_app.state.sessions) as s:
        row = s.query(WebhookDelivery).one()
        assert "not configured" in (row.last_error or "")


class FakeSMTP:
    """Stands in for smtplib.SMTP and records the conversation."""

    sent: ClassVar[list] = []
    calls: ClassVar[list] = []
    fail_with: ClassVar[Exception | None] = None

    def __init__(self, host, port, timeout):
        FakeSMTP.calls.append(("connect", host, port))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self, context):
        FakeSMTP.calls.append(("starttls",))

    def login(self, user, password):
        FakeSMTP.calls.append(("login", user))

    def send_message(self, message, to_addrs):
        if FakeSMTP.fail_with is not None:
            raise FakeSMTP.fail_with
        FakeSMTP.sent.append((message, to_addrs))


@pytest.fixture
def fake_smtp(monkeypatch):
    FakeSMTP.sent, FakeSMTP.calls, FakeSMTP.fail_with = [], [], None
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    return FakeSMTP


def test_smtp_send_uses_starttls_login_and_the_configured_sender(world, cloud_app, fake_smtp):
    add_email(world)
    failing_check(world)
    assert deliver(cloud_app, Recorder(), smtp=SMTP)["delivered"] == 1
    assert fake_smtp.calls == [("connect", "smtp.example.com", 587), ("starttls",), ("login", "u")]
    ((message, to),) = fake_smtp.sent
    assert (message["From"], to) == ("vectraudit@example.com", ["release-team@example.com"])


@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        (smtplib.SMTPRecipientsRefused({"x@example.com": (550, b"no such user")}), "failed"),
        (smtplib.SMTPResponseException(451, b"try later"), "retrying"),
        (smtplib.SMTPResponseException(554, b"rejected"), "failed"),
        (smtplib.SMTPServerDisconnected("gone"), "retrying"),
    ],
)
def test_refusals_fail_at_once_and_temporary_errors_retry(
    world, cloud_app, fake_smtp, error, outcome
):
    add_email(world)
    failing_check(world)
    fake_smtp.fail_with = error
    counts = deliver(cloud_app, Recorder(), smtp=SMTP)
    assert counts[outcome] == 1


def test_smtp_settings_come_from_the_environment():
    base = {"VECTRAUDIT_CLOUD_DATABASE_URL": "sqlite://", "VECTRAUDIT_CLOUD_SECRET_KEY": "x" * 40}
    assert Settings.from_env(base).smtp is None
    configured = Settings.from_env(
        {
            **base,
            "VECTRAUDIT_CLOUD_SMTP_HOST": "smtp.example.com",
            "VECTRAUDIT_CLOUD_SMTP_FROM": "va@example.com",
            "VECTRAUDIT_CLOUD_SMTP_SECURITY": "ssl",
        }
    ).smtp
    assert configured is not None
    assert (configured.port, configured.security) == (465, "ssl")
    with pytest.raises(SettingsError, match="both"):
        Settings.from_env({**base, "VECTRAUDIT_CLOUD_SMTP_HOST": "smtp.example.com"})
    with pytest.raises(SettingsError, match="starttls, ssl or none"):
        Settings.from_env(
            {
                **base,
                "VECTRAUDIT_CLOUD_SMTP_HOST": "h",
                "VECTRAUDIT_CLOUD_SMTP_FROM": "f@x.com",
                "VECTRAUDIT_CLOUD_SMTP_SECURITY": "tls",
            }
        )


def test_the_dashboard_adds_an_email_destination(world, cloud_app) -> None:
    owner = sign_in(cloud_app, world.tokens["acme_owner"])
    page = owner.get("/app/acme/search").text
    assert 'value="email"' in page
    csrf = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
    created = owner.post(
        "/app/acme/search/webhooks",
        data={
            "csrf": csrf,
            "url": "mailto:release-team@example.com",
            "style": "email",
            "event_decision": "1",
        },
    )
    assert created.status_code == 201
    assert "Email" in owner.get("/app/acme/search").text


@pytest.mark.parametrize("extra", ["", " because of a long headline" * 4])
def test_a_long_subject_is_never_folded_into_a_leading_space(extra) -> None:
    from types import SimpleNamespace

    from vectraudit.cloud.webhooks import _email

    headline = "nightly @ abc123 is blocked: 3 release rules fail." + extra
    raw = _email(
        SimpleNamespace(url="mailto:team@example.com"),
        {"event": "decision", "workspace": "acme", "project": "search", "headline": headline},
        SimpleNamespace(event="decision", event_key="decision:c1", id="d1"),
    )
    subject = message_from_bytes(raw, policy=email_policy)["Subject"]
    assert subject == f"[Vectraudit] acme/search: {headline}"[:200]
    assert b"Subject: [Vectraudit]" in raw  # on the header line itself, not a continuation
