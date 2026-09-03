from __future__ import annotations

import hashlib
import hmac
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webhook
from ledger import Ledger
from models import Diagnosis, FailureEvent, RootCause
from pipeline.allocate import (
    CONTACT_POOL,
    DEFAULT_RETRY_BUDGET,
    MID_MONTH_PENALTY,
    PAYDAY_DAYS,
    PAYDAY_UPLIFT,
    RETRY_POOL,
    SUPPRESSED_FOR_BUDGET,
    SUPPRESSED_FOR_NO_HEADROOM,
    recovery_probability,
)
from pipeline.govern import IST, NPCI_RETRY_CAP, to_ist

WEBHOOK_SECRET = "whsec_test_secret"
EVENT_ID = "evt_live_001"


def _sign(body: bytes, secret: str = WEBHOOK_SECRET) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _payment_failed_body() -> dict:
    return {
        "event": "payment.failed",
        "created_at": 1_726_000_000,
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_live_test_01",
                    "amount": 49_900,
                    "currency": "INR",
                    "status": "failed",
                    "method": "upi",
                    "vpa": "user@hdfcbank",
                    "subscription_id": "sub_live_01",
                    "error_code": "BAD_REQUEST_ERROR",
                    "error_source": "customer",
                    "error_step": "payment_authorization",
                    "error_reason": "insufficient_funds",
                    "created_at": 1_726_000_000,
                }
            }
        },
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", WEBHOOK_SECRET)
    monkeypatch.setenv("NAKAD_LLM_PROVIDER", "none")
    monkeypatch.setenv("LIVE_LEDGER_PATH", str(tmp_path / "ledger_live.db"))
    webhook.reset_live_state()
    webhook.LIVE_LEDGER_PATH = str(tmp_path / "ledger_live.db")
    return TestClient(webhook.app, raise_server_exceptions=False)


def _restart(tmp_path) -> TestClient:
    """Drop every cached handle, as a redeploy would, keeping the same file."""
    webhook.reset_live_state()
    webhook.LIVE_LEDGER_PATH = str(tmp_path / "ledger_live.db")
    return TestClient(webhook.app, raise_server_exceptions=False)


def _post(client: TestClient, body: bytes, event_id: str = EVENT_ID, signature: str | None = None):
    return client.post(
        "/webhook",
        content=body,
        headers={
            "X-Razorpay-Event-Id": event_id,
            "X-Razorpay-Signature": signature if signature is not None else _sign(body),
            "Content-Type": "application/json",
        },
    )


def test_a_valid_signature_is_accepted(client):
    body = json.dumps(_payment_failed_body(), separators=(",", ":")).encode()
    response = _post(client, body)
    assert response.status_code == 200
    assert response.json()["status"] == "processed"

    ledger = Ledger(webhook.LIVE_LEDGER_PATH)
    assert len(ledger.read_all()) == 4

    live = client.get("/live").json()
    assert len(live) == 1
    assert live[0]["diagnosis"]["cause"] == "INSUFFICIENT_FUNDS"
    assert live[0]["action"]["action"] in {"RETRY", "SUPPRESS", "PAYMENT_LINK", "NUDGE", "MANDATE_REPRESENT"}


def test_an_invalid_signature_is_rejected_with_400(client):
    body = json.dumps(_payment_failed_body()).encode()
    response = _post(client, body, signature="deadbeef")
    assert response.status_code == 400
    assert client.get("/live").json() == []


def test_a_reserialised_body_fails_verification(client):
    original = json.dumps(_payment_failed_body(), separators=(",", ":")).encode()
    signature = _sign(original)
    # Same logical payload, different bytes — the integration bug.
    reserialised = json.dumps(json.loads(original)).encode()
    assert reserialised != original

    response = _post(client, reserialised, signature=signature)
    assert response.status_code == 400
    assert client.get("/live").json() == []


PAYDAY_IST_DAY = 27
MID_MONTH_IST_DAY = 11
MANDATE = "sub_live_01"


def _body_on(day: int, payment_id: str = "pay_live_test_01") -> dict:
    """A payment.failed body timestamped 09:00 IST on ``day`` of Sep 2026.

    Day 27 is inside PAYDAY_DAYS, so INSUFFICIENT_FUNDS routes to RETRY and
    draws on the retry pool; day 11 routes to PAYMENT_LINK and the contact pool.
    """
    stamp = int(datetime(2026, 9, day, 9, 0, tzinfo=IST).timestamp())
    body = _payment_failed_body()
    body["created_at"] = stamp
    entity = body["payload"]["payment"]["entity"]
    entity["created_at"] = stamp
    entity["id"] = payment_id
    return body


def _send(client: TestClient, body: dict, event_id: str):
    raw = json.dumps(body, separators=(",", ":")).encode()
    return _post(client, raw, event_id=event_id)


def test_two_events_on_one_mandate_consume_two_attempts_not_two_budgets(client):
    # The bug this guards: a stateless caller hands allocate the full 120/60
    # every time, so every webhook wins an uncontested ranking and the scarcity
    # the policy exists to model never materialises in production.
    for index in range(2):
        response = _send(
            client, _body_on(PAYDAY_IST_DAY, f"pay_{index}"), f"evt_{index}"
        )
        assert response.json()["status"] == "processed"

    budget = webhook._live_budget()

    assert budget.attempts_for(MANDATE, "2026-09") == 2
    assert budget.spent(RETRY_POOL) == 2
    assert budget.remaining(RETRY_POOL, DEFAULT_RETRY_BUDGET) == DEFAULT_RETRY_BUDGET - 2


def test_the_cap_is_enforced_against_our_count_not_the_payload(client):
    # Every payload here carries retries_used=0, because payment.notes is a
    # merchant-controlled free-text field. Only our own record knows better.
    for index in range(NPCI_RETRY_CAP):
        response = _send(
            client, _body_on(PAYDAY_IST_DAY, f"pay_{index}"), f"evt_{index}"
        )
        assert response.json()["status"] == "processed", index

    budget = webhook._live_budget()
    assert budget.attempts_for(MANDATE, "2026-09") == NPCI_RETRY_CAP

    over = _send(client, _body_on(PAYDAY_IST_DAY, "pay_over"), "evt_over")
    assert over.json()["status"] == "processed"

    latest = client.get("/live").json()[0]
    assert latest["action"]["action"] == "SUPPRESS"
    assert latest["action"]["rationale"].startswith(SUPPRESSED_FOR_NO_HEADROOM)
    # Refused without spending anything, which is the point of the pre-screen.
    assert budget.attempts_for(MANDATE, "2026-09") == NPCI_RETRY_CAP
    assert budget.spent(RETRY_POOL) == NPCI_RETRY_CAP

    ingested = [
        entry
        for entry in Ledger(webhook.LIVE_LEDGER_PATH).read_all()
        if entry.entry_type == "ingested"
    ]
    assert ingested[0].payload["retries_used_source"] == webhook.SOURCE_PAYLOAD
    assert ingested[-1].payload["retries_used_source"] == webhook.SOURCE_LEDGER
    assert ingested[-1].payload["retries_used"] == NPCI_RETRY_CAP


def test_budget_state_survives_a_restart(client, tmp_path):
    assert (
        _send(client, _body_on(PAYDAY_IST_DAY, "pay_0"), "evt_0").json()["status"]
        == "processed"
    )

    restarted = _restart(tmp_path)
    budget = webhook._live_budget()

    assert budget.spent(RETRY_POOL) == 1
    assert budget.attempts_for(MANDATE, "2026-09") == 1

    # A second event after the restart accumulates rather than starting over.
    _send(restarted, _body_on(PAYDAY_IST_DAY, "pay_1"), "evt_1")
    assert budget.attempts_for(MANDATE, "2026-09") == 2
    assert budget.remaining(RETRY_POOL, DEFAULT_RETRY_BUDGET) == DEFAULT_RETRY_BUDGET - 2


def test_an_exhausted_pool_suppresses_with_the_batch_rationale(client):
    budget = webhook._live_budget()
    for index in range(DEFAULT_RETRY_BUDGET):
        budget.record_spend(RETRY_POOL, f"earlier_{index}")
    assert budget.remaining(RETRY_POOL, DEFAULT_RETRY_BUDGET) == 0

    response = _send(client, _body_on(PAYDAY_IST_DAY, "pay_0"), "evt_0")
    assert response.json()["status"] == "processed"

    latest = client.get("/live").json()[0]
    assert latest["action"]["action"] == "SUPPRESS"
    assert latest["action"]["rationale"].startswith(SUPPRESSED_FOR_BUDGET)
    assert budget.attempts_for(MANDATE, "2026-09") is None


def test_the_contact_pool_is_tracked_separately_from_the_retry_pool(client):
    # Day 11 routes INSUFFICIENT_FUNDS to PAYMENT_LINK, which draws on contacts.
    _send(client, _body_on(MID_MONTH_IST_DAY, "pay_0"), "evt_0")
    budget = webhook._live_budget()

    assert budget.spent(CONTACT_POOL) == 1
    assert budget.spent(RETRY_POOL) == 0
    assert budget.attempts_for(MANDATE, "2026-09") is None


def test_spend_rows_age_out_of_the_rolling_window(client):
    budget = webhook._live_budget()
    budget.record_spend(RETRY_POOL, "old")

    assert budget.spent(RETRY_POOL, window=timedelta(hours=24)) == 1
    assert budget.spent(RETRY_POOL, window=timedelta(seconds=0)) == 0
    assert budget.prune(older_than=timedelta(seconds=0)) == 1
    assert budget.spent(RETRY_POOL) == 0


def test_a_psp_vpa_is_tagged_and_excluded_from_bank_outage_correlation():
    body = _payment_failed_body()
    body["payload"]["payment"]["entity"]["vpa"] = "user@ybl"
    # No bank/card issuer fields — the VPA is the only identity available.
    event, provenance = webhook.failure_event_from_webhook("payment.failed", body)

    assert event.issuer == f"{webhook.PSP_ISSUER_PREFIX}ybl"
    assert "PSP" in provenance["issuer"]
    assert "user@ybl" in provenance["issuer"]
    assert "excluded from bank-outage correlation" in provenance["issuer"]


def test_an_unambiguous_bank_vpa_maps_to_the_issuing_bank():
    body = _payment_failed_body()
    body["payload"]["payment"]["entity"]["vpa"] = "customer@okhdfcbank"
    event, provenance = webhook.failure_event_from_webhook("payment.failed", body)

    assert event.issuer == "HDFC"
    assert "unambiguously" in provenance["issuer"]
    assert "HDFC" in provenance["issuer"]
    assert "customer@okhdfcbank" in provenance["issuer"]


def test_a_missing_issuer_defaults_to_unknown_with_provenance():
    body = _payment_failed_body()
    entity = body["payload"]["payment"]["entity"]
    entity.pop("vpa", None)
    event, provenance = webhook.failure_event_from_webhook("payment.failed", body)

    assert event.issuer == webhook.UNKNOWN_ISSUER
    assert "excluded from bank-outage correlation" in provenance["issuer"]


def test_an_ifsc_payload_resolves_to_the_right_bank():
    body = _payment_failed_body()
    entity = body["payload"]["payment"]["entity"]
    entity.pop("vpa", None)
    entity["ifsc"] = "SBIN0005943"
    event, provenance = webhook.failure_event_from_webhook("payment.failed", body)

    assert event.issuer == "SBIN"
    assert "SBIN0005943" in provenance["issuer"]
    assert "State Bank of India" in provenance["issuer"]
    assert "ifsc-banknames.json" in provenance["issuer"]


def test_an_unknown_bank_code_falls_back_rather_than_raising():
    body = _payment_failed_body()
    entity = body["payload"]["payment"]["entity"]
    entity["vpa"] = "user@ybl"
    entity["ifsc"] = "ZZZZ0000001"  # not in the vendored map
    event, provenance = webhook.failure_event_from_webhook("payment.failed", body)

    assert event.issuer == f"{webhook.PSP_ISSUER_PREFIX}ybl"
    assert "unknown bank code 'ZZZZ'" in provenance["issuer"]
    assert "falling through" in provenance["issuer"]
    assert "PSP" in provenance["issuer"]


def test_an_unknown_ifsc_with_no_vpa_becomes_unknown():
    body = _payment_failed_body()
    entity = body["payload"]["payment"]["entity"]
    entity.pop("vpa", None)
    entity["ifsc"] = "ZZZZ0000001"
    event, provenance = webhook.failure_event_from_webhook("payment.failed", body)

    assert event.issuer == webhook.UNKNOWN_ISSUER
    assert "ZZZZ" in provenance["issuer"]


def _technical_live_event(index: int, when: datetime) -> FailureEvent:
    return FailureEvent(
        event_id=f"buf_{index}",
        payment_id=f"pay_buf_{index}",
        subscription_id=MANDATE,
        amount_paise=49_900,
        method="upi_autopay",
        issuer="HDFC",
        error_code="BAD_REQUEST_ERROR",
        error_source="issuer_bank",
        error_step="payment_authorization",
        error_reason="bank_technical_error",
        occurred_at=when,
        retries_used=0,
        days_since_mandate_created=45,
        day_of_month=to_ist(when).day,
        issuer_recent_failure_rate=0.05,
        amount_vs_customer_avg=1.0,
        true_cause=None,
    )


def test_a_single_live_event_correlates_against_buffered_recent_events(client):
    # Nine prior technical failures in the buffer + this webhook = denser than
    # min_events. Without the buffer, diagnose_batch([event]) always returns {}.
    now = datetime.now(timezone.utc)
    buffer = webhook._recent_failures()
    for index in range(9):
        buffer.record(_technical_live_event(index, now))

    stamp = int(now.timestamp())
    body = _payment_failed_body()
    body["created_at"] = stamp
    entity = body["payload"]["payment"]["entity"]
    entity.update(
        {
            "id": "pay_live_fleet",
            "created_at": stamp,
            "vpa": "customer@okhdfcbank",
            "error_reason": "bank_technical_error",
            "error_source": "issuer_bank",
        }
    )
    response = _send(client, body, "evt_fleet")
    assert response.json()["status"] == "processed"

    latest = client.get("/live").json()[0]
    assert latest["diagnosis"]["method"] == "fleet"
    assert latest["diagnosis"]["cause"] == "ISSUER_DOWNTIME"


def _boom(*_a, **_kw):
    raise RuntimeError("provider timeout")


def test_a_transient_failure_leaves_the_event_unseen_and_returns_non_200(
    client, monkeypatch
):
    monkeypatch.setattr(webhook, "diagnose_batch", _boom)
    body = json.dumps(_payment_failed_body(), separators=(",", ":")).encode()

    response = _post(client, body)

    assert response.status_code == 500
    # Unseen is the whole point: acknowledging here would tell Razorpay the
    # event was handled and it would never be delivered again.
    assert EVENT_ID not in webhook._seen_events()


def test_a_redelivery_after_a_transient_failure_processes_successfully(
    client, monkeypatch
):
    body = json.dumps(_payment_failed_body(), separators=(",", ":")).encode()
    working = webhook.diagnose_batch

    monkeypatch.setattr(webhook, "diagnose_batch", _boom)
    assert _post(client, body).status_code == 500

    # Only the provider recovers; the secret and the ledger path stay as they
    # were, exactly as they would across a Razorpay retry.
    monkeypatch.setattr(webhook, "diagnose_batch", working)
    retry = _post(client, body)

    assert retry.status_code == 200
    assert retry.json()["status"] == "processed"
    assert webhook._seen_events().outcome_of(EVENT_ID) == webhook.OUTCOME_PROCESSED
    assert len(Ledger(webhook.LIVE_LEDGER_PATH).read_all()) == 4


@pytest.mark.parametrize(
    "raw, label",
    [
        (b"{not json at all", "malformed json"),
        # Well-formed JSON that cannot be mapped: no payment or subscription.
        (json.dumps({"event": "payment.failed", "payload": {}}).encode(), "unmappable"),
    ],
)
def test_an_unprocessable_payload_is_marked_seen_and_returns_200(client, raw, label):
    response = _post(client, raw)

    assert response.status_code == 200, label
    assert response.json()["status"] == "accepted", label
    # Retrying identical bytes fails identically, so burning Razorpay's
    # 24-hour retry budget on it achieves nothing.
    assert webhook._seen_events().outcome_of(EVENT_ID) == webhook.OUTCOME_UNPROCESSABLE


def test_dedupe_survives_a_restart(client, tmp_path):
    body = json.dumps(_payment_failed_body(), separators=(",", ":")).encode()
    assert _post(client, body).json()["status"] == "processed"

    restarted = _restart(tmp_path)

    assert EVENT_ID in webhook._seen_events()
    assert _post(restarted, body).json()["status"] == "duplicate"
    # Still four: the redelivery did not re-run the pipeline.
    assert len(Ledger(webhook.LIVE_LEDGER_PATH).read_all()) == 4


def test_old_dedupe_rows_are_prunable(client):
    body = json.dumps(_payment_failed_body(), separators=(",", ":")).encode()
    _post(client, body)
    seen = webhook._seen_events()
    assert EVENT_ID in seen

    assert seen.prune(older_than=timedelta(hours=48)) == 0
    assert seen.prune(older_than=timedelta(seconds=0)) == 1
    assert EVENT_ID not in seen


def test_day_of_month_is_the_ist_day_not_the_utc_one():
    ist_early_morning = datetime(2026, 9, 25, 1, 0, tzinfo=IST)
    assert ist_early_morning.astimezone(timezone.utc).day == 24

    body = _payment_failed_body()
    stamp = int(ist_early_morning.timestamp())
    body["created_at"] = stamp
    body["payload"]["payment"]["entity"]["created_at"] = stamp

    event, _ = webhook.failure_event_from_webhook("payment.failed", body)

    assert event.day_of_month == 25
    assert 25 in PAYDAY_DAYS and 24 not in PAYDAY_DAYS


def test_the_ist_day_carries_through_to_the_recovery_probability():
    def probability_for(when: datetime) -> float:
        body = _payment_failed_body()
        stamp = int(when.timestamp())
        body["created_at"] = stamp
        body["payload"]["payment"]["entity"]["created_at"] = stamp
        event, _ = webhook.failure_event_from_webhook("payment.failed", body)
        diagnosis = Diagnosis(
            event_id=event.event_id,
            cause=RootCause.INSUFFICIENT_FUNDS,
            confidence=0.9,
            evidence=["x"],
            method="rule",
        )
        return recovery_probability(event, diagnosis)

    on_payday = probability_for(datetime(2026, 9, 25, 1, 0, tzinfo=IST))
    mid_month = probability_for(datetime(2026, 9, 15, 1, 0, tzinfo=IST))

    assert round(on_payday - mid_month, 10) == PAYDAY_UPLIFT - MID_MONTH_PENALTY


def test_a_duplicate_event_id_is_processed_exactly_once(client, monkeypatch):
    calls = {"diagnose": 0}
    real = webhook.diagnose_batch

    def counting(events, **_kw):
        calls["diagnose"] += 1
        return real(events, **_kw)

    monkeypatch.setattr(webhook, "diagnose_batch", counting)

    body = json.dumps(_payment_failed_body(), separators=(",", ":")).encode()
    first = _post(client, body)
    second = _post(client, body)

    assert first.status_code == 200
    assert first.json()["status"] == "processed"
    assert second.status_code == 200
    assert second.json()["status"] == "duplicate"
    assert calls["diagnose"] == 1
    assert len(client.get("/live").json()) == 1
    assert len(Ledger(webhook.LIVE_LEDGER_PATH).read_all()) == 4
