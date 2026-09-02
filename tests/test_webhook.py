from __future__ import annotations

import hashlib
import hmac
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webhook
from ledger import Ledger

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
    return TestClient(webhook.app)


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


def test_a_duplicate_event_id_is_processed_exactly_once(client, monkeypatch):
    calls = {"diagnose": 0}
    real = webhook.diagnose_batch

    def counting(events):
        calls["diagnose"] += 1
        return real(events)

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
