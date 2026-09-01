from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import FailureEvent, RootCause
from pipeline import diagnose as diagnose_mod
from pipeline.diagnose import _LlmItem, _LlmResponse, diagnose_batch


def _event(**kw) -> FailureEvent:
    defaults = dict(
        event_id="e1",
        payment_id="p1",
        subscription_id=None,
        amount_paise=100,
        method="emandate",
        issuer="hdfc",
        error_code="BAD_REQUEST_ERROR",
        error_source="issuer_bank",
        error_step="payment_authorization",
        error_reason="payment_failed",
        occurred_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        days_since_mandate_created=40,
        day_of_month=15,
        issuer_recent_failure_rate=0.05,
        amount_vs_customer_avg=1.0,
    )
    defaults.update(kw)
    return FailureEvent(**defaults)


def _stub_gemini(monkeypatch, generate_content):
    monkeypatch.setenv("NAKAD_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        "google.genai.Client",
        lambda api_key: SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        ),
    )


def _ok_items(n: int) -> _LlmResponse:
    return _LlmResponse(
        items=[
            _LlmItem(cause=RootCause.DEAD_MANDATE, confidence=0.7, evidence=["sig"])
            for _ in range(n)
        ]
    )


def test_gemini_structured_output_and_fallback(monkeypatch):
    monkeypatch.setenv("NAKAD_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    captured: dict = {}

    def generate_content(*, model, contents, config):
        captured["model"] = model
        captured["schema"] = config.response_schema
        captured["mime"] = config.response_mime_type
        captured["afc_disabled"] = bool(
            getattr(getattr(config, "automatic_function_calling", None), "disable", False)
        )
        return SimpleNamespace(
            parsed=_LlmResponse(
                items=[
                    _LlmItem(
                        cause=RootCause.DEAD_MANDATE,
                        confidence=0.7,
                        evidence=["payment_failed"],
                    )
                ]
            ),
            text="",
        )

    monkeypatch.setattr(
        "google.genai.Client",
        lambda api_key: SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
    )

    diagnoses, stats = diagnose_batch([_event()])
    assert captured["model"] == "gemini-2.0-flash"
    assert captured["mime"] == "application/json"
    assert captured["schema"] is _LlmResponse
    assert captured["afc_disabled"] is True
    assert stats["unique_signatures"] == 1
    assert stats["unique_cache_keys"] == 1
    assert len(diagnoses) == 1
    assert diagnoses[0].method == "llm"
    assert diagnoses[0].cause is RootCause.DEAD_MANDATE
    assert stats["llm_calls"] == 1
    assert stats["fallbacks"] == 0

    def boom(**_kw):
        raise RuntimeError("gemini down")

    monkeypatch.setattr(
        "google.genai.Client",
        lambda api_key: SimpleNamespace(models=SimpleNamespace(generate_content=boom)),
    )
    diagnose_mod._cache.clear()
    diagnoses, stats = diagnose_batch([_event(event_id="e2", payment_id="p2")])
    assert diagnoses[0].cause is RootCause.INSUFFICIENT_FUNDS
    assert diagnoses[0].confidence == 0.0
    assert stats["fallbacks"] == 1
    assert any(err.startswith("RuntimeError: gemini down") for err in stats["llm_errors"])


def test_identical_signatures_share_one_llm_call(monkeypatch):
    def generate_content(*, model, contents, config):
        catalog = json.loads(contents.split("Input: ", 1)[1])
        return SimpleNamespace(parsed=_ok_items(len(catalog)), text="")

    _stub_gemini(monkeypatch, generate_content)
    events = [
        _event(event_id="e1", payment_id="p1"),
        _event(event_id="e2", payment_id="p2"),
    ]
    diagnoses, stats = diagnose_batch(events)
    assert stats["llm_calls"] == 1
    assert stats["cache_hits"] == 1
    assert stats["unique_signatures"] == 1
    assert stats["unique_cache_keys"] == 1
    assert [d.event_id for d in diagnoses] == ["e1", "e2"]
    assert {d.cause for d in diagnoses} == {RootCause.DEAD_MANDATE}


def test_forty_five_distinct_signatures_use_three_batches(monkeypatch):
    sizes: list[int] = []
    constructed: list[object] = []

    def generate_content(*, model, contents, config):
        catalog = json.loads(contents.split("Input: ", 1)[1])
        sizes.append(len(catalog))
        return SimpleNamespace(parsed=_ok_items(len(catalog)), text="")

    def fake_client(api_key):
        constructed.append(api_key)
        return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    monkeypatch.setenv("NAKAD_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("google.genai.Client", fake_client)
    events = [
        _event(event_id=f"e{i}", payment_id=f"p{i}", error_code=f"E{i}")
        for i in range(45)
    ]
    _, stats = diagnose_batch(events)
    assert stats["llm_calls"] == 3
    assert stats["unique_signatures"] == 45
    assert stats["unique_cache_keys"] == 45
    assert stats["cache_hits"] == 0
    assert sizes == [20, 20, 5]
    assert len(constructed) == 1


def test_invalid_enum_in_free_text_falls_back(monkeypatch):
    def generate_content(*, model, contents, config):
        return SimpleNamespace(
            parsed=None,
            text='{"items":[{"cause":"NO_MONEY","confidence":0.9,"evidence":["x"]}]}',
        )

    _stub_gemini(monkeypatch, generate_content)
    diagnoses, stats = diagnose_batch([_event()])
    assert diagnoses[0].cause is RootCause.INSUFFICIENT_FUNDS
    assert diagnoses[0].confidence == 0.0
    assert stats["fallbacks"] == 1
    assert stats["llm_errors"]
    assert any("NO_MONEY" in err for err in stats["llm_errors"])


def test_cache_buckets_not_precise_values(monkeypatch):
    captured: list[dict] = []

    def generate_content(*, model, contents, config):
        catalog = json.loads(contents.split("Input: ", 1)[1])
        captured.extend(catalog)
        return SimpleNamespace(parsed=_ok_items(len(catalog)), text="")

    _stub_gemini(monkeypatch, generate_content)
    events = [
        _event(
            event_id="e1",
            payment_id="p1",
            days_since_mandate_created=40,
            day_of_month=15,
            issuer_recent_failure_rate=0.06,
            amount_vs_customer_avg=1.0,
        ),
        _event(
            event_id="e2",
            payment_id="p2",
            days_since_mandate_created=80,
            day_of_month=12,
            issuer_recent_failure_rate=0.10,
            amount_vs_customer_avg=1.2,
        ),
    ]
    _, stats = diagnose_batch(events)
    assert stats["unique_signatures"] == 2
    assert stats["unique_cache_keys"] == 1
    assert stats["llm_calls"] == 1
    assert stats["cache_hits"] == 1
    assert len(captured) == 1
    assert captured[0]["days_since_mandate_created"] == 40
    assert captured[0]["day_of_month"] == 15
    assert captured[0]["issuer_recent_failure_rate"] == 0.06
    assert captured[0]["amount_vs_customer_avg"] == 1.0
