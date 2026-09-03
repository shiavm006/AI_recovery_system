from __future__ import annotations

import json
import logging
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
    assert diagnoses[0].cause is RootCause.UNKNOWN
    assert diagnoses[0].confidence == 0.0
    assert stats["fallbacks"] == 1
    assert any(err.startswith("RuntimeError: gemini down") for err in stats["llm_errors"])
    assert "gemini down" in " ".join(diagnoses[0].evidence)


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
    monkeypatch.setenv("LLM_BATCH_DELAY_SECONDS", "0")
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
    assert diagnoses[0].cause is RootCause.UNKNOWN
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


def test_a_failed_call_never_fabricates_a_real_cause(monkeypatch):
    # The old fallback returned INSUFFICIENT_FUNDS, which is both the most
    # common true cause and the highest-scoring one downstream, so failure
    # looked exactly like the most valuable kind of success.
    def boom(**_kw):
        raise RuntimeError("gemini down")

    _stub_gemini(monkeypatch, boom)
    events = [_event(event_id=f"e{i}", payment_id=f"p{i}") for i in range(5)]
    diagnoses, stats = diagnose_batch(events)

    assert stats["fallbacks"] == 5
    assert {d.cause for d in diagnoses} == {RootCause.UNKNOWN}
    assert all(d.cause is not RootCause.INSUFFICIENT_FUNDS for d in diagnoses)
    assert all(d.confidence == 0.0 for d in diagnoses)


def test_unknown_is_not_offered_to_the_classifier(monkeypatch):
    prompts: list[str] = []

    def generate_content(*, model, contents, config):
        prompts.append(contents)
        return SimpleNamespace(parsed=_ok_items(1), text="")

    _stub_gemini(monkeypatch, generate_content)
    diagnose_batch([_event()])
    offered = prompts[0].split("(", 1)[1].split(")", 1)[0]
    assert "INSUFFICIENT_FUNDS" in offered
    assert "UNKNOWN" not in offered


def test_evidence_names_the_specific_failure_not_a_category(monkeypatch, caplog):
    # A missing package and a rejected request are different operational
    # problems: one is a deploy fault, the other a credential fault. Collapsing
    # both to "api or parse failure" tells the operator nothing actionable.
    monkeypatch.setenv("NAKAD_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "anthropic", None)
    diagnose_mod._cache.clear()
    missing_package, _ = diagnose_batch([_event()])
    package_evidence = " ".join(missing_package[0].evidence)

    monkeypatch.setenv("NAKAD_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv(
        "OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"
    )

    # The body is what distinguishes a bad model from a rate limit from a
    # rejected parameter — the status line alone is always "403 Forbidden".
    response_body = '{"error":{"message":"model_not_found","type":"invalid_request_error"}}'

    def forbidden(*_a, **_kw):
        return _FakeResponse(403, "Forbidden", response_body)

    monkeypatch.setattr(diagnose_mod.requests, "post", forbidden)
    diagnose_mod._cache.clear()
    with caplog.at_level(logging.ERROR, logger=diagnose_mod.log.name):
        http_failure, _ = diagnose_batch([_event()])
    http_evidence = " ".join(http_failure[0].evidence)

    assert missing_package[0].cause is RootCause.UNKNOWN
    assert http_failure[0].cause is RootCause.UNKNOWN
    assert package_evidence != http_evidence
    assert "anthropic" in package_evidence
    assert "403" in http_evidence and "Forbidden" in http_evidence
    assert "model_not_found" in http_evidence
    assert any(
        "url=https://api.openai.com/v1/chat/completions" in record.message
        and "model=gpt-4o-mini" in record.message
        and "model_not_found" in record.message
        for record in caplog.records
    )


class _FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(
        self, status_code: int, reason: str, text: str, headers: dict | None = None
    ) -> None:
        self.status_code = status_code
        self.reason = reason
        self.text = text
        self.headers = headers or {}

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self):
        return json.loads(self.text)


def _capture_openai(monkeypatch, content: str) -> list[dict]:
    """Stub requests.post and return the request bodies that were posted."""
    posted: list[dict] = []

    def post(url, **kwargs):
        posted.append(kwargs["json"])
        return _FakeResponse(
            200,
            "OK",
            json.dumps({"choices": [{"message": {"content": content}}]}),
        )

    monkeypatch.setenv("NAKAD_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(diagnose_mod.requests, "post", post)
    return posted


def test_openai_json_schema_mode_sends_strict_schema(monkeypatch):
    posted = _capture_openai(
        monkeypatch,
        '{"items":[{"cause":"INSUFFICIENT_FUNDS","confidence":0.9,"evidence":["x"]}]}',
    )
    monkeypatch.setenv("OPENAI_STRUCTURED_MODE", "json_schema")
    diagnoses, _ = diagnose_batch([_event()])

    assert diagnoses[0].cause is RootCause.INSUFFICIENT_FUNDS
    fmt = posted[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    enum = fmt["json_schema"]["schema"]["properties"]["items"]["items"]["properties"][
        "cause"
    ]["enum"]
    assert "UNKNOWN" not in enum
    assert "INSUFFICIENT_FUNDS" in enum
    assert "requested schema" in posted[0]["messages"][0]["content"]


def test_openai_json_object_mode_puts_shape_in_the_system_prompt(monkeypatch):
    posted = _capture_openai(
        monkeypatch,
        '{"items":[{"cause":"INSUFFICIENT_FUNDS","confidence":0.9,"evidence":["x"]}]}',
    )
    monkeypatch.setenv("OPENAI_STRUCTURED_MODE", "json_object")
    diagnoses, _ = diagnose_batch([_event()])

    assert diagnoses[0].cause is RootCause.INSUFFICIENT_FUNDS
    assert posted[0]["response_format"] == {"type": "json_object"}
    system = posted[0]["messages"][0]["content"]
    assert "INSUFFICIENT_FUNDS" in system
    assert "UNKNOWN" not in system
    assert '"items"' in system


def test_openai_invalid_cause_falls_back_in_both_structured_modes(monkeypatch):
    # Enum constraint is enforced by Pydantic after the call either way, so a
    # free-text NO_MONEY yields UNKNOWN under both modes rather than inventing
    # a real cause.
    for mode in ("json_schema", "json_object"):
        _capture_openai(
            monkeypatch,
            '{"items":[{"cause":"NO_MONEY","confidence":0.9,"evidence":["x"]}]}',
        )
        monkeypatch.setenv("OPENAI_STRUCTURED_MODE", mode)
        diagnose_mod._cache.clear()
        diagnoses, stats = diagnose_batch([_event()])
        assert diagnoses[0].cause is RootCause.UNKNOWN, mode
        assert diagnoses[0].confidence == 0.0, mode
        assert stats["fallbacks"] == 1, mode


_OK_BODY = json.dumps(
    {
        "choices": [
            {
                "message": {
                    "content": '{"items":[{"cause":"INSUFFICIENT_FUNDS",'
                    '"confidence":0.9,"evidence":["x"]}]}'
                }
            }
        ]
    }
)


def _rate_limited(body: str, headers: dict | None = None) -> _FakeResponse:
    return _FakeResponse(429, "Too Many Requests", body, headers)


def _stub_openai_sequence(monkeypatch, responses: list) -> list[float]:
    """Serve canned responses in order; return the sleeps that were taken."""
    slept: list[float] = []
    queue = list(responses)

    monkeypatch.setenv("NAKAD_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_STRUCTURED_MODE", "json_object")
    monkeypatch.setattr(diagnose_mod.requests, "post", lambda url, **kw: queue.pop(0))
    monkeypatch.setattr(diagnose_mod.time, "sleep", slept.append)
    diagnose_mod._cache.clear()
    return slept


def test_a_429_is_retried_after_the_delay_named_in_the_body(monkeypatch, caplog):
    # The free tier's 429 is a tokens-per-minute ceiling, not a dead request:
    # the identical call succeeds once the named window has passed.
    slept = _stub_openai_sequence(
        monkeypatch,
        [
            _rate_limited('{"error":{"message":"Please try again in 7.66s."}}'),
            _FakeResponse(200, "OK", _OK_BODY),
        ],
    )
    with caplog.at_level(logging.WARNING, logger=diagnose_mod.log.name):
        diagnoses, stats = diagnose_batch([_event()])

    assert diagnoses[0].cause is RootCause.INSUFFICIENT_FUNDS
    assert stats["fallbacks"] == 0
    assert slept == [7.66]
    assert any(
        "rate limited" in r.getMessage() and "waiting=7.66s" in r.getMessage()
        for r in caplog.records
    )


def test_a_retry_after_header_wins_over_the_body(monkeypatch):
    slept = _stub_openai_sequence(
        monkeypatch,
        [
            _rate_limited(
                '{"error":{"message":"try again in 60s"}}', {"retry-after": "3"}
            ),
            _FakeResponse(200, "OK", _OK_BODY),
        ],
    )
    diagnoses, _ = diagnose_batch([_event()])

    assert diagnoses[0].cause is RootCause.INSUFFICIENT_FUNDS
    assert slept == [3.0]


def test_a_wait_beyond_the_ceiling_degrades_instead_of_blocking(monkeypatch, caplog):
    # A daily-quota 429 asks for the rest of the day. Sleeping it out stalls
    # every remaining batch behind one exhausted bucket.
    slept = _stub_openai_sequence(
        monkeypatch,
        [_rate_limited("", {"retry-after": "1435"}), _FakeResponse(200, "OK", _OK_BODY)],
    )
    with caplog.at_level(logging.ERROR, logger=diagnose_mod.log.name):
        diagnoses, stats = diagnose_batch([_event()])

    assert slept == []
    assert diagnoses[0].cause is RootCause.UNKNOWN
    assert stats["fallbacks"] == 1
    assert "1435s" in " ".join(diagnoses[0].evidence)
    assert any("beyond retry ceiling" in r.getMessage() for r in caplog.records)


def test_a_wait_exactly_at_the_ceiling_is_still_taken(monkeypatch):
    slept = _stub_openai_sequence(
        monkeypatch,
        [
            _rate_limited(
                "", {"retry-after": str(int(diagnose_mod.OPENAI_MAX_WAIT_SECONDS))}
            ),
            _FakeResponse(200, "OK", _OK_BODY),
        ],
    )
    diagnoses, _ = diagnose_batch([_event()])

    assert slept == [diagnose_mod.OPENAI_MAX_WAIT_SECONDS]
    assert diagnoses[0].cause is RootCause.INSUFFICIENT_FUNDS


def _items_body(count: int) -> str:
    items = [{"cause": "INSUFFICIENT_FUNDS", "confidence": 0.9, "evidence": ["x"]}] * count
    return json.dumps(
        {"choices": [{"message": {"content": json.dumps({"items": items})}}]}
    )


def test_a_long_response_is_truncated_to_the_requested_count(monkeypatch):
    # Items are positional, so the first n still answer the n inputs in order.
    events = [
        _event(event_id=f"e{i}", payment_id=f"p{i}", error_code=f"ERR_{i}")
        for i in range(3)
    ]
    _stub_openai_sequence(monkeypatch, [_FakeResponse(200, "OK", _items_body(4))])
    diagnoses, stats = diagnose_batch(events)

    assert len(diagnoses) == 3
    assert stats["fallbacks"] == 0
    assert all(d.cause is RootCause.INSUFFICIENT_FUNDS for d in diagnoses)


def test_a_short_response_still_falls_back(monkeypatch):
    # Truncating cannot help here: position 2 has no answer, and borrowing
    # another event's item would put a confident wrong cause on it.
    events = [
        _event(event_id=f"e{i}", payment_id=f"p{i}", error_code=f"ERR_{i}")
        for i in range(3)
    ]
    _stub_openai_sequence(monkeypatch, [_FakeResponse(200, "OK", _items_body(2))])
    diagnoses, stats = diagnose_batch(events)

    assert stats["fallbacks"] == 3
    assert all(d.cause is RootCause.UNKNOWN for d in diagnoses)
    assert "item count 2 != 3" in " ".join(diagnoses[0].evidence)


def test_an_unparseable_rate_limit_body_falls_back_to_the_default_wait(monkeypatch):
    slept = _stub_openai_sequence(
        monkeypatch,
        [_rate_limited("slow down"), _FakeResponse(200, "OK", _OK_BODY)],
    )
    diagnose_batch([_event()])

    assert slept == [diagnose_mod.RATE_LIMIT_DEFAULT_WAIT_SECONDS]


def test_compound_and_millisecond_delays_parse():
    # Parser only: 2m59.56s is above the retry ceiling, so asserting on the
    # sleep would conflate format handling with the degrade-instead-of-block
    # policy.
    for body, expected in (
        ("try again in 7.66s", 7.66),
        ("try again in 2m59.56s", 179.56),
        ("try again in 500ms", 0.5),
    ):
        assert diagnose_mod._rate_limit_wait(_rate_limited(body)) == expected, body


def test_fallback_happens_only_after_retries_are_exhausted(monkeypatch):
    # Three attempts, two waits between them, and the final body still reaches
    # the evidence string so the operator sees why the run degraded.
    slept = _stub_openai_sequence(
        monkeypatch,
        [_rate_limited('{"error":{"message":"try again in 1s"}}')] * 3,
    )
    diagnoses, stats = diagnose_batch([_event()])

    assert len(slept) == diagnose_mod.OPENAI_MAX_ATTEMPTS - 1
    assert diagnoses[0].cause is RootCause.UNKNOWN
    assert stats["fallbacks"] == 1
    assert "429" in " ".join(diagnoses[0].evidence)


def test_sequential_batches_are_spaced_but_the_first_is_not(monkeypatch):
    monkeypatch.setenv("LLM_BATCH_DELAY_SECONDS", "2")
    slept = _stub_openai_sequence(monkeypatch, [])
    posted: list[dict] = []

    def post(url, **kw):
        posted.append(kw["json"])
        return _FakeResponse(200, "OK", _batch_body(kw["json"]))

    monkeypatch.setattr(diagnose_mod.requests, "post", post)
    # 25 distinct cache keys spill into two batches of 20 and 5.
    events = [
        _event(event_id=f"e{i}", payment_id=f"p{i}", error_code=f"ERR_{i}")
        for i in range(25)
    ]
    diagnose_batch(events)

    assert len(posted) == 2
    assert slept == [2.0]


def _batch_body(request_json: dict) -> str:
    """One INSUFFICIENT_FUNDS item per event named in the user prompt."""
    prompt = request_json["messages"][1]["content"]
    count = prompt.count('"index"')
    items = [
        {"cause": "INSUFFICIENT_FUNDS", "confidence": 0.9, "evidence": ["x"]}
    ] * count
    return json.dumps(
        {"choices": [{"message": {"content": json.dumps({"items": items})}}]}
    )


def test_fallback_reasons_are_counted_and_logged(monkeypatch, caplog):
    monkeypatch.setenv("NAKAD_LLM_PROVIDER", "none")
    with caplog.at_level(logging.ERROR, logger=diagnose_mod.log.name):
        _, stats = diagnose_batch(
            [_event(event_id=f"e{i}", payment_id=f"p{i}") for i in range(3)]
        )

    assert stats["fallbacks"] == 3
    assert stats["fallback_reasons"] == {"RuntimeError: NAKAD_LLM_PROVIDER=none": 3}
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "degraded run" in logged
    assert "3 of 3" in logged
    assert "NAKAD_LLM_PROVIDER=none" in logged
