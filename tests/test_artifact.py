from __future__ import annotations
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import write_artifact
from models import RootCause
from pipeline import diagnose as diagnose_mod
from pipeline.diagnose import run_config

def _config(**overrides) -> dict:
    base = {
        "provider": "openai",
        "model": "openai/gpt-oss-120b",
        "events": 500,
        "by_method": {"llm": 124, "rule": 313},
        "unknown": 0,
        "degraded": False,
    }
    return {**base, **overrides}


def _artifact(**overrides) -> dict:
    base = {
        "seed": 42,
        "events": 500,
        "run_config": _config(),
        "multi_seed": {"ratio": {"mean": 2.35}, "run_config": _config(provider="none")},
    }
    return {**base, **overrides}


def test_a_populated_artifact_is_written(tmp_path):
    path = tmp_path / "console.json"
    write_artifact(_artifact(), path)

    written = json.loads(path.read_text())
    assert written["run_config"]["provider"] == "openai"
    assert written["multi_seed"]["run_config"]["provider"] == "none"


@pytest.mark.parametrize(
    "artifact, expected",
    [
        (_artifact(run_config=None), "run_config"),
        (_artifact(run_config={}), "run_config"),
        (_artifact(multi_seed={"ratio": {}}), "multi_seed.run_config"),
        (_artifact(run_config=_config(provider="")), "provider"),
        (_artifact(run_config=_config(model=None)), "model"),
        (_artifact(run_config=_config(by_method=None)), "by_method"),
        (_artifact(run_config=_config(unknown=None)), "unknown"),
    ],
)
def test_an_artifact_without_populated_provenance_is_refused(
    tmp_path, artifact, expected
):
    path = tmp_path / "console.json"
    with pytest.raises(ValueError, match=expected):
        write_artifact(artifact, path)
    assert not path.exists()


def test_the_sweep_provenance_is_checked_too(tmp_path):
    # The failure mode being guarded is a headline run with the LLM live and a
    # sweep quietly run with it off, so the sweep's block is not optional.
    path = tmp_path / "console.json"
    artifact = _artifact(
        multi_seed={"ratio": {}, "run_config": _config(model="")},
    )
    with pytest.raises(ValueError, match="multi_seed.run_config"):
        write_artifact(artifact, path)
    assert not path.exists()


def test_run_config_reports_the_provider_and_method_mix(monkeypatch):
    monkeypatch.setenv("NAKAD_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "openai/gpt-oss-120b")
    diagnoses = [
        diagnose_mod.Diagnosis(
            event_id="e0", cause=RootCause.INSUFFICIENT_FUNDS, confidence=0.9,
            evidence=["x"], method="rule",
        ),
        diagnose_mod.Diagnosis(
            event_id="e1", cause=RootCause.UNKNOWN, confidence=0.0,
            evidence=["x"], method="llm",
        ),
    ]
    config = run_config(diagnoses)

    # The UNKNOWN event is counted as undiagnosed, not as "llm": the LLM layer
    # failed it rather than diagnosing it, and by_method is a record of work
    # done. The buckets still sum to events.
    assert config == {
        "provider": "openai",
        "model": "openai/gpt-oss-120b",
        "events": 2,
        "by_method": {"rule": 1, "undiagnosed": 1},
        "unknown": 1,
        "degraded": True,
    }
    assert sum(config["by_method"].values()) == config["events"]


def test_run_config_reports_the_model_default_when_unset(monkeypatch):
    # The reported model must be the one the branch would actually call, or the
    # artifact records a model that was never used.
    monkeypatch.setenv("NAKAD_LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    assert run_config([])["model"] == diagnose_mod._MODEL_ENV["openai"][1]
