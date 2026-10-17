"""
How each structured LLM call is made: token cap, reasoning effort, and which
errors are retried. The Groq client is faked; nothing leaves the machine.
"""

import os
import sys
from types import SimpleNamespace

import pytest

os.environ["API_KEY"] = "test-key-for-pytest"
os.environ["ADMIN_API_KEY"] = "admin-key-for-pytest"
os.environ["GROQ_API_KEY"] = "not-used-in-tests"
os.environ["GROQ_MODEL"] = "openai/gpt-oss-120b"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

JSON_FAILED = RuntimeError(
    "Error code: 400 - {'error': {'message': \"Failed to validate JSON. Please adjust "
    "your prompt. See 'failed_generation' for more details.\", 'code': "
    "'json_validate_failed', 'failed_generation': ''}}")


class FakeClient:
    """Plays back a list of outcomes: an exception to raise, or text to return."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))],
                               usage=usage)


@pytest.fixture
def llm(monkeypatch):
    for module in list(sys.modules):
        if module.split(".")[0] in {"services", "utils"}:
            del sys.modules[module]
    import services.llm_service as llm
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    return llm


def test_default_cap_is_4096_with_no_reasoning_override(llm, monkeypatch):
    fake = FakeClient(['{"ok": true}'])
    monkeypatch.setattr(llm, "client", fake)
    llm._call_with_retry("prompt")
    assert fake.calls[0]["max_tokens"] == 4096
    assert "reasoning_effort" not in fake.calls[0]


def test_reasoning_effort_is_sent_when_set(llm, monkeypatch):
    fake = FakeClient(['{"ok": true}'])
    monkeypatch.setattr(llm, "client", fake)
    monkeypatch.setattr(llm, "LLM_REASONING_EFFORT", "low")
    llm._call_with_retry("prompt")
    assert fake.calls[0]["reasoning_effort"] == "low"


def test_invalid_json_is_retried_once(llm, monkeypatch):
    fake = FakeClient([JSON_FAILED, '{"ok": true}'])
    monkeypatch.setattr(llm, "client", fake)
    assert llm._call_with_retry("prompt") == '{"ok": true}'
    assert len(fake.calls) == 2


def test_invalid_json_twice_is_a_502(llm, monkeypatch):
    from fastapi import HTTPException
    fake = FakeClient([JSON_FAILED, JSON_FAILED, '{"never": "reached"}'])
    monkeypatch.setattr(llm, "client", fake)
    with pytest.raises(HTTPException) as exc:
        llm._call_with_retry("prompt", trace_id="t-1")
    assert exc.value.status_code == 502
    assert len(fake.calls) == 2


def test_rate_limit_is_still_retried_after_a_json_retry(llm, monkeypatch):
    fake = FakeClient([JSON_FAILED, RuntimeError("Error code: 429 - rate_limit_exceeded"),
                       '{"ok": true}'])
    monkeypatch.setattr(llm, "client", fake)
    assert llm._call_with_retry("prompt") == '{"ok": true}'
    assert len(fake.calls) == 3


def test_an_error_that_merely_contains_rate_is_not_a_rate_limit(llm, monkeypatch):
    """'generate' contains 'rate'; that must not trigger rate-limit retries."""
    from fastapi import HTTPException
    fake = FakeClient([RuntimeError("Error code: 400 - could not generate a response")])
    monkeypatch.setattr(llm, "client", fake)
    with pytest.raises(HTTPException) as exc:
        llm._call_with_retry("prompt")
    assert exc.value.status_code == 502
    assert len(fake.calls) == 1


@pytest.mark.parametrize("raw,expected", [(None, 4096), ("8192", 8192), ("10", 256),
                                          ("999999", 32768), ("lots", 4096)])
def test_max_tokens_setting(monkeypatch, raw, expected):
    from utils.config import _as_int
    assert _as_int(raw, 4096, 256, 32768) == expected


@pytest.mark.parametrize("raw,expected", [("", None), ("LOW", "low"), ("high", "high"),
                                          ("extreme", None)])
def test_reasoning_effort_setting(monkeypatch, raw, expected):
    monkeypatch.setenv("LLM_REASONING_EFFORT", raw)
    for module in list(sys.modules):
        if module.startswith("utils.config"):
            del sys.modules[module]
    import utils.config as config
    assert config.LLM_REASONING_EFFORT == expected
