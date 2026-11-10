"""The OpenRouter client (studio_tool_service.llm), with its HTTP session replaced by a fake."""

import json

import pytest
import requests

from studio_tool_service import llm as llm_module
from studio_tool_service.llm import LLM, LLMError, LLMRefused

SCHEMA = {"title": "reply", "type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"],
          "additionalProperties": False}


class FakeResponse:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self.body, self.headers = status, body or {}, headers or {}
        self.text = json.dumps(self.body)

    def json(self):
        return self.body


def answer(data, reason="stop", tokens=(10, 5)):
    return FakeResponse(200, {
        "choices": [{"message": {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)},
                     "finish_reason": reason}],
        "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1]},
    })


class FakeSession:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)


def test_asks_for_json_in_the_given_shape_with_the_key_in_a_header():
    session = FakeSession(answer({"title": "སྐྱབས་འགྲོ།"}))
    client = LLM("secret-key", "google/gemini-2.5-flash", session)

    assert client.json("Write a title.", SCHEMA, system="Answer in Tibetan.") == {"title": "སྐྱབས་འགྲོ།"}

    call = session.calls[0]
    assert call["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer secret-key" and "secret-key" not in call["url"]
    body = call["json"]
    assert body["model"] == "google/gemini-2.5-flash"
    assert body["messages"] == [
        {"role": "system", "content": "Answer in Tibetan."}, {"role": "user", "content": "Write a title."},
    ]
    assert body["response_format"] == {
        "type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": SCHEMA},
    }
    assert body["provider"] == {"require_parameters": True}  # only providers that keep to the schema
    assert (client.tokens_in, client.tokens_out) == (10, 5)


def test_uses_the_ai_course_creators_model_when_none_is_set():
    session = FakeSession(answer({}))
    LLM("key", "", session).json("x", SCHEMA)
    assert session.calls[0]["json"]["model"] == "google/gemini-2.5-flash-lite"


def test_tries_again_while_busy_or_unreachable():
    session = FakeSession(
        FakeResponse(503, {"error": {"message": "overloaded"}}),
        requests.ConnectionError("reset"),
        FakeResponse(429, headers={"Retry-After": "2"}),
        FakeResponse(200, {"error": {"code": 502, "message": "provider returned an error"}}),
        answer({"title": "ok"}),
    )
    assert LLM("key", "m", session).json("x", SCHEMA) == {"title": "ok"}
    assert len(session.calls) == 5


def test_gives_up_after_a_few_tries():
    session = FakeSession(*[FakeResponse(503)] * 5)
    with pytest.raises(LLMError, match="did not answer"):
        LLM("key", "m", session).json("x", SCHEMA)
    assert len(session.calls) == 5


@pytest.mark.parametrize("status, message", [
    (401, "No auth credentials found"),
    (402, "Insufficient credits"),
    (400, "google/gemini-9 is not a valid model ID"),
])
def test_a_refusal_stops_straight_away(status, message):
    session = FakeSession(FakeResponse(status, {"error": {"code": status, "message": message}}))
    with pytest.raises(LLMRefused) as refused:
        LLM("key", "m", session).json("x", SCHEMA)
    assert refused.value.status == status and message in str(refused.value)
    assert len(session.calls) == 1


def test_a_reply_cut_off_failed_or_not_json_is_an_error():
    with pytest.raises(LLMError, match="cut off"):
        LLM("key", "m", FakeSession(answer({"title": "x"}, reason="length"))).json("x", SCHEMA)
    broken = FakeResponse(200, {"choices": [{"message": {"content": "{not json"}, "finish_reason": "stop"}]})
    with pytest.raises(LLMError, match="not valid JSON"):
        LLM("key", "m", FakeSession(broken)).json("x", SCHEMA)
    empty = FakeResponse(200, {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]})
    with pytest.raises(LLMError, match="empty reply"):
        LLM("key", "m", FakeSession(empty)).json("x", SCHEMA)
    failed = FakeResponse(200, {"choices": [{"error": {"message": "upstream failed"}}]})
    with pytest.raises(LLMError, match="upstream failed"):
        LLM("key", "m", FakeSession(failed)).json("x", SCHEMA)


class NotJSON(FakeResponse):
    def __init__(self):
        super().__init__(200)
        self.text = "<html>502 Bad Gateway</html>"

    def json(self):
        raise ValueError("not JSON")


def test_an_answer_that_isnt_json_is_tried_again():
    session = FakeSession(NotJSON(), answer({"title": "ok"}))
    assert LLM("key", "m", session).json("x", SCHEMA) == {"title": "ok"}
    assert len(session.calls) == 2


def test_an_error_given_as_plain_text_is_still_a_refusal():
    session = FakeSession(FakeResponse(401, {"error": "Unauthorized"}))
    with pytest.raises(LLMRefused, match="Unauthorized"):
        LLM("key", "m", session).json("x", SCHEMA)


def test_a_request_that_timed_out_is_sent_again_only_once():
    session = FakeSession(requests.ReadTimeout("slow"), requests.ReadTimeout("slow"), answer({"title": "ok"}))
    with pytest.raises(LLMError, match="took longer"):
        LLM("key", "m", session).json("x", SCHEMA)
    assert len(session.calls) == 2  # each one may be billed: no more
