"""
The AI model for the writing steps (course_from_recording phase 2: the course outline and lessons), reached
through OpenRouter (https://openrouter.ai), which serves Google's Gemini models (and others) behind one
OpenAI-style API. The model is a setting (OPENROUTER_MODEL), e.g. "google/gemini-2.5-flash-lite".

Called with requests, like Monlam, so the image needs no extra library. Every reply is JSON in a fixed shape:
a strict JSON Schema makes the model answer in that shape (OpenRouter only uses providers that support it),
and the caller still checks what it gets. The key goes in a header, never in the URL, so it can't end up in
a log.
"""

import json
import logging
import threading
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"  # the AI course creator's model, if the setting is empty
APP_TITLE = "WeBuddhist Studio tools"  # shown in the OpenRouter account's activity list

# A long teaching's outline is one big request: give the model time to answer.
TIMEOUT_SECONDS = 300
# Answers worth trying again: busy (429) or briefly down (5xx). Anything else won't change by itself.
RETRY_STATUSES = {429, 500, 502, 503, 504}
RETRY_DELAYS = (5, 15, 30, 60)  # seconds between tries, unless OpenRouter says how long to wait
# A request that timed out may still be running (and billed) at the provider, and a long one is the whole
# transcript: send it again only this many times.
TIMEOUT_RETRIES = 1


class LLMError(Exception):
    """The model gave no usable reply (it kept failing, cut its reply off, or didn't answer in JSON)."""


class LLMRefused(LLMError):
    """OpenRouter refused the request itself: a wrong key (401), no credit left (402), an unknown model..."""

    def __init__(self, status: int, message: str):
        super().__init__(f"OpenRouter refused the request (HTTP {status}): {message}")
        self.status = status


class LLM:
    def __init__(self, api_key: str, model: str = "", session: Optional[requests.Session] = None):
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL
        self.session = session or requests.Session()
        # Tokens used by this client, for the log (cost tracking). Lessons are asked for a few at a time, from
        # threads, so the counts are added up under a lock.
        self.tokens_in = self.tokens_out = 0
        self._count = threading.Lock()

    def json(self, prompt: str, schema: dict, system: str = "", temperature: float = 0.4) -> dict:
        """Ask once; the reply is the JSON object ``schema`` (a JSON Schema) describes."""
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema.get("title", "reply"), "strict": True, "schema": schema},
            },
            # Only providers that can keep to the schema.
            "provider": {"require_parameters": True},
        }
        reply = self._post(body)

        usage = reply.get("usage") or {}
        with self._count:
            self.tokens_in += usage.get("prompt_tokens") or 0
            self.tokens_out += usage.get("completion_tokens") or 0
        choice = (reply.get("choices") or [{}])[0]
        if choice.get("error"):
            raise LLMError(f"The model failed: {choice['error'].get('message', choice['error'])}")
        reason = choice.get("finish_reason")
        text = (choice.get("message") or {}).get("content") or ""
        if reason == "length":
            raise LLMError("The model's reply was cut off: it was too long")
        if not text:
            raise LLMError(f"The model gave an empty reply ({reason or 'no reason'})")
        try:
            return json.loads(text)
        except ValueError as error:
            raise LLMError("The model's reply was not valid JSON") from error

    def _post(self, body: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}", "X-Title": APP_TITLE}
        problem, timeouts = "", 0
        for attempt in range(len(RETRY_DELAYS) + 1):
            wait = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
            try:
                resp = self.session.post(API_URL, json=body, headers=headers, timeout=TIMEOUT_SECONDS)
            except requests.ReadTimeout as error:
                # Sent, but no answer in time (unlike a connection that never opened, which costs nothing).
                timeouts += 1
                if timeouts > TIMEOUT_RETRIES:
                    raise LLMError(f"OpenRouter took longer than {TIMEOUT_SECONDS} s, {timeouts} times") from error
                problem = str(error)
            except (requests.ConnectionError, requests.Timeout) as error:
                problem = str(error)
            else:
                if resp.status_code == 200:
                    reply = _json(resp)
                    if not isinstance(reply, dict):
                        # E.g. an error page from a proxy in front of OpenRouter: worth another try.
                        problem = f"an answer that isn't JSON: {resp.text[:100]!r}"
                    else:
                        # OpenRouter can report a provider's failure inside a 200 answer.
                        status, message = _error_in(reply)
                        if not status:
                            return reply
                        if status not in RETRY_STATUSES:
                            raise LLMRefused(status, message)
                        problem = f"HTTP {status}: {message}"
                elif resp.status_code not in RETRY_STATUSES:
                    raise LLMRefused(resp.status_code, _error_message(resp))
                else:
                    problem = f"HTTP {resp.status_code}: {_error_message(resp)}"
                    wait = _retry_after(resp, wait)
            if attempt < len(RETRY_DELAYS):
                log.warning("OpenRouter did not answer (%s); trying again in %s s", problem, wait)
                time.sleep(wait)
        raise LLMError(f"OpenRouter did not answer after {len(RETRY_DELAYS) + 1} tries: {problem}")


def _json(resp):
    """The answer's JSON, or None if it isn't JSON."""
    try:
        return resp.json()
    except ValueError:
        return None


def _error_in(reply: dict):
    """
    (status, message) of an error OpenRouter reports in an answer's body, or (0, "") without one. The error is
    usually {"code": 429, "message": ...}, but may be just a text; a code that isn't a number counts as 502.
    """
    error = reply.get("error")
    if not error:
        return 0, ""
    if not isinstance(error, dict):
        return 502, str(error)[:300]
    code = str(error.get("code", ""))
    return (int(code) if code.isdigit() else 502), str(error.get("message", ""))[:300]


def _error_message(resp) -> str:
    """The message of an error answer: OpenRouter's own, or the start of the text (e.g. an error page)."""
    reply = _json(resp)
    if isinstance(reply, dict):
        message = _error_in(reply)[1]
        if message:
            return message
    return resp.text[:300]


def _retry_after(resp, default: float) -> float:
    try:
        return min(120.0, float(resp.headers.get("Retry-After", default)))
    except (TypeError, ValueError):
        return default
