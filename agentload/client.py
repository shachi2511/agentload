"""Async client for OpenAI-compatible APIs (Coral by default).

Two call styles, both streamed:
- chat():    Chat Completions. The full history is re-sent every turn.
- respond(): Responses API. The server keeps the history (previous_response_id).
Each call measures time to first token, captures usage and cost, retries 429s with
exponential backoff + jitter, and enforces the budget guard (hold before, settle after).
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import openai
from openai import AsyncOpenAI

from agentload.budget import BudgetGuard
from agentload.pricing import worst_case_cost, worst_case_cost_for_chars

CORAL_BASE_URL = "https://inference.coralbricks.ai/v1"
DEFAULT_MODEL = "glm-5.3-flash-fast"


@dataclass
class RequestResult:
    request_id: str
    model: str
    scenario: str
    turn: int | None
    started_at: float                       # unix time the request was first sent
    api: str = "chat"                       # "chat" (history re-sent) or "responses"
    request_chars: int | None = None        # size of what we actually sent (JSON characters)
    response_id: str | None = None          # Responses API id, used as previous_response_id
    ttft_s: float | None = None             # first generated token of any kind (thinking or answer)
    first_content_s: float | None = None    # first visible answer token
    total_s: float | None = None            # successful attempt only; excludes retry waits
    last_token_s: float | None = None       # when the last generated token arrived
    text_chunks: int = 0                    # stream events that carried thinking or answer text
    max_gap_s: float = 0.0                  # longest pause between two text chunks
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None   # physically cached; NOT what you are billed for
    billable_cache_write_tokens: int | None = None
    cache_write_blocks: int | None = None   # 10-minute retention blocks billed (Chat only)
    cost_usd: float | None = None           # usage.cost: the real charge
    cost_details: dict[str, Any] | None = None
    output_text: str = ""
    attempts: int = 1
    retry_wait_s: float = 0.0               # total time spent backing off after 429s
    status_code: int | None = None          # HTTP status of the final error, if any
    error: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)   # scenario labels: variant, gap...

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def decode_tokens_per_s(self) -> float | None:
        """Output tokens / time from first to last token. Unreliable if delivery was bursty."""
        if not self.completion_tokens or self.ttft_s is None or self.last_token_s is None:
            return None
        span = self.last_token_s - self.ttft_s
        return self.completion_tokens / span if span > 0 else None


def _get(obj: Any, name: str) -> Any:
    return getattr(obj, name, None) if obj is not None else None


def apply_usage(result: RequestResult, usage: Any) -> None:
    """Chat Completions usage fields (confirmed in Phase 1)."""
    p = _get(usage, "prompt_tokens_details")
    c = _get(usage, "completion_tokens_details")
    result.prompt_tokens = _get(usage, "prompt_tokens")
    result.completion_tokens = _get(usage, "completion_tokens")
    result.reasoning_tokens = _get(c, "reasoning_tokens")
    result.cached_tokens = _get(p, "cached_tokens")
    result.cache_write_tokens = _get(p, "cache_write_tokens")
    result.billable_cache_write_tokens = _get(p, "billable_cache_write_tokens")
    result.cache_write_blocks = _get(p, "cache_write_blocks")
    _apply_cost(result, usage)


def apply_response_usage(result: RequestResult, usage: Any) -> None:
    """Responses API usage fields (confirmed in Phase 3.8a)."""
    p = _get(usage, "input_tokens_details")
    c = _get(usage, "output_tokens_details")
    result.prompt_tokens = _get(usage, "input_tokens")
    result.completion_tokens = _get(usage, "output_tokens")
    result.reasoning_tokens = _get(c, "reasoning_tokens")
    result.cached_tokens = _get(p, "cached_tokens")
    result.cache_write_tokens = _get(p, "cache_write_tokens")
    result.billable_cache_write_tokens = _get(p, "billable_cache_write_tokens")
    result.cache_write_blocks = _get(p, "cache_write_blocks")
    _apply_cost(result, usage)


def _apply_cost(result: RequestResult, usage: Any) -> None:
    result.cost_usd = _get(usage, "cost")
    details = _get(usage, "cost_details")
    result.cost_details = dict(details) if details is not None else None


def _record_token(result: RequestResult, now: float, last_at: float | None) -> float:
    """Update first-token time, longest gap and chunk count; return the new 'last' time."""
    if result.ttft_s is None:
        result.ttft_s = now
    if last_at is not None:
        result.max_gap_s = max(result.max_gap_s, now - last_at)
    result.text_chunks += 1
    return now


def _retry_after_seconds(error: Exception) -> float | None:
    """Seconds from a Retry-After header, if the server sent one in numeric form."""
    headers = _get(_get(error, "response"), "headers")
    value = headers.get("retry-after") if headers is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None  # HTTP-date form: fall back to our own backoff


class AgentLoadClient:
    def __init__(
        self,
        budget: BudgetGuard,
        base_url: str = CORAL_BASE_URL,
        api_key: str | None = None,
        timeout_s: float = 300.0,
        max_attempts: int = 5,
        base_backoff_s: float = 1.0,
        max_backoff_s: float = 30.0,
    ) -> None:
        key = api_key or os.environ.get("CORAL_API_KEY")
        if not key:
            raise RuntimeError("CORAL_API_KEY is not set")
        # max_retries=0: the SDK would otherwise retry 429s silently and hide them from us.
        self._openai = AsyncOpenAI(base_url=base_url, api_key=key, timeout=timeout_s,
                                   max_retries=0)
        self.budget = budget
        self.max_attempts = max_attempts
        self.base_backoff_s = base_backoff_s
        self.max_backoff_s = max_backoff_s
        # Seams so tests can run without network, real sleeps or randomness.
        self._create = self._openai.chat.completions.create
        self._create_response = self._openai.responses.create
        self._sleep = asyncio.sleep
        self._rand = random.random

    def _backoff(self, attempt: int, error: Exception) -> float:
        """Honor Retry-After; otherwise 'full jitter': random wait in [0, base * 2^(n-1)], capped."""
        retry_after = _retry_after_seconds(error)
        if retry_after is not None:
            return min(retry_after, self.max_backoff_s)
        ceiling = min(self.max_backoff_s, self.base_backoff_s * 2 ** (attempt - 1))
        return self._rand() * ceiling

    @staticmethod
    def _new_result(model: str, scenario: str, turn: int | None, api: str, request_chars: int,
                    extra_body: dict[str, Any] | None,
                    tags: dict[str, Any] | None) -> RequestResult:
        return RequestResult(
            request_id=uuid.uuid4().hex, model=model, scenario=scenario, turn=turn,
            started_at=time.time(), api=api, request_chars=request_chars,
            extra_body=dict(extra_body or {}), tags=dict(tags or {}),
        )

    async def _with_retries(self, result: RequestResult, hold: float,
                            attempt: Callable[[], Awaitable[None]]) -> None:
        try:
            for n in range(1, self.max_attempts + 1):
                result.attempts = n
                try:
                    await attempt()
                    result.error, result.status_code = None, None
                    break
                except openai.RateLimitError as e:
                    result.status_code = e.status_code
                    result.error = f"RateLimitError: {e}"
                    if n == self.max_attempts:
                        break
                    wait = self._backoff(n, e)
                    result.retry_wait_s += wait
                    await self._sleep(wait)
        except (openai.APIError, RuntimeError) as e:  # API failures are recorded; our bugs crash
            result.status_code = getattr(e, "status_code", None)
            result.error = f"{type(e).__name__}: {e}"
        finally:
            # If we never learned the real cost, charge the full hold so the budget errs safe.
            charged = result.cost_usd if result.cost_usd is not None else hold
            await self.budget.settle(hold, charged)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1000,
        scenario: str = "adhoc",
        turn: int | None = None,
        extra_body: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> RequestResult:
        result = self._new_result(model, scenario, turn, "chat", len(json.dumps(messages)),
                                  extra_body, tags)
        hold = await self.budget.reserve(worst_case_cost(model, messages, max_tokens))
        await self._with_retries(result, hold, lambda: self._stream_chat(
            result, messages, model, max_tokens, extra_body))
        return result

    async def respond(
        self,
        input_text: str,
        *,
        history_chars: int,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1000,
        scenario: str = "adhoc",
        turn: int | None = None,
        extra_body: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> RequestResult:
        """Responses API turn. history_chars = size of the WHOLE conversation the server will
        process (it bills all of it), so the budget hold covers more than what we send."""
        sent = {"instructions": instructions, "input": input_text,
                "previous_response_id": previous_response_id}
        result = self._new_result(model, scenario, turn, "responses", len(json.dumps(sent)),
                                  extra_body, tags)
        billable_chars = max(history_chars, len(input_text) + len(instructions or ""))
        hold = await self.budget.reserve(
            worst_case_cost_for_chars(model, billable_chars, max_tokens))
        await self._with_retries(result, hold, lambda: self._stream_response(
            result, input_text, instructions, previous_response_id, model, max_tokens,
            extra_body))
        return result

    async def _stream_chat(
        self,
        result: RequestResult,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        extra_body: dict[str, Any] | None,
    ) -> None:
        t0 = time.perf_counter()
        stream = await self._create(
            model=model, messages=messages, max_tokens=max_tokens, stream=True,
            stream_options={"include_usage": True}, extra_body=extra_body or None,
        )
        parts: list[str] = []
        last_at: float | None = None
        async for chunk in stream:
            now = time.perf_counter() - t0
            if chunk.choices:
                delta = chunk.choices[0].delta
                content = delta.content
                if content or getattr(delta, "reasoning_content", None):
                    last_at = _record_token(result, now, last_at)
                if content:
                    if result.first_content_s is None:
                        result.first_content_s = now
                    parts.append(content)
            if chunk.usage is not None:
                apply_usage(result, chunk.usage)
        result.total_s = time.perf_counter() - t0
        result.last_token_s = last_at
        result.output_text = "".join(parts)
        if result.cost_usd is None:
            raise RuntimeError("stream ended without usage/cost")

    async def _stream_response(
        self,
        result: RequestResult,
        input_text: str,
        instructions: str | None,
        previous_response_id: str | None,
        model: str,
        max_tokens: int,
        extra_body: dict[str, Any] | None,
    ) -> None:
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {"model": model, "input": input_text,
                                  "max_output_tokens": max_tokens, "stream": True,
                                  "extra_body": extra_body or None}
        if instructions is not None:
            kwargs["instructions"] = instructions
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        stream = await self._create_response(**kwargs)
        parts: list[str] = []
        last_at: float | None = None
        final = None
        async for event in stream:
            now = time.perf_counter() - t0
            etype = getattr(event, "type", "") or ""
            if etype.endswith(".delta"):
                last_at = _record_token(result, now, last_at)
                if etype == "response.output_text.delta":
                    if result.first_content_s is None:
                        result.first_content_s = now
                    parts.append(getattr(event, "delta", "") or "")
            elif etype in ("response.completed", "response.incomplete"):
                final = getattr(event, "response", None)
        result.total_s = time.perf_counter() - t0
        result.last_token_s = last_at
        result.output_text = "".join(parts)
        if final is None:
            raise RuntimeError("stream ended without a final response")
        result.response_id = getattr(final, "id", None)
        if not result.output_text:
            result.output_text = getattr(final, "output_text", "") or ""
        apply_response_usage(result, getattr(final, "usage", None))
        if result.cost_usd is None:
            raise RuntimeError("stream ended without usage/cost")
