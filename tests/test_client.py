"""Client tests against a fake streaming endpoint. No network, no spend, no real sleeping."""
import asyncio
from types import SimpleNamespace as NS

import openai
import pytest

from agentload.budget import BudgetExceeded, BudgetGuard
from agentload.client import AgentLoadClient
from agentload.pricing import worst_case_cost

MESSAGES = [{"role": "user", "content": "hi"}]

USAGE = NS(
    prompt_tokens=100, completion_tokens=20,
    prompt_tokens_details=NS(cached_tokens=64, cache_write_tokens=36,
                             billable_cache_write_tokens=36, cache_write_blocks=3),
    completion_tokens_details=NS(reasoning_tokens=15),
    cost=0.0000183,
    cost_details={"currency": "USD", "input": 0.0, "cached_input": 0.0,
                  "cache_write": 8.28e-06, "output": 1e-05},
)
OK_STREAM = [NS(choices=[NS(delta=NS(content="ok", reasoning_content=None))], usage=None),
             NS(choices=[], usage=USAGE)]


def chunk(content=None, reasoning=None, usage=None):
    has_delta = content is not None or reasoning is not None
    choices = [NS(delta=NS(content=content, reasoning_content=reasoning))] if has_delta else []
    return NS(choices=choices, usage=usage)


def make_429(retry_after=None):
    """A RateLimitError built without a real HTTP response (avoids network-library internals)."""
    err = openai.RateLimitError.__new__(openai.RateLimitError)
    Exception.__init__(err, "rate limited")
    err.status_code = 429
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    err.response = NS(headers=headers)
    return err


def make_client(tmp_path, chunks=(), errors=(), run_cap=5.0, max_attempts=5):
    budget = BudgetGuard(run_cap_usd=run_cap, ledger_path=tmp_path / "ledger.json")
    client = AgentLoadClient(budget, api_key="test-key", max_attempts=max_attempts)
    client.calls, client.sleeps = [], []
    pending = list(errors)

    async def fake_create(**kwargs):
        client.calls.append(kwargs)
        if pending:
            raise pending.pop(0)

        async def gen():
            for c in chunks:
                yield c

        return gen()

    async def fake_sleep(seconds):
        client.sleeps.append(seconds)

    client._create, client._sleep, client._rand = fake_create, fake_sleep, lambda: 0.5
    return client, budget


def test_records_timing_usage_and_cost(tmp_path):
    chunks = [chunk(reasoning="thinking"), chunk(content="Hel"), chunk(content="lo"),
              chunk(usage=USAGE)]
    client, budget = make_client(tmp_path, chunks)
    r = asyncio.run(client.chat(MESSAGES, scenario="t", turn=1))
    assert r.error is None and r.attempts == 1
    assert r.output_text == "Hello"
    assert r.ttft_s is not None and r.first_content_s is not None
    assert r.ttft_s <= r.first_content_s
    assert r.cached_tokens == 64 and r.cache_write_blocks == 3 and r.reasoning_tokens == 15
    assert r.cost_usd == pytest.approx(0.0000183)
    assert budget.run_spent == pytest.approx(0.0000183)
    sent = client.calls[0]
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}


def test_missing_usage_charges_worst_case(tmp_path):
    client, budget = make_client(tmp_path, [chunk(content="hi")])
    r = asyncio.run(client.chat(MESSAGES, max_tokens=100))
    assert "without usage" in r.error
    assert budget.run_spent == pytest.approx(worst_case_cost("glm-5.3-flash-fast", MESSAGES, 100))


def test_non_429_error_recorded_and_not_retried(tmp_path):
    client, budget = make_client(tmp_path, errors=[RuntimeError("boom")])
    r = asyncio.run(client.chat(MESSAGES))
    assert "boom" in r.error
    assert r.attempts == 1 and len(client.calls) == 1 and client.sleeps == []
    assert budget.run_spent > 0


def test_budget_refuses_before_sending(tmp_path):
    client, _ = make_client(tmp_path, [chunk(usage=USAGE)], run_cap=1e-9)
    with pytest.raises(BudgetExceeded):
        asyncio.run(client.chat(MESSAGES))
    assert client.calls == []


def test_unknown_model_refused(tmp_path):
    client, _ = make_client(tmp_path, [chunk(usage=USAGE)])
    with pytest.raises(ValueError, match="no price"):
        asyncio.run(client.chat(MESSAGES, model="mystery-model"))
    assert client.calls == []


def test_extra_body_passed_through(tmp_path):
    client, _ = make_client(tmp_path, OK_STREAM)
    ttl = {"prompt_cache_options": {"ttl": "10m"}}
    r = asyncio.run(client.chat(MESSAGES, extra_body=ttl))
    assert client.calls[0]["extra_body"] == ttl and r.extra_body == ttl


def test_retries_429_with_growing_backoff_then_succeeds(tmp_path):
    client, _ = make_client(tmp_path, OK_STREAM, errors=[make_429(), make_429()])
    r = asyncio.run(client.chat(MESSAGES))
    assert r.error is None and r.status_code is None
    assert r.attempts == 3 and len(client.calls) == 3
    assert client.sleeps == [0.5, 1.0]          # 0.5 x (1s, 2s): ceiling doubles each retry
    assert r.retry_wait_s == pytest.approx(1.5)


def test_honors_retry_after_header(tmp_path):
    client, _ = make_client(tmp_path, OK_STREAM, errors=[make_429(retry_after="2")])
    r = asyncio.run(client.chat(MESSAGES))
    assert r.error is None and client.sleeps == [2.0]


def test_gives_up_after_max_attempts(tmp_path):
    client, budget = make_client(tmp_path, OK_STREAM, errors=[make_429()] * 3, max_attempts=3)
    r = asyncio.run(client.chat(MESSAGES))
    assert r.attempts == 3 and r.status_code == 429
    assert r.error.startswith("RateLimitError")
    assert len(client.sleeps) == 2              # no pointless sleep after the final attempt
    assert budget.run_spent > 0                 # cost unknown, so the hold is kept


def test_backoff_is_capped(tmp_path):
    client, _ = make_client(tmp_path)
    client._rand = lambda: 1.0
    assert client._backoff(10, make_429()) == client.max_backoff_s


def test_records_chunk_timing_for_burst_detection(tmp_path):
    chunks = [chunk(reasoning="a"), chunk(reasoning="b"), chunk(content="c"),
              chunk(usage=USAGE)]
    client, _ = make_client(tmp_path, chunks)
    r = asyncio.run(client.chat(MESSAGES))
    assert r.text_chunks == 3
    assert r.last_token_s is not None and r.last_token_s >= r.ttft_s
    assert r.max_gap_s >= 0.0


RUSAGE = NS(
    input_tokens=7042, output_tokens=113,
    input_tokens_details=NS(cached_tokens=7040, cache_write_tokens=2,
                            billable_cache_write_tokens=2),
    output_tokens_details=NS(reasoning_tokens=59), cost=0.001,
    cost_details={"currency": "USD", "input": 0.0, "cached_input": 0.0,
                  "cache_write": 5e-07, "output": 5.65e-05},
)


def make_response_client(tmp_path, events, run_cap=5.0):
    client, budget = make_client(tmp_path, run_cap=run_cap)
    client.response_calls = []

    async def fake_create_response(**kwargs):
        client.response_calls.append(kwargs)

        async def gen():
            for e in events:
                yield e

        return gen()

    client._create_response = fake_create_response
    return client, budget


def test_respond_streams_text_and_maps_usage(tmp_path):
    events = [NS(type="response.created"),
              NS(type="response.output_text.delta", delta="Hel"),
              NS(type="response.output_text.delta", delta="lo"),
              NS(type="response.completed",
                 response=NS(id="resp_1", output_text="Hello", usage=RUSAGE))]
    client, budget = make_response_client(tmp_path, events)
    r = asyncio.run(client.respond("new tool output", history_chars=20000,
                                   instructions="be brief", previous_response_id="resp_0"))
    assert r.error is None and r.api == "responses" and r.response_id == "resp_1"
    assert r.output_text == "Hello" and r.first_content_s is not None
    assert r.prompt_tokens == 7042 and r.cached_tokens == 7040 and r.reasoning_tokens == 59
    assert budget.run_spent == pytest.approx(0.001)
    sent = client.response_calls[0]
    assert sent["previous_response_id"] == "resp_0" and sent["instructions"] == "be brief"
    assert sent["stream"] is True and r.request_chars < 200


def test_respond_without_final_event_is_an_error(tmp_path):
    client, budget = make_response_client(tmp_path, [NS(type="response.created")])
    r = asyncio.run(client.respond("x", history_chars=10))
    assert "final response" in r.error and budget.run_spent > 0


def test_respond_hold_covers_server_side_history(tmp_path):
    client, _ = make_response_client(tmp_path, [], run_cap=1.0)
    with pytest.raises(BudgetExceeded):
        asyncio.run(client.respond("tiny", history_chars=10_000_000))
    assert client.response_calls == []
