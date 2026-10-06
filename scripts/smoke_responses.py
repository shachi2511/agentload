"""Phase 3.8a: smoke-test Coral's Responses API (~$0.002) and confirm its real fields.
Run: PYTHONPATH=. python scripts/smoke_responses.py"""
import asyncio
import json
import os
import sys
import time

import openai
from openai import OpenAI

from agentload.budget import BudgetGuard
from agentload.client import CORAL_BASE_URL
from agentload.workload import ToolOutputGenerator, new_run_nonce

MODEL = "glm-5.3-flash-fast"
EXTRA = {"prompt_cache_options": {"ttl": "10m"}}


def show(label: str, r, elapsed: float, sent_chars: int) -> float:
    usage = r.usage.model_dump() if getattr(r, "usage", None) else None
    print(f"\n--- {label}: {elapsed:.2f}s | input chars sent this request: {sent_chars}")
    print("id:", getattr(r, "id", None), "| status:", getattr(r, "status", None))
    print("output_text:", (getattr(r, "output_text", "") or "")[:120])
    print("RAW USAGE:", json.dumps(usage, indent=2, default=str))
    return float((usage or {}).get("cost") or 0.0)


async def main() -> None:
    key = os.environ.get("CORAL_API_KEY")
    if not key:
        sys.exit("CORAL_API_KEY is not set.")
    guard = BudgetGuard(run_cap_usd=0.05)
    hold = await guard.reserve(0.02)
    client = OpenAI(base_url=CORAL_BASE_URL, api_key=key, max_retries=0)
    gen = ToolOutputGenerator()
    instructions = f"Run {new_run_nonce()}. You are a coding agent. Reply with one short sentence."
    spent = 0.0
    try:
        first = gen.make(5000, step=1)
        t = time.perf_counter()
        r1 = client.responses.create(model=MODEL, instructions=instructions, input=first,
                                     max_output_tokens=500, extra_body=EXTRA)
        spent += show("turn 1: new conversation", r1, time.perf_counter() - t, len(first))

        second = gen.make(2000, step=2)
        t = time.perf_counter()
        r2 = client.responses.create(model=MODEL, previous_response_id=r1.id, input=second,
                                     max_output_tokens=500, extra_body=EXTRA)
        spent += show("turn 2: previous_response_id", r2, time.perf_counter() - t, len(second))

        third = gen.make(1000, step=3)
        t = time.perf_counter()
        first_delta, event_types, final = None, [], None
        stream = client.responses.create(model=MODEL, previous_response_id=r2.id, input=third,
                                         max_output_tokens=500, stream=True, extra_body=EXTRA)
        for event in stream:
            if event.type not in event_types:
                event_types.append(event.type)
            if first_delta is None and event.type.endswith(".delta"):
                first_delta = time.perf_counter() - t
            if event.type == "response.completed":
                final = event.response
        print(f"\n--- turn 3: streaming | event types seen: {event_types}")
        print(f"first .delta event at: {first_delta}")
        if final is not None:
            spent += show("turn 3: final response", final, time.perf_counter() - t, len(third))
        else:
            print("NO response.completed EVENT RECEIVED")
    except openai.APIStatusError as e:
        print(f"\nREJECTED ({e.status_code}): {e.message}")
    finally:
        await guard.settle(hold, spent if spent else hold)
        print(f"\nbudget: this run ${guard.run_spent:.6f} | all-time ${guard.total_spent:.6f}")


asyncio.run(main())
