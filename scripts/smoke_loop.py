"""Phase 1, step 5: a 20-turn agent loop on GLM 5.3 Flash.
Each turn appends ~1.5K tokens of synthetic tool output; logs timing, cache and cost per turn."""
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

MODEL = "glm-5.3-flash-fast"
TURNS = 20
LINES_PER_TURN = 60      # ~25 tokens per line measured in step 4 -> ~1.5K tokens/turn
BUDGET_USD = 0.50        # simple safety stop for this smoke test
OUT = Path("results/smoke_loop.jsonl")

key = os.environ.get("CORAL_API_KEY")
if not key:
    sys.exit("CORAL_API_KEY is not set.")
client = OpenAI(base_url="https://inference.coralbricks.ai/v1", api_key=key)

messages: list[dict] = [
    {"role": "system", "content": "You are a coding agent. After each tool output, reply with one short sentence: the next action."}
]
spent = 0.0
OUT.parent.mkdir(exist_ok=True)
out = OUT.open("w")

print(f"{'turn':>4} {'prompt':>7} {'cached':>7} {'write':>6} {'hit%':>6} {'out':>5} "
      f"{'1st data':>8} {'1st text':>8} {'total':>6} {'cost $':>10}")
for turn in range(1, TURNS + 1):
    tool_output = "\n".join(
        f"[turn {turn:02d} log {i:03d}] test_{(turn * 7 + i) % 53} passed in {(turn + i) % 11}.{i % 7}s"
        for i in range(LINES_PER_TURN)
    )
    messages.append({"role": "user", "content": f"Tool output from step {turn}:\n{tool_output}"})

    start = time.perf_counter()
    first_data = first_text = None
    answer: list[str] = []
    usage = None
    stream = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=1000,
        stream=True, stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.choices:
            d = chunk.choices[0].delta
            now = time.perf_counter() - start
            if first_data is None and (d.content or getattr(d, "reasoning_content", None)):
                first_data = now
            if d.content:
                if first_text is None:
                    first_text = now
                answer.append(d.content)
        if chunk.usage is not None:
            usage = chunk.usage
    total = time.perf_counter() - start

    reply = "".join(answer).strip()
    messages.append({"role": "assistant", "content": reply})
    p = usage.prompt_tokens_details
    hit = p.cached_tokens / usage.prompt_tokens if usage.prompt_tokens else 0.0
    spent += usage.cost
    row = {
        "turn": turn, "prompt_tokens": usage.prompt_tokens, "cached_tokens": p.cached_tokens,
        "cache_write_tokens": p.cache_write_tokens, "output_tokens": usage.completion_tokens,
        "reasoning_tokens": usage.completion_tokens_details.reasoning_tokens,
        "hit_rate": round(hit, 4), "first_data_s": first_data, "first_text_s": first_text,
        "total_s": round(total, 3), "cost": usage.cost, "cost_details": usage.cost_details,
        "reply": reply,
    }
    out.write(json.dumps(row) + "\n")
    fd = f"{first_data:.2f}" if first_data is not None else "-"
    ft = f"{first_text:.2f}" if first_text is not None else "-"
    print(f"{turn:>4} {usage.prompt_tokens:>7} {p.cached_tokens:>7} {p.cache_write_tokens:>6} "
          f"{hit * 100:>5.1f}% {usage.completion_tokens:>5} {fd:>8} {ft:>8} {total:>6.2f} {usage.cost:>10.6f}")
    if spent > BUDGET_USD:
        print(f"STOP: spent ${spent:.4f} > budget ${BUDGET_USD}")
        break

out.close()
print(f"\nTOTAL SPENT: ${spent:.6f} | rows saved to {OUT}")
