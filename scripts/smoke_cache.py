"""Phase 1, step 4: send the same ~5K-token prompt twice.
Checks (a) whether chunks trickle in or arrive in one burst, (b) whether the 2nd call hits the cache."""
import os
import sys
import time

from openai import OpenAI

key = os.environ.get("CORAL_API_KEY")
if not key:
    sys.exit("CORAL_API_KEY is not set.")

client = OpenAI(base_url="https://inference.coralbricks.ai/v1", api_key=key)

# Synthetic "tool output": ~400 numbered log lines, identical on both calls.
filler = "\n".join(
    f"[log {i:04d}] build step completed, module_{i % 37} compiled in {i % 9}.{i % 7}s, no warnings"
    for i in range(400)
)
messages = [
    {"role": "system", "content": "You are a coding agent reviewing build logs."},
    {"role": "user", "content": filler + "\n\nIn one short sentence: did any step fail?"},
]


def run(label: str) -> None:
    start = time.perf_counter()
    data_times: list[float] = []  # arrival time of every chunk that carried text
    usage = None
    stream = client.chat.completions.create(
        model="glm-5.3-flash-fast",
        messages=messages,
        max_tokens=1000,
        stream=True,
        stream_options={"include_usage": True},
    )
    answer = []
    for chunk in stream:
        if chunk.choices:
            d = chunk.choices[0].delta
            text = d.content or getattr(d, "reasoning_content", None)
            if text:
                data_times.append(time.perf_counter() - start)
            if d.content:
                answer.append(d.content)
        if chunk.usage is not None:
            usage = chunk.usage
    total = time.perf_counter() - start
    p = usage.prompt_tokens_details
    spread = data_times[-1] - data_times[0] if data_times else 0.0
    print(f"--- {label} ---")
    print(f"reply: {''.join(answer).strip()}")
    print(f"first data: {data_times[0]:.3f}s | last data: {data_times[-1]:.3f}s | "
          f"spread: {spread:.3f}s | text chunks: {len(data_times)} | total: {total:.3f}s")
    print(f"prompt: {usage.prompt_tokens} | cached: {p.cached_tokens} | "
          f"cache_write: {p.cache_write_tokens} | blocks: {p.cache_write_blocks} | "
          f"output: {usage.completion_tokens} | cost: ${usage.cost}")
    print(f"cost_details: {usage.cost_details}")


run("call 1 (cold)")
time.sleep(3)
run("call 2 (should be warm)")
