"""Phase 1, step 3: one streaming request; record which delta fields carry data and when."""
import json
import os
import sys
import time

from openai import OpenAI

key = os.environ.get("CORAL_API_KEY")
if not key:
    sys.exit("CORAL_API_KEY is not set.")

client = OpenAI(base_url="https://inference.coralbricks.ai/v1", api_key=key)

start = time.perf_counter()
first_seen: dict[str, float] = {}   # field name -> seconds until it first carried data
chars: dict[str, int] = {}          # field name -> total characters streamed
answer: list[str] = []
usage = None
chunks = 0

stream = client.chat.completions.create(
    model="glm-5.3-flash-fast",
    messages=[{"role": "user", "content": "Say hello in five words."}],
    max_tokens=1000,
    stream=True,
    stream_options={"include_usage": True},
)
for chunk in stream:
    chunks += 1
    if chunk.choices:
        delta = chunk.choices[0].delta.model_dump()
        for field, value in delta.items():
            if field == "role" or value in (None, "", []):
                continue
            first_seen.setdefault(field, time.perf_counter() - start)
            chars[field] = chars.get(field, 0) + len(str(value))
        if chunk.choices[0].delta.content:
            answer.append(chunk.choices[0].delta.content)
    if chunk.usage is not None:
        usage = chunk.usage
total = time.perf_counter() - start

print("REPLY:", "".join(answer))
print(f"total: {total:.3f}s | chunks: {chunks}")
print("FIELDS THAT CARRIED DATA (first seen at, total chars):")
for field in first_seen:
    print(f"  {field}: {first_seen[field]:.3f}s, {chars[field]} chars")
print("RAW USAGE:")
print(json.dumps(usage.model_dump(), indent=2) if usage else "NO USAGE RETURNED")
