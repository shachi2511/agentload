"""Phase 1, step 6: does prompt_cache_options.ttl work, and what does prompt_cache_retention='off' do?
A: ttl=1m,  wait 90s -> expect MISS if TTL is honored
B: ttl=10m, wait 90s -> expect HIT (control: rules out plain eviction)
C: retention off, two calls back to back -> expect no caching; check how input is billed
"""
import os
import sys
import time
import uuid

import openai
from openai import OpenAI

MODEL = "glm-5.3-flash-fast"
WAIT_S = 90

key = os.environ.get("CORAL_API_KEY")
if not key:
    sys.exit("CORAL_API_KEY is not set.")
client = OpenAI(base_url="https://inference.coralbricks.ai/v1", api_key=key)


def big_prompt(tag: str) -> list[dict]:
    """~7K-token prompt with a unique random prefix so tests never share cache."""
    nonce = uuid.uuid4().hex
    lines = "\n".join(
        f"[{tag} log {i:04d}] step ok, module_{i % 37} built in {i % 9}.{i % 7}s" for i in range(300)
    )
    return [
        {"role": "system", "content": f"Run {nonce}. You review build logs."},
        {"role": "user", "content": lines + "\n\nOne short sentence: any failures?"},
    ]


def call(messages: list[dict], extra: dict, label: str) -> None:
    try:
        r = client.chat.completions.create(
            model=MODEL, messages=messages, max_tokens=1000, extra_body=extra
        )
    except openai.APIStatusError as e:
        print(f"{label:<26} REJECTED ({e.status_code}): {e.message}")
        return
    u = r.usage
    p = u.prompt_tokens_details
    print(f"{label:<26} prompt {u.prompt_tokens:>6} | cached {p.cached_tokens:>6} | "
          f"write {p.cache_write_tokens:>6} | cost ${u.cost:.6f}")
    print(f"{'':<26} cost_details: {u.cost_details}")


a = big_prompt("A")
b = big_prompt("B")
c = big_prompt("C")
ttl_1m = {"prompt_cache_options": {"ttl": "1m"}}
ttl_10m = {"prompt_cache_options": {"ttl": "10m"}}
off = {"prompt_cache_retention": "off"}

call(a, ttl_1m, "A1 ttl=1m (first)")
call(b, ttl_10m, "B1 ttl=10m (first)")
print(f"... waiting {WAIT_S}s ...")
time.sleep(WAIT_S)
call(a, ttl_1m, "A2 ttl=1m after 90s")
call(b, ttl_10m, "B2 ttl=10m after 90s")
call(c, off, "C1 retention=off")
call(c, off, "C2 retention=off again")
