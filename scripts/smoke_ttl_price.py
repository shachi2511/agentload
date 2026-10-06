"""Phase 1, step 7: does the cache-write price depend on prompt_cache_options.ttl?
One cold call per setting, each with a unique prompt; prints the effective $/1M rates."""
import os
import sys
import uuid

import openai
from openai import OpenAI

MODEL = "glm-5.3-flash-fast"
key = os.environ.get("CORAL_API_KEY")
if not key:
    sys.exit("CORAL_API_KEY is not set.")
client = OpenAI(base_url="https://inference.coralbricks.ai/v1", api_key=key)

SETTINGS: list[tuple[str, dict]] = [
    ("default (none)", {}),
    ("ttl=1m", {"prompt_cache_options": {"ttl": "1m"}}),
    ("ttl=5m", {"prompt_cache_options": {"ttl": "5m"}}),
    ("ttl=10m", {"prompt_cache_options": {"ttl": "10m"}}),
    ("ttl=15m", {"prompt_cache_options": {"ttl": "15m"}}),
    ("ttl=30m", {"prompt_cache_options": {"ttl": "30m"}}),
    ("ttl=60m", {"prompt_cache_options": {"ttl": "60m"}}),
    ("ttl=240m", {"prompt_cache_options": {"ttl": "240m"}}),
    ("retention=off", {"prompt_cache_retention": "off"}),
]

print(f"{'setting':<16} {'prompt':>6} {'write':>6} {'billable':>8} {'blocks':>6} "
      f"{'input $/M':>9} {'write $/M':>9}")
for label, extra in SETTINGS:
    lines = "\n".join(f"[{label} {i:03d}] step ok, module_{i % 37} built" for i in range(150))
    messages = [
        {"role": "system", "content": f"Run {uuid.uuid4().hex}. Review build logs."},
        {"role": "user", "content": lines + "\n\nOne word: any failures?"},
    ]
    try:
        r = client.chat.completions.create(model=MODEL, messages=messages,
                                           max_tokens=1000, extra_body=extra)
    except openai.APIStatusError as e:
        print(f"{label:<16} REJECTED ({e.status_code}): {e.message}")
        continue
    u, p, cd = r.usage, r.usage.prompt_tokens_details, r.usage.cost_details
    billable = getattr(p, "billable_cache_write_tokens", None)
    blocks = getattr(p, "cache_write_blocks", None)
    in_rate = cd["input"] / u.prompt_tokens * 1e6
    wr_rate = cd["cache_write"] / p.cache_write_tokens * 1e6 if p.cache_write_tokens else 0.0
    print(f"{label:<16} {u.prompt_tokens:>6} {p.cache_write_tokens:>6} {billable!s:>8} "
          f"{blocks!s:>6} {in_rate:>9.4f} {wr_rate:>9.4f}")
