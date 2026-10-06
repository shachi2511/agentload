"""A tiny fake OpenAI-compatible server for tests and CI: no network, no API key, no spend.

It streams Chat Completions the way Coral does (a thinking chunk, answer chunks, then a usage
chunk with Coral's cache fields and cost) and imitates what we measured about Coral:
  - prefix caching in 64-token chunks (cached = longest earlier prompt that is a prefix)
  - cache writes billed per 10-minute block, max 3 blocks; retention "off" billed as input
  - charges rounded to 8 decimals
So the whole pipeline (client, runner, JSONL log, billing check, report) can run end to end.

Run standalone (for the CI smoke step):  PYTHONPATH=. python tests/mock_server.py --port 8099
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from agentload.pricing import PRICES

CHUNK_TOKENS = 64
CHARS_PER_TOKEN = 3
DEFAULT_MODEL = "glm-5.3-flash-fast"


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.seen: list[str] = []


def _blocks(body: dict[str, Any]) -> int:
    if body.get("prompt_cache_retention") == "off":
        return 0
    ttl = (body.get("prompt_cache_options") or {}).get("ttl")
    if not ttl:
        return 3
    minutes = int(str(ttl).rstrip("m"))
    return min(3, max(1, math.ceil(minutes / 10)))


def make_usage(body: dict[str, Any], state: _State) -> dict[str, Any]:
    model = body.get("model") or DEFAULT_MODEL
    price = PRICES.get(model, PRICES[DEFAULT_MODEL])
    flat = json.dumps(body.get("messages", []))
    prompt = max(1, len(flat) // CHARS_PER_TOKEN)
    with state.lock:
        common = max((len(os.path.commonprefix([s, flat])) for s in state.seen), default=0)
        state.seen.append(flat)
    cached = min(prompt, (common // CHARS_PER_TOKEN) // CHUNK_TOKENS * CHUNK_TOKENS)
    new = prompt - cached
    blocks = _blocks(body)
    billable = new if blocks else 0
    reasoning, completion = 8, 12
    cost_details = {
        "currency": "USD",
        "input": round((new - billable) * price.input / 1e6, 8),
        "cached_input": 0.0,
        "cache_write": round(billable * price.cache_write / 3 * blocks / 1e6, 8),
        "output": round(completion * price.output / 1e6, 8),
    }
    return {
        "prompt_tokens": prompt, "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "completion_tokens_details": {"reasoning_tokens": reasoning},
        "prompt_tokens_details": {"cached_tokens": cached, "cache_write_tokens": new,
                                  "billable_cache_write_tokens": billable,
                                  "cache_write_blocks": blocks},
        "cost": round(sum(v for k, v in cost_details.items() if k != "currency"), 8),
        "cost_details": cost_details,
    }


class _Handler(BaseHTTPRequestHandler):
    state: _State

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404, "only /v1/chat/completions is mocked")
            return
        usage = make_usage(body, self.state)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        base = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 0,
                "model": body.get("model") or DEFAULT_MODEL}
        deltas = [{"role": "assistant", "content": ""}, {"reasoning_content": "thinking"},
                  {"content": "Next: "}, {"content": "run the tests."}]
        for delta in deltas:
            self._send({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
        self._send({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        if (body.get("stream_options") or {}).get("include_usage"):
            self._send({**base, "choices": [], "usage": usage})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _send(self, obj: dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()

    def log_message(self, *args: Any) -> None:  # keep test output quiet
        pass


def make_server(port: int = 0) -> ThreadingHTTPServer:
    handler = type("MockHandler", (_Handler,), {"state": _State()})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def start_mock_server() -> tuple[str, ThreadingHTTPServer]:
    """Start on a free port in a background thread; returns (base_url, server)."""
    server = make_server()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}/v1", server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake OpenAI-compatible server (no spend).")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()
    print(f"mock server on http://127.0.0.1:{args.port}/v1", flush=True)
    make_server(args.port).serve_forever()
