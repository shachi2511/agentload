"""Request logging. Phase 2: one JSON line per request (JSONL). Phase 6 adds Prometheus metrics."""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from agentload.client import RequestResult


def new_run_id(label: str) -> str:
    """Readable, unique run id, e.g. 20261005-225501-check-a1b2c3."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{label}-{uuid.uuid4().hex[:6]}"


class JsonlWriter:
    """Appends one JSON object per request. Each line is written and closed immediately,
    so a crash mid-run loses nothing that already finished."""

    def __init__(self, path: Path | str, run_id: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.count = 0

    def write(self, result: RequestResult) -> None:
        record = {"run_id": self.run_id, "logged_at": time.time(), **result.to_dict()}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        self.count += 1


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
