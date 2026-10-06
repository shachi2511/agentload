"""Phase 6: Prometheus metrics for live runs.

Prometheus 'scrapes' (polls) http://<host>:<port>/metrics every few seconds and stores the
numbers over time; Grafana draws them. Every finished request updates the metrics below.
Metric values contain no prompts, replies or keys: only counts, costs and timings.
"""
from __future__ import annotations

from pathlib import Path

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

from agentload.client import RequestResult
from agentload.metrics import JsonlWriter

LABELS = ("scenario", "model", "api")


class Exporter:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.requests = Counter("agentload_requests", "Requests finished, by status",
                                (*LABELS, "status"), registry=r)
        self.retries = Counter("agentload_retries", "Extra attempts after 429 responses",
                               LABELS, registry=r)
        self.cost = Counter("agentload_cost_usd", "Reported cost (usage.cost) in USD",
                            LABELS, registry=r)
        self.prompt_tokens = Counter("agentload_prompt_tokens", "Prompt tokens", LABELS,
                                     registry=r)
        self.cached_tokens = Counter("agentload_cached_tokens", "Prompt tokens read from cache",
                                     LABELS, registry=r)
        self.write_tokens = Counter("agentload_billable_cache_write_tokens",
                                    "Prompt tokens billed as cache writes", LABELS, registry=r)
        self.output_tokens = Counter("agentload_output_tokens", "Output tokens (incl. thinking)",
                                     LABELS, registry=r)
        self.ttft = Histogram("agentload_ttft_seconds", "Time to first token", LABELS,
                              buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32), registry=r)
        self.latency = Histogram("agentload_request_seconds", "Total request time", LABELS,
                                 buckets=(0.5, 1, 2, 4, 8, 16, 32, 64), registry=r)
        self.reuse = Histogram(
            "agentload_cache_reuse_ratio",
            "Cached tokens / previous prompt tokens in the same conversation (1.0 = healthy)",
            LABELS, buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.01), registry=r)
        self.tps = Gauge("agentload_decode_tokens_per_second",
                         "Output tokens per second of the most recent request", LABELS, registry=r)
        self._last_prompt: dict[str, int] = {}

    def prepare(self, scenario: str, model: str, api: str) -> None:
        """Create this scenario's counters at 0 before any request. Prometheus only counts
        increases it sees between scrapes, so a counter that first appears already at 3
        would lose those 3 in increase()/rate(). Starting at 0 fixes that."""
        labels = (scenario, model, api)
        for status in ("ok", "error"):
            self.requests.labels(*labels, status)
        for counter in (self.retries, self.cost, self.prompt_tokens, self.cached_tokens,
                        self.write_tokens, self.output_tokens):
            counter.labels(*labels)

    def observe(self, r: RequestResult) -> float | None:
        """Record one finished request. Returns its cache-reuse ratio, if it has one."""
        labels = (r.scenario, r.model, r.api)
        self.requests.labels(*labels, "error" if r.error else "ok").inc()
        if r.attempts > 1:
            self.retries.labels(*labels).inc(r.attempts - 1)
        if r.error:
            return None
        self.cost.labels(*labels).inc(r.cost_usd or 0.0)
        self.prompt_tokens.labels(*labels).inc(r.prompt_tokens or 0)
        self.cached_tokens.labels(*labels).inc(r.cached_tokens or 0)
        self.write_tokens.labels(*labels).inc(r.billable_cache_write_tokens or 0)
        self.output_tokens.labels(*labels).inc(r.completion_tokens or 0)
        if r.ttft_s is not None:
            self.ttft.labels(*labels).observe(r.ttft_s)
        if r.total_s is not None:
            self.latency.labels(*labels).observe(r.total_s)
        tps = r.decode_tokens_per_s()
        if tps:
            self.tps.labels(*labels).set(tps)
        return self._observe_reuse(r, labels)

    def _observe_reuse(self, r: RequestResult, labels: tuple[str, str, str]) -> float | None:
        """Reuse only makes sense within one growing conversation (tag 'conversation')."""
        key = r.tags.get("conversation")
        if not key or r.prompt_tokens is None:
            return None
        previous = self._last_prompt.get(key)
        self._last_prompt[key] = r.prompt_tokens
        if not previous or r.cached_tokens is None:
            return None
        ratio = r.cached_tokens / previous
        self.reuse.labels(*labels).observe(ratio)
        return ratio

    def serve(self, port: int, addr: str = "0.0.0.0") -> None:
        """Expose /metrics. 0.0.0.0 so the Prometheus container can reach it."""
        start_http_server(port, addr=addr, registry=self.registry)


class ExportingWriter(JsonlWriter):
    """Writes the JSONL log exactly as before, and also updates the live metrics."""

    def __init__(self, path: Path | str, run_id: str, exporter: Exporter) -> None:
        super().__init__(path, run_id)
        self.exporter = exporter

    def write(self, result: RequestResult) -> None:
        super().write(result)
        self.exporter.observe(result)
