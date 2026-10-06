"""Prometheus exporter tests. No network, no spend, no real HTTP server."""
import time

import pytest
from prometheus_client import CollectorRegistry

from agentload.client import RequestResult
from agentload.exporter import Exporter, ExportingWriter
from agentload.metrics import read_jsonl

LABELS = {"scenario": "s/v", "model": "m", "api": "chat"}


def result(turn, prompt, cached, conversation="s/v", error=None, attempts=1):
    return RequestResult(request_id=str(turn), model="m", scenario="s/v", turn=turn,
                         started_at=time.time(), prompt_tokens=prompt, cached_tokens=cached,
                         billable_cache_write_tokens=prompt - cached, completion_tokens=20,
                         cost_usd=0.001, ttft_s=0.5, total_s=1.0, error=error,
                         attempts=attempts, tags={"conversation": conversation})


def value(exporter, name, **extra):
    return exporter.registry.get_sample_value(name, {**LABELS, **extra})


def test_counters_and_histograms_update():
    ex = Exporter(CollectorRegistry())
    ex.observe(result(1, 1000, 0))
    ex.observe(result(2, 2000, 960, attempts=3))
    assert value(ex, "agentload_requests_total", status="ok") == 2
    assert value(ex, "agentload_retries_total") == 2
    assert value(ex, "agentload_prompt_tokens_total") == 3000
    assert value(ex, "agentload_cached_tokens_total") == 960
    assert value(ex, "agentload_cost_usd_total") == pytest.approx(0.002)
    assert value(ex, "agentload_ttft_seconds_count") == 2


def test_reuse_is_per_conversation():
    ex = Exporter(CollectorRegistry())
    assert ex.observe(result(1, 1000, 0)) is None               # first turn: nothing to reuse
    assert ex.observe(result(2, 2000, 960)) == pytest.approx(0.96)
    assert ex.observe(result(1, 5000, 0, conversation="other")) is None   # separate thread
    assert ex.observe(result(3, 3000, 0)) == 0.0                 # broken cache shows as 0
    assert value(ex, "agentload_cache_reuse_ratio_count") == 2
    assert value(ex, "agentload_cache_reuse_ratio_sum") == pytest.approx(0.96)


def test_no_conversation_tag_means_no_reuse():
    ex = Exporter(CollectorRegistry())
    r = result(1, 1000, 0)
    r.tags = {}
    ex.observe(r)
    assert ex.observe(result(2, 2000, 1000, conversation="")) is None


def test_errors_counted_without_tokens():
    ex = Exporter(CollectorRegistry())
    ex.observe(result(1, 1000, 0, error="boom"))
    assert value(ex, "agentload_requests_total", status="error") == 1
    assert value(ex, "agentload_prompt_tokens_total") is None


def test_exporting_writer_logs_and_observes(tmp_path):
    ex = Exporter(CollectorRegistry())
    writer = ExportingWriter(tmp_path / "run.jsonl", "r", ex)
    writer.write(result(1, 1000, 0))
    assert len(list(read_jsonl(tmp_path / "run.jsonl"))) == 1
    assert value(ex, "agentload_requests_total", status="ok") == 1


def test_prepare_creates_zero_counters():
    ex = Exporter(CollectorRegistry())
    assert value(ex, "agentload_requests_total", status="ok") is None
    ex.prepare("s/v", "m", "chat")
    assert value(ex, "agentload_requests_total", status="ok") == 0
    assert value(ex, "agentload_requests_total", status="error") == 0
    assert value(ex, "agentload_cost_usd_total") == 0
