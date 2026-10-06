"""JSONL logging tests. No network, no spend."""
import time

from agentload.client import RequestResult
from agentload.metrics import JsonlWriter, new_run_id, read_jsonl


def make_result(turn, cost):
    return RequestResult(request_id=f"r{turn}", model="glm-5.3-flash-fast", scenario="test",
                         turn=turn, started_at=time.time(), cost_usd=cost, cached_tokens=64,
                         cost_details={"output": cost})


def test_writes_one_line_per_request_and_reads_back(tmp_path):
    path = tmp_path / "run.jsonl"
    writer = JsonlWriter(path, run_id="run-1")
    writer.write(make_result(1, 0.001))
    writer.write(make_result(2, 0.002))
    rows = list(read_jsonl(path))
    assert writer.count == 2 and len(rows) == 2
    assert rows[0]["run_id"] == "run-1" and rows[1]["turn"] == 2
    assert rows[1]["cost_usd"] == 0.002 and rows[0]["cost_details"] == {"output": 0.001}
    assert "logged_at" in rows[0]


def test_appends_instead_of_overwriting(tmp_path):
    path = tmp_path / "run.jsonl"
    JsonlWriter(path, run_id="a").write(make_result(1, 0.001))
    second = JsonlWriter(path, run_id="b")
    second.write(make_result(1, 0.001))
    second.write(make_result(2, 0.001))
    assert [r["run_id"] for r in read_jsonl(path)] == ["a", "b", "b"]


def test_run_id_is_readable_and_unique():
    a, b = new_run_id("check"), new_run_id("check")
    assert "-check-" in a and a != b
