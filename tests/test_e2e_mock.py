"""End to end: runner -> fake server -> JSONL log -> billing check -> report.
No network, no real key, no spend. The real spend ledger is untouched (runs in a temp dir)."""
import asyncio
import importlib.util
from pathlib import Path

from agentload.metrics import read_jsonl
from agentload.pricing_report import validate
from agentload.runner import run_scenario
from tests.mock_server import start_mock_server

SCENARIO = Path(__file__).parent.parent / "scenarios" / "ci_smoke.yaml"


def load_report():
    spec = importlib.util.spec_from_file_location(
        "report_module", Path(__file__).parent.parent / "report" / "report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_end_to_end_against_mock_server(tmp_path, monkeypatch):
    url, server = start_mock_server()
    try:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AGENTLOAD_BASE_URL", url)
        monkeypatch.setenv("CORAL_API_KEY", "test-key-not-real")
        asyncio.run(run_scenario(SCENARIO))
    finally:
        server.shutdown()

    logs = list((tmp_path / "results").glob("*-ci_smoke-*.jsonl"))
    assert len(logs) == 1
    rows = list(read_jsonl(logs[0]))
    assert len(rows) == 8 and all(r["error"] is None for r in rows)
    assert all(r["cached_tokens"] > 0 for r in rows if r["turn"] > 1)   # prefix cache imitated
    assert {r["tags"]["variant"]: r["cache_write_blocks"] for r in rows} == {"ttl10m": 1,
                                                                              "default": 3}
    check = validate(logs)
    assert check["checked"] == 8 and check["matched"] == 8   # billing model agrees exactly

    md = load_report().build(tmp_path / "results", tmp_path / "out").read_text()
    assert "reproduced **8 of 8**" in md
