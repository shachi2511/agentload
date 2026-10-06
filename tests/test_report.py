"""Report tests on a tiny synthetic log. No network, no spend."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "report_module", Path(__file__).parent.parent / "report" / "report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write_sequential(results):
    rows, prev = [], 0
    for variant, blocks in (("default", 3), ("ttl10m", 1)):
        prev = 0
        for turn in range(1, 13):
            prompt = prev + 2000
            cached = (prev // 64) * 64
            write = prompt - cached
            cost = (write * 0.23 / 3 * blocks + 50 * 0.5) / 1e6
            rows.append({"model": "glm-5.3-flash-fast", "turn": turn, "prompt_tokens": prompt,
                         "cached_tokens": cached, "billable_cache_write_tokens": write,
                         "cache_write_blocks": blocks, "completion_tokens": 50,
                         "cost_usd": cost, "ttft_s": 0.5, "started_at": 1791258000.0 + turn,
                         "error": None, "tags": {"variant": variant}})
            prev = prompt + 50
    path = results / "20261005-225939-sequential-abc123.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_build_with_partial_logs(tmp_path):
    results, out = tmp_path / "results", tmp_path / "out"
    results.mkdir()
    write_sequential(results)
    md = report.build(results, out).read_text()
    assert "| H1 flat cost per turn | **HELD** |" in md
    assert "reproduced **24 of 24**" in md          # billing model check ran on the log
    assert "_Not run yet._" in md                   # missing scenarios are reported, not faked
    assert (out / "h1_cost_per_turn.png").stat().st_size > 10_000


def test_money_delta_formatting():
    assert report.money_delta(-0.000004) == "~$0"
    assert report.money_delta(0.00147) == "+$0.0015"
    assert report.money_delta(None) == "-"
