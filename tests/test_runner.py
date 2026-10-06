"""Runner tests with a fake client. No network, no spend."""
import asyncio
import time

import pytest

from agentload.client import RequestResult
from agentload.metrics import JsonlWriter, read_jsonl
from agentload.runner import (
    Variant,
    load_scenario,
    run_long_context,
    run_sequential,
    tool_tokens_for_turn,
)
from agentload.workload import ToolOutputGenerator


class FakeClient:
    def __init__(self):
        self.seen = []

    async def chat(self, messages, *, turn, scenario, tags, **kwargs):
        self.seen.append([dict(m) for m in messages])
        return RequestResult(request_id=str(turn), model="m", scenario=scenario, turn=turn,
                             started_at=time.time(), prompt_tokens=1000 * turn,
                             cached_tokens=1000 * (turn - 1), cost_usd=0.001,
                             output_text=f"next {turn}", tags=tags)


def write_yaml(tmp_path, body):
    path = tmp_path / "s.yaml"
    path.write_text(body)
    return path


def test_load_scenario_reads_variants_and_params(tmp_path):
    path = write_yaml(tmp_path, "name: s\nturns: 3\ntool_tokens: [100, 200]\nrepeats: 2\n"
                                "variants:\n  - name: a\n  - name: b\n"
                                "    extra_body: {prompt_cache_options: {ttl: '10m'}}\n")
    sc = load_scenario(path)
    assert [v.name for v in sc.variants] == ["a", "b"]
    assert sc.variants[1].extra_body == {"prompt_cache_options": {"ttl": "10m"}}
    assert sc.params == {"repeats": 2}


def test_load_scenario_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError, match="budget"):
        load_scenario(write_yaml(tmp_path, "name: s\nbudget_usd: 50\n"))
    with pytest.raises(ValueError, match="tool_tokens"):
        load_scenario(write_yaml(tmp_path, "name: s\ntool_tokens: [5, 2]\n"))
    with pytest.raises(ValueError, match="unknown mode"):
        load_scenario(write_yaml(tmp_path, "name: s\nmode: chaos\n"))


def test_tool_tokens_stay_in_range():
    sizes = [tool_tokens_for_turn(t, 1000, 4000) for t in range(1, 60)]
    assert min(sizes) >= 1000 and max(sizes) <= 4000 and len(set(sizes)) > 10


def test_sequential_grows_history_and_logs_every_turn(tmp_path):
    sc = load_scenario(write_yaml(tmp_path, "name: s\nturns: 4\ntool_tokens: [50, 60]\n"))
    client = FakeClient()
    writer = JsonlWriter(tmp_path / "run.jsonl", run_id="r")
    results = asyncio.run(run_sequential(client, writer, sc, Variant("v"), ToolOutputGenerator()))
    assert len(results) == 4
    assert [len(m) for m in client.seen] == [2, 4, 6, 8]
    logged = list(read_jsonl(tmp_path / "run.jsonl"))
    assert [row["turn"] for row in logged] == [1, 2, 3, 4]
    assert logged[0]["tags"]["variant"] == "v"


def test_long_context_cold_warm_append(tmp_path):
    sc = load_scenario(write_yaml(
        tmp_path, "name: lc\nmode: long_context\ncontext_sizes: [100, 200]\nrepeats: 1\n"
                  "warm_delay_s: 0\nappend_tokens: 50\n"))
    client = FakeClient()
    writer = JsonlWriter(tmp_path / "run.jsonl", run_id="r")
    results = asyncio.run(run_long_context(client, writer, sc, Variant("v"),
                                           ToolOutputGenerator()))
    phases = [(r.tags["size"], r.tags["phase"]) for r in results]
    assert phases == [(100, "cold"), (100, "warm"), (100, "warm+append"),
                      (200, "cold"), (200, "warm"), (200, "warm+append")]
    assert client.seen[0] == client.seen[1]            # warm is the exact same prompt
    assert len(client.seen[2]) == len(client.seen[0]) + 2
    assert client.seen[0][0] != client.seen[3][0]      # each size gets a fresh, unique prefix


def test_fanout_shared_vs_distinct_prefixes(tmp_path):
    from agentload.runner import run_fanout
    body = ("name: f\nmode: fanout\nprefix_tokens: 100\nchildren: 2\nchild_turns: 2\n"
            "task_tokens: 50\nrepeats: 1\nvariants:\n"
            "  - name: w\n    prefix: shared_warm\n  - name: d\n    prefix: distinct\n")
    sc = load_scenario(write_yaml(tmp_path, body))
    writer = JsonlWriter(tmp_path / "run.jsonl", run_id="r")
    warm = FakeClient()
    res = asyncio.run(run_fanout(warm, writer, sc, sc.variants[0], ToolOutputGenerator()))
    phases = [r.tags["phase"] for r in res]
    assert phases.count("parent") == 1 and phases.count("child") == 4
    assert len({m[0]["content"] for m in warm.seen}) == 1     # one shared prefix for all
    distinct = FakeClient()
    res = asyncio.run(run_fanout(distinct, writer, sc, sc.variants[1], ToolOutputGenerator()))
    assert all(r.tags["phase"] == "child" for r in res)
    assert len({m[0]["content"] for m in distinct.seen}) == 2  # one prefix per child


def test_fanout_refuses_more_than_8_parallel(tmp_path):
    from agentload.runner import run_fanout
    sc = load_scenario(write_yaml(tmp_path, "name: f\nmode: fanout\nchildren: 9\n"))
    with pytest.raises(ValueError, match="1-8"):
        asyncio.run(run_fanout(FakeClient(), JsonlWriter(tmp_path / "x.jsonl", "r"), sc,
                               sc.variants[0], ToolOutputGenerator()))


def test_idle_gaps_writes_all_then_checks_in_time_order(tmp_path):
    from agentload.runner import run_idle_gaps
    body = ("name: i\nmode: idle_gaps\nprompt_tokens: 50\nseconds_per_min: 0.001\nprobes:\n"
            "  - {name: late, checks_min: [12]}\n"
            "  - {name: early, checks_min: [2, 8]}\n")
    sc = load_scenario(write_yaml(tmp_path, body))
    client = FakeClient()
    res = asyncio.run(run_idle_gaps(client, JsonlWriter(tmp_path / "r.jsonl", "r"), sc,
                                    sc.variants[0], ToolOutputGenerator()))
    order = [(r.tags["probe"], r.tags["check_min"]) for r in res]
    assert order == [("late", 0), ("early", 0), ("early", 2), ("early", 8), ("late", 12)]
    assert client.seen[0] == client.seen[4]       # the check re-sends the exact same prompt


def test_idle_gaps_rejects_long_waits(tmp_path):
    from agentload.runner import run_idle_gaps
    sc = load_scenario(write_yaml(tmp_path, "name: i\nmode: idle_gaps\nprobes:\n"
                                            "  - {name: x, checks_min: [90]}\n"))
    with pytest.raises(ValueError, match="checks_min"):
        asyncio.run(run_idle_gaps(FakeClient(), JsonlWriter(tmp_path / "r.jsonl", "r"), sc,
                                  sc.variants[0], ToolOutputGenerator()))


def test_early_edit_and_timestamp_variants(tmp_path):
    body = ("name: e\nturns: 4\ntool_tokens: [50, 60]\nvariants:\n"
            "  - name: edit\n    edit_at_turn: 3\n    edit_message_index: 1\n"
            "  - name: clock\n    system_timestamp: true\n")
    sc = load_scenario(write_yaml(tmp_path, body))
    writer = JsonlWriter(tmp_path / "r.jsonl", "r")
    edit = FakeClient()
    res = asyncio.run(run_sequential(edit, writer, sc, sc.variants[0], ToolOutputGenerator()))
    assert [r.tags["edited"] for r in res] == [False, False, True, False]
    assert not edit.seen[1][1]["content"].startswith("[edited]")
    assert edit.seen[2][1]["content"].startswith("[edited]")
    assert edit.seen[3][1]["content"].startswith("[edited]")   # the edit persists afterwards
    clock = FakeClient()
    asyncio.run(run_sequential(clock, writer, sc, sc.variants[1], ToolOutputGenerator()))
    systems = [m[0]["content"] for m in clock.seen]
    assert len(set(systems)) == 4 and all("Current time" in s for s in systems)


def test_edit_turn_must_be_in_range(tmp_path):
    sc = load_scenario(write_yaml(tmp_path, "name: e\nturns: 3\nvariants:\n"
                                            "  - name: x\n    edit_at_turn: 9\n"))
    with pytest.raises(ValueError, match="edit_at_turn"):
        asyncio.run(run_sequential(FakeClient(), JsonlWriter(tmp_path / "r.jsonl", "r"), sc,
                                   sc.variants[0], ToolOutputGenerator()))


def test_concurrency_levels_model_override_and_cap(tmp_path):
    from agentload.runner import run_concurrency
    sc = load_scenario(write_yaml(tmp_path, "name: c\nmode: concurrency\nlevels: [1, 2]\n"
                                            "rounds: 2\npause_s: 0\nvariants:\n"
                                            "  - name: v\n    model: m2\n"))
    res = asyncio.run(run_concurrency(FakeClient(), JsonlWriter(tmp_path / "r.jsonl", "r"), sc,
                                      sc.variants[0], ToolOutputGenerator()))
    assert sorted(r.tags["level"] for r in res) == [1, 1, 2, 2, 2, 2]
    assert all(r.tags["model"] == "m2" for r in res)
    bad = load_scenario(write_yaml(tmp_path, "name: c\nmode: concurrency\nlevels: [16]\n"))
    with pytest.raises(ValueError, match="levels"):
        asyncio.run(run_concurrency(FakeClient(), JsonlWriter(tmp_path / "r.jsonl", "r"), bad,
                                    bad.variants[0], ToolOutputGenerator()))


class FakeRespondClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.responds = []

    async def respond(self, input_text, *, history_chars, instructions, previous_response_id,
                      turn, scenario, tags, **kwargs):
        self.responds.append((instructions, previous_response_id, history_chars))
        return RequestResult(request_id=str(turn), model="m", scenario=scenario, turn=turn,
                             started_at=time.time(), api="responses", response_id=f"resp_{turn}",
                             prompt_tokens=1000 * turn, cached_tokens=1000 * (turn - 1),
                             cost_usd=0.001, output_text="ok", tags=tags)


def test_api_compare_chains_ids_and_instructions(tmp_path):
    from agentload.runner import run_api_compare
    body = ("name: a\nmode: api_compare\nturns: 3\ntool_tokens: [50, 60]\nvariants:\n"
            "  - name: once\n    api: responses\n    repeat_instructions: false\n"
            "  - name: chat\n    api: chat\n")
    sc = load_scenario(write_yaml(tmp_path, body))
    writer = JsonlWriter(tmp_path / "r.jsonl", "r")
    client = FakeRespondClient()
    asyncio.run(run_api_compare(client, writer, sc, sc.variants[0], ToolOutputGenerator()))
    instr, prev, chars = zip(*client.responds, strict=True)
    assert prev == (None, "resp_1", "resp_2")
    assert instr[0] is not None and instr[1] is None and instr[2] is None
    assert chars[0] < chars[1] < chars[2]
    chat = FakeRespondClient()
    asyncio.run(run_api_compare(chat, writer, sc, sc.variants[1], ToolOutputGenerator()))
    assert [len(m) for m in chat.seen] == [2, 4, 6] and chat.responds == []


def test_api_compare_rejects_unknown_api(tmp_path):
    from agentload.runner import run_api_compare
    sc = load_scenario(write_yaml(tmp_path, "name: a\nmode: api_compare\nvariants:\n"
                                            "  - name: x\n    api: fax\n"))
    with pytest.raises(ValueError, match="api"):
        asyncio.run(run_api_compare(FakeRespondClient(), JsonlWriter(tmp_path / "r.jsonl", "r"),
                                    sc, sc.variants[0], ToolOutputGenerator()))
