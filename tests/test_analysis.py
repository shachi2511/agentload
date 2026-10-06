"""Analysis helper tests. No network, no spend."""
import pytest

from agentload.analysis import cache_reuse, hit_rate, percentile, summarize


def rows(prompts, cached, costs):
    return [{"prompt_tokens": p, "cached_tokens": c, "cost_usd": x, "ttft_s": 0.5, "error": None}
            for p, c, x in zip(prompts, cached, costs, strict=True)]


def test_percentile_nearest_rank():
    data = [1.0, 2.0, 3.0, 4.0, 10.0]
    assert percentile(data, 50) == 3.0
    assert percentile(data, 95) == 10.0
    assert percentile([], 50) is None


def test_cache_reuse_vs_hit_rate():
    r = rows([1000, 2000, 3000], [0, 960, 1984], [0.1, 0.1, 0.1])
    assert cache_reuse(r)[0] is None
    assert cache_reuse(r)[1] == pytest.approx(0.96)    # 960 of 1000 reusable tokens reused
    assert hit_rate(r[1]) == pytest.approx(0.48)       # raw hit rate looks much worse


def test_summarize_excludes_errors_and_splits_costs():
    r = rows([1000, 2000, 3000], [0, 1000, 2000], [0.3, 0.1, 0.2])
    r.append({"prompt_tokens": None, "cached_tokens": None, "cost_usd": None,
              "ttft_s": None, "error": "boom"})
    s = summarize(r)
    assert s["requests"] == 4 and s["errors"] == 1
    assert s["total_cost"] == pytest.approx(0.6)
    assert s["mean_cost_warm"] == pytest.approx(0.15)  # turn 1 (cold) excluded
    assert s["reuse_mean"] == pytest.approx(1.0)
    assert s["context_first"] == 1000 and s["context_last"] == 3000


def test_summarize_long_context_groups_and_orders():
    from agentload.analysis import summarize_long_context

    def row(size, phase, ttft, cached):
        return {"prompt_tokens": size, "cached_tokens": cached, "cost_usd": 0.01, "ttft_s": ttft,
                "error": None, "tags": {"size": size, "phase": phase}}

    data = [row(200, "warm", 0.5, 190), row(100, "cold", 3.0, 0), row(100, "warm", 0.4, 96),
            row(100, "cold", 5.0, 0), row(200, "cold", 9.0, 0)]
    table = summarize_long_context(data)
    assert [(g["size"], g["phase"]) for g in table] == [(100, "cold"), (100, "warm"),
                                                         (200, "cold"), (200, "warm")]
    assert table[0]["n"] == 2 and table[0]["ttft_p50"] == 3.0
    assert table[1]["cached_share"] == pytest.approx(0.96)


def test_summarize_fanout_child_turn_one():
    from agentload.analysis import summarize_fanout

    def row(phase, turn, prompt, cached, cost):
        return {"prompt_tokens": prompt, "cached_tokens": cached, "cost_usd": cost,
                "ttft_s": 1.0, "error": None, "tags": {"phase": phase, "child_turn": turn}}

    data = [row("parent", None, 1000, 0, 0.01), row("child", 1, 1100, 960, 0.001),
            row("child", 1, 1100, 0, 0.003), row("child", 2, 1300, 1280, 0.001)]
    s = summarize_fanout(data)
    assert s["child1_cached_share"] == pytest.approx((960 / 1100) / 2)
    assert s["later_cached_share"] == pytest.approx(1280 / 1300)
    assert s["total_cost"] == pytest.approx(0.015)


def test_summarize_idle_verdicts():
    from agentload.analysis import summarize_idle

    def row(phase, cached, error=None):
        return {"prompt_tokens": 1000, "cached_tokens": cached, "cost_usd": 0.001,
                "ttft_s": 1.0, "error": error, "tags": {"phase": phase, "probe": "p",
                                                        "check_min": 5}}

    data = [row("write", 0), row("check", 990), row("check", 500), row("check", 0),
            row("check", None, error="boom")]
    assert [c["verdict"] for c in summarize_idle(data)] == ["HIT", "PARTIAL", "MISS", "ERROR"]


def test_summarize_concurrency_speeds():
    from agentload.analysis import summarize_concurrency

    def row(level, rnd, tokens, start):
        return {"completion_tokens": tokens, "ttft_s": 0.5, "last_token_s": 1.5, "total_s": 1.6,
                "started_at": start, "text_chunks": tokens // 2, "attempts": 1,
                "status_code": None, "error": None, "tags": {"level": level, "round": rnd}}

    table = summarize_concurrency([row(1, 1, 300, 100.0), row(2, 1, 200, 100.0),
                                   row(2, 1, 200, 100.1)])
    one, two = table
    assert one["tps_p50"] == pytest.approx(300.0)          # 300 tokens over 1.0s
    assert one["agg_tps"] == pytest.approx(300 / 1.6)
    assert two["tps_p50"] == pytest.approx(200.0)
    assert two["agg_tps"] == pytest.approx(400 / 1.7)      # both streams over the round's wall time
    assert two["tokens_per_chunk"] == pytest.approx(2.0) and two["rate_limited"] == 0


def test_summarize_api_compare_write_rate_and_request_size():
    from agentload.analysis import summarize_api_compare

    def row(prompt, cached, billable, write_cost, chars):
        return {"prompt_tokens": prompt, "cached_tokens": cached, "cost_usd": write_cost + 1e-4,
                "ttft_s": 0.5, "first_content_s": 0.8, "total_s": 1.0, "error": None,
                "billable_cache_write_tokens": billable, "request_chars": chars,
                "cost_details": {"cache_write": write_cost, "input": 0.0}}

    s = summarize_api_compare([row(1000, 0, 1000, 0.00023, 3000),
                               row(2000, 960, 1040, 0.0002392, 3100)])
    assert s["write_rate_per_m"] == pytest.approx(0.23)
    assert s["request_chars_mean"] == pytest.approx(3050) and s["request_chars_last"] == 3100
