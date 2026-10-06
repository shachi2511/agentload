"""Summaries computed from request records (dicts, as stored in the JSONL log)."""
from __future__ import annotations

import math
from collections import defaultdict
from itertools import pairwise
from typing import Any

Row = dict[str, Any]


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (p in 0-100). None if there are no values."""
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    rank = max(1, math.ceil(p / 100 * len(clean)))
    return clean[rank - 1]


def cache_reuse(rows: list[Row]) -> list[float | None]:
    """Per turn: cached tokens / previous request's prompt tokens.
    The share of the reusable prefix that was actually reused (1.0 = perfect).
    Unlike raw hit rate, it does not drift upward just because the conversation got longer.
    Can slightly exceed 1.0 when part of the previous reply was cached too."""
    out: list[float | None] = [None]
    for prev, cur in pairwise(rows):
        if prev.get("prompt_tokens") and cur.get("cached_tokens") is not None:
            out.append(cur["cached_tokens"] / prev["prompt_tokens"])
        else:
            out.append(None)
    return out


def hit_rate(row: Row) -> float | None:
    if not row.get("prompt_tokens") or row.get("cached_tokens") is None:
        return None
    return row["cached_tokens"] / row["prompt_tokens"]


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def summarize(rows: list[Row]) -> dict[str, Any]:
    """Headline numbers for one sequential run (rows in turn order)."""
    ok = [r for r in rows if not r.get("error")]
    reuse = [x for x in cache_reuse(ok)[1:] if x is not None]
    costs = [r["cost_usd"] for r in ok]
    warm_costs = costs[1:]
    return {
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "total_cost": sum(costs),
        "mean_cost_warm": _mean(warm_costs),
        "mean_cost_first10": _mean(warm_costs[:10]),
        "mean_cost_last10": _mean(warm_costs[-10:]),
        "context_first": ok[0]["prompt_tokens"] if ok else None,
        "context_last": ok[-1]["prompt_tokens"] if ok else None,
        "reuse_mean": _mean(reuse),
        "reuse_min": min(reuse) if reuse else None,
        "ttft_p50": percentile([r["ttft_s"] for r in ok], 50),
        "ttft_p95": percentile([r["ttft_s"] for r in ok], 95),
    }


PHASE_ORDER = {"cold": 0, "warm": 1, "warm+append": 2}


def summarize_long_context(rows: list[Row]) -> list[dict[str, Any]]:
    """Group long-context calls by (size, phase): count, size, cached share, TTFT p50/min/max, cost."""
    groups: dict[tuple[Any, Any], list[Row]] = defaultdict(list)
    for r in rows:
        if r.get("error"):
            continue
        tags = r.get("tags") or {}
        groups[(tags.get("size"), tags.get("phase"))].append(r)
    out: list[dict[str, Any]] = []
    for (size, phase), group in sorted(
        groups.items(), key=lambda kv: (kv[0][0] or 0, PHASE_ORDER.get(kv[0][1], 9))
    ):
        ttfts = [r["ttft_s"] for r in group if r.get("ttft_s") is not None]
        out.append({
            "size": size, "phase": phase, "n": len(group),
            "prompt_tokens": _mean([r["prompt_tokens"] for r in group]),
            "cached_share": _mean([hit_rate(r) for r in group]),
            "ttft_p50": percentile(ttfts, 50),
            "ttft_min": min(ttfts) if ttfts else None,
            "ttft_max": max(ttfts) if ttfts else None,
            "cost_mean": _mean([r["cost_usd"] for r in group]),
        })
    return out


def summarize_fanout(rows: list[Row]) -> dict[str, Any]:
    """Did sub-agents reuse the shared context? Child turn 1 is the key number."""
    ok = [r for r in rows if not r.get("error")]

    def child_rows(first_turn: bool) -> list[Row]:
        out = []
        for r in ok:
            tags = r.get("tags") or {}
            if tags.get("phase") == "child" and (tags.get("child_turn") == 1) == first_turn:
                out.append(r)
        return out

    first, later = child_rows(True), child_rows(False)
    return {
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "total_cost": sum(r["cost_usd"] for r in ok),
        "child1_cached_share": _mean([hit_rate(r) for r in first]),
        "child1_cost_mean": _mean([r["cost_usd"] for r in first]),
        "child1_ttft_p50": percentile([r["ttft_s"] for r in first], 50),
        "later_cached_share": _mean([hit_rate(r) for r in later]),
    }


def summarize_idle(rows: list[Row]) -> list[dict[str, Any]]:
    """One line per idle-gap check: was the prompt still cached after the wait?"""
    out: list[dict[str, Any]] = []
    for r in rows:
        tags = r.get("tags") or {}
        if tags.get("phase") != "check":
            continue
        share = hit_rate(r)
        if r.get("error"):
            verdict = "ERROR"
        elif share is not None and share >= 0.9:
            verdict = "HIT"
        elif share:
            verdict = "PARTIAL"
        else:
            verdict = "MISS"
        out.append({"probe": tags.get("probe"), "check_min": tags.get("check_min"),
                    "cached_share": share, "cost": r.get("cost_usd"), "ttft": r.get("ttft_s"),
                    "verdict": verdict})
    return out


def decode_tps(row: Row) -> float | None:
    """Output tokens / time from first to last token, for one request."""
    tokens, ttft, last = row.get("completion_tokens"), row.get("ttft_s"), row.get("last_token_s")
    if not tokens or ttft is None or last is None or last <= ttft:
        return None
    return tokens / (last - ttft)


def summarize_concurrency(rows: list[Row]) -> list[dict[str, Any]]:
    """Per concurrency level: latency percentiles, per-stream and total tokens/sec, 429s."""
    groups: dict[Any, list[Row]] = defaultdict(list)
    for r in rows:
        groups[(r.get("tags") or {}).get("level")].append(r)
    out: list[dict[str, Any]] = []
    for level in sorted(k for k in groups if k is not None):
        g = groups[level]
        ok = [r for r in g if not r.get("error")]
        tps = [t for t in (decode_tps(r) for r in ok) if t is not None]
        rounds: dict[Any, list[Row]] = defaultdict(list)
        for r in ok:
            rounds[(r.get("tags") or {}).get("round")].append(r)
        agg = []
        for rr in rounds.values():
            start = min(r["started_at"] for r in rr)
            end = max(r["started_at"] + (r.get("total_s") or 0) for r in rr)
            if end > start:
                agg.append(sum(r.get("completion_tokens") or 0 for r in rr) / (end - start))
        per_chunk = [r["completion_tokens"] / r["text_chunks"] for r in ok
                     if r.get("completion_tokens") and r.get("text_chunks")]
        ttfts = [r.get("ttft_s") for r in ok]
        totals = [r.get("total_s") for r in ok]
        out.append({
            "level": level, "n": len(g), "errors": len(g) - len(ok),
            "rate_limited": sum(1 for r in g
                                if (r.get("attempts") or 1) > 1 or r.get("status_code") == 429),
            "ttft_p50": percentile(ttfts, 50), "ttft_p95": percentile(ttfts, 95),
            "total_p50": percentile(totals, 50), "total_p95": percentile(totals, 95),
            "total_p99": percentile(totals, 99),
            "tps_p50": percentile(tps, 50), "tps_min": min(tps) if tps else None,
            "agg_tps": _mean(agg), "tokens_per_chunk": _mean(per_chunk),
        })
    return out


def summarize_api_compare(rows: list[Row]) -> dict[str, Any]:
    """H4: sequential summary plus effective write price, input cost and request size."""
    ok = [r for r in rows if not r.get("error")]
    writes = sum((r.get("cost_details") or {}).get("cache_write", 0.0) for r in ok)
    billable = sum(r.get("billable_cache_write_tokens") or 0 for r in ok)
    chars = [r["request_chars"] for r in ok if r.get("request_chars")]
    return {
        **summarize(rows),
        "write_rate_per_m": writes / billable * 1e6 if billable else None,
        "input_cost": sum((r.get("cost_details") or {}).get("input", 0.0) for r in ok),
        "request_chars_mean": _mean(chars),
        "request_chars_last": chars[-1] if chars else None,
        "first_content_p50": percentile([r.get("first_content_s") for r in ok], 50),
        "total_p50": percentile([r.get("total_s") for r in ok], 50),
    }
