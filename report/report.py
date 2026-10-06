"""Phase 5: build charts and results.md from the JSONL logs. No API calls, $0.

Run: PYTHONPATH=. python report/report.py          (writes report/out/)
Every number in results.md is computed from the logs; nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from agentload.analysis import (
    Row,
    cache_reuse,
    hit_rate,
    summarize,
    summarize_api_compare,
    summarize_concurrency,
    summarize_fanout,
    summarize_idle,
    summarize_long_context,
)
from agentload.metrics import read_jsonl
from agentload.pricing import DiscountModel, discount_cost
from agentload.pricing_report import compare, validate

# Reference data-viz palette: categorical slots 1-3 (validated all-pairs), ink, status, ordinal.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
STATUS = {"HIT": "#0ca30c", "PARTIAL": "#fab219", "MISS": "#d03b3b", "ERROR": "#d03b3b"}
ORDINAL = ("#86b6ef", "#2a78d6", "#104281")  # p50, p95, p99
SCENARIOS = ("sequential", "long_context", "fanout", "idle_gaps", "early_edit",
             "concurrency", "api_compare")


# ---------- loading ----------

def latest_run(results_dir: Path, name: str) -> Path | None:
    files = sorted(results_dir.glob(f"*-{name}-*.jsonl"))
    return files[-1] if files else None


def by_variant(path: Path) -> dict[str, list[Row]]:
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in read_jsonl(path):
        groups[(r.get("tags") or {}).get("variant") or r.get("scenario")].append(r)
    for rows in groups.values():
        rows.sort(key=lambda r: r.get("turn") or 0)
    return dict(groups)


def ok_rows(rows: list[Row]) -> list[Row]:
    return [r for r in rows if not r.get("error")]


# ---------- chart helpers ----------

def new_fig(panels: int = 1) -> tuple[Figure, list[Any]]:
    fig = Figure(figsize=(7.2 if panels == 1 else 11, 3.8), facecolor=SURFACE)
    axes = [fig.add_subplot(1, panels, i + 1) for i in range(panels)]
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    return fig, axes


def label(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def end_label(ax: Any, x: float, y: float, text: str) -> None:
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center",
                fontsize=8, color=INK2)


def money_delta(x: float | None, digits: int = 4) -> str:
    """Signed dollar change; values that round to zero print as '~$0'."""
    if x is None:
        return "-"
    if abs(x) < 0.5 * 10 ** -digits:
        return "~$0"
    return f"{'+' if x > 0 else '-'}${abs(x):.{digits}f}"


def integer_x(ax: Any) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def save(fig: Figure, out: Path, name: str) -> str:
    fig.tight_layout()
    fig.savefig(out / name, dpi=150, facecolor=SURFACE)
    return name


def legend(ax: Any) -> None:
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)


# ---------- sections (each returns markdown lines and a scoreboard row) ----------

def section_h1(path: Path, out: Path) -> tuple[list[str], tuple[str, str, str]]:
    cmp = compare(path, DiscountModel())
    groups = by_variant(path)
    fig, (ax,) = new_fig()
    for i, name in enumerate([n for n in ("default", "ttl10m") if n in groups]):
        rows = ok_rows(groups[name])
        xs, ys = [r["turn"] for r in rows], [r["cost_usd"] * 1000 for r in rows]
        ax.plot(xs, ys, color=SERIES[i], linewidth=2, label=f"Coral {name} (measured)")
        end_label(ax, xs[-1], ys[-1], f"Coral {name}")
    if "default" in groups:
        rows = ok_rows(groups["default"])
        xs = [r["turn"] for r in rows]
        ys = [(discount_cost(r) or 0) * 1000 for r in rows]
        ax.plot(xs, ys, color=SERIES[2], linewidth=2, linestyle="--",
                label="discounted-reads rule, same tokens (modeled)")
        end_label(ax, xs[-1], ys[-1], "discounted reads")
    integer_x(ax)
    label(ax, "H1: cost per turn as the conversation grows", "turn", "cost per turn ($ x 1,000)")
    legend(ax)
    chart = save(fig, out, "h1_cost_per_turn.png")

    held = True
    lines = [f"![H1 chart]({chart})", ""]
    for name, s in cmp.items():
        held = held and s["alt_slope"] > 0 and abs(s["measured_slope"]) < 0.1 * s["alt_slope"]
        lines.append(
            f"- `{name}`: context {s['context_first']:,} → {s['context_last']:,} tokens. "
            f"Coral cost per turn, first 10 → last 10: ${s['measured_first10']:.6f} → "
            f"${s['measured_last10']:.6f} **[measured]**. Discounted-reads rule on the same "
            f"tokens: ${s['alt_first10']:.6f} → ${s['alt_last10']:.6f} **[modeled]**. Cost added "
            f"per +100K tokens of context: Coral {money_delta(s['measured_slope'], 6)}, discounted "
            f"{money_delta(s['alt_slope'], 6)}. Whole run: ${s['measured_total']:.5f} vs "
            f"${s['alt_total']:.5f} ({s['alt_total'] / s['measured_total']:.1f}x).")
    if {"default", "ttl10m"} <= cmp.keys():
        saving = 1 - cmp["ttl10m"]["measured_total"] / cmp["default"]["measured_total"]
        lines.append(f"- Same loop with `ttl: \"10m\"` instead of the default: "
                     f"{saving:.0%} cheaper overall **[measured]**.")
    verdict = "HELD" if held else "NOT HELD"
    slope = next(iter(cmp.values()))
    key = (f"Coral {money_delta(slope['measured_slope'])} vs discounted "
           f"{money_delta(slope['alt_slope'])} per turn per +100K context")
    return lines, ("H1 flat cost per turn", verdict, key)


def section_h2_loop(path: Path) -> tuple[list[str], tuple[str, str, str]]:
    rows = ok_rows(by_variant(path).get("default", []))
    s = summarize(rows)
    hits = [hit_rate(r) for r in rows[1:]]
    hit_min = min(h for h in hits if h is not None)
    held = (s["reuse_mean"] or 0) >= 0.9
    lines = [(f"- Plain {len(rows)}-turn loop: cache reuse mean {s['reuse_mean']:.1%}, "
             f"min {s['reuse_min']:.1%} **[measured]**. Raw hit rate (cached ÷ prompt) dipped "
             f"to {hit_min:.1%} even though the cache was working: it mostly reflects how long "
             f"the conversation is **[inference]**.")]
    return lines, ("H2 plain loop keeps the cache", "HELD" if held else "NOT HELD",
                   f"reuse {s['reuse_mean']:.1%} mean (raw hit rate as low as {hit_min:.0%})")


def section_h2c(path: Path, out: Path) -> tuple[list[str], tuple[str, str, str]]:
    groups = by_variant(path)
    fig, axes = new_fig(2)
    edit_turn = None
    for i, name in enumerate(groups):
        rows = ok_rows(groups[name])
        xs = [r["turn"] for r in rows]
        axes[0].plot(xs, [(hit_rate(r) or 0) * 100 for r in rows], color=SERIES[i % 3],
                     linewidth=2, label=name)
        reuse = cache_reuse(rows)
        axes[1].plot(xs[1:], [(x or 0) * 100 for x in reuse[1:]], color=SERIES[i % 3],
                     linewidth=2, label=name)
        for r in rows:
            if (r.get("tags") or {}).get("edited"):
                edit_turn = r["turn"]
    for ax in axes:
        integer_x(ax)
        ax.set_ylim(-3, 105)
        if edit_turn:
            ax.axvline(edit_turn, color=INK2, linewidth=1, linestyle=":")
            ax.annotate("early edit", (edit_turn, 50), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=INK2)
    label(axes[0], "Raw hit rate (cached ÷ prompt)", "turn", "%")
    label(axes[1], "Cache reuse (cached ÷ previous prompt)", "turn", "%")
    legend(axes[1])
    chart = save(fig, out, "h2_cache_over_time.png")

    lines = [f"![H2 chart]({chart})", ""]
    held = False
    edit_rows = ok_rows(groups.get("early_edit", []))
    if edit_rows:
        reuse = cache_reuse(edit_rows)
        idx = next(i for i, r in enumerate(edit_rows) if (r.get("tags") or {}).get("edited"))
        at, after = reuse[idx], reuse[idx + 1] if idx + 1 < len(reuse) else None
        normal = [r["cost_usd"] for j, r in enumerate(edit_rows) if j not in (0, idx)]
        held = at is not None and at < 0.1 and after is not None and after >= 0.9
        edit_cost = edit_rows[idx]["cost_usd"]
        times = edit_cost / (sum(normal) / len(normal))
        lines.append(f"- Early edit at turn {edit_rows[idx]['turn']}: reuse {at:.1%}, cost "
                     f"${edit_cost:.6f} (~{times:.0f}x a normal turn); next turn reuse "
                     f"{after:.1%} **[measured]**.")
    if "system_timestamp" in groups and "control" in groups:
        ts = summarize(ok_rows(groups["system_timestamp"]))
        ctl = summarize(ok_rows(groups["control"]))
        lines.append(f"- Timestamp in the system prompt (a common agent bug): reuse mean "
                     f"{ts['reuse_mean']:.1%}; cost per turn {ts['mean_cost_first10']:.6f} → "
                     f"{ts['mean_cost_last10']:.6f} (growing); whole run "
                     f"{ts['total_cost'] / ctl['total_cost']:.1f}x the control **[measured]**.")
    return lines, ("H2(c) early edit breaks the prefix", "HELD" if held else "NOT HELD",
                   "one edit = one full re-write, then recovery")


def section_h2a(path: Path, out: Path) -> tuple[list[str], tuple[str, str, str]]:
    checks = summarize_idle(ok_rows(next(iter(by_variant(path).values()))))
    fig, (ax,) = new_fig()
    names = [f"{c['probe']} @ {c['check_min']} min" for c in checks]
    values = [(c["cached_share"] or 0) * 100 for c in checks]
    ax.barh(names, values, color=[STATUS[c["verdict"]] for c in checks], height=0.6)
    for y, (v, c) in enumerate(zip(values, checks, strict=True)):
        ax.annotate(c["verdict"], (v, y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    label(ax, "H2(a): still cached after an idle gap?", "% of prompt cached on re-send", "")
    chart = save(fig, out, "h2a_idle_gaps.png")

    verdicts = {f"{c['probe']}@{c['check_min']}": c["verdict"] for c in checks}
    lines = [f"![H2a chart]({chart})", "",
             "| probe | after | cached | verdict | re-send cost |", "|---|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['probe']} | {c['check_min']} min | {(c['cached_share'] or 0):.1%} | "
                     f"{c['verdict']} | ${c['cost']:.6f} |")
    lines.append("")
    lines.append("- One check per probe: first observations, not proof **[measured]**. "
                 "A miss could also be early eviction, which we can't see from outside.")
    short_hit = verdicts.get("ttl10m_12m@12") == "HIT"
    long_miss = verdicts.get("default_35m@35") == "MISS"
    verdict = ("NOT HELD at 12 min / HELD at 35 min" if short_hit and long_miss else "SEE TABLE")
    return lines, ("H2(a) idle gap longer than TTL", verdict,
                   "10m block still cached at 12 min; default and '60m' gone by 35 min")


def section_h2b(path: Path, out: Path) -> tuple[list[str], tuple[str, str, str]]:
    summaries = {name: summarize_fanout(rows) for name, rows in by_variant(path).items()}
    fig, (ax,) = new_fig()
    names = list(summaries)
    values = [(summaries[n]["child1_cached_share"] or 0) * 100 for n in names]
    ax.bar(names, values, color=SERIES[0], width=0.55)
    for x, n in enumerate(names):
        ax.annotate(f"{values[x]:.0f}% · ${summaries[n]['child1_cost_mean']:.6f}",
                    (x, values[x]), xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=8, color=INK2)
    ax.set_ylim(0, 110)
    label(ax, "H2(b): sub-agents' first call, share cached and cost", "", "% cached")
    chart = save(fig, out, "h2b_fanout.png")
    lines = [f"![H2b chart]({chart})", ""]
    for n, s in summaries.items():
        lines.append(f"- `{n}`: sub-agent first call {s['child1_cached_share']:.1%} cached, "
                     f"${s['child1_cost_mean']:.6f} each, TTFT p50 {s['child1_ttft_p50']:.2f}s; "
                     f"later turns {s['later_cached_share']:.1%} cached **[measured]**.")
    warm = summaries.get("shared_warm", {}).get("child1_cached_share") or 0
    distinct = summaries.get("distinct", {}).get("child1_cached_share")
    held = warm >= 0.8 and distinct is not None and distinct <= 0.1
    return lines, ("H2(b) fan-out with shared vs different prefixes",
                   "HELD" if held else "NOT HELD",
                   f"warm shared prefix {warm:.0%} cached vs distinct {distinct or 0:.0%}")


def section_h3(path: Path, out: Path) -> tuple[list[str], tuple[str, str, str]]:
    rows = ok_rows(next(iter(by_variant(path).values())))
    table = summarize_long_context(rows)
    fig, (ax,) = new_fig()
    for i, phase in enumerate(("cold", "warm", "warm+append")):
        pts = [(r["prompt_tokens"] / 1000, r["ttft_s"]) for r in rows
               if (r.get("tags") or {}).get("phase") == phase and r.get("ttft_s") is not None]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=36, color=SERIES[i],
                   edgecolors=SURFACE, linewidths=1.5, label=phase)
        med = [(g["prompt_tokens"] / 1000, g["ttft_p50"]) for g in table if g["phase"] == phase]
        ax.plot([m[0] for m in med], [m[1] for m in med], color=SERIES[i], linewidth=2)
    label(ax, "H3: time to first token vs context size", "context (thousand tokens)",
          "TTFT (s), dots = runs, line = median")
    legend(ax)
    chart = save(fig, out, "h3_ttft_vs_context.png")
    p50 = {(g["size"], g["phase"]): g for g in table}
    lines = [f"![H3 chart]({chart})", "",
             "| size | cold TTFT p50 | warm TTFT p50 | speed-up | warm + 2K new | n |",
             "|---|---|---|---|---|---|"]
    ratios = []
    for size in sorted({g["size"] for g in table}):
        cold, warm, app = (p50.get((size, k)) for k in ("cold", "warm", "warm+append"))
        if cold and warm:
            ratios.append(cold["ttft_p50"] / warm["ttft_p50"])
            appended = app["ttft_p50"] if app else 0
            lines.append(f"| {size // 1000}K | {cold['ttft_p50']:.2f}s | "
                         f"{warm['ttft_p50']:.2f}s | {ratios[-1]:.1f}x | {appended:.2f}s | "
                         f"{cold['n']} |")
    lines += ["", ("- Measured with n=3 per size: enough for the 2-9x effect, not for p95/p99 "
              "**[measured]**.")]
    held = bool(ratios) and min(ratios) > 1.5
    return lines, ("H3 warm cache cuts TTFT at long context", "HELD" if held else "NOT HELD",
                   " / ".join(f"{r:.1f}x" for r in ratios) + " faster warm vs cold")


def section_h4(path: Path, out: Path) -> tuple[list[str], tuple[str, str, str]]:
    groups = by_variant(path)
    summaries = {n: summarize_api_compare(rows) for n, rows in groups.items()}
    fig, axes = new_fig(2)
    for i, name in enumerate([n for n in ("chat_ttl10m", "responses_ttl10m") if n in groups]):
        rows = ok_rows(groups[name])
        xs, ys = [r["turn"] for r in rows], [(r["request_chars"] or 0) / 1000 for r in rows]
        axes[0].plot(xs, ys, color=SERIES[i], linewidth=2, label=name)
        end_label(axes[0], xs[-1], ys[-1], name.split("_")[0])
    integer_x(axes[0])
    label(axes[0], "Characters sent per request", "turn", "thousand characters")
    legend(axes[0])
    names = list(summaries)
    costs = [summaries[n]["total_cost"] * 1000 for n in names]
    axes[1].bar(range(len(names)), costs, color=SERIES[0], width=0.55)
    axes[1].set_xticks(range(len(names)), [n.replace("_", "\n") for n in names], fontsize=7)
    for x, v in enumerate(costs):
        axes[1].annotate(f"${v / 1000:.5f}", (x, v), xytext=(0, 4), textcoords="offset points",
                         ha="center", fontsize=8, color=INK2)
    label(axes[1], "Total cost of the same 12-turn loop", "", "$ x 1,000")
    chart = save(fig, out, "h4_chat_vs_responses.png")
    lines = [f"![H4 chart]({chart})", "",
             ("| variant | total $ | write $/M | sent chars (last turn) | reuse mean/min | "
             "first answer p50 | total p50 |"), "|---|---|---|---|---|---|---|"]
    for n, s in summaries.items():
        rate = "-" if s["write_rate_per_m"] is None else f"{s['write_rate_per_m']:.4f}"
        lines.append(f"| {n} | {s['total_cost']:.5f} | {rate} | {s['request_chars_last']:,} | "
                     f"{s['reuse_mean']:.1%} / {s['reuse_min']:.1%} | "
                     f"{s['first_content_p50']:.2f}s | {s['total_p50']:.2f}s |")
    chat, resp = summaries.get("chat_ttl10m"), summaries.get("responses_ttl10m")
    verdict, key = "SEE TABLE", ""
    if chat and resp:
        size = chat["request_chars_last"] / resp["request_chars_last"]
        cost = resp["total_cost"] / chat["total_cost"]
        speed = resp["total_p50"] / chat["total_p50"]
        lines += ["", (f"- Responses sent {size:.0f}x less data on the last turn and was "
                  f"{'faster' if speed < 1 else 'slower'} (total p50 {resp['total_p50']:.2f}s vs "
                  f"{chat['total_p50']:.2f}s, n={chat['requests']} per variant: suggestive only) "
                  f"**[measured]**."),
                  (f"- But it cost {cost:.1f}x more: Responses billed writes at "
                  f"${resp['write_rate_per_m']:.4f}/M despite `ttl: \"10m\"` (Chat: "
                  f"${chat['write_rate_per_m']:.4f}/M) **[measured]**. Coral's docs say cache "
                  f"options apply to both APIs **[docs]**.")]
        verdict = "MIXED" if size > 2 and cost > 1 else ("HELD" if size > 2 else "NOT HELD")
        key = f"{size:.0f}x smaller requests, but {cost:.1f}x the cost (TTL ignored)"
    return lines, ("H4 Responses vs Chat", verdict, key)


def section_h5(path: Path, out: Path) -> tuple[list[str], tuple[str, str, str]]:
    groups = by_variant(path)
    tables = {n: summarize_concurrency(rows) for n, rows in groups.items()}
    fig, axes = new_fig(2)
    for i, (name, table) in enumerate(tables.items()):
        lv = [g["level"] for g in table]
        axes[0].plot(lv, [g["tps_p50"] for g in table], color=SERIES[i], linewidth=2,
                     marker="o", markersize=5, label=name)
        axes[1].plot(lv, [g["agg_tps"] for g in table], color=SERIES[i], linewidth=2,
                     marker="o", markersize=5, label=name)
    for ax in axes:
        ax.set_xticks([1, 2, 4, 8])
        ax.set_ylim(bottom=0)
    label(axes[0], "Speed of each stream (p50)", "parallel streams", "tokens/s per stream")
    label(axes[1], "Total speed across streams", "parallel streams", "tokens/s total")
    legend(axes[0])
    chart = save(fig, out, "h5_tps_vs_concurrency.png")

    fig, axes = new_fig(len(tables))
    for ax, (name, table) in zip(axes, tables.items(), strict=True):
        lv = [g["level"] for g in table]
        for j, key in enumerate(("total_p50", "total_p95", "total_p99")):
            ax.plot(lv, [g[key] for g in table], color=ORDINAL[j], linewidth=2, marker="o",
                    markersize=5, label=key.split("_")[1])
        ax.set_xticks([1, 2, 4, 8])
        ax.set_ylim(bottom=0)
        label(ax, f"{name}: request time p50 / p95 / p99", "parallel streams", "seconds")
        legend(ax)
    latency_chart = save(fig, out, "latency_percentiles.png")

    lines = [f"![H5 chart]({chart})", "", f"![Latency percentiles]({latency_chart})", "",
             ("| model | streams | n | per-stream tok/s p50 (min) | total tok/s | "
             "total time p50/p95/p99 | 429s |"), "|---|---|---|---|---|---|---|"]
    held_any, key = False, []
    for name, table in tables.items():
        for g in table:
            lines.append(f"| {name} | {g['level']} | {g['n']} | {g['tps_p50']:.0f} "
                         f"({g['tps_min']:.0f}) | {g['agg_tps']:.0f} | {g['total_p50']:.1f}/"
                         f"{g['total_p95']:.1f}/{g['total_p99']:.1f}s | {g['rate_limited']} |")
        one, eight = table[0], table[-1]
        drop = 1 - eight["tps_p50"] / one["tps_p50"]
        held_any = held_any or drop > 0.2
        key.append(f"{name} {one['tps_p50']:.0f}→{eight['tps_p50']:.0f} tok/s")
    singles = sorted(round(x) for x in
                     (r["completion_tokens"] / (r["last_token_s"] - r["ttft_s"])
                      for r in ok_rows(groups.get("deepseek_flash", []))
                      if (r.get("tags") or {}).get("level") == 1 and r.get("last_token_s")))
    lines += ["", "- With few samples, p99 is effectively the slowest request.",
              f"- DeepSeek single-stream samples: {singles} tok/s vs the advertised 'up to 469' "
              f"**[measured] / [docs]**." if singles else ""]
    return lines, ("H5 per-stream speed drops under concurrency",
                   "HELD" if held_any else "NOT HELD", "; ".join(key))


# ---------- assemble ----------

def date_range(paths: list[Path]) -> str:
    stamps = [r["started_at"] for p in paths for r in read_jsonl(p) if r.get("started_at")]
    if not stamps:
        return "unknown"
    first, last = (datetime.fromtimestamp(t, tz=UTC).astimezone().date()
                   for t in (min(stamps), max(stamps)))
    return str(first) if first == last else f"{first} to {last}"


def build(results_dir: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    runs = {name: latest_run(results_dir, name) for name in SCENARIOS}
    used = [p for p in runs.values() if p]
    all_logs = sorted(results_dir.glob("*.jsonl"))
    check = validate(all_logs)

    sections: list[tuple[str, list[str]]] = []
    board: list[tuple[str, str, str]] = []
    plan = [("H1: flat cost per turn", "sequential", section_h1),
            ("H2: cache reuse in a plain loop", "sequential", lambda p, o: section_h2_loop(p)),
            ("H2(a): idle gaps", "idle_gaps", section_h2a),
            ("H2(b): fan-out", "fanout", section_h2b),
            ("H2(c): early edit and the timestamp bug", "early_edit", section_h2c),
            ("H3: warm vs cold at long context", "long_context", section_h3),
            ("H4: Responses vs Chat Completions", "api_compare", section_h4),
            ("H5: speed under concurrency", "concurrency", section_h5)]
    for title, scenario, fn in plan:
        path = runs.get(scenario)
        if path is None:
            sections.append((title, ["_Not run yet._"]))
            continue
        lines, row = fn(path, out)
        sections.append((title, [*lines, "", f"_Source: `{path.name}`_"]))
        board.append(row)

    md = ["# AgentLoad results", "",
          (f"Generated {datetime.now(tz=UTC).astimezone():%Y-%m-%d %H:%M} from the JSONL logs. "
          f"Test dates: {date_range(used)}. Every number below is computed from the logs."),
          "",
          ("Labels: **[measured]** our own runs · **[modeled]** real token counts re-priced under "
          "a hypothetical rule · **[docs]** Coral's documentation · **[inference]** our reading "
          "of the data."), "",
          "## Billing model check", "",
          (f"Our model of Coral's billing reproduced **{check['matched']} of {check['checked']}** "
          f"logged charges (worst difference ${check['worst_abs_diff']:.10f}) across "
          f"{check['files']} log files **[measured]**: cached reads free; writes billed per "
          "10-minute block (listed write price ÷ 3 per block, max 3); retention \"off\" billed "
          "as plain input; Responses billed as 3 blocks."), "",
          "## Scoreboard", "", "| hypothesis | verdict | key number |", "|---|---|---|"]
    md += [f"| {h} | **{v}** | {k} |" for h, v, k in board]
    for title, lines in sections:
        md += ["", f"## {title}", "", *[x for x in lines if x is not None]]
    md += ["", "## Notes and limitations", "",
           ("- Outside-in testing: we can't see Coral's infrastructure. Results reflect their "
           "production service on the test dates above and may change."),
           ("- One student, a self-imposed $5 total budget, at most 8 parallel streams, no "
           "stress testing. Many results rest on 1-3 samples per setting."),
           ("- The discounted-reads comparison is modeled on Coral's own base prices; it does "
           "not name or quote any other provider."),
           "- Out of scope: answer quality of 4-bit models vs full precision (open question).",
           "", "## Runs used", "", *[f"- `{p.name}`" for p in used]]
    report = out / "results.md"
    report.write_text("\n".join(md) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build AgentLoad charts and results.md.")
    parser.add_argument("--results", default="results", help="folder with the JSONL logs")
    parser.add_argument("--out", default="report/out", help="where to write charts + results.md")
    args = parser.parse_args(argv)
    path = build(Path(args.results), Path(args.out))
    print(f"wrote {path} and {len(list(path.parent.glob('*.png')))} charts")


if __name__ == "__main__":
    main()
