"""Scenario runner: load a YAML scenario and replay it against the API.

Usage: PYTHONPATH=. python -m agentload.runner scenarios/<name>.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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
from agentload.budget import BudgetExceeded, BudgetGuard
from agentload.client import DEFAULT_MODEL, AgentLoadClient, RequestResult
from agentload.exporter import Exporter, ExportingWriter
from agentload.metrics import JsonlWriter, new_run_id
from agentload.workload import ToolOutputGenerator, new_run_nonce, system_prompt

MAX_RUN_BUDGET_USD = 1.5
MAX_CONCURRENCY = 8  # etiquette: never more than 8 parallel streams
KNOWN_KEYS = {"name", "mode", "model", "turns", "tool_tokens", "max_tokens", "budget_usd",
              "variants", "description"}


@dataclass
class Variant:
    name: str
    extra_body: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)  # per-variant mode settings


@dataclass
class Scenario:
    name: str
    mode: str
    model: str = DEFAULT_MODEL
    turns: int = 1
    tool_tokens: tuple[int, int] = (1000, 4000)
    max_tokens: int = 1000
    budget_usd: float = 1.0
    variants: list[Variant] = field(default_factory=lambda: [Variant("default")])
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)  # mode-specific settings


def load_scenario(path: Path | str) -> Scenario:
    raw = yaml.safe_load(Path(path).read_text())
    lo, hi = (int(x) for x in raw.get("tool_tokens", (1000, 4000)))
    if not 0 < lo <= hi:
        raise ValueError("tool_tokens must be [min, max] with 0 < min <= max")
    budget = float(raw.get("budget_usd", 1.0))
    if not 0 < budget <= MAX_RUN_BUDGET_USD:
        raise ValueError(f"budget_usd must be in (0, {MAX_RUN_BUDGET_USD}]")
    mode = raw.get("mode", "sequential")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; known: {sorted(MODES)}")
    variants = [
        Variant(v["name"], v.get("extra_body") or {},
                {k: x for k, x in v.items() if k not in ("name", "extra_body")})
        for v in raw.get("variants") or [{"name": "default"}]
    ]
    return Scenario(
        name=raw["name"], mode=mode, model=raw.get("model", DEFAULT_MODEL),
        turns=int(raw.get("turns", 1)), tool_tokens=(lo, hi),
        max_tokens=int(raw.get("max_tokens", 1000)), budget_usd=budget, variants=variants,
        description=raw.get("description", ""),
        params={k: v for k, v in raw.items() if k not in KNOWN_KEYS},
    )


def tool_tokens_for_turn(turn: int, lo: int, hi: int) -> int:
    """Deterministic spread of tool-output sizes across [lo, hi]."""
    return lo if lo == hi else lo + (turn * 7919) % (hi - lo + 1)


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


ROW_HEADER = (f"{'call':<20} {'prompt':>7} {'cached':>7} {'hit%':>6} {'reuse%':>7} "
              f"{'ttft':>6} {'total':>6} {'out':>5} {'cost $':>10}")


def print_row(label: str, r: RequestResult, reuse: float | None = None) -> None:
    hr = hit_rate(r.to_dict())
    print(f"{label:<20} {_fmt(r.prompt_tokens, 'd'):>7} {_fmt(r.cached_tokens, 'd'):>7} "
          f"{_fmt(hr and hr * 100, '.1f'):>6} {_fmt(reuse and reuse * 100, '.1f'):>7} "
          f"{_fmt(r.ttft_s, '.2f'):>6} {_fmt(r.total_s, '.2f'):>6} "
          f"{_fmt(r.completion_tokens, 'd'):>5} {_fmt(r.cost_usd, '.6f'):>10}")
    if r.error:
        print(f"  error: {r.error}")


async def run_sequential(client: AgentLoadClient, writer: JsonlWriter, sc: Scenario,
                         variant: Variant, gen: ToolOutputGenerator) -> list[RequestResult]:
    """One agent loop: each turn appends tool output, the model replies, history grows."""
    base_system = system_prompt(new_run_nonce())
    messages: list[dict[str, Any]] = [{"role": "system", "content": base_system}]
    edit_turn = variant.params.get("edit_at_turn")
    edit_index = int(variant.params.get("edit_message_index", 1))
    if edit_turn is not None and not 2 <= int(edit_turn) <= sc.turns:
        raise ValueError("edit_at_turn must be between 2 and turns")
    results: list[RequestResult] = []
    for turn in range(1, sc.turns + 1):
        size = tool_tokens_for_turn(turn, *sc.tool_tokens)
        messages.append({"role": "user", "content": gen.make(size, step=turn)})
        if variant.params.get("system_timestamp"):  # common agent bug: a clock in the prompt
            messages[0]["content"] = f"{base_system} Current time (ns): {time.time_ns()}"
        edited = edit_turn == turn
        if edited:  # change an early message: everything after it no longer matches the cache
            if not 1 <= edit_index < len(messages) - 1:
                raise ValueError("edit_message_index must point at an earlier message")
            messages[edit_index]["content"] = "[edited] " + messages[edit_index]["content"]
        r = await client.chat(
            messages, model=sc.model, max_tokens=sc.max_tokens,
            scenario=f"{sc.name}/{variant.name}", turn=turn, extra_body=variant.extra_body,
            tags={"variant": variant.name, "tool_tokens_target": size, "edited": edited,
                  "conversation": f"{sc.name}/{variant.name}"},
        )
        writer.write(r)
        results.append(r)
        reuse = cache_reuse([x.to_dict() for x in results[-2:]])[-1] if turn > 1 else None
        print_row(f"turn {turn}" + (" EDIT" if edited else ""), r, reuse)
        if r.error:
            break
        messages.append({"role": "assistant", "content": r.output_text})
    return results


async def run_long_context(client: AgentLoadClient, writer: JsonlWriter, sc: Scenario,
                           variant: Variant, gen: ToolOutputGenerator) -> list[RequestResult]:
    """Per size and repeat: a cold prompt, the same prompt warm, then warm plus a small append."""
    sizes = [int(s) for s in sc.params.get("context_sizes", [100_000])]
    repeats = int(sc.params.get("repeats", 1))
    delay = float(sc.params.get("warm_delay_s", 5))
    append_tokens = int(sc.params.get("append_tokens", 2000))
    question = "\n\nIn one short sentence: which file appears most often above?"
    results: list[RequestResult] = []

    async def call(messages: list[dict[str, Any]], phase: str, size: int,
                   rep: int) -> RequestResult:
        r = await client.chat(
            messages, model=sc.model, max_tokens=sc.max_tokens,
            scenario=f"{sc.name}/{variant.name}", turn=len(results) + 1,
            extra_body=variant.extra_body,
            tags={"variant": variant.name, "phase": phase, "size": size, "rep": rep},
        )
        writer.write(r)
        results.append(r)
        print_row(f"{size // 1000}K r{rep} {phase}", r)
        return r

    for rep in range(1, repeats + 1):
        for i, size in enumerate(sizes):
            step = rep * 100 + i
            messages = [
                {"role": "system", "content": system_prompt(new_run_nonce())},
                {"role": "user", "content": gen.make(size, step=step) + question},
            ]
            cold = await call(messages, "cold", size, rep)
            if cold.error:
                return results
            await asyncio.sleep(delay)
            warm = await call(messages, "warm", size, rep)
            if warm.error:
                return results
            followup = [*messages, {"role": "assistant", "content": warm.output_text},
                        {"role": "user", "content": gen.make(append_tokens, step=step + 50)
                         + question}]
            appended = await call(followup, "warm+append", size, rep)
            if appended.error:
                return results
    return results


def print_sequential_summary(rows_by_variant: dict[str, list[Row]]) -> None:
    print("\n=== summary ===")
    print(f"{'variant':<10} {'reqs':>4} {'err':>3} {'total $':>9} {'mean $/turn':>11} "
          f"{'first10 $':>10} {'last10 $':>10} {'context':>16} {'reuse mean/min':>15} "
          f"{'ttft p50/p95':>13}")
    for name, rows in rows_by_variant.items():
        s = summarize(rows)
        context = f"{_fmt(s['context_first'], 'd')}->{_fmt(s['context_last'], 'd')}"
        reuse = (f"{_fmt(s['reuse_mean'] and s['reuse_mean'] * 100, '.1f')}/"
                 f"{_fmt(s['reuse_min'] and s['reuse_min'] * 100, '.1f')}%")
        ttft = f"{_fmt(s['ttft_p50'], '.2f')}/{_fmt(s['ttft_p95'], '.2f')}s"
        print(f"{name:<10} {s['requests']:>4} {s['errors']:>3} {s['total_cost']:>9.5f} "
              f"{_fmt(s['mean_cost_warm'], '.6f'):>11} {_fmt(s['mean_cost_first10'], '.6f'):>10} "
              f"{_fmt(s['mean_cost_last10'], '.6f'):>10} {context:>16} {reuse:>15} {ttft:>13}")


def print_long_context_summary(rows_by_variant: dict[str, list[Row]]) -> None:
    for name, rows in rows_by_variant.items():
        table = summarize_long_context(rows)
        print(f"\n=== summary: {name} ===")
        print(f"{'size':>6} {'phase':<12} {'n':>2} {'prompt':>8} {'cached%':>8} {'ttft p50':>8} "
              f"{'min-max':>13} {'cost $':>9}")
        for g in table:
            spread = f"{_fmt(g['ttft_min'], '.2f')}-{_fmt(g['ttft_max'], '.2f')}"
            share = g["cached_share"]
            print(f"{g['size'] // 1000:>5}K {g['phase']:<12} {g['n']:>2} "
                  f"{_fmt(g['prompt_tokens'], '.0f'):>8} "
                  f"{_fmt(share and share * 100, '.1f'):>8} {_fmt(g['ttft_p50'], '.2f'):>8} "
                  f"{spread:>13} {_fmt(g['cost_mean'], '.5f'):>9}")
        p50 = {(g["size"], g["phase"]): g["ttft_p50"] for g in table}
        for size in sorted({g["size"] for g in table}):
            cold, warm = p50.get((size, "cold")), p50.get((size, "warm"))
            if cold and warm:
                print(f"{size // 1000}K: cold TTFT / warm TTFT = {cold / warm:.1f}x")


FANOUT_STYLES = ("shared_warm", "shared_cold", "distinct")


async def run_fanout(client: AgentLoadClient, writer: JsonlWriter, sc: Scenario,
                     variant: Variant, gen: ToolOutputGenerator) -> list[RequestResult]:
    """A main agent hands a big shared context to N parallel sub-agents (H2b)."""
    prefix_tokens = int(sc.params.get("prefix_tokens", 30_000))
    children = int(sc.params.get("children", 4))
    child_turns = int(sc.params.get("child_turns", 3))
    task_tokens = int(sc.params.get("task_tokens", 2000))
    repeats = int(sc.params.get("repeats", 1))
    style = variant.params.get("prefix", "shared_warm")
    if style not in FANOUT_STYLES:
        raise ValueError(f"prefix must be one of {FANOUT_STYLES}")
    if not 1 <= children <= MAX_CONCURRENCY:
        raise ValueError(f"children must be 1-{MAX_CONCURRENCY}")
    results: list[RequestResult] = []
    counter = itertools.count(1)

    async def call(messages: list[dict[str, Any]], label: str,
                   tags: dict[str, Any]) -> RequestResult:
        r = await client.chat(
            messages, model=sc.model, max_tokens=sc.max_tokens,
            scenario=f"{sc.name}/{variant.name}", turn=next(counter),
            extra_body=variant.extra_body, tags={"variant": variant.name, "style": style, **tags},
        )
        writer.write(r)
        results.append(r)
        print_row(label, r)
        return r

    def shared_context(nonce: str) -> list[dict[str, Any]]:
        return [{"role": "system", "content": system_prompt(nonce)},
                {"role": "user", "content": "Repository context:\n" + gen.make(prefix_tokens, step=7)}]

    async def child(rep: int, idx: int, nonce: str) -> None:
        msgs = shared_context(nonce)
        for t in range(1, child_turns + 1):
            task = gen.make(task_tokens, step=rep * 1000 + idx * 10 + t)
            msgs.append({"role": "user", "content": f"Sub-agent {idx}, step {t}:\n{task}"})
            r = await call(msgs, f"r{rep} child{idx} t{t}",
                           {"phase": "child", "child": idx, "child_turn": t, "rep": rep,
                            "conversation": f"{variant.name}/r{rep}/c{idx}"})
            if r.error:
                return
            msgs.append({"role": "assistant", "content": r.output_text})

    for rep in range(1, repeats + 1):
        nonce = new_run_nonce()
        if style == "shared_warm":
            parent = [*shared_context(nonce),
                      {"role": "user", "content": "In one sentence: summarize the repository."}]
            p = await call(parent, f"r{rep} parent", {"phase": "parent", "rep": rep})
            if p.error:
                return results
        nonces = [new_run_nonce() if style == "distinct" else nonce for _ in range(children)]
        await asyncio.gather(*(child(rep, i + 1, nonces[i]) for i in range(children)))
    return results


def print_fanout_summary(rows_by_variant: dict[str, list[Row]]) -> None:
    print("\n=== summary ===")
    print(f"{'variant':<12} {'reqs':>4} {'err':>3} {'total $':>9} {'child t1 cached%':>16} "
          f"{'child t1 $':>11} {'child t1 ttft':>13} {'later cached%':>13}")
    for name, rows in rows_by_variant.items():
        s = summarize_fanout(rows)
        c1, lt = s["child1_cached_share"], s["later_cached_share"]
        print(f"{name:<12} {s['requests']:>4} {s['errors']:>3} {s['total_cost']:>9.5f} "
              f"{_fmt(c1 and c1 * 100, '.1f'):>16} {_fmt(s['child1_cost_mean'], '.6f'):>11} "
              f"{_fmt(s['child1_ttft_p50'], '.2f'):>13} {_fmt(lt and lt * 100, '.1f'):>13}")


MAX_IDLE_CHECK_MIN = 60


async def run_idle_gaps(client: AgentLoadClient, writer: JsonlWriter, sc: Scenario,
                        variant: Variant, gen: ToolOutputGenerator) -> list[RequestResult]:
    """H2(a): write each probe's prompt once, wait, re-send it and see if it is still cached."""
    probes = sc.params.get("probes") or []
    size = int(sc.params.get("prompt_tokens", 10_000))
    seconds_per_min = float(sc.params.get("seconds_per_min", 60.0))  # tests shrink this
    if not probes:
        raise ValueError("idle_gaps needs at least one probe")
    for p in probes:
        checks = p.get("checks_min") or []
        if not checks or not all(0 < m <= MAX_IDLE_CHECK_MIN for m in checks):
            raise ValueError(f"probe {p.get('name')}: checks_min must be in 1-{MAX_IDLE_CHECK_MIN}")
    results: list[RequestResult] = []
    counter = itertools.count(1)

    async def call(msgs: list[dict[str, Any]], probe: dict[str, Any],
                   minute: float) -> RequestResult:
        r = await client.chat(
            msgs, model=sc.model, max_tokens=sc.max_tokens,
            scenario=f"{sc.name}/{probe['name']}", turn=next(counter),
            extra_body=probe.get("extra_body") or {},
            tags={"variant": variant.name, "probe": probe["name"], "check_min": minute,
                  "phase": "write" if minute == 0 else "check"},
        )
        writer.write(r)
        results.append(r)
        print_row(f"{probe['name']} @{minute}m", r)
        return r

    prompts: dict[str, list[dict[str, Any]]] = {}
    written_at: dict[str, float] = {}
    for i, p in enumerate(probes):
        msgs = [{"role": "system", "content": system_prompt(new_run_nonce())},
                {"role": "user", "content": gen.make(size, step=500 + i)
                 + "\n\nOne word: any failures?"}]
        prompts[p["name"]] = msgs
        w = await call(msgs, p, 0)
        if w.error:
            return results
        written_at[p["name"]] = time.monotonic()

    schedule = sorted(
        ((written_at[p["name"]] + m * seconds_per_min, m, p)
         for p in probes for m in p["checks_min"]),
        key=lambda item: item[0],
    )
    for due, minute, p in schedule:
        wait = due - time.monotonic()
        if wait > 0:
            print(f"... waiting {wait / 60:.1f} min for {p['name']} @{minute}m")
            await asyncio.sleep(wait)
        await call(prompts[p["name"]], p, minute)
    return results


def print_idle_summary(rows_by_variant: dict[str, list[Row]]) -> None:
    print("\n=== summary ===")
    print(f"{'probe':<16} {'after':>6} {'cached%':>8} {'verdict':>8} {'ttft':>6} {'cost $':>10}")
    for rows in rows_by_variant.values():
        for c in summarize_idle(rows):
            share = c["cached_share"]
            print(f"{c['probe']:<16} {c['check_min']:>5}m {_fmt(share and share * 100, '.1f'):>8} "
                  f"{c['verdict']:>8} {_fmt(c['ttft'], '.2f'):>6} {_fmt(c['cost'], '.6f'):>10}")


CONCURRENCY_PROMPT = ("Write a detailed, roughly 500-word technical explanation of how a hash map "
                      "handles collisions, with a short example.")


async def run_concurrency(client: AgentLoadClient, writer: JsonlWriter, sc: Scenario,
                          variant: Variant, gen: ToolOutputGenerator) -> list[RequestResult]:
    """H5: N identical-shape requests at once; short unique prompts, long answers."""
    levels = [int(x) for x in sc.params.get("levels", [1, 2, 4, 8])]
    rounds = int(sc.params.get("rounds", 3))
    pause = float(sc.params.get("pause_s", 2))
    model = variant.params.get("model", sc.model)
    prompt = sc.params.get("prompt", CONCURRENCY_PROMPT)
    if not levels or not all(1 <= lv <= MAX_CONCURRENCY for lv in levels):
        raise ValueError(f"levels must be between 1 and {MAX_CONCURRENCY}")
    results: list[RequestResult] = []
    counter = itertools.count(1)

    async def stream(level: int, rnd: int, idx: int) -> None:
        msgs = [{"role": "system",
                 "content": f"Run {new_run_nonce()}. You are a helpful technical writer."},
                {"role": "user", "content": prompt}]
        r = await client.chat(
            msgs, model=model, max_tokens=sc.max_tokens,
            scenario=f"{sc.name}/{variant.name}", turn=next(counter),
            extra_body=variant.extra_body,
            tags={"variant": variant.name, "model": model, "level": level, "round": rnd,
                  "stream": idx},
        )
        writer.write(r)
        results.append(r)
        tps = r.decode_tokens_per_s()
        print_row(f"c{level} r{rnd} s{idx} " + (f"{tps:.0f}t/s" if tps else "-"), r)

    for level in levels:
        for rnd in range(1, rounds + 1):
            await asyncio.gather(*(stream(level, rnd, i + 1) for i in range(level)))
            await asyncio.sleep(pause)
    return results


def print_concurrency_summary(rows_by_variant: dict[str, list[Row]]) -> None:
    for name, rows in rows_by_variant.items():
        print(f"\n=== summary: {name} ===")
        print(f"{'streams':>7} {'n':>3} {'err':>3} {'429':>3} {'ttft p50/p95':>13} "
              f"{'total p50/p95/p99':>18} {'tok/s/stream p50 (min)':>23} {'total tok/s':>11} "
              f"{'tok/chunk':>9}")
        for g in summarize_concurrency(rows):
            ttft = f"{_fmt(g['ttft_p50'], '.2f')}/{_fmt(g['ttft_p95'], '.2f')}"
            tot = (f"{_fmt(g['total_p50'], '.1f')}/{_fmt(g['total_p95'], '.1f')}/"
                   f"{_fmt(g['total_p99'], '.1f')}")
            tps = f"{_fmt(g['tps_p50'], '.0f')} ({_fmt(g['tps_min'], '.0f')})"
            print(f"{g['level']:>7} {g['n']:>3} {g['errors']:>3} {g['rate_limited']:>3} "
                  f"{ttft:>13} {tot:>18} {tps:>23} {_fmt(g['agg_tps'], '.0f'):>11} "
                  f"{_fmt(g['tokens_per_chunk'], '.1f'):>9}")


async def run_api_compare(client: AgentLoadClient, writer: JsonlWriter, sc: Scenario,
                          variant: Variant, gen: ToolOutputGenerator) -> list[RequestResult]:
    """H4: the same agent loop via Chat (history re-sent) or Responses (previous_response_id)."""
    api = variant.params.get("api", "chat")
    if api not in ("chat", "responses"):
        raise ValueError("api must be 'chat' or 'responses'")
    repeat_instructions = bool(variant.params.get("repeat_instructions", True))
    instructions = system_prompt(new_run_nonce())
    messages: list[dict[str, Any]] = [{"role": "system", "content": instructions}]
    previous_id: str | None = None
    history_chars = len(instructions)
    results: list[RequestResult] = []
    for turn in range(1, sc.turns + 1):
        size = tool_tokens_for_turn(turn, *sc.tool_tokens)
        text = gen.make(size, step=turn)
        common: dict[str, Any] = {
            "model": sc.model, "max_tokens": sc.max_tokens,
            "scenario": f"{sc.name}/{variant.name}", "turn": turn,
            "extra_body": variant.extra_body,
            "tags": {"variant": variant.name, "api": api, "tool_tokens_target": size,
                     "conversation": f"{sc.name}/{variant.name}"},
        }
        if api == "chat":
            messages.append({"role": "user", "content": text})
            r = await client.chat(messages, **common)
        else:
            history_chars += len(text)
            send = instructions if (repeat_instructions or turn == 1) else None
            r = await client.respond(text, history_chars=history_chars, instructions=send,
                                     previous_response_id=previous_id, **common)
        writer.write(r)
        results.append(r)
        reuse = cache_reuse([x.to_dict() for x in results[-2:]])[-1] if turn > 1 else None
        print_row(f"turn {turn} {(r.request_chars or 0) / 1000:.1f}k", r, reuse)
        if r.error:
            break
        if api == "chat":
            messages.append({"role": "assistant", "content": r.output_text})
        else:
            previous_id = r.response_id
            history_chars += len(r.output_text)
    return results


def print_api_compare_summary(rows_by_variant: dict[str, list[Row]]) -> None:
    print("\n=== summary ===")
    print(f"{'variant':<28} {'total $':>9} {'write $/M':>9} {'input $':>8} "
          f"{'reuse mean/min':>15} {'sent chars mean/last':>21} {'1st answer p50':>14} "
          f"{'total p50':>9}")
    for name, rows in rows_by_variant.items():
        s = summarize_api_compare(rows)
        reuse = (f"{_fmt(s['reuse_mean'] and s['reuse_mean'] * 100, '.1f')}/"
                 f"{_fmt(s['reuse_min'] and s['reuse_min'] * 100, '.1f')}%")
        sent = f"{_fmt(s['request_chars_mean'], '.0f')}/{_fmt(s['request_chars_last'], 'd')}"
        print(f"{name:<28} {s['total_cost']:>9.5f} {_fmt(s['write_rate_per_m'], '.4f'):>9} "
              f"{s['input_cost']:>8.5f} {reuse:>15} {sent:>21} "
              f"{_fmt(s['first_content_p50'], '.2f'):>14} {_fmt(s['total_p50'], '.2f'):>9}")


Mode = Callable[[AgentLoadClient, JsonlWriter, Scenario, Variant, ToolOutputGenerator],
                Awaitable[list[RequestResult]]]


@dataclass(frozen=True)
class ModeSpec:
    run: Mode
    summary: Callable[[dict[str, list[Row]]], None]


MODES: dict[str, ModeSpec] = {
    "sequential": ModeSpec(run_sequential, print_sequential_summary),
    "long_context": ModeSpec(run_long_context, print_long_context_summary),
    "fanout": ModeSpec(run_fanout, print_fanout_summary),
    "idle_gaps": ModeSpec(run_idle_gaps, print_idle_summary),
    "concurrency": ModeSpec(run_concurrency, print_concurrency_summary),
    "api_compare": ModeSpec(run_api_compare, print_api_compare_summary),
}


async def run_scenario(path: Path | str, metrics_port: int | None = None,
                       hold_s: float = 15.0, warmup_s: float = 6.0) -> None:
    sc = load_scenario(path)
    spec = MODES[sc.mode]
    run_id = new_run_id(sc.name)
    log_path = Path("results") / f"{run_id}.jsonl"
    writer: JsonlWriter = JsonlWriter(log_path, run_id=run_id)
    if metrics_port:
        exporter = Exporter()
        exporter.serve(metrics_port)
        writer = ExportingWriter(log_path, run_id, exporter)
        print(f"live metrics: http://localhost:{metrics_port}/metrics")
        for v in sc.variants:
            exporter.prepare(f"{sc.name}/{v.name}", v.params.get("model", sc.model),
                             v.params.get("api", "chat"))
        print(f"waiting {warmup_s:.0f}s so Prometheus scrapes the zeroed counters first...")
        await asyncio.sleep(warmup_s)
    budget = BudgetGuard(run_cap_usd=sc.budget_usd)
    client = AgentLoadClient(budget)
    gen = ToolOutputGenerator()
    print(f"scenario: {sc.name} | mode: {sc.mode} | model: {sc.model} | "
          f"run cap ${sc.budget_usd:.2f}")
    rows_by_variant: dict[str, list[Row]] = {}
    try:
        for variant in sc.variants:
            print(f"\n=== {sc.name} / {variant.name} | extra_body={variant.extra_body} ===")
            print(ROW_HEADER)
            results = await spec.run(client, writer, sc, variant, gen)
            rows_by_variant[variant.name] = [r.to_dict() for r in results]
    except BudgetExceeded as e:
        print(f"\nSTOPPED by budget guard: {e}")
    spec.summary(rows_by_variant)
    print(f"\nlog: {writer.path} | this run ${budget.run_spent:.4f} | "
          f"all-time ${budget.total_spent:.4f}")
    if metrics_port:
        print(f"keeping metrics up {hold_s:.0f}s so Prometheus gets a final scrape...")
        await asyncio.sleep(hold_s)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run an AgentLoad scenario.")
    parser.add_argument("scenario", help="path to a scenario YAML file")
    parser.add_argument("--metrics-port", type=int, default=None,
                        help="serve live Prometheus metrics on this port (e.g. 9108)")
    parser.add_argument("--hold-s", type=float, default=15.0,
                        help="seconds to keep metrics up after the run")
    parser.add_argument("--warmup-s", type=float, default=6.0,
                        help="seconds to wait before the first request (one scrape)")
    args = parser.parse_args(argv)
    asyncio.run(run_scenario(args.scenario, args.metrics_port, args.hold_s, args.warmup_s))


if __name__ == "__main__":
    main()
