# AgentLoad results

Generated 2026-10-06 02:57 from the JSONL logs. Test dates: 2026-10-05 to 2026-10-06. Every number below is computed from the logs.

Labels: **[measured]** our own runs · **[modeled]** real token counts re-priced under a hypothetical rule · **[docs]** Coral's documentation · **[inference]** our reading of the data.

## Billing model check

Our model of Coral's billing reproduced **1054 of 1054** logged charges (worst difference $0.0000000033) across 23 log files **[measured]**: cached reads free; writes billed per 10-minute block (listed write price ÷ 3 per block, max 3); retention "off" billed as plain input; Responses billed as 3 blocks.

## Scoreboard

| hypothesis | verdict | key number |
|---|---|---|
| H1 flat cost per turn | **HELD** | Coral ~$0 vs discounted +$0.0015 per turn per +100K context; 10m TTL saves 61%-63% (3 runs) |
| H2 plain loop keeps the cache | **HELD** | reuse 99.9%+ mean in 6 conversations (raw hit rate as low as 55%) |
| H2(a) idle gap longer than TTL | **NOT HELD** | 10m block still cached at 30 min; default gone at 35 min |
| H2(b) fan-out with shared vs different prefixes | **HELD** | warm shared prefix 91% cached vs distinct 0% |
| H2(c) early edit breaks the prefix | **HELD** | one edit = one full re-write, then recovery |
| H3 warm cache cuts TTFT at long context | **HELD** | 3.3x / 5.7x / 7.7x faster warm vs cold |
| H4 Responses vs Chat | **MIXED** | 15x smaller requests, but 2.4-2.6x the cost in 4 run(s) (writes billed as 3 blocks even with a 10m TTL) |
| H5 per-stream speed drops under concurrency | **HELD** | glm_flash 284→188 tok/s; deepseek_flash 408→258 tok/s |

## H1: flat cost per turn

![H1 chart](h1_cost_per_turn.png)

- Tool outputs alternate between ~1K and ~3K tokens, so cost zigzags turn to turn; the thick lines are 5-turn averages.

- `default`: context 2,987 → 128,119 tokens. Coral cost per turn, first 10 → last 10: $0.000585 → $0.000578 **[measured]**. Discounted-reads rule on the same tokens: $0.000700 → $0.002186 **[modeled]**. Cost added per +100K tokens of context: Coral +$0.000002, discounted +$0.001497. Whole run: $0.03096 vs $0.07242 (2.3x).
- `ttl10m`: context 2,990 → 128,639 tokens. Coral cost per turn, first 10 → last 10: $0.000229 → $0.000210 **[measured]**. Discounted-reads rule on the same tokens: $0.000718 → $0.002204 **[modeled]**. Cost added per +100K tokens of context: Coral -$0.000017, discounted +$0.001490. Whole run: $0.01162 vs $0.07319 (6.3x).
- Same loop with `ttl: "10m"` instead of the default: 62% cheaper overall **[measured]**.
- Repeated 3 times: the 10m TTL saved 61% / 63% / 62% per run **[measured]**. The chart and numbers above are the latest run.

_Source: `20261005-225939-sequential-4382da.jsonl`, `20261006-015350-sequential-1cd41e.jsonl`, `20261006-015743-sequential-9f90c6.jsonl`_

## H2: cache reuse in a plain loop

- Plain 50-turn loop, 6 conversations across 3 run(s) (default and ttl10m): cache reuse mean 99.9%-99.9%, lowest single turn 98.4% **[measured]**.
- Raw hit rate (cached ÷ prompt) dipped to 55.5% even though the cache was working: it mostly reflects how long the conversation is **[inference]**.

_Source: `20261005-225939-sequential-4382da.jsonl`, `20261006-015350-sequential-1cd41e.jsonl`, `20261006-015743-sequential-9f90c6.jsonl`_

## H2(a): idle gaps

![H2a chart](h2a_idle_gaps.png)

| probe | after | cached | verdict | re-send cost |
|---|---|---|---|---|
| default_12m | 12 min | 99.9% | HIT | $0.000030 |
| default_20m | 20 min | 99.9% | HIT | $0.000057 |
| default_25m | 25 min | 99.8% | HIT | $0.000049 |
| default_30m | 30 min | 99.8% | HIT | $0.000043 |
| default_35m | 35 min | 0.0% | MISS | $0.002412 |
| off_2m | 2 min | 99.8% | HIT | $0.000023 |
| off_12m | 12 min | 99.8% | HIT | $0.000032 |
| off_20m | 20 min | 99.8% | HIT | $0.000044 |
| off_30m | 30 min | 99.8% | HIT | $0.000044 |
| ttl10m_5m | 5 min | 99.9% | HIT | $0.000027 |
| ttl10m_refresh | 8 min | 99.9% | HIT | $0.000039 |
| ttl10m_12m | 12 min | 99.8% | HIT | $0.000030 |
| ttl10m_refresh | 16 min | 99.9% | HIT | $0.000059 |
| ttl10m_20m | 20 min | 99.8% | HIT | $0.000042 |
| ttl10m_25m | 25 min | 99.8% | HIT | $0.000051 |
| ttl10m_30m | 30 min | 99.9% | HIT | $0.000027 |
| ttl60m_25m | 25 min | 99.8% | HIT | $0.000036 |
| ttl60m_35m | 35 min | 0.0% | MISS | $0.002339 |

- One check per probe and time, across 2 runs: observations, not proof **[measured]**. A miss could also be early eviction, which we can't see from outside.
- Cost to write the ~10K-token prompt, by setting: `default` $0.002356, `off` $0.001566, `ttl10m` $0.000813, `ttl60m` $0.002333 **[measured]**.
- A `ttl: "10m"` block was still cached after 30 min; no check missed before 35 min, when `default` and `ttl60m` missed **[measured]**. So the TTL changed what we paid, not how long the cache lived in these runs **[inference]**.

_Source: `20261005-231533-idle_gaps-8d4480.jsonl`, `20261006-020322-idle_followup-954ff7.jsonl`_

## H2(b): fan-out

![H2b chart](h2b_fanout.png)

- Setup: a ~30K-token shared context handed to 4 parallel sub-agents, 3 turns each. `shared_warm`: the main agent caches the context first. `shared_cold`: all 4 start at the same moment, nothing cached yet. `distinct`: each gets a different prefix (control).

- `shared_warm`: sub-agent first call 91.4% cached, $0.000324 each, TTFT p50 1.07s; later turns 93.7% cached **[measured]**.
- `shared_cold`: sub-agent first call 59.8% cached, $0.001145 each, TTFT p50 2.05s; later turns 91.2% cached **[measured]**.
- `distinct`: sub-agent first call 0.0% cached, $0.002620 each, TTFT p50 2.11s; later turns 94.1% cached **[measured]**.

_Source: `20261005-231221-fanout-642d3c.jsonl`_

## H2(c): early edit and the timestamp bug

![H2 chart](h2_cache_over_time.png)

- Early edit at turn 11: reuse 0.0%, cost $0.002111 (~8x a normal turn); next turn reuse 99.8% **[measured]**.
- Timestamp in the system prompt (a common agent bug): reuse mean 0.3%; cost per turn $0.001321 → $0.003025 (growing); whole run 7.8x the control **[measured]**.

_Source: `20261005-235235-early_edit-195b1b.jsonl`_

## H3: warm vs cold at long context

![H3 chart](h3_ttft_vs_context.png)

| size | cold TTFT p50 | warm TTFT p50 | speed-up | warm + 2K new | n |
|---|---|---|---|---|---|
| 100K | 3.81s | 1.14s | 3.3x | 1.13s | 6 |
| 250K | 7.19s | 1.27s | 5.7x | 2.22s | 6 |
| 500K | 15.71s | 2.05s | 7.7x | 3.34s | 6 |

- Pooled over 2 runs, n=6 per size and phase: enough for the effect, not for p95/p99 **[measured]**.
- Cold TTFT was steady run to run; warm TTFT was noisier, so the exact speed-up moves between runs **[measured]**.

_Source: `20261005-230550-long_context-cbfeee.jsonl`, `20261006-014912-long_context-446e9e.jsonl`_

## H4: Responses vs Chat Completions

![H4 chart](h4_chat_vs_responses.png)

| variant | total $ | write $/M | sent chars (last turn) | reuse mean/min | first answer p50 | total p50 |
|---|---|---|---|---|---|---|
| chat_ttl10m | 0.00338 | 0.0767 | 98,276 | 99.6% / 98.6% | 0.73s | 0.86s |
| responses_ttl10m | 0.00814 | 0.2300 | 6,685 | 99.7% / 98.6% | 0.78s | 1.02s |
| responses_ttl10m_instr_once | 0.00897 | 0.2300 | 6,514 | 99.5% / 96.2% | 0.85s | 1.57s |
| responses_off | 0.00551 | - | 6,685 | 99.6% / 98.5% | 0.94s | 1.05s |

- The table above is the latest run. Pooled over 4 runs (n=48 per variant), first answer p50 / p95: Responses 0.68s / 2.34s vs Chat 1.03s / 4.05s **[measured]**. Chat always ran first in each run, so part of the gap may be warm-up **[inference]**.
- Responses sent 15x less data on the last turn **[measured]**.
- Responses cost more in every run: 2.45 / 2.48 / 2.65 / 2.41x Chat **[measured]**.
- Instructions sent only on turn 1: turn 2's cache broke in 2 of 4 runs (cached share per run: 0% / 0% / 49% / 49%) **[measured]**. Inconsistent, so it's a question for Coral, not a conclusion.
- Why it costs more: Responses billed writes at $0.2300/M despite `ttl: "10m"` (Chat: $0.0767/M) **[measured]**. Coral's docs say cache options apply to both APIs **[docs]**.

_Source: `20261006-000442-api_compare-509331.jsonl`, `20261006-013415-api_compare-08a4b4.jsonl`, `20261006-013625-api_compare-a4d8e3.jsonl`, `20261006-013803-api_compare-29c398.jsonl`_

## H5: speed under concurrency

![H5 chart](h5_tps_vs_concurrency.png)

![Latency percentiles](latency_percentiles.png)

| model | streams | n | per-stream tok/s p50 (min) | total tok/s | total time p50/p95/p99 | 429s |
|---|---|---|---|---|---|---|
| glm_flash | 1 | 9 | 284 (240) | 226 | 7.0/7.4/7.4s | 0 |
| glm_flash | 2 | 18 | 276 (193) | 426 | 6.4/8.2/8.2s | 0 |
| glm_flash | 4 | 36 | 226 (132) | 712 | 7.2/13.3/13.3s | 0 |
| glm_flash | 8 | 72 | 188 (131) | 1175 | 9.1/11.9/11.9s | 0 |
| deepseek_flash | 1 | 9 | 408 (291) | 284 | 2.7/3.8/3.8s | 0 |
| deepseek_flash | 2 | 18 | 301 (95) | 439 | 2.8/7.9/7.9s | 0 |
| deepseek_flash | 4 | 36 | 301 (139) | 741 | 3.4/5.5/5.8s | 0 |
| deepseek_flash | 8 | 72 | 258 (140) | 1322 | 3.4/5.1/5.6s | 0 |

- Pooled over 3 runs. With few samples, p99 is effectively the slowest request.
- Per-stream speed is measured from first to last token. Total output per second uses wall-clock time per round, including the wait for the first token, so at 1 stream it is lower than the per-stream speed.
- Coral's homepage: "GLM 5.3 and DeepSeek V4.1 at up to 469 output tokens per second" **[docs]**.
- Single stream, DeepSeek V4.1 Flash: p50 408 tok/s, range 291-608 (n=9) **[measured]**.
- Single stream, GLM 5.3 Flash: p50 284 tok/s, range 240-325 (n=9) **[measured]**.

_Source: `20261005-235603-concurrency-959236.jsonl`, `20261006-014127-concurrency-696072.jsonl`, `20261006-014443-concurrency-e135cb.jsonl`_

## Notes and limitations

- Outside-in testing: we can't see Coral's infrastructure. Results reflect their production service on the test dates above and may change.
- One student, a self-imposed $5 total budget, at most 8 parallel streams, no stress testing. Repeated runs: sequential x3, long_context x2, api_compare x4, concurrency x3; fan-out, early edit and each idle check are single observations.
- The discounted-reads comparison is modeled on Coral's own base prices; it does not name or quote any other provider.
- Out of scope: answer quality of 4-bit models vs full precision (open question).

## Runs used

- `20261005-225939-sequential-4382da.jsonl`
- `20261005-230550-long_context-cbfeee.jsonl`
- `20261005-231221-fanout-642d3c.jsonl`
- `20261005-231533-idle_gaps-8d4480.jsonl`
- `20261005-235235-early_edit-195b1b.jsonl`
- `20261005-235603-concurrency-959236.jsonl`
- `20261006-000442-api_compare-509331.jsonl`
- `20261006-013415-api_compare-08a4b4.jsonl`
- `20261006-013625-api_compare-a4d8e3.jsonl`
- `20261006-013803-api_compare-29c398.jsonl`
- `20261006-014127-concurrency-696072.jsonl`
- `20261006-014443-concurrency-e135cb.jsonl`
- `20261006-014912-long_context-446e9e.jsonl`
- `20261006-015350-sequential-1cd41e.jsonl`
- `20261006-015743-sequential-9f90c6.jsonl`
- `20261006-020322-idle_followup-954ff7.jsonl`
