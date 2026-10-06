# Runbook: AgentLoadCacheReuseLow

**Alert:** prompt cache reuse below 50% for a scenario, averaged over the last minute (at least
3 turns), for 30 seconds. Defined in `alerts/cache_health.yml`.

## What it means

Each turn of an agent conversation should reuse almost all of the previous prompt from the
prefix cache. Reuse = cached tokens ÷ previous prompt tokens in the same conversation. Healthy
loops measured 97-100% (small prompts ~91% because the cache stores 64-token chunks). Below 50%,
most of every prompt is being written to the cache again.

## Why it matters (measured in this project, Oct 2026, GLM 5.3 Flash)

- A timestamp in the system prompt broke the cache on every turn: cost per turn grew linearly
  with the conversation instead of staying flat; a 20-turn run cost **7.8x** the healthy control,
  a 15-turn demo **6.7x**; time to first token rose from ~0.3s to ~1.8s.
- Nothing errors. Replies look normal. Without this alert the only signal is the bill.

## Confirm it (2 minutes)

1. Grafana → **AgentLoad** dashboard → **Cache reuse** panel: is the scenario's line near 0%
   (persistent) or did it dip once and recover (an early edit, which is normal)?
2. **Cost per request** panel: a broken cache shows cost per request climbing turn after turn.
3. Prometheus: `agentload:cache_reuse:ratio_1m` and `ALERTS{alertname="AgentLoadCacheReuseLow"}`
   at http://localhost:9090.
4. Raw evidence from the request log (no API calls):
   ```bash
   python - << 'PY'
   import json, glob
   path = sorted(glob.glob("results/*.jsonl"))[-1]
   rows = [json.loads(l) for l in open(path)]
   for prev, cur in zip(rows, rows[1:]):
       if prev["prompt_tokens"] and cur.get("cached_tokens") is not None:
           print(cur["scenario"], cur["turn"], cur["prompt_tokens"], cur["cached_tokens"],
                 f"{cur['cached_tokens'] / prev['prompt_tokens']:.1%}")
   PY
   ```
   A cached count stuck at a small constant (here it was 64 every turn) means only the very
   start of the prompt matches: something early in the prompt changes every turn.

## Likely causes, most likely first

| Cause | How to check | Fix |
|---|---|---|
| A changing value early in the prompt (timestamp, request id, random tool order) | Diff two consecutive prompts; look at the first ~100 tokens | Move changing values to the end of the prompt, or remove them |
| Responses API without `instructions` on every turn | Turn 2 cached 0; later turns lag one turn behind | Send the same `instructions` with every `previous_response_id` call |
| History rewritten every turn (summaries, edits near the start) | Many dips, not one | Append instead of rewriting; edit late in the prompt |
| Idle gap longer than the cache retention | Gap before the drop; first turn after the gap cached 0 | Expected; consider a longer `prompt_cache_options.ttl` (observed: default and "60m" both gone by 35 min) |
| Provider-side eviction or incident | **All** scenarios drop at the same moment, with no client change | Collect request ids, timestamps and cached counts; report to the provider |

Not a cause: a single early edit. It costs one full re-write, then reuse recovers on the next
turn (measured). The 3-turn minimum and 1-minute average keep it from alerting.

## Tuning

- Threshold 0.5: well below healthy (≥91%), well above broken (~0-2%).
- `for: 30s` is for demos. In production use about `5m` to avoid paging on short blips.
- To test the alert end to end: `./scripts/obs_up.sh`, then
  `PYTHONPATH=. python -m agentload.runner scenarios/dashboard_demo.yaml --metrics-port 9108`
  (about $0.02). The `system_timestamp` variant triggers it; `control` does not.
